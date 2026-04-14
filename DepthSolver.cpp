/********************************************************************
file base:      DepthSolver.cpp
author:         LZD
created:        2025/03/12
purpose:
*********************************************************************/
#include "DepthSolver.h"

#include <opencv2/xfeatures2d.hpp>
#include <random>
#include "boost/filesystem.hpp"
#include "Util/tinyxml2.h"
#include "Util/Logger.h"
#include "Util/Distribution_analyzer.h"
#include "Util/LightFieldPlotter.h"

#include "MINeighbor/SelectNeighbors.h"
#include "Util/ImageQualityEstimate.h"
#include "MIStereo/MIStereoMatch.h"
#include "FocalStack/LFRefocus.h"
#include "FocalStack/VirtualToRealDepthFunc.h"

namespace LFMVS {

    // 计算85% 密度
    std::pair<float, float>
    findHighDensityInterval(KDE &kde, float min_x, float max_x, int num_points = 1000, float target_mass = 0.85) {
        std::vector<kde_point> sampled;

        float dx = (max_x - min_x) / num_points;
        for (int i = 0; i <= num_points; ++i) {
            float x = min_x + i * dx;
            float p = kde(x);
            sampled.push_back({x, p});
        }

        // 按密度降序排序
        std::sort(sampled.begin(), sampled.end(), [](const kde_point &a, const kde_point &b) {
            return a.density > b.density;
        });

        // 累加概率密度面积
        float cumulative_mass = 0.0;
        std::vector<float> selected_x;
        for (const auto &pt: sampled) {
            cumulative_mass += pt.density * dx;
            selected_x.push_back(pt.x);
            if (cumulative_mass >= target_mass)
                break;
        }

        // 找到包含85%概率质量的区间
        auto [min_it, max_it] = std::minmax_element(selected_x.begin(), selected_x.end());
        return {*min_it, *max_it};
    }

    // 众数
    float findKDEMode(KDE &kde, float min_x, float max_x, int num_points = 1000) {
        float dx = (max_x - min_x) / num_points;
        float mode_x = min_x;
        float max_density = -1.0;

        for (int i = 0; i <= num_points; ++i) {
            float x = min_x + i * dx;
            float d = kde(x);
            if (d > max_density) {
                max_density = d;
                mode_x = x;
            }
        }
        return mode_x;
    }

    // fastMode
    float FastOMPMode(std::vector<float> &data, float binWidth = 0.01f, float minVal = 0.0f,
                      float maxVal = 4.0f) {
        if (data.empty())
            return std::numeric_limits<float>::quiet_NaN();
        int numBins = static_cast<int>((maxVal - minVal) / binWidth) + 1;
        int numThreads = omp_get_max_threads();

        // 每个线程维护一个私有的局部bin 计数组
        std::vector<std::vector<int>> localBins(numThreads, std::vector<int>(numBins, 0));

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<int> &bins = localBins[tid];
#pragma omp for  nowait
            for (size_t i = 0; i < data.size(); ++i) {
                float val = data[i];
                if (val < minVal || val > maxVal)
                    continue;
                int binIdx = static_cast<int>((val - minVal) / binWidth);
                ++bins[binIdx];
            }
        }
        //  合并所有线程的结果
        std::vector<int> globalBins(numBins, 0);
        for (int t = 0; t < numThreads; ++t) {
            for (int i = 0; i < numBins; ++i) {
                globalBins[i] += localBins[t][i];
            }
        }
        // 找出最大 bin
        int maxIdx = 0;
        for (int i = 1; i < numBins; ++i) {
            if (globalBins[i] > globalBins[maxIdx]) {
                maxIdx = i;
            }
        }
        return minVal + binWidth * maxIdx;
    }

    //////////////////////////////////////////////////////
    DepthSolver::DepthSolver(std::string &strRootPath, std::string &strSavePath)
            : m_iGarbageRows(1), m_bPlannar(false), m_bLRCheck(false), m_strRootPath(strRootPath),
              m_strSavePath(strSavePath), m_RawImage_Width(0), m_RawImage_Height(0), m_eStereoType(eST_ACMH),
              m_top_device(-1), m_MLA_valid_image_count(-1) {
        Init();

        // 处理写出结果的文件夹
        if (m_strSavePath.empty())
            m_strSavePath = m_strRootPath + LF_DEPTH_INTRA_NAME;
        boost::filesystem::path dir_save_path(m_strSavePath);
        if (!boost::filesystem::exists(dir_save_path)) {
            if (!boost::filesystem::create_directory(dir_save_path)) {
                std::cout << "dir failed to create: " << m_strSavePath << std::endl;
            }
        }

        // test
        m_iCount = 0;
        m_iCount_2 = 0;
    }

    void DepthSolver::SetRawImageWidth(int width) {
        m_RawImage_Width = width;
    }

    const int DepthSolver::GetRawImageWidth() {
        return m_RawImage_Width;
    }

    void DepthSolver::SetRawImageHeight(int height) {
        m_RawImage_Height = height;
    }

    const int DepthSolver::GetRawImageHeight() {
        return m_RawImage_Height;
    }

    LightFieldParams DepthSolver::GetParams() {
        return m_Params;
    }
    
    LightFieldParams& DepthSolver::GetLightFieldParams() {
        return m_Params;
    }

    std::string &DepthSolver::GetRootPath() {
        return m_strRootPath;
    }

    void DepthSolver::Init() {
        m_top_device = -1;
        SelectGPUDevice();

        if (!ComputeRawImageFullPath()) {
            return;
        }
        if (!ComputeWhiteImageFullPath()) {
            return;
        }

        m_MLA_info_map.clear();
        m_MIA_problem_map_map.clear();
        m_MIA_dispNormal_map_map.clear();

        m_Colors.push_back(cv::Vec3b(0, 0, 255)); // red
        m_Colors.push_back(cv::Vec3b(0, 255, 0)); // green
        m_Colors.push_back(cv::Vec3b(255, 0, 0)); // blue
        m_Colors.push_back(cv::Vec3b(0, 255, 255)); // yellow
        m_Colors.push_back(cv::Vec3b(255, 255, 0)); // cyan

        m_Colors.push_back(cv::Vec3b(88, 87, 86)); // 象牙黑
        m_Colors.push_back(cv::Vec3b(128, 138, 135)); // 冷灰
        m_Colors.push_back(cv::Vec3b(245, 245, 245)); // 白烟灰
        m_Colors.push_back(cv::Vec3b(156, 102, 31)); // 砖红
        m_Colors.push_back(cv::Vec3b(202, 235, 216)); // 天蓝灰
        m_Colors.push_back(cv::Vec3b(255, 235, 205)); // 杏仁灰
        m_Colors.push_back(cv::Vec3b(156, 102, 31)); // 镉黄
        m_Colors.push_back(cv::Vec3b(227, 207, 87)); // 香蕉黄

        m_Colors.push_back(cv::Vec3b(255, 255, 255)); // white
        //m_Colors.push_back(cv::Vec3b(0,0,0)); // black
    }

    DepthSolver::~DepthSolver() {
        // // 在析构前确保资源已经释放
        // try {
        //     Reset();
        // } catch (...) {
        //     // 忽略析构过程中可能出现的异常
        // }
    }

    void DepthSolver::Reset() {
        // 检查CUDA上下文是否仍然有效
        cudaError_t cuda_status = cudaGetLastError();
        bool cuda_context_valid = (cuda_status != cudaErrorCudartUnloading &&
                                   cuda_status != 0x30 /* cudaErrorNotInitialized */ &&
                                   cuda_status != cudaErrorUnknown);
        
        // 清理ACMP对象，无论CUDA上下文是否有效
        try {
            // 清理problems map，确保每个MLA_Problem对象都调用Release方法
            for (auto& frame_pair : m_MIA_problem_map_map) {
                for (auto& problem_pair : frame_pair.second) {
                    problem_pair.second.Release(); // 显式调用Release方法
                }
                frame_pair.second.clear();
            }
            m_MIA_problem_map_map.clear();
            
            // 清理disparity and normal map
            for (auto& frame_pair : m_MIA_dispNormal_map_map) {
                for (auto& dn_pair : frame_pair.second) {
                    if (dn_pair.second) {
                        dn_pair.second->Release();
                    }
                }
                frame_pair.second.clear();
            }
            m_MIA_dispNormal_map_map.clear();

            m_RawImageMap.clear();
            m_BlurscoreImageMap.clear();
            m_RichnessImageMap.clear();
            m_MLA_info_map.clear();
            
            m_strRawImagePathVec.clear();
            m_WhiteImage.release();
            
            m_raw_image_key_gray.release();
            
            m_disparityRangeMap.clear();
        }
        catch (...) {
            // 忽略资源清理过程中的任何异常
        }
    }

    // CUDA显存信息打印函数
    void DepthSolver::PrintCudaMemoryInfo(const std::string& tag)
    {
        size_t free_mem, total_mem;
        cudaError_t cuda_status = cudaMemGetInfo(&free_mem, &total_mem);

        if (cuda_status == cudaSuccess)
        {
            size_t used_mem = total_mem - free_mem;
            LOG_ERROR( "[CUDA Memory Info][" , tag , "] "
                      , "Total: " , total_mem / (1024 * 1024) , " MB, "
                      , "Free: " , free_mem / (1024 * 1024) , " MB, "
                      , "Used: " , used_mem / (1024 * 1024) , " MB");
        }
        else {
            LOG_ERROR("[CUDA Memory Info][" , tag ,
                        "] Error getting memory info: ",
                        cudaGetErrorString(cuda_status));
        }
    }

    void DepthSolver::PrintMemoryInfo(const std::string& tag)
    {
        if(g_Debug_Static >= 1)
        {
            PrintSystemMemoryInfo(tag);
            PrintCudaMemoryInfo(tag);
        }
    }

    void DepthSolver::PrintSystemMemoryInfo(const std::string& tag)
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;

        long mem_total = 0, mem_free = 0, mem_available = 0;

        while (std::getline(meminfo, line)) {
            if (line.substr(0, 9) == "MemTotal:") {
                mem_total = std::stol(line.substr(10));
            } else if (line.substr(0, 8) == "MemFree:") {
                mem_free = std::stol(line.substr(9));
            } else if (line.substr(0, 13) == "MemAvailable:") {
                mem_available = std::stol(line.substr(14));
            }
        }

        // 获取当前进程的内存使用情况
        long proc_mem_used = 0;
        std::ifstream proc_status("/proc/self/status");
        std::string proc_line;
        while (std::getline(proc_status, proc_line)) {
            if (proc_line.substr(0, 6) == "VmRSS:") {
                // VmRSS是进程实际使用的物理内存
                size_t pos = proc_line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    proc_mem_used = std::stol(proc_line.substr(pos));
                }
                break;
            }
        }

        LOG_ERROR("[System Memory Info][" , tag , "] "
                  , "Total: " , mem_total / 1024 , " MB, "
                  , "Free: " , mem_free / 1024 , " MB, "
                  , "Available: " , mem_available / 1024 , " MB, "
                  , "Process Used: " , proc_mem_used / 1024 , " MB");
    }

    /**
     * 释放当前处理问题所占用的资源
     * 在处理大量微图像时，每个微图像处理完成后应及时释放资源，避免内存和显存累积
     */
    void DepthSolver::ReleaseResources(std::string& strFrame)
    {
        PrintMemoryInfo("Release begin");

        RemoveProblemsAndDisNormals_Frame(strFrame);

        // step1: 释放内存
        try {
            std::map<std::string, cv::Mat>::iterator itrRA = m_RawImageMap.find(strFrame);
            if(itrRA != m_RawImageMap.end())
            {
                itrRA->second.release();
                m_RawImageMap.erase(itrRA);
            }

            std::map<std::string, cv::Mat>::iterator itrB = m_BlurscoreImageMap.find(strFrame);
            if(itrB != m_BlurscoreImageMap.end())
            {
                itrB->second.release();
                m_BlurscoreImageMap.erase(itrB);
            }

            std::map<std::string, cv::Mat>::iterator itrR = m_RichnessImageMap.find(strFrame);
            if(itrR != m_RichnessImageMap.end())
            {
                itrR->second.release();
                m_RichnessImageMap.erase(itrR);
            }

            // 清理该帧相关的其他资源
            m_WhiteImage.release();
            m_raw_image_key_gray.release();
            
            // 清理该帧的视差范围映射
            m_disparityRangeMap.clear();
        }
        catch (...) {
            // 忽略资源清理过程中的任何异常
        }
        
        // 释放显存
        try {
            // 强制进行CUDA垃圾回收
            cudaDeviceSynchronize();
            cudaDeviceReset();
        } catch (...) {
            // 忽略任何可能的异常
        }

        PrintMemoryInfo("Release Finish");
    }

    void DepthSolver::SetPlannar(bool b) {
        m_bPlannar = b;
    }

    bool DepthSolver::GetPlannar() {
        return m_bPlannar;
    }

    void DepthSolver::SetLRCheck(bool b) {
        m_bLRCheck = b;
    }

    bool DepthSolver::GetLRCheck() {
        return m_bLRCheck;
    }

    void DepthSolver::SetStereoType(eStereoType type) {
        m_eStereoType = type;
    }

    eStereoType DepthSolver::GetStereoType() {
        return m_eStereoType;
    }

    // 选择设备
    void DepthSolver::SelectGPUDevice() {
        cudaDeviceProp deviceProp;
        int deviceCount = 0;
        cudaError_t cudaError = cudaGetDeviceCount(&deviceCount);
        int device_Major = 0;
        int device_Min = 0;
        int device_GlobalMemSize = 0;
        for (int i = 0; i < deviceCount; i++) {
            cudaError = cudaGetDeviceProperties(&deviceProp, i);
            if (cudaError != cudaSuccess)
                continue;

            if (deviceProp.major > device_Major) {
                m_top_device = i;
            } else if (deviceProp.major == device_Major) {
                // 当大版本号相同，小版本号大且显存大，才会切换最优device
                if (deviceProp.minor > device_Min) {
                    if (deviceProp.totalGlobalMem / 1024 / 1024 >= device_GlobalMemSize) {
                        m_top_device = i;
                    }
                }
            }
            device_Major = deviceProp.major;
            device_Min = deviceProp.minor;
            device_GlobalMemSize = deviceProp.totalGlobalMem / 1024 / 1024;
        }
#if RELWITHDEBINFO
        VERBOSE("Selected GPU is %d", m_top_device);
#endif
    }

    void DepthSolver::GetDeviceInfo() {
        // 获取 GPU 设备信息
        cudaDeviceProp deviceProp;
        int deviceCount = 0;
        cudaError_t cudaError;
        cudaError = cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            cudaError = cudaGetDeviceProperties(&deviceProp, i);
            if (cudaError != cudaSuccess) {
                continue;
            }

            std::cout << "设备 " << i + 1 << " 的主要属性： " << std::endl;
            std::cout << "设备显卡型号： " << deviceProp.name << std::endl;
            std::cout << "设备全局内存总量（以MB为单位）" << deviceProp.totalGlobalMem / 1024 / 1024 << std::endl;
            std::cout << "设备中一个Block中可用的最大共享内存（以KB为单位）：" << deviceProp.sharedMemPerBlock / 1024
                      << std::endl;
            std::cout << "设备中一个Block中可用的32位寄存器数量：" << deviceProp.regsPerBlock << std::endl;
            std::cout << "设备中一个Block中可包含的最大线程数量：" << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "设备中的计算功能集（Compute Capability）的版本号：" << deviceProp.major << std::endl;
            std::cout << "设备上多处理器的数量" << deviceProp.multiProcessorCount << std::endl;
        }

        // 存在显卡，才进一步选择合适的显卡
        if (deviceCount >= 1) {
            // 定义需要的设备属性
            cudaDeviceProp devicePropDefined;
            memset(&devicePropDefined, 0, sizeof(cudaDeviceProp)); // 设置devicepropDefined的值
            devicePropDefined.major = 5;
            devicePropDefined.minor = 2;

            int devicedChoosed = -1; // 选中的设备ID
            cudaError_t cudaError_choose;
            cudaGetDevice(&devicedChoosed);  //获取当前设备ID
            std::cout << "当前使用设备的编号： " << devicedChoosed << std::endl;

            cudaChooseDevice(&devicedChoosed, &devicePropDefined);  //查找符合要求的设备ID
            std::cout << "满足指定属性要求的设备的编号： " << devicedChoosed << std::endl;

            cudaError_choose = cudaSetDevice(devicedChoosed); //设置选中的设备为下文的运行设备
        }

        // 获取GPU显存使用信息
        size_t mem_avail = 0;
        size_t mem_total = 0;
        cudaMemGetInfo(&mem_avail, &mem_total);
        std::cout << "Device memory total(MB): " << mem_total / 1024 / 1024 << std::endl;
        std::cout << "Device memory avail(MB): " << mem_avail / 1024 / 1024 << std::endl;
        std::cout << "Device memory used(MB): " << (mem_total - mem_avail) / 1024 / 1024 << std::endl;
    }

    int DepthSolver::Otsu(cv::Mat &image) {
        int width = image.cols;//图像的宽度
        int height = image.rows;//图像的长度
        int x = 0, y = 0;
        int pixelCount[256];
        float pixelPro[256];
        int i, j, pixelSum = width * height, threshold = 0;

        uchar *data = (uchar *) image.data;

        //初始化
        for (i = 0; i < 256; i++) {
            pixelCount[i] = 0;
            pixelPro[i] = 0;
        }

        //统计灰度级中每个像素在整幅图像中的个数
        for (i = y; i < height; i++) {
            for (j = x; j < width; j++) {
                pixelCount[data[i * image.step + j]]++;
            }
        }

        //计算每个像素在整幅图像中的比例
        for (i = 0; i < 256; i++) {
            pixelPro[i] = (float) (pixelCount[i]) / (float) (pixelSum);
        }

        //经典ostu算法,得到前景和背景的分割
        //遍历灰度级[0,255],计算出方差最大的灰度值,为最佳阈值
        float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
        for (i = 0; i < 256; i++) {
            w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;

            for (j = 0; j < 256; j++) {
                if (j <= i) //背景部分
                {
                    //以i为阈值分类，第一类总的概率
                    w0 += pixelPro[j];
                    u0tmp += j * pixelPro[j];
                } else       //前景部分
                {
                    //以i为阈值分类，第二类总的概率
                    w1 += pixelPro[j];
                    u1tmp += j * pixelPro[j];
                }
            }

            u0 = u0tmp / w0;        //第一类的平均灰度
            u1 = u1tmp / w1;        //第二类的平均灰度
            u = u0tmp + u1tmp;      //整幅图像的平均灰度
            //计算类间方差
            deltaTmp = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u);
            //找出最大类间方差以及对应的阈值
            if (deltaTmp > deltaMax) {
                deltaMax = deltaTmp;
                threshold = i;
            }
        }
        //返回最佳阈值;
        return threshold;
    }

    bool comp(cv::Point2f a, cv::Point2f b) {
        return a.x < b.x;
    }

    // 中心点排序
    void DepthSolver::SortCenter(std::vector<cv::Point2f> &Center,
                                 std::vector<std::vector<cv::Point2f>> &center_1)//参数分别为：上一个函数求出的中心点向量，双重vector
    {
        std::vector<cv::Point2f> center_row;//每一行的中心点坐标的vector
        for (int i = Center.size() - 1; i > 0; i--) {
            if (std::abs(Center[i].y - Center[i - 1].y) <
                10)//纵坐标相差小于15代表是同一行（）&&(Center[i].x-Center[i-1].x)>3000//横坐标突然变化很大，也可以判断进入下一行，两个可以同时进行判断）
            {

                center_row.push_back(Center[i]);//第一行加入到向量中
            } else//当纵坐标变化大于10，一般可能是30左右时
            {
                center_row.push_back(Center[i]);//将这一行最后一个微透镜中心点加入vector

                //对每行的中心点进行排序
                sort(center_row.begin(), center_row.end(), comp);

                center_1.push_back(center_row);//将每一行的排好序的中心点vector加入到双重向量中
                center_row.clear();//清空每一行的微透镜中心点，用于下一行
            }
        }
    }

    bool DepthSolver::ComputeCenterFromWhiteImage_Special() {
        MLA_InfoVec mla_info_vec;

        std::ifstream infile;
        std::string strCenterPath = m_strRootPath + LF_RAW_DATASET_NAME + MLA_WHITE_CENTERS_INFO_NAME;

        float y_min = -1;
        float y_max = -1;
        int count = 0;
        infile.open(strCenterPath);
        if (infile.is_open()) {
            // 读取文件
            std::string line;
            while (std::getline(infile, line)) {
                if (line.find("#") != std::string::npos)
                    continue;
                std::istringstream iss(line);
                float x = -1.0;
                float y = -1.0;
                iss >> x >> std::ws >> y;

                cv::Point2f center_coord;
                center_coord.x = x;
                center_coord.y = y;


                if (center_coord.y >= 0 && center_coord.y <= m_Params.baseline * m_Params.mla_v_size &&
                    center_coord.x >= 0 && center_coord.x <= m_Params.baseline * m_Params.mla_u_size) {
                    if (count == 0) // 赋初值
                    {
                        y_min = center_coord.y;
                        y_max = center_coord.y;
                    } else {
                        if (y_min > center_coord.y) {
                            y_min = center_coord.y;
                        } else if (y_max < center_coord.y) {
                            y_max = center_coord.y;
                        }
                    }
                    count++;
                    MLA_InfoPtr ptrInfo = std::make_shared<MLA_Info>();
                    ptrInfo->SetCenter(center_coord);
                    mla_info_vec.push_back(ptrInfo);
                } else {
                    int kk = 0;
                }
            }
        }
        infile.close();

        // 中心点坐标排序：行优先
        typedef int ROW_Index;
        std::map<ROW_Index, MLA_InfoVec> ColsCentersMap;

        float delta = 0.0000001;
        MLA_InfoVec Center_Coords_Vec_tmp = mla_info_vec;
        float y_ideal_next = 0.0;
        int y_ideal_next_count = 0;

        int total_count = 0;
        for (int row = 0; row < m_Params.mla_v_size; row++) {
            MLA_InfoVec Col_Centers_Vec;
            float y_ideal = 0.0;

            if (row == 0) {
                y_ideal = y_min;
            } else {
                y_ideal = y_ideal_next / y_ideal_next_count;
                y_ideal = y_ideal + 0.87 * m_Params.baseline;
                y_ideal_next = 0.0;
                y_ideal_next_count = 0;
            }

            for (int index = 0; index < Center_Coords_Vec_tmp.size(); ++index) {
                MLA_InfoPtr ptrInfo = Center_Coords_Vec_tmp.at(index);
                cv::Point2f &center_coord = ptrInfo->GetCenter();

                if (abs(center_coord.y - y_ideal) < m_Params.baseline * 0.5) {
                    Col_Centers_Vec.push_back(ptrInfo);
                    ptrInfo->SetAbandonByArea(true);
                    y_ideal_next += center_coord.y;
                    y_ideal_next_count++;
                }
            }

            // 同一行中所有的坐标值，从小到大排序
            MLA_InfoVec Col_Centers_Sort_Vec;
            std::map<float, int> Sort_coords_map; // 按 x（列）排序
            for (int i = 0; i < Col_Centers_Vec.size(); ++i) {
                MLA_InfoPtr ptrInfo = Col_Centers_Vec.at(i);
                cv::Point2f &coord = ptrInfo->GetCenter();
                Sort_coords_map[coord.x] = i;
            }
            for (std::map<float, int>::iterator itr = Sort_coords_map.begin(); itr != Sort_coords_map.end(); itr++) {
                MLA_InfoPtr ptrInfo = Col_Centers_Vec[itr->second];
                Col_Centers_Sort_Vec.push_back(ptrInfo);
            }
            ColsCentersMap[row] = Col_Centers_Sort_Vec;
            total_count += Col_Centers_Sort_Vec.size();
        }

        for (int index = 0; index < Center_Coords_Vec_tmp.size(); ++index) {
            MLA_InfoPtr ptrInfo = Center_Coords_Vec_tmp.at(index);
            if (ptrInfo->IsAbandonByArea() == false) {
                std::cout << ptrInfo->GetCenter().x << ", " << ptrInfo->GetCenter().y << std::endl;
            }
        }

        std::string strCenterPath_new = m_strRootPath + LF_RAW_DATASET_NAME + "new_" + MLA_WHITE_CENTERS_INFO_NAME;
        std::ofstream outFile(strCenterPath_new);
        outFile << std::fixed;
        outFile << "# row col center_x center_y m_Area. " << std::endl;
        std::map<ROW_Index, MLA_InfoVec>::iterator itrCV = ColsCentersMap.begin();
        for (; itrCV != ColsCentersMap.end(); itrCV++) {
            MLA_InfoVec &Cols_coord_Vec = itrCV->second;
            for (int col = 0; col < Cols_coord_Vec.size(); ++col) {
                MLA_InfoPtr ptrInfo = Cols_coord_Vec[col];
                ptrInfo->SetCol(col);
                ptrInfo->SetRow(itrCV->first);
                ptrInfo->SetAbandonByArea(false);
                outFile << "(" << ptrInfo->GetRow() << "," << ptrInfo->GetCol() << ") " <<
                        ptrInfo->GetCenter().x << " " << ptrInfo->GetCenter().y << " "
                        << ptrInfo->GetArea() << " 0" << std::endl;
                // 存入key中
                QuadTreeTileKeyPtr ptrKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                                                                            ptrInfo->GetCol(), ptrInfo->GetRow());
                m_MLA_info_map[ptrKey] = ptrInfo;
            }
        }
        outFile.close();
        return true;
    }

    bool DepthSolver::ComputeMLACentersFromWhiteImage() {
        // 注：坐标系的原点在左上角，x正轴水平向右，y正轴竖直向下
        m_MLA_info_map.clear();
        if (!ReadWhiteImage()) {
            LOG_ERROR("Read WhiteImage Error!");
            return false;
        }

        // Step: 提取白图像的轮廓
        int threshold_white = Otsu(m_WhiteImage); // 阈值计算，利用otsu
        cv::Mat gray;
        cvtColor(m_WhiteImage, gray, cv::COLOR_BGR2GRAY); // 转化成灰度图像
        cv::Mat thresholded;/* = Mat::zeros(White.size(), White.type());*/
        threshold(gray, thresholded, threshold_white, 255, cv::THRESH_BINARY);//二值化
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(thresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//查找轮廓

        // 设置路径
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibFullPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME + MLA_WHITE_CENTERS_INFO_NAME;

        bool bTest = true;
        if (bTest) {
            cv::Mat output = cv::Mat::zeros(m_WhiteImage.rows, m_WhiteImage.cols, CV_8U);
            for (int i = 0; i < contours.size(); ++i) {
                cv::drawContours(output, contours, i, cv::Scalar(255));
            }

            std::string strCircleFullPath =
                    root_path_parent.string() + LF_CALIB_FOLDER_NAME + MLA_WHITE_CENTERS_CIRCLE_INFO_NAME;
            cv::imwrite(strCircleFullPath, output);
            std::cout << "contours size is: " << contours.size() << std::endl;
        }

        MLA_InfoVec mla_info_vec;
        // 计算白圆的重心坐标
        cv::Point2f left_up_center;
        float y_min = -1;
        float y_max = -1;
        cv::Moments moment; // 矩
        double average_area = 0.0;
        for (int index = 0; index < contours.size(); index++) {
            cv::Mat temp(contours[index]);
            cv::Scalar color(0, 0, 255);
            moment = moments(temp, false);

            float x = -1;
            float y = -1;
            if (moment.m00 != 0) // 除数不能为0
            {
                x = moment.m10 / moment.m00; // 计算重心横坐标
                y = moment.m01 / moment.m00; // 计算重心纵坐标
            }

            cv::Point2f center_coord = cv::Point2f(x, y); // 重心坐标
            if (center_coord.y >= m_Params.baseline * 0.5 &&
                center_coord.y <= m_Params.baseline * m_Params.mla_v_size &&
                center_coord.x >= m_Params.baseline * 0.5 &&
                center_coord.x <= m_Params.baseline * m_Params.mla_u_size) {
                // 计算轮廓面积
                double area = contourArea(contours[index]);  //获得轮廓面积
                average_area += area;

                MLA_InfoPtr ptrInfo = std::make_shared<MLA_Info>();
                ptrInfo->SetArea(area);
                ptrInfo->SetCenter(center_coord);
                mla_info_vec.push_back(ptrInfo);

                circle(m_WhiteImage, center_coord, 2, color, 2, 8);

                if (index == 0) // 赋初值
                {
                    y_min = center_coord.y;
                    y_max = center_coord.y;
                } else {
                    if (y_min > center_coord.y) {
                        y_min = center_coord.y;
                    } else if (y_max < center_coord.y) {
                        y_max = center_coord.y;
                    }
                }
            }
        }
        average_area /= contours.size();
        std::cout << "contours size is: " << contours.size() << std::endl;

        // 中心点坐标排序：行优先
        std::map<g_row_index, MLA_InfoVec> ColsCentersMap;
        float delta = 0.0000001;
        float y_ideal_next = 0.0;
        int y_ideal_next_count = 0;
        for (int row = 0; row < m_Params.mla_v_size; row++) {
            MLA_InfoVec Col_Centers_Vec;
            float y_ideal = 0.0;

            if (row == 0) {
                y_ideal = m_Params.baseline * 0.5; //-1
            } else {
                y_ideal = y_ideal_next / y_ideal_next_count;
                y_ideal += m_Params.baseline;
                y_ideal_next = 0.0;
                y_ideal_next_count = 0;
            }

            int centers_count = mla_info_vec.size();
            for (int index = 0; index < centers_count; ++index) {
                MLA_InfoPtr ptrInfo = mla_info_vec.at(index);
                cv::Point2f &center_coord = ptrInfo->GetCenter();
                if (abs(center_coord.y - y_ideal) <
                    m_Params.baseline * 0.5) //&& m_Center.y > g_MIA_fBase/2 && m_Center.y <(g_MIA_fBase * g_MLA_row)
                {
                    Col_Centers_Vec.push_back(ptrInfo);
                    y_ideal_next += center_coord.y;
                    y_ideal_next_count++;
                }
                //std::cout << 'y_ideal_next_count=      ' << y_ideal_next_count <<std::endl;
            }

            // 同一行中所有的坐标值，从小到大排序
            MLA_InfoVec Col_Centers_Sort_Vec;
            std::map<float, int> Sort_coords_map; // 按x（列）排序
            for (int i = 0; i < Col_Centers_Vec.size(); ++i) {
                MLA_InfoPtr ptrInfo = Col_Centers_Vec.at(i);
                cv::Point2f &coord = ptrInfo->GetCenter();
                Sort_coords_map[coord.x] = i;
            }
            for (std::map<float, int>::iterator itr = Sort_coords_map.begin(); itr != Sort_coords_map.end(); itr++) {
                MLA_InfoPtr ptrInfo = Col_Centers_Vec[itr->second];
                Col_Centers_Sort_Vec.push_back(ptrInfo);
            }
            ColsCentersMap[row] = Col_Centers_Sort_Vec;
        }

        // 创建MLA_Info队列
        std::map<g_row_index, MLA_InfoVec>::iterator itrCV = ColsCentersMap.begin();
        for (; itrCV != ColsCentersMap.end(); itrCV++) {
            MLA_InfoVec &Cols_coord_Vec = itrCV->second;
            for (int col = 0; col < Cols_coord_Vec.size(); ++col) {
                MLA_InfoPtr ptrInfo = Cols_coord_Vec[col];
                ptrInfo->SetCol(col);
                ptrInfo->SetRow(itrCV->first);

                // 剔除较小面积的轮廓
                if (ptrInfo->GetArea() < average_area * 0.8) // 0.7
                {
                    ptrInfo->SetAbandonByArea(true);
                } else {
                    ptrInfo->SetAbandonByArea(false);
                }
                // 存入key中
                QuadTreeTileKeyPtr ptrKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                                                                            ptrInfo->GetCol(), ptrInfo->GetRow());
                m_MLA_info_map[ptrKey] = ptrInfo;
            }
        }
        // 写出
        bool bWrite = WriteCalibrationParamsXML(strCalibFullPath);
        return bWrite;
    }

    bool DepthSolver::ReadIntrinsicsFromXML() {
        // 准备路径
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibXMLFullPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME + LF_INTRINSICS_NAME;

        tinyxml2::XMLDocument xmlDoc;
        tinyxml2::XMLError eResult = xmlDoc.LoadFile(strCalibXMLFullPath.c_str());
        if (eResult != tinyxml2::XML_SUCCESS) {
            std::cerr << "Failed to load XML file: " << strCalibXMLFullPath.c_str() << ", " << xmlDoc.ErrorID()
                      << std::endl;
            return false;
        }
        // 获取根节点
        tinyxml2::XMLElement *pRoot = xmlDoc.RootElement();
        if (pRoot == nullptr) {
            std::cerr << "Failed to get root element" << std::endl;
            return false;
        }
        std::string strRootName = pRoot->Name();
        if (strRootName != "_") {
            std::cerr << "Error root element: " << pRoot->Name() << std::endl;
            return false;
        }

        bool bMIA = false;
        // 遍历根元素下的子元素
        for (tinyxml2::XMLElement *pGroupElement = pRoot->FirstChildElement();
             pGroupElement != nullptr; pGroupElement = pGroupElement->NextSiblingElement()) {
            //if (bMIA)
            //break;
            std::string strGroupName = pGroupElement->Name();
            if (strGroupName == "D") {
                std::string strTmp = pGroupElement->GetText();
                sscanf(strTmp.c_str(), "%f", &LFMVS::g_bl0);
            } else if (strGroupName == "d") {
                std::string strTmp = pGroupElement->GetText();
                sscanf(strTmp.c_str(), "%f", &LFMVS::g_B);
            } else if (strGroupName == "sensor") {
                for (tinyxml2::XMLElement *pElement = pGroupElement->FirstChildElement();
                     pElement != nullptr; pElement = pElement->NextSiblingElement()) {
                    std::string strName = pElement->Name();
                    if (strName == "pose") {
                        for (tinyxml2::XMLElement *pChildElement = pElement->FirstChildElement();
                             pChildElement != nullptr; pChildElement = pChildElement->NextSiblingElement()) {
                            std::string strChildName = pChildElement->Name();
                            if (strChildName == "rotation") {
                                tinyxml2::XMLElement *pChild_1 = pChildElement->FirstChildElement();
                                if (pChild_1 != nullptr) {
                                    std::string strChild1 = pChild_1->Name();
                                    if (strChild1 == "_") {
                                        // 第0行 第0列
                                        tinyxml2::XMLElement *pChild_00 = pChild_1->FirstChildElement();
                                        std::string strTmp00 = pChild_00->GetText();
                                        sscanf(strTmp00.c_str(), "%lf", &m_Params.sensor_rotation(0, 0));

                                        // 第1行 第0列
                                        tinyxml2::XMLElement *pChild_10 = pChild_00->NextSiblingElement();
                                        std::string strTmp10 = pChild_10->GetText();
                                        sscanf(strTmp10.c_str(), "%lf", &m_Params.sensor_rotation(1, 0));

                                        // 第2行 第0列
                                        tinyxml2::XMLElement *pChild_20 = pChild_10->NextSiblingElement();
                                        std::string strTmp20 = pChild_20->GetText();
                                        sscanf(strTmp20.c_str(), "%lf", &m_Params.sensor_rotation(2, 0));
                                    }
                                }
                                tinyxml2::XMLElement *pChild_2 = pChild_1->NextSiblingElement();
                                if (pChild_2 != nullptr) {
                                    //第1列
                                    std::string strChild1Name = pChild_2->Name();
                                    if (strChild1Name == "_") {
                                        // 第0行 第1列
                                        tinyxml2::XMLElement *pChild_01 = pChild_2->FirstChildElement();
                                        std::string strTmp01 = pChild_01->GetText();
                                        sscanf(strTmp01.c_str(), "%lf", &m_Params.sensor_rotation(0, 1));

                                        // 第1行 第1列
                                        tinyxml2::XMLElement *pChild_11 = pChild_01->NextSiblingElement();
                                        std::string strTmp11 = pChild_11->GetText();
                                        sscanf(strTmp11.c_str(), "%lf", &m_Params.sensor_rotation(1, 1));

                                        // 第2行 第1列
                                        tinyxml2::XMLElement *pChild_21 = pChild_11->NextSiblingElement();
                                        std::string strTmp21 = pChild_21->GetText();
                                        sscanf(strTmp21.c_str(), "%lf", &m_Params.sensor_rotation(2, 1));
                                    }
                                }
                                tinyxml2::XMLElement *pChild_3 = pChild_2->NextSiblingElement();
                                if (pChild_3 != nullptr) {
                                    //第1列
                                    std::string strChild3Name = pChild_3->Name();
                                    if (strChild3Name == "_") {
                                        // 第0行 第1列
                                        tinyxml2::XMLElement *pChild_02 = pChild_3->FirstChildElement();
                                        std::string strTmp02 = pChild_02->GetText();
                                        sscanf(strTmp02.c_str(), "%lf", &m_Params.sensor_rotation(0, 2));

                                        // 第1行 第1列
                                        tinyxml2::XMLElement *pChild_12 = pChild_02->NextSiblingElement();
                                        std::string strTmp12 = pChild_12->GetText();
                                        sscanf(strTmp12.c_str(), "%lf", &m_Params.sensor_rotation(1, 2));

                                        // 第2行 第1列
                                        tinyxml2::XMLElement *pChild_22 = pChild_12->NextSiblingElement();
                                        std::string strTmp22 = pChild_22->GetText();
                                        sscanf(strTmp22.c_str(), "%lf", &m_Params.sensor_rotation(2, 2));
                                    }
                                }
                            } else if (strChildName == "translation") {
                                tinyxml2::XMLElement *pChild_1 = pChildElement->FirstChildElement();
                                if (pChild_1 != nullptr) {
                                    std::string strChild1 = pChild_1->Name();
                                    if (strChild1 == "_") {
                                        tinyxml2::XMLElement *pChild_0 = pChild_1->FirstChildElement();
                                        std::string strTmp0 = pChild_0->GetText();
                                        sscanf(strTmp0.c_str(), "%lf", &m_Params.sensor_translate[0]);

                                        tinyxml2::XMLElement *pChild_1 = pChild_0->NextSiblingElement();
                                        std::string strTmp1 = pChild_1->GetText();
                                        sscanf(strTmp1.c_str(), "%lf", &m_Params.sensor_translate[1]);

                                        tinyxml2::XMLElement *pChild_2 = pChild_1->NextSiblingElement();
                                        std::string strTmp2 = pChild_2->GetText();
                                        sscanf(strTmp2.c_str(), "%lf", &m_Params.sensor_translate[2]);
                                    }
                                }
                            }
                        }
                    } else if (strName == "scale") {
                        std::string strTmp = pElement->GetText();
                        sscanf(strTmp.c_str(), "%f", &m_Params.sensor_pixel_size);
                    }
                }
            } else if (strGroupName == "main_lens") {
                for (tinyxml2::XMLElement *pElement = pGroupElement->FirstChildElement();
                     pElement != nullptr; pElement = pElement->NextSiblingElement()) {
                    std::string strName = pElement->Name();
                    if (strName == "f") {
                        //tinyxml2::XMLElement* pChild = pElement->FirstChildElement();
                        std::string strTmp = pElement->GetText();
                        // 单位：毫米
                        sscanf(strTmp.c_str(), "%f", &m_Params.mainlen_flocal_length);
                    }
                }
            } else if (strGroupName == "mia") {
                for (tinyxml2::XMLElement *pElement = pGroupElement->FirstChildElement();
                     pElement != nullptr; pElement = pElement->NextSiblingElement()) {
                    std::string strName = pElement->Name();
                    if (strName == "mesh") {
                        for (tinyxml2::XMLElement *pChildElement = pElement->FirstChildElement();
                             pChildElement != nullptr;
                             pChildElement = pChildElement->NextSiblingElement()) {
                            std::string strChildName = pChildElement->Name();
                            if (strChildName == "height") {
                                std::string strTmp = pChildElement->GetText();
                                sscanf(strTmp.c_str(), "%d", &m_Params.mla_v_size);
                            } else if (strChildName == "width") {
                                std::string strTmp = pChildElement->GetText();
                                sscanf(strTmp.c_str(), "%d", &m_Params.mla_u_size);
                            } else if (strChildName == "pitch") {
                                for (tinyxml2::XMLElement *pChildChildElement = pChildElement->FirstChildElement();
                                     pChildChildElement != nullptr;
                                     pChildChildElement = pChildChildElement->NextSiblingElement()) {
                                    std::string strChildChildName = pChildChildElement->Name();
                                    if (strChildChildName == "_") {
                                        // LZD: 目前，标定的内参文件中包含了x方向和y方向两个基线长度值。
                                        // 两个值基本一致，因此只需要取一个值作为基线即可
                                        bool bBaseLine = false;
                                        for (tinyxml2::XMLElement *pGrandChildElement = pChildChildElement->FirstChildElement();
                                             pGrandChildElement != nullptr;
                                             pGrandChildElement = pGrandChildElement->NextSiblingElement()) {
                                            std::string strGrandChildName = pGrandChildElement->Name();
                                            if (strChildChildName == "_") {
                                                // 单位：像素
                                                std::string strTmp = pGrandChildElement->GetText();
                                                sscanf(strTmp.c_str(), "%f", &m_Params.baseline);
                                                bBaseLine = true;
                                            }
                                        }
                                        //if (bBaseLine)
                                        //{
                                        //    break;
                                        //}
                                    }
                                }
                            }
                        }
                    }
                }
                bMIA = true;
            } else if (strGroupName == "distortions") {
                for (tinyxml2::XMLElement *pElement = pGroupElement->FirstChildElement();
                     pElement != nullptr; pElement = pElement->NextSiblingElement()) {
                    std::string strName = pElement->Name();
                    if (strName == "depth") {
                        tinyxml2::XMLElement *pChild = pElement->FirstChildElement();

                        tinyxml2::XMLElement *pChild_0 = pChild->FirstChildElement();
                        std::string strTmp0 = pChild_0->GetText();
                        sscanf(strTmp0.c_str(), "%lf", &m_Params.distor_depth(0));

                        tinyxml2::XMLElement *pChild_1 = pChild_0->NextSiblingElement();
                        std::string strTmp1 = pChild_1->GetText();
                        sscanf(strTmp1.c_str(), "%lf", &m_Params.distor_depth(1));

                        tinyxml2::XMLElement *pChild_2 = pChild_1->NextSiblingElement();
                        std::string strTmp2 = pChild_2->GetText();
                        sscanf(strTmp2.c_str(), "%lf", &m_Params.distor_depth(2));
                    } else if (strName == "radial") {
                        tinyxml2::XMLElement *pChild = pElement->FirstChildElement();

                        tinyxml2::XMLElement *pChild_0 = pChild->FirstChildElement();
                        std::string strTmp0 = pChild_0->GetText();
                        sscanf(strTmp0.c_str(), "%lf", &m_Params.distor_radial(0));

                        tinyxml2::XMLElement *pChild_1 = pChild_0->NextSiblingElement();
                        std::string strTmp1 = pChild_1->GetText();
                        sscanf(strTmp1.c_str(), "%lf", &m_Params.distor_radial(1));

                        tinyxml2::XMLElement *pChild_2 = pChild_1->NextSiblingElement();
                        std::string strTmp2 = pChild_2->GetText();
                        sscanf(strTmp2.c_str(), "%lf", &m_Params.distor_radial(2));
                    } else if (strName == "tangential") {
                        tinyxml2::XMLElement *pChild = pElement->FirstChildElement();

                        tinyxml2::XMLElement *pChild_0 = pChild->FirstChildElement();
                        std::string strTmp0 = pChild_0->GetText();
                        sscanf(strTmp0.c_str(), "%lf", &m_Params.distor_tangential(0));

                        tinyxml2::XMLElement *pChild_1 = pChild_0->NextSiblingElement();
                        std::string strTmp1 = pChild_1->GetText();
                        sscanf(strTmp1.c_str(), "%lf", &m_Params.distor_tangential(1));
                    }
                }
            }
        }
        return true;
    }

    bool DepthSolver::ReadCalibrationParamsFromXML() {
        // 准备路径
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibXMLFullPath =
                root_path_parent.string() + LF_CALIB_FOLDER_NAME + MLA_WHITE_CENTERS_INFO_NAME;

        tinyxml2::XMLDocument xmlDoc;
        tinyxml2::XMLError eResult = xmlDoc.LoadFile(strCalibXMLFullPath.c_str());
        if (eResult != tinyxml2::XML_SUCCESS) {
            std::cerr << "Failed to load XML file: " << strCalibXMLFullPath.c_str() << ", " << xmlDoc.ErrorID()
                      << std::endl;
            return false;
        }

        // 获取根节点
        tinyxml2::XMLElement *pRoot = xmlDoc.RootElement();
        if (pRoot == nullptr) {
            std::cerr << "Failed to get root element" << std::endl;
            return false;
        }

        std::string strRootName = pRoot->Name();
        if (strRootName != "Calib") {
            std::cerr << "Error root element: " << pRoot->Name() << std::endl;
            return false;
        }

        std::map<int, float> blur_type_radius_map; // <eMLABlurType, 模糊半径>
        // 遍历根元素下的子元素
        for (tinyxml2::XMLElement *pGroupElement = pRoot->FirstChildElement();
             pGroupElement != nullptr; pGroupElement = pGroupElement->NextSiblingElement()) {
            // 输出元素名和文本内容
            std::string strGroupName = pGroupElement->Name();
            if (strGroupName == "BlurInfo") {
                for (tinyxml2::XMLElement *pElement = pGroupElement->FirstChildElement();
                     pElement != nullptr; pElement = pElement->NextSiblingElement()) {
                    std::string strName = pElement->Name();
                    if (strName == "pose") {
                        // 阵列相对于传感器的姿态：旋转和平移
                        double rotation[4];
                        double translation[2];
                        for (tinyxml2::XMLElement *pChildElement = pElement->FirstChildElement();
                             pChildElement != nullptr; pChildElement = pChildElement->NextSiblingElement()) {
                            std::string strName = pChildElement->Name();
                            std::string strTmp = pChildElement->GetText();
                            if (strName == "r") {
                                sscanf(strTmp.c_str(), "%lf,%lf,%lf,%lf", &rotation[0], &rotation[1], &rotation[2],
                                       &rotation[3]);
                            } else if (strName == "t") {
                                sscanf(strTmp.c_str(), "%lf,%lf", &translation[0], &translation[1]);
                            }
                        }
                    } else if (strName == "MLAType") {
                        std::string text = pElement->GetText();
                        float blur_radius = -1.0;
                        sscanf(text.c_str(), "%f", &blur_radius);
                        std::string strID = pElement->Attribute("id");
                        int type = -1;
                        sscanf(strID.c_str(), "%i", &type);
                        blur_type_radius_map[type] = blur_radius;
                    }
                }
            } else if (strGroupName == "MLAInfo") {
                for (tinyxml2::XMLElement *pElement = pGroupElement->FirstChildElement();
                     pElement != nullptr; pElement = pElement->NextSiblingElement()) {
                    std::string strName = pElement->Name();
                    if (strName == "MLA") {
                        int row = -1;
                        int col = -1;
                        cv::Point2f center_coord;
                        int MLA_type = -1;
                        float area = 0.0;
                        for (tinyxml2::XMLElement *pChilidElement = pElement->FirstChildElement();
                             pChilidElement != nullptr; pChilidElement = pChilidElement->NextSiblingElement()) {
                            std::string strChilidName = pChilidElement->Name();
                            if (strChilidName == "key") {
                                if (pChilidElement->FirstChild()) {
                                    std::string strTmp = pChilidElement->GetText();
                                    sscanf(strTmp.c_str(), "%i,%i", &row, &col);
                                }
                            } else if (strChilidName == "center") {
                                if (pChilidElement->FirstChild()) {
                                    std::string strTmp = pChilidElement->GetText();
                                    sscanf(strTmp.c_str(), "%f,%f", &(center_coord.x), &(center_coord.y));
                                }
                            } else if (strChilidName == "MLAType") {
                                if (pChilidElement->FirstChild()) {
                                    std::string strTmp = pChilidElement->GetText();
                                    sscanf(strTmp.c_str(), "%i", &MLA_type);
                                }
                            } else if (strChilidName == "aera") {
                                if (pChilidElement->FirstChild()) {
                                    std::string strTmp = pChilidElement->GetText();
                                    sscanf(strTmp.c_str(), "%f", &area);
                                }
                            }
                        }

                        MLA_InfoPtr ptrInfo = std::make_shared<MLA_Info>();
                        ptrInfo->SetCol(col);
                        ptrInfo->SetRow(row);
                        ptrInfo->SetCenter(center_coord);
                        ptrInfo->SetArea(area);
                        float blur_radius = blur_type_radius_map[MLA_type];
                        ptrInfo->SetBlurRadius(blur_radius);
                        switch (MLA_type) {
                            case 0:
                                ptrInfo->SetBlurType(eBT_Level0);
                                break;
                            case 1:
                                ptrInfo->SetBlurType(eBT_Level1);
                                break;
                            case 2:
                                ptrInfo->SetBlurType(eBT_Level2);
                                break;
                            case 3:
                                ptrInfo->SetBlurType(eBT_Level3);
                                break;
                            case 4:
                                ptrInfo->SetBlurType(eBT_Level4);
                                break;
                            case 5:
                                ptrInfo->SetBlurType(eBT_Level5);
                                break;
                            default:
                                break;
                        }
                        ptrInfo->SetBlurRadius(blur_type_radius_map[MLA_type]);
                        QuadTreeTileKeyPtr ptrKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0, col, row);
                        m_MLA_info_map[ptrKey] = ptrInfo;
                    }
                }
            } else {
                std::cout << "error group element" << std::endl;
            }
        }
        std::cout << "MLA_Infos_map size: " << GetMLAInfoMap().size() << std::endl;
        return true;
    }

    bool DepthSolver::WriteCalibrationParamsXML(std::string &strCalibFullPath) {
        if (m_MLA_info_map.empty())
            return false;

        // 写出XML
        char temp[256];
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLElement *root = doc.NewElement("Calib");
        doc.InsertFirstChild(root);

        // 模糊信息
        tinyxml2::XMLElement *pBlurInfo = doc.NewElement("BlurInfo");
        root->InsertEndChild(pBlurInfo);
        for (int index = 0; index < 3; index++) {
            tinyxml2::XMLElement *pBlurType = doc.NewElement("MLAType");
            pBlurInfo->InsertEndChild(pBlurType);

            std::ostringstream oss;
            oss << index;
            std::string strRadius = oss.str();

            pBlurType->SetAttribute("id", strRadius.c_str());
            tinyxml2::XMLText *pBlurType_text = doc.NewText(strRadius.c_str());
            pBlurType->InsertEndChild(pBlurType_text);
        }

        // MLA info
        tinyxml2::XMLElement *pMLAGroupInfo = doc.NewElement("MLAInfo");
        root->InsertEndChild(pMLAGroupInfo);
        std::ostringstream oss_output;
        std::string strTxt = "";
        for (QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.begin(); itr != m_MLA_info_map.end(); itr++) {
            QuadTreeTileKeyPtr ptrKey = itr->first;
            MLA_InfoPtr ptrInfo = itr->second;
            // 剔除较小面积的轮廓
            if (ptrInfo->IsAbandonByArea()) // 0.7
                continue;

            // 创建MLA(微透镜图像)节点
            tinyxml2::XMLElement *pMLA = doc.NewElement("MLA");
            pMLAGroupInfo->InsertEndChild(pMLA);

            // 行列号
            tinyxml2::XMLElement *pMLA_key = doc.NewElement("key");
            pMLA->InsertEndChild(pMLA_key);
            oss_output.str("");
            strTxt = "";
            oss_output << ptrInfo->GetRow() << "," << ptrInfo->GetCol();
            strTxt = oss_output.str();
            tinyxml2::XMLText *pText = doc.NewText(strTxt.c_str());
            pMLA_key->InsertEndChild(pText);

            // 中心点坐标
            tinyxml2::XMLElement *pMLA_center = doc.NewElement("center");
            pMLA->InsertEndChild(pMLA_center);
            oss_output.str("");
            strTxt = "";
            oss_output << ptrInfo->GetCenter().x << "," << ptrInfo->GetCenter().y;
            strTxt = oss_output.str();
            tinyxml2::XMLText *pCenterText = doc.NewText(strTxt.c_str());
            pMLA_center->InsertEndChild(pCenterText);

            // 微透镜类型
            tinyxml2::XMLElement *pMLA_Type = doc.NewElement("MLAType");
            pMLA->InsertEndChild(pMLA_Type);
            tinyxml2::XMLText *pTypeText = doc.NewText("1");
            pMLA_Type->InsertEndChild(pTypeText);

            // 面积
            tinyxml2::XMLElement *pMLA_Aera = doc.NewElement("aera");
            pMLA->InsertEndChild(pMLA_Aera);
            oss_output.str("");
            strTxt = "";
            oss_output << ptrInfo->GetArea();
            strTxt = oss_output.str();
            tinyxml2::XMLText *pAeraText = doc.NewText(strTxt.c_str());
            pMLA_Aera->InsertEndChild(pAeraText);
        }

        // 将XML文档保存到文件中
        tinyxml2::XMLError eResult = doc.SaveFile(strCalibFullPath.c_str());
        if (eResult != tinyxml2::XML_SUCCESS) {
            std::cerr << "Failed to save file: " << doc.ErrorID() << std::endl;
            return false;
        }
        return true;
    }

    bool DepthSolver::GetOrComputeMLAInfo(eParseMLACentersType parse_type) {
        bool bReadIntri = ReadIntrinsicsFromXML();
        if (!bReadIntri) {
            LOG_ERROR("ReadIntrinsicsFromXML Failed!");
            return false;
        }

        // 创建绘图对象并绘制
        // CameraParams params;
        // params.f = m_Params.mainlen_flocal_length;
        // params.bL0 = g_bl0;
        // params.D = g_B;
        // params.p = m_Params.baseline*m_Params.sensor_pixel_size;
        // LightFieldPlotter plotter(params, m_strRootPath);
        // plotter.plotRelations(1, 35);

        // 根据基线计算微透镜图像中，可用于深度估计的像素区域
        m_Params.ComputeMIA_Math_Info();
        switch (parse_type) {
            case ePMLACT_wts: // wts给索引方式
            {
                bool bCompute = ComputeCenterFromWhiteImage_Special();
                if (!bCompute) {
                    LOG_ERROR("Compute Center From WhiteImage Error!");
                    return false;
                }
            }
                break;
            case ePMLACT_Auto || ePMLACT_ParseFromCalib: {
                // 先尝试从标定结果XML文件中读取，否则直接根据白图像计算中心点坐标
                // 然后，尝试根据白图像自动化计算
                if (!ReadCalibrationParamsFromXML()) {
                    bool bCompute = ComputeMLACentersFromWhiteImage();
                    if (!bCompute) {
                        LOG_ERROR("Compute Center From WhiteImage Error!");
                        return false;
                    }
                }
            }
                break;
            default:
                LOG_ERROR("parse_type Error!");
                break;
        }
        return true;
    }

    bool DepthSolver::GetMLAImages() {
        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strNameLessExt = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;

            std::string strMLAPath = m_strRootPath + LF_DEPTH_INTRA_NAME;
            {
                boost::filesystem::path dir_path(strMLAPath);
                if (!boost::filesystem::exists(dir_path)) {
                    if (!boost::filesystem::create_directory(dir_path)) {
                        std::cout << "dir failed to create: " << strMLAPath << std::endl;
                    }
                }
            }
            strMLAPath += strNameLessExt;

            {
                boost::filesystem::path dir_path(strMLAPath);
                if (!boost::filesystem::exists(dir_path)) {
                    if (!boost::filesystem::create_directory(dir_path)) {
                        std::cout << "dir failed to create: " << strMLAPath << std::endl;
                    }
                }
            }
            strMLAPath += LF_RAW_MLA_IMAGES_NAME;

            bool bWriteMLAImages = false;
            {
                boost::filesystem::path dir_path(strMLAPath);
                if (!boost::filesystem::exists(dir_path)) {
                    if (!boost::filesystem::create_directory(dir_path)) {
                        std::cout << "dir failed to create: " << strMLAPath << std::endl;
                    }
                }
            }

            // 先尝试读取
            if (!ReadMLAImages(strMLAPath, m_MLA_valid_image_count, problem_map))
            {
                // 若读取失败，则一边分割一边存储到内存和外存
                if (!Slice_RawMLAImage(strNameLessExt, strMLAPath, problem_map, bWriteMLAImages))
                {
                    continue;
                }
            }
        }
        return true;
    }

    bool DepthSolver::GetMLAImagesSequence(std::string strNameLessExt,
        QuadTreeProblemMap &problem_map)
    {
        PrintMemoryInfo("GetMLAImagesSequence Begin");

        std::string strMLAPath = m_strRootPath + LF_DEPTH_INTRA_NAME;
        {
            boost::filesystem::path dir_path(strMLAPath);
            if (!boost::filesystem::exists(dir_path)) {
                if (!boost::filesystem::create_directory(dir_path)) {
                    std::cout << "dir failed to create: " << strMLAPath << std::endl;
                }
            }
        }
        strMLAPath += strNameLessExt;

        {
            boost::filesystem::path dir_path(strMLAPath);
            if (!boost::filesystem::exists(dir_path)) {
                if (!boost::filesystem::create_directory(dir_path)) {
                    std::cout << "dir failed to create: " << strMLAPath << std::endl;
                }
            }
        }
        strMLAPath += LF_RAW_MLA_IMAGES_NAME;
        bool bWriteMLAImages = false;
        {
            boost::filesystem::path dir_path(strMLAPath);
            if (!boost::filesystem::exists(dir_path)) {
                if (!boost::filesystem::create_directory(dir_path)) {
                    std::cout << "dir failed to create: " << strMLAPath << std::endl;
                }
            }
        }

        // 先尝试读取
        if (!ReadMLAImages(strMLAPath, m_MLA_valid_image_count, problem_map)) {
            // 若读取失败，则边分割边存储
            if (!Slice_RawMLAImage(strNameLessExt, strMLAPath, problem_map, bWriteMLAImages))
            {
                PrintMemoryInfo("GetMLAImagesSequence-2 Finish");
                return false;
            }
        }
        PrintMemoryInfo("GetMLAImagesSequence Finish");
        return true;
    }

    void DepthSolver::TestRawImageTilekeyWithCircleLine(bool bTest_tilekey, QuadTreeTileKeyPtr ptrCenterKey,
                                                        QuadTreeTileKeyPtrCircles &circleKeyMap) {
        if (!bTest_tilekey)
            return;

        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;

            // 构造写出路径
            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
            std::string strImgFullPath = strSavePath + std::string("/raw_key_color_neighbors.png");
            cv::Mat &raw_image = m_RawImageMap[strName];
            cv::Mat raw_image_key = raw_image.clone();
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;

                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;
                MLA_InfoPtr ptrInfo = itrInfo->second;
                if (ptrInfo->IsAbandonByArea())
                    continue;

                // MLA_Tilekey: 微透镜编码
                // 设置字体和颜色
                int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
                double fontScale = 0.4;                  // 字体大小
                int thickness = 1;                        // 线条粗细
                cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
                // 文字内容
                std::string text = problem.m_ptrKey->StrRemoveLOD();
                // 文字位置，(x, y)为文字左下角的坐标
                cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
                // 将文字写入图片
                cv::putText(raw_image_key, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);

                // 绘制中心点
                cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
                cv::circle(raw_image_key, center_p, 2, (255, 0, 255), -1);
            }

            // 绘制圆圈
            // 自身
            QuadTreeTileInfoMap::iterator itrInfo_center = m_MLA_info_map.find(ptrCenterKey);
            if (itrInfo_center == m_MLA_info_map.end())
                continue;
            MLA_InfoPtr ptrCenterInfo = itrInfo_center->second;

            int circle_index = 0;
            for (QuadTreeTileKeyPtrCircles::iterator itrC = circleKeyMap.begin();
                 itrC != circleKeyMap.end(); itrC++) {
                QuadTreeTileKeyPtrVec &circle_Keys = itrC->second;
                for (int i = 0; i < circle_Keys.size(); i++) {
                    QuadTreeTileKeyPtr ptrKey_c = circle_Keys.at(i);
                    QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrKey_c);
                    if (itrInfo == m_MLA_info_map.end())
                        continue;

                    MLA_InfoPtr ptrInfo = itrInfo->second;
                    cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
                    if (circle_index == 0) {
                        cv::circle(raw_image_key, center_p, 10, cv::Scalar(255, 0, 0), 2);
                    } else if (circle_index == 1) {
                        cv::circle(raw_image_key, center_p, 20, cv::Scalar(0, 255, 0), 20);
                    } else if (circle_index == 2) {
                        cv::circle(raw_image_key, center_p, 20, cv::Scalar(0, 0, 255), 20);
                    } else if (circle_index == 3) {
                        cv::circle(raw_image_key, center_p, 20, cv::Scalar(0, 255, 255), 20);
                    } else if (circle_index == 4) {
                        cv::circle(raw_image_key, center_p, 20, cv::Scalar(255, 0, 255), 20);
                    }

                }
                circle_index++;
            }
            cv::imwrite(strImgFullPath, raw_image_key);
        }
    }

    void DepthSolver::TestRawImageTilekeyWithSortNeighForRefocus(bool bTest_tilekey,
                                                                 QuadTreeTileKeyPtr ptrCenterKey,
                                                                 MLA_Problem &curr_problem) {
        if (!bTest_tilekey)
            return;

        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;

            // 构造写出路径
            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
            std::string strImgFullPath = strSavePath + std::string("/raw_key_color_neighbors_refocus.png");
            cv::Mat &raw_image = m_RawImageMap[strName];
            cv::Mat raw_image_key = raw_image.clone();
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;

                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;
                MLA_InfoPtr ptrInfo = itrInfo->second;
                if (ptrInfo->IsAbandonByArea())
                    continue;

                // MLA_Tilekey: 微透镜编码
                // 设置字体和颜色
                int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
                double fontScale = 0.4;                  // 字体大小
                int thickness = 1;                        // 线条粗细
                cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
                // 文字内容
                std::string text = problem.m_ptrKey->StrRemoveLOD();
                // 文字位置，(x, y)为文字左下角的坐标
                cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
                // 将文字写入图片
                cv::putText(raw_image_key, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);

                // 绘制中心点
                cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
                cv::circle(raw_image_key, center_p, 2, (255, 0, 255), -1);
            }

            // 绘制圆圈
            int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
            double fontScale = 0.5;                  // 字体大小
            int thickness = 2;                        // 线条粗细
            cv::Scalar color = cv::Scalar(0, 128, 255); // 字体颜色，BGR格式
            for (int neig_index = 0; neig_index < curr_problem.m_NeighsSortVecForRefocus.size(); neig_index++) {
                QuadTreeTileKeyPtr ptrNeigKey = curr_problem.m_NeighsSortVecForRefocus.at(neig_index);
                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrNeigKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;

                MLA_InfoPtr ptrInfo = itrInfo->second;
                cv::Point pos_text(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - 5);

                cv::putText(raw_image_key, std::to_string(neig_index), pos_text, fontFace, fontScale, color, thickness,
                            cv::LINE_AA);
            }
            cv::imwrite(strImgFullPath, raw_image_key);
        }
    }

    void DepthSolver::TestRawImageTilekeyWithSortNeighForMatch(bool bTest_tilekey, QuadTreeTileKeyPtr ptrCenterKey,
                                                               MLA_Problem &curr_problem) {
        if (!bTest_tilekey)
            return;

        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;

            // 构造写出路径
            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
            std::string strImgFullPath = strSavePath + std::string("/raw_key_color_neighbors_match.png");
            cv::Mat &raw_image = m_RawImageMap[strName];
            cv::Mat raw_image_key = raw_image.clone();
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;

                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;
                MLA_InfoPtr ptrInfo = itrInfo->second;
                if (ptrInfo->IsAbandonByArea())
                    continue;

                // MLA_Tilekey: 微透镜编码
                // 设置字体和颜色
                int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
                double fontScale = 0.4;                  // 字体大小
                int thickness = 1;                        // 线条粗细
                cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
                // 文字内容
                std::string text = problem.m_ptrKey->StrRemoveLOD();
                // 文字位置，(x, y)为文字左下角的坐标
                cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
                // 将文字写入图片
                cv::putText(raw_image_key, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);

                // 绘制中心点
                cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
                cv::circle(raw_image_key, center_p, 2, (255, 0, 255), -1);
            }

            // 绘制圆圈
            int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
            double fontScale = 0.5;                  // 字体大小
            int thickness = 2;                        // 线条粗细
            cv::Scalar color = cv::Scalar(0, 128, 255); // 字体颜色，BGR格式
            for (int neig_index = 0; neig_index < curr_problem.m_NeighsSortVecForMatch.size(); neig_index++) {
                QuadTreeTileKeyPtr ptrNeigKey = curr_problem.m_NeighsSortVecForMatch.at(neig_index);
                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrNeigKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;

                MLA_InfoPtr ptrInfo = itrInfo->second;
                cv::Point pos_text(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - 5);

                cv::putText(raw_image_key, std::to_string(neig_index), pos_text, fontFace, fontScale, color, thickness,
                            cv::LINE_AA);
            }
            cv::imwrite(strImgFullPath, raw_image_key);
        }
    }

    cv::Mat &DepthSolver::GetWhiteImage() {
        return m_WhiteImage;
    }

    void DepthSolver::TestRawImageTilekey(bool bTest_tilekey) {
        if (!bTest_tilekey)
            return;

        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;

            // 构造写出路径
            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
            std::string strImgFullPath = strSavePath + std::string("/raw_key_color.png");
            std::string strImgFullPath_gray = strSavePath + std::string("/raw_key_gray.png");
            if (boost::filesystem::exists(strImgFullPath) &&
                boost::filesystem::exists(strImgFullPath_gray)) {
                // 文件存在，则无需重新创建，直接处理下一个
                continue;
            }

            cv::Mat &raw_image = m_RawImageMap[strName];
            cv::Mat raw_image_key = raw_image.clone();
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;

                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;
                MLA_InfoPtr ptrInfo = itrInfo->second;
                if (ptrInfo->IsAbandonByArea())
                    continue;

                // MLA_Tilekey: 微透镜编码
                // 设置字体和颜色
                int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
                double fontScale = 0.4;                  // 字体大小
                int thickness = 1;                        // 线条粗细
                cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
                // 文字内容
                std::string text = problem.m_ptrKey->StrRemoveLOD();
                // 文字位置，(x, y)为文字左下角的坐标
                cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
                // 将文字写入图片
                cv::putText(raw_image_key, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);

                // 绘制中心点
                cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
                cv::circle(raw_image_key, center_p, 2, (255, 0, 255), -1);
            }

            cv::imwrite(strImgFullPath, raw_image_key);
            cv::cvtColor(raw_image_key, m_raw_image_key_gray, cv::COLOR_BGR2GRAY);
            cv::imwrite(strImgFullPath_gray, m_raw_image_key_gray);
        }
    }

    void DepthSolver::TestRawImageTilekeySequence(bool bTest_tilekey,
                                                  std::string &strName,
                                                  QuadTreeProblemMap &problem_map) {
        if (!bTest_tilekey)
            return;

        // 构造写出路径
        std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(strSavePath);
            if (!boost::filesystem::exists(dir_save_path)) {
                if (!boost::filesystem::create_directory(dir_save_path)) {
                    std::cout << "dir failed to create: " << strSavePath << std::endl;
                }
            }
        }
        std::string strImgFullPath = strSavePath + std::string("/raw_key.png");
        std::string strImgFullPath_gray = strSavePath + std::string("/raw_key_gray.png");
        if (boost::filesystem::exists(strImgFullPath) &&
            boost::filesystem::exists(strImgFullPath_gray)) {
            // 文件存在，则无需重新创建，直接返回
            return;
        }

        cv::Mat &raw_image = m_RawImageMap[strName];
        cv::Mat raw_image_key = raw_image.clone();
        for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
            MLA_Problem &problem = itr->second;

            QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
            if (itrInfo == m_MLA_info_map.end())
                continue;
            MLA_InfoPtr ptrInfo = itrInfo->second;
            if (ptrInfo->IsAbandonByArea())
                continue;

            // MLA_Tilekey: 微透镜编码
            // 设置字体和颜色
            int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
            double fontScale = 0.4;                  // 字体大小
            int thickness = 1;                        // 线条粗细
            cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
            // 文字内容
            std::string text = problem.m_ptrKey->StrRemoveLOD();
            // 文字位置，(x, y)为文字左下角的坐标
            cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
            // 将文字写入图片
            cv::putText(raw_image_key, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);

            // 绘制中心点
            cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
            cv::circle(raw_image_key, center_p, 2, (255, 0, 255), -1);

            // 绘制圆环：根据标定的微透镜直径
            cv::circle(raw_image_key, center_p, m_Params.baseline * 0.5, (255, 0, 255), 1);
        }

        // 写出
        cv::imwrite(strImgFullPath, raw_image_key);
        cv::Mat raw_image_key_gray;
        cv::cvtColor(raw_image_key, raw_image_key_gray, cv::COLOR_BGR2GRAY);
        cv::imwrite(strImgFullPath_gray, raw_image_key_gray);
    }

    void DepthSolver::RemoveProblemsAndDisNormals_Frame(std::string str_frame)
    {
        // 清理problems map
        QuadTreeProblemMapMap::iterator itrP = m_MIA_problem_map_map.find(str_frame);
        if (itrP != m_MIA_problem_map_map.end())
        {
            QuadTreeProblemMap& problem_map = itrP->second;
            for (auto& pair : problem_map) {
                pair.second.Release(); // 调用MLA_Problem的Release方法
            }
            problem_map.clear();
            //m_MIA_problem_map_map.erase(itrP);
        }

        // 清理disparity and normal map
        QuadTreeDisNormalMapMap::iterator itrD = m_MIA_dispNormal_map_map.find(str_frame);
        if(itrD != m_MIA_dispNormal_map_map.end())
        {
            QuadTreeDisNormalMap& dispNormal_map = itrD->second;
            QuadTreeDisNormalMap::iterator itrN = dispNormal_map.begin();
            for(; itrN != itrD->second.end(); ++itrN)
            {
                itrN->second->Release();
            }
            dispNormal_map.clear();
            //m_MIA_dispNormal_map_map.erase(itrD);
        }
    }

    void DepthSolver::CreateProblemsAndDisNormals_Frame(std::string str_frame)
    {
        PrintMemoryInfo("MIPD Begin");

        // 根据微透镜阵列的硬件信息创建深度估计需要的变量
        // 根据CalibInfo信息统计被抛弃的微透镜
        int total_abandon = 0;
        for (QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.begin(); itr != m_MLA_info_map.end(); itr++) {
            QuadTreeTileKeyPtr ptrkey = itr->first;
            MLA_InfoPtr ptrInfo = itr->second;
            if (ptrInfo->IsAbandonByArea()) {
                total_abandon++;
                continue;
            }
            float x_leftDown = ptrInfo->GetCenter().x - (m_Params.mi_width_for_match - 1) * 0.5;
            float y_leftDown = ptrInfo->GetCenter().y - (m_Params.mi_height_for_match - 1) * 0.5;
            float x_rightUp = ptrInfo->GetCenter().x + (m_Params.mi_width_for_match - 1) * 0.5;
            float y_rightUp = ptrInfo->GetCenter().y + (m_Params.mi_height_for_match - 1) * 0.5;
            if (x_leftDown < 0 || y_leftDown < 0 || x_rightUp > m_Params.mla_u_size * m_Params.baseline
                || y_rightUp > m_Params.mla_v_size * m_Params.baseline) {
                ptrInfo->SetAbandonByArea(true);
                total_abandon++;
                continue;
                }
            ptrInfo->GetLeftDownCorner().x = x_leftDown;
            ptrInfo->GetLeftDownCorner().y = y_leftDown;

            // 创建problem和disNormal
            CreateProblem_frame(ptrkey, str_frame);
            CreateDisNormal_frame(ptrkey, str_frame);
        }
        LOG_ERROR("MLA_Centers_map size is: ", m_MLA_info_map.size(), ", abandon count is " , total_abandon);
        m_MLA_valid_image_count = m_MLA_info_map.size() - total_abandon;

        PrintMemoryInfo("MIPD Finish");
    }

    void DepthSolver::CreateMIAofProblemDisNormals() {
        // 根据微透镜阵列的硬件信息创建深度估计需要的变量
        // 根据CalibInfo信息统计被抛弃的微透镜
        int total_abandon = 0;
        for (QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.begin(); itr != m_MLA_info_map.end(); itr++) {
            QuadTreeTileKeyPtr ptrkey = itr->first;
            MLA_InfoPtr ptrInfo = itr->second;
            if (ptrInfo->IsAbandonByArea()) {
                total_abandon++;
                continue;
            }
            float x_leftDown = ptrInfo->GetCenter().x - (m_Params.mi_width_for_match - 1) * 0.5;
            float y_leftDown = ptrInfo->GetCenter().y - (m_Params.mi_height_for_match - 1) * 0.5;
            float x_rightUp = ptrInfo->GetCenter().x + (m_Params.mi_width_for_match - 1) * 0.5;
            float y_rightUp = ptrInfo->GetCenter().y + (m_Params.mi_height_for_match - 1) * 0.5;
            if (x_leftDown < 0 || y_leftDown < 0 || x_rightUp > m_Params.mla_u_size * m_Params.baseline
                || y_rightUp > m_Params.mla_v_size * m_Params.baseline) {
                ptrInfo->SetAbandonByArea(true);
                total_abandon++;
                continue;
            }
            ptrInfo->GetLeftDownCorner().x = x_leftDown;
            ptrInfo->GetLeftDownCorner().y = y_leftDown;

            // 创建problem和disNormal
            CreateProblems(ptrkey);
            CreateDisNormals(ptrkey);
        }
        std::cout << "MLA_Centers_map size is: " << m_MLA_info_map.size() << ", abandon count is " << total_abandon
                  << std::endl;
        m_MLA_valid_image_count = m_MLA_info_map.size() - total_abandon;
    }

    bool
    DepthSolver::Slice_RawMLAImage(std::string &strName, std::string &strMLAPath,
        QuadTreeProblemMap &problem_map, bool bWriteMLAImages)
    {
        std::map<std::string, cv::Mat>::iterator itrR = m_RawImageMap.find(strName);
        if (itrR == m_RawImageMap.end())
            return false;

        cv::Mat &raw_image = itrR->second;
        int total_abandon = 0;
        for (QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.begin(); itr != m_MLA_info_map.end(); itr++)
        {
            QuadTreeTileKeyPtr ptrKey = itr->first;
            MLA_InfoPtr ptrInfo = itr->second;
            if (ptrInfo->IsAbandonByArea())
            {
                total_abandon++;
                continue;
            }

            QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
            if (itrP != problem_map.end())
            {
                MLA_Problem &problem = itrP->second;
                std::string strSlice_MLAFullPath = strMLAPath + "/" + ptrKey->StrRemoveLOD() + ".png";

                // 切割
                cv::Rect rect(ptrInfo->GetLeftDownCorner().x, ptrInfo->GetLeftDownCorner().y,
                              m_Params.mi_width_for_match, m_Params.mi_height_for_match);
                raw_image(rect).copyTo(problem.m_Image_rgb);

                // 写出微图像。因写出的微图像数量太多且每个微图像占用空间较小，会导致磁盘碎片过多，所以不建议使用
                if (bWriteMLAImages)
                {
                    imwrite(strSlice_MLAFullPath, problem.m_Image_rgb);
                    problem.m_Image_gray = cv::imread(strSlice_MLAFullPath, cv::IMREAD_GRAYSCALE);
                }
                else
                {
                    cv::cvtColor(problem.m_Image_rgb, problem.m_Image_gray, cv::COLOR_BGR2GRAY);
                }
            }
        }

        int slice_count = m_MLA_info_map.size() - total_abandon;
        LOG_ERROR("slice mla_images number: ", slice_count, ", abandon count is " , total_abandon);
        return true;
    }

    void DepthSolver::QuantizeBlurLevelForMI(cv::Mat &gray_MI, std::string &strMI_BlurValue_path) {
        // SMD2算子：灰度方差乘积函数。对每一个像素邻域两个灰度差相乘后再逐个像素累加
        cv::Mat_<float> mi_blur_value = cv::Mat::zeros(gray_MI.rows, gray_MI.cols, CV_32FC1);
        for (int col = 0; col < gray_MI.cols - 1; col++) {
            for (int row = 0; row < gray_MI.rows - 1; row++) {
                float v1 = fabs(gray_MI.at<uchar>(row, col) - gray_MI.at<uchar>(row + 1, col));
                float v2 = fabs(gray_MI.at<uchar>(row, col) - gray_MI.at<uchar>(row, col + 1));
                float va = v1 * v2;
                mi_blur_value.at<float>(row, col) = va;
            }
        }

        std::string strMI_BlurValue_gray_path = strMI_BlurValue_path + "_blur.png";
        imwrite(strMI_BlurValue_gray_path, mi_blur_value);

        //cv::Mat disp_color;
        //applyColorMap(mi_blur_value, disp_color, cv::COLORMAP_JET);
        //std::string strMI_BlurValue_color_path = strMI_BlurValue_path + "_blur_color.png";
        //imwrite(strMI_BlurValue_color_path, disp_color);
    }

    bool DepthSolver::ComputeRawImageFullPath() {
        m_strRawImagePathVec.clear();

        std::string strRawImagesPath = m_strRootPath + LF_RAW_DATASET_NAME;
        // 遍历并搜集文件夹中所有的图像文件
        boost::filesystem::path img_path(strRawImagesPath);
        for (const auto &entry: boost::filesystem::directory_iterator(img_path)) {
            if (boost::filesystem::is_regular_file(entry.status())) // 检查是否为文件
            {
                std::string filename = entry.path().filename().string();
                size_t dot_pos = filename.rfind('.');
                if (dot_pos != std::string::npos) {
                    std::string extension = filename.substr(dot_pos);
                    if (std::find(g_Common_image_formats.begin(), g_Common_image_formats.end(), extension)
                        != g_Common_image_formats.end()) {
                        m_strRawImagePathVec.push_back(strRawImagesPath + filename); // 存储完整路径
                    }
                }
            }
        }

        // for (int i = 0; i < m_strRawImagePathVec.size(); i++)
        // {
        //     boost::filesystem::path path(m_strRawImagePathVec[i]);
        //     std::string strName = path.filename().string();
        //     std::string strExt = path.extension().string();
        //     std::size_t pos = strName.find(strExt);
        //     std::string strNameLessExt = strName.substr(0, pos);
        //     if (strNameLessExt == LF_RAW_IMAGE_NAME)
        //     {
        //         m_strRawImagePath = strPath + strName;
        //         return true;
        //     }
        // }

        if (m_strRawImagePathVec.empty()) {
            std::cout << " not find RawImage: " << strRawImagesPath << std::endl;
            return false;
        }
        return true;
    }

    bool DepthSolver::ComputeWhiteImageFullPath() {
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME;

        // 遍历并搜集文件夹中所有的图像文件
        std::vector<std::string> strFilesPath_vec;
        boost::filesystem::path img_path(strCalibPath);
        for (const auto &entry: boost::filesystem::directory_iterator(img_path)) {
            if (boost::filesystem::is_regular_file(entry.status())) // 检查是否为文件
            {
                std::string filename = entry.path().filename().string();
                size_t dot_pos = filename.rfind('.');
                if (dot_pos != std::string::npos) {
                    std::string extension = filename.substr(dot_pos);
                    if (std::find(g_Common_image_formats.begin(), g_Common_image_formats.end(), extension)
                        != g_Common_image_formats.end()) {
                        strFilesPath_vec.push_back(strCalibPath + filename); // 存储完整路径
                    }
                }
            }
        }

        for (int i = 0; i < strFilesPath_vec.size(); i++) {
            boost::filesystem::path path(strFilesPath_vec[i]);
            std::string strName = path.filename().string();
            std::string strExt = path.extension().string();
            std::size_t pos = strName.find(strExt);
            std::string strNameLessExt = strName.substr(0, pos);
            if (strNameLessExt == LF_WHITE_IMAGE_NAME) {
                m_strWhiteImagePath = strCalibPath + strName;
                return true;
            }
        }
        std::cout << " not find WhiteImage: " << strCalibPath << std::endl;
        return true; // false
    }

    std::vector<std::string> &DepthSolver::GetRawImagePathVec() {
        return m_strRawImagePathVec;
    }

    std::string &DepthSolver::GetWhiteImagePath() {
        return m_strWhiteImagePath;
    }

    std::vector<std::string> DepthSolver::splitStrings(const std::string &str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    bool DepthSolver::ReadMLAImages(std::string strMLAPath, int MLA_valid_image_count,
                                    QuadTreeProblemMap &problem_map) {
        bool bRead = false;

        // 遍历并搜集文件夹中所有的图像文件
        std::vector<std::string> strFilesPath_vec;
        boost::filesystem::path img_path(strMLAPath);
        for (const auto &entry: boost::filesystem::directory_iterator(img_path)) {
            if (boost::filesystem::is_regular_file(entry.status())) // 检查是否为文件
            {
                std::string filename = entry.path().filename().string();
                size_t dot_pos = filename.rfind('.');
                if (dot_pos != std::string::npos) {
                    std::string extension = filename.substr(dot_pos);
                    if (std::find(g_Common_image_formats.begin(), g_Common_image_formats.end(), extension)
                        != g_Common_image_formats.end()) {
                        strFilesPath_vec.push_back(strMLAPath + filename); // 存储完整路径
                    }
                }
            }
        }

        // 数量不相同，则清除后直接返回
        if (strFilesPath_vec.size() != MLA_valid_image_count) {
            if (!strFilesPath_vec.empty()) {
                boost::filesystem::path dir_path(strMLAPath);
                if (boost::filesystem::exists(dir_path) && boost::filesystem::is_directory(dir_path)) {
                    boost::filesystem::remove_all(dir_path);
                    boost::filesystem::create_directory(dir_path);
                }
            }
            return bRead;
        }

        // 读取分割后的微透镜图像
        for (int i = 0; i < strFilesPath_vec.size(); i++) {
            boost::filesystem::path path(strFilesPath_vec[i]);
            std::string strName = path.filename().string();
            std::string strExt = path.extension().string();
            std::size_t pos = strName.find(strExt);
            strName = strName.substr(0, pos);
            std::vector<std::string> strKey = splitStrings(strName, '_');
            cv::Mat image_gray = cv::imread(strFilesPath_vec[i], cv::IMREAD_GRAYSCALE);
            cv::Mat image_rgb = cv::imread(strFilesPath_vec[i], cv::IMREAD_COLOR);
            if (image_gray.empty()) {
                std::cout << "read image Failed! " << strFilesPath_vec[i] << std::endl;
                continue;
            }
            bRead = true;
            int tile_x = atoi(strKey[0].c_str());
            int tile_y = atoi(strKey[1].c_str());

            QuadTreeTileKeyPtr ptrKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_x, tile_y);
            QuadTreeProblemMap::iterator itr = problem_map.find(ptrKey);
            if (itr != problem_map.end()) {
                MLA_Problem &problem = itr->second;
                problem.m_Image_gray = image_gray; // lzd0705
                problem.m_Image_rgb = image_rgb;
            } else {
                std::cout << "ReadMLAImages: ptrKey is not contained map" << tile_x << ", " << tile_y << std::endl;
            }
        }
        return bRead;
    }

    //计算特征匹配对需要满足的阈值
    std::vector<double> DepthSolver::slope(cv::Point2f point1, cv::Point2f point2) {
        std::vector<double> k_l;

        //中心点平行
        if (point1.y == point2.y) {
            double y1 = point1.y + 1, y2 = point1.y - 1;
            double x1 = point2.x - (point2.x - point1.x) / 4.0;

            double k_a = (y2 - point1.y) / (x1 - point1.x);
            double k_b = (y1 - point1.y) / (x1 - point1.x);
            std::cout << "ka: " << k_a << k_b << std::endl;

            if (k_a > k_b) {
                k_l.push_back(k_b);
                k_l.push_back(k_a);

            } else {
                k_l.push_back(k_a);
                k_l.push_back(k_b);

            }
        }

        //中心点垂直
        if (point1.x == point2.x) {
            double y1 = point2.y - (point2.y - point1.y) / 4.0;
            double x1 = point2.x - 1, x2 = point2.x + 1;
            double k_a = (y1 - point1.y) / (x1 - point1.x);
            double k_b = (y1 - point1.y) / (x2 - point1.x);
            //cout << "ka: " << k_a << k_b << endl;
            if (k_a > k_b) {
                k_l.push_back(k_b);
                k_l.push_back(k_a);
            } else {
                k_l.push_back(k_a);
                k_l.push_back(k_b);
            }
        }

        if (point1.x != point2.x && point1.y != point2.y) {
            double distance = sqrt(pow(point2.y - point1.y, 2) + pow(point2.x - point1.x, 2));
            double d_x = 1.5 * (point2.y - point1.y) / distance;
            double d_y = 1.5 * (point2.x - point1.x) / distance;
            double x1 = point2.x - d_x, x2 = point2.x + d_x;
            double y1 = point2.y + d_y, y2 = point2.y - d_y;
            double k_a = (y1 - point1.y) / (x1 - point1.x);
            double k_b = (y2 - point1.y) / (x2 - point1.x);
            //cout << "ka: " << k_a << k_b << endl;
            if (k_a > k_b) {
                k_l.push_back(k_b);
                k_l.push_back(k_a);
            } else {
                k_l.push_back(k_a);
                k_l.push_back(k_b);
            }
        }
        return k_l;
    }

    bool myCompare_LF(cv::DMatch &matches, cv::DMatch &matches2) {
        return matches.distance < matches2.distance;
    }

    bool Compare1_LF(MLA_img &MLa_img, MLA_img &MLa_img1) {
        return MLa_img.Base_line < MLa_img1.Base_line;
    }

    QuadTreeTileInfoMap &DepthSolver::GetMLAInfoMap() {
        return m_MLA_info_map;
    }

    bool DepthSolver::AddMLAInfo(QuadTreeTileKeyPtr ptrKey, MLA_InfoPtr ptrInfo) {
        m_MLA_info_map[ptrKey] = ptrInfo;
    }

    std::map<std::string, cv::Mat> &DepthSolver::GetRawImageMap() {
        return m_RawImageMap;
    }

    QuadTreeProblemMapMap &DepthSolver::GetMIAProblemsMapMap() {
        return m_MIA_problem_map_map;
    }

    QuadTreeProblemMap &DepthSolver::GetMIAProblemsMap(std::string &strName) {
        QuadTreeProblemMap problemsMap;
        QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.find(strName);
        if (itr == m_MIA_problem_map_map.end())
            return problemsMap;
        return itr->second;
    }

    std::string &DepthSolver::GetSavePath() {
        return m_strSavePath;
    }

    std::map<std::string, cv::Mat> &DepthSolver::GetBlurScoreImageMap() {
        return m_BlurscoreImageMap;
    }

    void DepthSolver::SetBlurScoreImageMap(std::string &strFrameName, cv::Mat &blurScoreImage) {
        std::map<std::string, cv::Mat>::iterator itr = m_BlurscoreImageMap.find(strFrameName);
        if (itr == m_BlurscoreImageMap.end()) {
            m_BlurscoreImageMap[strFrameName] = blurScoreImage;
        }
    }

    void DepthSolver::GetBlurScoreMI(std::string &strFrameName, QuadTreeTileKeyPtr ptrKey, cv::Mat &target) {
        QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrKey);
        if (itr == m_MLA_info_map.end())
            return;
        MLA_InfoPtr ptrInfo = itr->second;
        if (ptrInfo->IsAbandonByArea())
            return;
        cv::Rect rect(ptrInfo->GetLeftDownCorner().x, ptrInfo->GetLeftDownCorner().y,
                      m_Params.mi_width_for_match, m_Params.mi_height_for_match);

        std::map<std::string, cv::Mat>::iterator itrR = m_BlurscoreImageMap.find(strFrameName);
        if (itrR == m_BlurscoreImageMap.end())
            return;
        cv::Mat &blurImage = itrR->second;
        blurImage(rect).copyTo(target);
    }

    void DepthSolver::GetRichnessScoreMI(std::string &strFrameName, QuadTreeTileKeyPtr ptrKey, cv::Mat &target) {
        QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrKey);
        if (itr == m_MLA_info_map.end())
            return;
        MLA_InfoPtr ptrInfo = itr->second;
        if (ptrInfo->IsAbandonByArea())
            return;
        cv::Rect rect(ptrInfo->GetLeftDownCorner().x, ptrInfo->GetLeftDownCorner().y,
                      m_Params.mi_width_for_match, m_Params.mi_height_for_match);

        std::map<std::string, cv::Mat>::iterator itrR = m_RichnessImageMap.find(strFrameName);
        if (itrR == m_RichnessImageMap.end())
            return;
        cv::Mat &richnessImage = itrR->second;
        richnessImage(rect).copyTo(target);
    }

    QuadTreeDisNormalMapMap &DepthSolver::GetMLADisNormalMapMap() {
        return m_MIA_dispNormal_map_map;
    }

    std::map<std::string, cv::Mat> &DepthSolver::GetRichnessImageMap() {
        return m_RichnessImageMap;
    }

    void DepthSolver::SetRichnessImageMap(std::string &strFrameName, cv::Mat &richnessImage) {
        std::map<std::string, cv::Mat>::iterator itr = m_RichnessImageMap.find(strFrameName);
        if (itr == m_RichnessImageMap.end()) {
            m_RichnessImageMap[strFrameName] = richnessImage;
        }
    }

    void DepthSolver::CreateProblem_frame(QuadTreeTileKeyPtr ptrKey, std::string strFrame)
    {
        QuadTreeProblemMapMap::iterator itrM = m_MIA_problem_map_map.find(strFrame);
        if (itrM == m_MIA_problem_map_map.end())
            return;
        QuadTreeProblemMap& problem_map = itrM->second;
        QuadTreeProblemMap::iterator itrD = problem_map.find(ptrKey);
        if (itrD != problem_map.end())
            return;

        MLA_Problem problem;
        problem.m_ptrKey = ptrKey;
        problem_map[ptrKey] = problem;

        // 写出
        std::string strSavePath = m_strSavePath + strFrame;
        {
            boost::filesystem::path dir_save_path(strSavePath);
            if (!boost::filesystem::exists(dir_save_path)) {
                if (!boost::filesystem::create_directory(dir_save_path)) {
                    std::cout << "dir failed to create: " << strSavePath << std::endl;
                }
            }
        }
    }

    void DepthSolver::CreateDisNormal_frame(QuadTreeTileKeyPtr ptrKey, std::string strFrame)
    {
        QuadTreeDisNormalMapMap::iterator itrM = m_MIA_dispNormal_map_map.find(strFrame);
        if (itrM == m_MIA_dispNormal_map_map.end())
            return;

        QuadTreeDisNormalMap& dis_normal_map = itrM->second;
        QuadTreeDisNormalMap::iterator itrD = dis_normal_map.find(ptrKey);
        if (itrD != dis_normal_map.end())
            return;

        DisparityAndNormalPtr ptrDisNormal = std::make_shared<DisparityAndNormal>(m_Params);
        ptrDisNormal->m_ptrKey = ptrKey;
        dis_normal_map[ptrKey] = ptrDisNormal;
    }

    void DepthSolver::CreateProblems(QuadTreeTileKeyPtr ptrKey) {
        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;

            QuadTreeProblemMap::iterator itrD = problem_map.find(ptrKey);
            if (itrD != problem_map.end())
                continue;

            MLA_Problem problem;
            problem.m_ptrKey = ptrKey;
            problem_map[ptrKey] = problem;

            // 写出
            std::string strSavePath = m_strSavePath + strName;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
        }
    }

    void DepthSolver::CreateDisNormals(QuadTreeTileKeyPtr ptrKey) {
        for (QuadTreeDisNormalMapMap::iterator itr = m_MIA_dispNormal_map_map.begin();
             itr != m_MIA_dispNormal_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeDisNormalMap &dis_normal_map = itr->second;

            QuadTreeDisNormalMap::iterator itrD = dis_normal_map.find(ptrKey);
            if (itrD != dis_normal_map.end())
                continue;

            DisparityAndNormalPtr ptrDisNormal = std::make_shared<DisparityAndNormal>(m_Params);
            ptrDisNormal->m_ptrKey = ptrKey;
            dis_normal_map[ptrKey] = ptrDisNormal;
        }
    }

    void DepthSolver::Undistortion() {
        return;

        // 主透镜造成的畸变
        bool bOk = UndistortionWithMainLens();

        // TODO：微透镜造成的畸变
        //UndistortionWithMicroLens();
    }

    // 主透镜造成的畸变
    bool DepthSolver::UndistortionWithMainLens() {
        std::map<std::string, cv::Mat> Undistor_imageMap = m_RawImageMap;

        std::map<std::string, cv::Mat>::iterator itr = m_RawImageMap.begin();
        for (; itr != m_RawImageMap.end(); itr++) {
            std::string strName = itr->first;
            cv::Mat &ori_image = itr->second;
            cv::Mat undis_img = cv::Mat::zeros(ori_image.rows, ori_image.cols, CV_8UC3);
            if (ori_image.empty() || undis_img.empty())
                continue;

            // 畸变参数
            Distortions mainLens_distor;
            mainLens_distor.radial() = m_Params.distor_radial;
            mainLens_distor.tangential() = m_Params.distor_tangential;
            mainLens_distor.depth() = m_Params.distor_depth;
            for (int row = 0; row < ori_image.rows; row++) {
                for (int col = 0; col < ori_image.cols; col++) {
                    double metric_row = row * m_Params.sensor_pixel_size;
                    double metric_col = col * m_Params.sensor_pixel_size;
                    double z = 0;

                    Eigen::Matrix<double, static_cast<int>(3), 1> imgPoint{metric_row, metric_col, z};
                    //传感器坐标系转换为相机坐标系
                    Eigen::Matrix<double, static_cast<int>(3), 1> imgPoint_Camera =
                            (m_Params.sensor_rotation * imgPoint) + m_Params.sensor_translate;
                    //添加畸变，计算畸变后点坐标
                    mainLens_distor.apply(imgPoint_Camera);
                    //相机坐标系转换为传感器坐标系
                    Eigen::Matrix<double, static_cast<int>(3), 1> imgPoint_Sensor =
                            m_Params.sensor_rotation.transpose() * (imgPoint_Camera - m_Params.sensor_translate);

                    int distor_row = imgPoint_Sensor.coeff(0) / m_Params.sensor_pixel_size;
                    int distor_col = imgPoint_Sensor.coeff(1) / m_Params.sensor_pixel_size;

                    //判断点是否超出图像范围
                    if (distor_col < ori_image.cols && distor_col > 0 &&
                        distor_row < ori_image.rows && distor_row > 0) {
                        undis_img.at<cv::Vec3b>(row, col) = ori_image.at<cv::Vec3b>(distor_row, distor_col);
                    } else {
                        undis_img.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
                    }
                }
            }
            cv::Mat dif_img = undis_img - ori_image;
            std::string strUndistor_Path = m_strRootPath + LF_RAW_DATASET_NAME + LF_UNDISTOR_NAME;
            boost::filesystem::create_directory(strUndistor_Path);
            strUndistor_Path += strName;
            cv::imwrite(strUndistor_Path + "_undistor.png", undis_img);
            cv::imwrite(strUndistor_Path + "_undistor_dif.png", dif_img);

            Undistor_imageMap[itr->first] = undis_img;
        }
        Undistor_imageMap.swap(m_RawImageMap);
        return true;
    }

    // 微透镜造成的畸变
    bool DepthSolver::UndistortionWithMicroLens(cv::Mat &ori_image, cv::Mat &undis_image) {
        return false;
    }

    bool DepthSolver::ReadRawImagesAndCreateProblemMaps() {
        // 根据原始光场影像数据的帧数，创建深度估计流程中需要的遍历（几个map）
        for (int index = 0; index < m_strRawImagePathVec.size(); index++) {
            std::string &strRawImagePath = m_strRawImagePathVec.at(index);

            // 读取原始光场影像
            cv::Mat rawImage = cv::imread(strRawImagePath, 1);
            if (rawImage.empty()) {
                std::cout << "Read RawImage Failed! " << strRawImagePath << std::endl;
                continue;
            }
            SetRawImageHeight(rawImage.rows);
            SetRawImageWidth(rawImage.cols);

            boost::filesystem::path raw_path(strRawImagePath);
            std::string strName = raw_path.filename().string();
            std::string strExt = raw_path.extension().string();
            std::size_t pos = strName.find(strExt);
            std::string strNameLessExt = strName.substr(0, pos);

            // 创建对象
            m_RawImageMap[strNameLessExt] = rawImage;
            // 创建problem
            QuadTreeProblemMap problem_map;
            m_MIA_problem_map_map[strNameLessExt] = problem_map;
            // 创建视差和法线结果
            QuadTreeDisNormalMap dis_normal_map;
            m_MIA_dispNormal_map_map[strNameLessExt] = dis_normal_map;
        }
        return true;
    }

    bool DepthSolver::ReadWhiteImage() {
        m_WhiteImage = cv::imread(m_strWhiteImagePath, 1);
        if (m_WhiteImage.empty()) {
            LOG_ERROR("Read WhiteImage Failed! ", m_strWhiteImagePath);
            return false;
        }
        return true;
    }

    void DepthSolver::ProcessProblemsByACMH_LF_Tilekey() {
        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;
            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];

            int garbge = 0;
            std::cout << "PPBACMHLFT: current problemMap: " << strName << ", counts: " << problem_map.size()
                      << std::endl;
            for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); itrP++) {
                MLA_Problem &problem = itrP->second;
                if (problem.m_bGarbage) {
                    garbge++;
                    continue;
                }
                ProcessProblem_LF_TileKey(problem, problem_map, dis_normal_map, strName);
            }
            std::cout << "PPBACMHLFT: garbge is: " << garbge << std::endl;

            // 左右一致性检测
            if (m_bLRCheck) {
                LRCheckImp_Tilekey(problem_map, dis_normal_map);
            }
        }
    }

    void DepthSolver::ProcessProblemsByACMH_LF_TilekeySequence(std::string &strName, QuadTreeProblemMap &problem_map) {
        DepthEstimateByACMH_LF_Tilekey(strName, problem_map);
        WriteDisMap_TileKey_new_AccuSequence(strName, problem_map);
        Virtual_depth_map_TileKeySequence(strName, problem_map);
    }

    void DepthSolver::DepthEstimateByACMH_LF_Tilekey(std::string &strName, QuadTreeProblemMap &problem_map) {
        QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];

        int garbge = 0;
        std::cout << "PPBACMHLFT: current problemMap: " << strName << ", counts: " << problem_map.size() << std::endl;
        for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); itrP++) {
            MLA_Problem &problem = itrP->second;
            if (problem.m_bGarbage) {
                garbge++;
                continue;
            }
            ProcessProblem_LF_TileKey(problem, problem_map, dis_normal_map, strName);
        }
        std::cout << "PPBACMHLFT: garbge is: " << garbge << std::endl;

        // 左右一致性检测
        if (m_bLRCheck) {
            LRCheckImp_Tilekey(problem_map, dis_normal_map);
        }
    }

    void DepthSolver::ProcessProblemsByPlannerPrior_LF_Tilekey(bool geom_consistency) {
        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strFrameName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;
            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strFrameName];
            std::cout << "PPBPPLFT: current problemMap: " << strFrameName << ", counts: " << problem_map.size()
                      << std::endl;

            int garbge = 0;
            for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); itrP++) {
                MLA_Problem &problem = itrP->second;
                if (problem.m_bGarbage) {
                    garbge++;
                    continue;
                }
                ProcessProblem_planner_LF_TileKey(problem, problem_map, dis_normal_map, strFrameName, geom_consistency);
            }
            std::cout << "PPBPPLFT: garbge is: " << garbge << std::endl;

            // 左右一致性检测
            if (m_bLRCheck) {
                LRCheckImp_Tilekey(problem_map, dis_normal_map);
            }
        }
    }

    void
    DepthSolver::ProcessProblemsByPlannerPrior_LF_TilekeySequence(std::string &strName, QuadTreeProblemMap &problem_map,
                                                                  bool geom_consistency) {
        DepthEstimateByPlannerPrior_LF_Tilekey(strName, problem_map, geom_consistency);
        WriteDisMap_TileKey_new_AccuSequence(strName, problem_map);
        Virtual_depth_map_TileKeySequence(strName, problem_map);
    }

    void DepthSolver::DepthEstimateByPlannerPrior_LF_Tilekey(std::string &strName, QuadTreeProblemMap &problem_map,
                                                             bool geom_consistency) {
        QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];
        std::cout << "PPBPPLFT: current problemMap: " << strName << ", counts: " << problem_map.size() << std::endl;

        int garbge = 0;
        for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); ++itrP) {
            MLA_Problem &problem = itrP->second;
            if (problem.m_bGarbage) {
                garbge++;
                continue;
            }
            ProcessProblem_planner_LF_TileKey(problem, problem_map, dis_normal_map, strName, geom_consistency);
        }
        std::cout << "PPBPPLFT: garbge is: " << garbge << std::endl;

        // 左右一致性检测
        if (m_bLRCheck) {
            LRCheckImp_Tilekey(problem_map, dis_normal_map);
        }
    }

    void DepthSolver::ProcessProblemsByHP_LF_Tilekey(bool geom_consistency) {
        for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
             itr != m_MIA_problem_map_map.end(); itr++) {
            std::string strName = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;
            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];
            std::cout << "PPBHPLFT: current problemMap: " << strName << ", counts: " << problem_map.size() << std::endl;

            int garbge = 0;
            for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); itrP++) {
                MLA_Problem &problem = itrP->second;
                if (problem.m_bGarbage) {
                    garbge++;
                    continue;
                }
                ProcessProblem_HP_LF_TileKey(problem, problem_map, dis_normal_map, strName, geom_consistency);
            }
            std::cout << "PPBHPLFT: garbge is: " << garbge << std::endl;

            // 左右一致性检测
            if (m_bLRCheck) {
                LRCheckImp_Tilekey(problem_map, dis_normal_map);
            }
        }
    }

    void DepthSolver::ProcessProblemsByHP_LF_TilekeySequence(std::string &strName, QuadTreeProblemMap &problem_map,
                                                             bool geom_consistency) {
        DepthEstimateByHP_LF_Tilekey(strName, problem_map, geom_consistency);
        WriteDisMap_TileKey_new_AccuSequence(strName, problem_map);
        Virtual_depth_map_TileKeySequence(strName, problem_map);
    }

    void DepthSolver::DepthEstimateByHP_LF_Tilekey(std::string &strName, QuadTreeProblemMap &problem_map,
                                                   bool geom_consistency) {
        QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];
        std::cout << "PPBHPLFT: current problemMap: " << strName << ", counts: " << problem_map.size() << std::endl;

        int garbge = 0;
        for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); itrP++) {
            MLA_Problem &problem = itrP->second;
            if (problem.m_bGarbage) {
                garbge++;
                continue;
            }
            ProcessProblem_HP_LF_TileKey(problem, problem_map, dis_normal_map, strName, geom_consistency);
        }
        std::cout << "PPBHPLFT: garbge is: " << garbge << std::endl;

        // 左右一致性检测
        if (m_bLRCheck) {
            LRCheckImp_Tilekey(problem_map, dis_normal_map);
        }
    }

    void DepthSolver::ProcessProblemsByBlurFeature_LF_Tilekey() {
        std::cout << "ProcessProblemsByBlurFeature, " << std::endl;
        return;
    }

    void DepthSolver::SliceAndIndicatlizeForMI(QuadTreeProblemMapMap::iterator &itrFrame) {
        QuadTreeProblemMap &problem_map = itrFrame->second;
        const std::string &strFrameName = itrFrame->first;

        cv::Mat &blurscore_img = m_BlurscoreImageMap[strFrameName];
        cv::Mat &richenss_img = m_RichnessImageMap[strFrameName];

        // 二值化
        double thresh = 10; // 阈值
        double maxValue = 255;
        cv::Mat blureness_img_Binarized = cv::Mat::zeros(blurscore_img.rows, blurscore_img.cols, CV_8UC1);
        cv::threshold(blurscore_img, blureness_img_Binarized, thresh, maxValue, cv::THRESH_BINARY);
        cv::Mat richness_img_Binarized = cv::Mat::zeros(richenss_img.rows, richenss_img.cols, CV_8UC1);
        cv::threshold(richenss_img, richness_img_Binarized, thresh, maxValue, cv::THRESH_BINARY);

        // std::string strTMP = "/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/";
        // cv::imwrite(strTMP+"blurscore.png", blurscore_img);
        // cv::imwrite(strTMP+"richness.png", richenss_img);
        // cv::imwrite(strTMP+"blurscore_binar.png", blureness_img_Binarized);
        // cv::imwrite(strTMP+"richness_binar.png", richness_img_Binarized);

#pragma omp parallel for schedule(dynamic)
        for (int id_problem = 0; id_problem < problem_map.size(); id_problem++) {
            QuadTreeProblemMap::iterator itrP = problem_map.begin();
            std::advance(itrP, id_problem);

            QuadTreeTileKeyPtr ptrKey = itrP->first;
            MLA_Problem &problem = itrP->second;
            MLA_InfoPtr ptrInfo = m_MLA_info_map[ptrKey];

            // 切割
            cv::Rect rect(ptrInfo->GetLeftDownCorner().x, ptrInfo->GetLeftDownCorner().y,
                          m_Params.mi_width_for_match, m_Params.mi_height_for_match);
            blureness_img_Binarized(rect).copyTo(problem.m_Image_Blureness_Bianry);
            blurscore_img(rect).copyTo(problem.m_Image_Blureness);
            richness_img_Binarized(rect).copyTo(problem.m_Image_Richness);
        }

        for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); ++itrP) {
            MLA_Problem &problem = itrP->second;
            //cv::imwrite(strTMP+"blurscore_binar_0_0.png", problem.m_Image_Blureness_Bianry);
            problem.ComputeBlurenessValue();
            problem.ComputeRichnessValue();
            // std::cout<<(itrP->first)->Str().c_str()<<"blur_value: "<<problem.m_BlurenessValue<<
            //     ", rich_value: "<<problem.m_RichnessValue<<std::endl;
        }
    }

    void DepthSolver::ProcessDepthInfo_BlurFeature_LFFrame(QuadTreeProblemMapMap::iterator &itrFrame)
    {
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap &problems_map = itrFrame->second;

        // Step 1: 量化微图像的模糊程度和纹理丰富度
        {
            PrintMemoryInfo("Image_QE Begin");
            bool bWrite  = true;
            ImageQualityEstimate image_QE(this);
            image_QE.SetBlurEstimateType(BST_Laplacian);
            image_QE.QuantizeBlurLevelForMIC(itrFrame, true, bWrite);
            image_QE.SetRichnessEstimateType(RST_GLCM);
            image_QE.QuantizeRichnessLevelForMIC(itrFrame, true, bWrite);
            // 计算每个微图像的指标值
            SliceAndIndicatlizeForMI(itrFrame);
            PrintMemoryInfo("Image_QE Finish");
        }

        // Step 2: 邻域微图像集合的选择
        {
            PrintMemoryInfo("SelectNeighbors Begin");
            // (本质是选择出重叠度较高，可匹配性较好的微图像：photo、基线，二者结合计算出一个score, 存下来作为step3的引导)
            SelectNeighbors selectNeighborsImp(this);
            selectNeighborsImp.CollectMIANeighImagesForMatch(itrFrame);
            //SelectNeighborsForProblems();
            PrintMemoryInfo("SelectNeighbors Finish");
        }

        // Step 3: 视差匹配
        {
            PrintMemoryInfo("Stereo Begin");
            MIStereoMatch miStereoImp(this);
            //miStereoImp.StereoMatchingForMIA(itrFrame);
            //miStereoImp.StereoMatchingForMIA_SoftProxyRepair(itrFrame);
            miStereoImp.StereoMatchingForMIA_FrameCrossViews(itrFrame);
            //miStereoImp.StereoMatchingForMIA_SoftProxyPGRRepair(itrFrame);
            WriteDisMapForMIA(strFrameName, problems_map);
            PrintMemoryInfo("Stereo Finish");
        }

        // Step 4: 焦点堆栈的计算
        {
            PrintMemoryInfo("Refocus Begin");
            LFRefocus refocusImp(this);
//            refocusImp.FuseVirtualDepth_BackProject_OctreeVoxel(itrFrame);
            refocusImp.FuseVirtualDepth_BackProject(itrFrame);
            Focus_AIF_VD(itrFrame);
            refocusImp.AIFImageCompositeForMIA(itrFrame);
            PrintMemoryInfo("Refocus Finish");
        }

        // Step 5: 虚拟深度图转真实深度图
        // TODO: 1.统计功能用到这里，来去噪声
        // 2.实现定量验证
         // {
         //     VirtualToRealDepthFunc VTRDFunction(this);
         //     VTRDFunction.SetVirtualToRealDepthType(VTORD_SegmentBehavioralmodel);
         //     VTRDFunction.SetSamplePointSelectType(SPSelectByRandom);
         //     VTRDFunction.VirtualToRealDepth(itrFrame);
         // }
    }

        void DepthSolver::ProcessProblemsImp(bool geom_consistency)
        {
            switch (m_eStereoType) {
                case eST_ACMH: {
                    m_bLRCheck = true; //
                    m_bPlannar = false;
                    ProcessProblemsByACMH_LF_Tilekey();
                }
                    break;
                case eST_PlannerPrior: {
                    m_bLRCheck = true;
                    m_bPlannar = true;
                    ProcessProblemsByPlannerPrior_LF_Tilekey(geom_consistency);
                }
                    break;
                case eST_Horizontal_Propagation: {
                    m_bLRCheck = true;
                    m_bPlannar = true;
                    ProcessProblemsByHP_LF_Tilekey(geom_consistency);
                }
                    break;
                case eST_BlurFeature: {
                    m_bLRCheck = true;
                    ProcessProblemsByBlurFeature_LF_Tilekey();
                }
                    break;
                default:
                    break;
            }
        }

        void DepthSolver::ProcessProblemsImpSequence(QuadTreeProblemMapMap::iterator itr, bool geom_consistency)
        {
            std::string strNameLessExt = itr->first;
            QuadTreeProblemMap &problem_map = itr->second;
            switch (m_eStereoType)
            {
                case eST_ACMH:
                    {
                    m_bLRCheck = true;
                    m_bPlannar = false;

                    // Step 3: 微透镜图像的邻居选择
                    eSelectNeighborsType eSelectType = eSNT_FixedPosition; // eSNT_Features
                    SelectNeighbors SN_algorthim(this);
                    SN_algorthim.SelectNeighborsForMIA(itr, eSelectType);
                    // Step 4: 深度初始估计;转虚拟深度图和真实深度图
                    ProcessProblemsByACMH_LF_TilekeySequence(strNameLessExt, problem_map);
                }
                    break;
                case eST_PlannerPrior:
                {
                    m_bLRCheck = true;
                    m_bPlannar = true;
                    // Step 3: 微透镜图像的邻居选择
                    eSelectNeighborsType eSelectType = eSNT_FixedPosition; // eSNT_Features
                    SelectNeighbors SN_algorthim(this);
                    SN_algorthim.SelectNeighborsForMIA(itr, eSelectType);
                    // Step 4: 深度初始估计;转虚拟深度图和真实深度图
                    ProcessProblemsByPlannerPrior_LF_TilekeySequence(strNameLessExt, problem_map,
                                                                     geom_consistency);
                }
                    break;
                case eST_Horizontal_Propagation:
                {
                    m_bLRCheck = true;
                    m_bPlannar = true;
                    // Step 3: 微透镜图像的邻居选择
                    eSelectNeighborsType eSelectType = eSNT_FixedPosition; // eSNT_Features
                    SelectNeighbors SN_algorthim(this);
                    SN_algorthim.SelectNeighborsForMIA(itr, eSelectType);
                    // Step 4: 深度初始估计;转虚拟深度图和真实深度图
                    ProcessProblemsByHP_LF_TilekeySequence(strNameLessExt, problem_map, geom_consistency);
                }
                    break;
                case eST_BlurFeature:
                {
                    m_bLRCheck = true;
                    ProcessDepthInfo_BlurFeature_LFFrame(itr);
                }
                    break;
                default:
                    break;
            }
        }

        void DepthSolver::ProcessProblems_planarImp(QuadTreeProblemMap &problem_map,
                                                    QuadTreeDisNormalMap &dis_normal_map,
                                                    bool geom_consistency,
                                                    bool planar_prior,
                                                    bool multi_geometry,
                                                    std::string &strName) {
            for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); itrP++) {
                QuadTreeTileKeyPtr ptrKey = itrP->first;
                MLA_Problem &problem = itrP->second;
                if (!problem.m_bGarbage) {
                    ProcessProblem_planar_TileKey(problem, problem_map, dis_normal_map, strName,
                                                  geom_consistency, planar_prior, multi_geometry);
                }
            }
        }

        void DepthSolver::LRCheckImp_Tilekey(QuadTreeProblemMap &problem_map, QuadTreeDisNormalMap &dis_normal_map) {
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;
                if (!problem.m_bGarbage) {
                    LRCheck_TileKey(problem, dis_normal_map);
                }
            }
        }

        void DepthSolver::ConfirmProblemForEstimation(MLA_Problem &problem) {
            if (problem.m_Res_Image_KeyVec.size() > 0) {
                if (problem.m_ptrKey->GetTileX() < m_iGarbageRows ||
                    problem.m_ptrKey->GetTileY() < m_iGarbageRows) {
                    problem.m_bGarbage = true;
                    m_iCount++;
                } else {
                    problem.m_bGarbage = false;
                }
            } else {
                problem.m_bGarbage = true;
                m_iCount_2++;
            }
        }

        std::vector<float> DepthSolver::Compute_avg_std_key(std::vector<Res_image_Key> &res_img_tilekey) {
        }

        void DepthSolver::SelectNeighborsForProblems() {
            eSelectNeighborsType eSelectType = eSNT_FixedPosition; // eSNT_Features

            for (QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.begin(); itr != m_MLA_info_map.end(); itr++) {
                QuadTreeTileKeyPtr ptrKey = itr->first;
                MLA_InfoPtr ptrInfo = itr->second;
                if (ptrInfo->IsAbandonByArea() == true)
                    continue;

                switch (eSelectType) {
                    case eSNT_FixedPosition: // 固定位置的邻居
                    {
                        SelectNeighborsFromFixedPosition(ptrKey);
                    }
                        break;
                    case eSNT_Features: // 根据特征点及匹配的结果，自适应选择邻域
                    {
                        SelectNeighborsFromFeatures(ptrKey);
                    }
                        break;
                    default:
                        break;
                }
            }

            bool bTest = false;
            if (bTest) {
                for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
                     itr != m_MIA_problem_map_map.end(); itr++) {
                    std::string strName = itr->first;
                    QuadTreeProblemMap &problem_map = itr->second;
                    TestSelectNeighbors(strName, problem_map);
                }
            }
        }

        void DepthSolver::SelectNeighborsForProblemsBySequence(std::string &strName,
                                                               QuadTreeProblemMap &problem_map) {
            eSelectNeighborsType eSelectType = eSNT_FixedPosition; // eSNT_Features

            for (QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.begin(); itr != m_MLA_info_map.end(); itr++) {
                QuadTreeTileKeyPtr ptrKey = itr->first;
                MLA_InfoPtr ptrInfo = itr->second;
                if (ptrInfo->IsAbandonByArea() == true)
                    continue;

                switch (eSelectType) {
                    case eSNT_FixedPosition: // 固定位置的邻居
                    {
                        SelectNeighborsFromFixedPositionSequence(ptrKey, strName, problem_map);
                    }
                        break;
                    case eSNT_Features: // 根据特征点及匹配的结果，自适应选择邻域
                    {
                        SelectNeighborsFromFeaturesSequence(ptrKey, strName, problem_map);
                    }
                        break;
                    default:
                        break;
                }
            }

            bool bTest = false;
            if (bTest) {
                TestSelectNeighbors(strName, problem_map);
            }
        }

        void DepthSolver::Generate_MlaList(std::vector<cv::Mat> &img, std::vector<cv::Point2f> &center,
                                           std::vector<MLA_Problem> &MLA_problems, cv::Mat &BackGroundImage) {
            MLA_Problem problem;

            for (int i = 0; i < m_Params.mla_v_size; i++) {
                for (int j = 0; j < m_Params.mla_u_size; ++j) {
                    if (i > 2 && j > 2 && i < m_Params.mla_v_size - 2 && j < m_Params.mla_u_size - 2) {
                        problem.main_img = i * m_Params.mla_u_size + j;
                        // 自适应选择
//                    Get_len_img1(img, center, problem,i, j);
//                    Sort_len_img1(center, problem);
//                    sift_match1(img, center, problem);
//                    compute_NCC1(img,problem);
//                    Select_img1(img, center, problem);

                        // 固定范围
                        Select_img(img, center, problem, i, j);
                        //std::cout << problem.number.size() << std::endl;
                        MLA_problems.push_back(problem);
                        problem.number.clear();
                    } else {
                        problem.main_img = i * m_Params.mla_u_size + j;
                        MLA_problems.push_back(problem);
                        problem.number.clear();
                    }
                }
            }
            //checkNeiborhoods(center,MLA_problems,BackGroundImage);
        }

        void DepthSolver::TestSelectNeighbors(std::string &strName, QuadTreeProblemMap &problems_map) {
            cv::Mat BackGroundImage = m_WhiteImage;

            // 生成随机数
            int count = 10;
            std::random_device rd;
            std::mt19937 gen(rd());

            // 设置参考点和邻居点的颜色
            cv::Scalar refColor(0, 0, 255); // 红色
            cv::Scalar neighborColor(0, 255, 255); // 黄色
            for (int k = 0; k < count;) {
                std::uniform_int_distribution<> dis_tileX(1, m_Params.mla_u_size - 1);
                int tile_x = dis_tileX(gen);

                std::uniform_int_distribution<> dis_tileY(1, m_Params.mla_v_size - 1);
                int tile_y = dis_tileY(gen);

                QuadTreeTileKeyPtr ptrKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_x, tile_y);
                QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrKey);
                if (itr == m_MLA_info_map.end())
                    continue;

                QuadTreeProblemMap::iterator itrP = problems_map.find(ptrKey);
                if (itrP == problems_map.end())
                    continue;

                ++k;
                MLA_InfoPtr ptrInfo = itr->second;
                MLA_Problem &problem = itrP->second;

                float radius = 15;
                // 绘制参考点圆（红色）
                cv::circle(BackGroundImage, ptrInfo->GetCenter(), radius, refColor, cv::FILLED);
                // 绘制邻居点圆（黄色）
                std::vector<Res_image_Key> &res_img_vec = problem.m_Res_Image_KeyVec;
                for (size_t idx = 0; idx < res_img_vec.size(); idx++) {
                    Res_image_Key &res_img = res_img_vec[idx];
                    QuadTreeTileKeyPtr ptrNeigKey = res_img.m_ptrKey;
                    QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrNeigKey);
                    if (itr == m_MLA_info_map.end())
                        continue;
                    MLA_InfoPtr ptrNeigInfo = itr->second;
                    cv::circle(BackGroundImage, ptrNeigInfo->GetCenter(), radius, neighborColor, cv::FILLED);
                }
            }

            // 显示图像
            std::string m_strRawImagePath = m_strRootPath + LF_DEPTH_INTRA_NAME;
            boost::filesystem::path path(m_strRawImagePath);
            std::string strPath = path.parent_path().string();
            std::string strTestNeigPath = strPath + "/" + strName + "/TestNeig.png";
            bool bSave = cv::imwrite(strTestNeigPath, BackGroundImage);
            if (bSave) {
                std::cout << "TestNeig Image saved successfully" << std::endl;
            }
        }

        void DepthSolver::checkNeiborhoods(std::vector<cv::Point2f> &center, std::vector<MLA_Problem> &MLA_problems,
                                           cv::Mat &BackGroundImage) {
            //生成随机数
            int count = 10;
            std::random_device rd;
            std::mt19937 gen(rd());
            // 设置参考点和邻居点的颜色
            cv::Scalar refColor(0, 0, 255); // 红色
            cv::Scalar neighborColor(0, 255, 255); // 黄色
            for (int k = 0; k < count; ++k) {

                std::uniform_int_distribution<> dis(1, 50);
                int random_number1 = dis(gen);
                int random_number2 = dis(gen);
                int col = random_number1;
                int row = random_number2;
                float radius = 15;
                // 绘制参考点圆（红色）
                cv::circle(BackGroundImage, cv::Point(center[row * m_Params.mla_u_size + col].x,
                                                      center[row * m_Params.mla_u_size + col].y), radius, refColor,
                           cv::FILLED);
                // 绘制邻居点圆（黄色）
                for (size_t idx = 0; idx < MLA_problems[row * m_Params.mla_u_size + col].number.size(); idx++) {
                    int current_index = MLA_problems[row * m_Params.mla_u_size + col].number[idx];
                    cv::circle(BackGroundImage, cv::Point(center[current_index].x, center[current_index].y), radius,
                               neighborColor, cv::FILLED);
                }
            }
            // 显示图像
            //cv::imshow("Circles", BackGroundImage);
            bool saveSuccess = cv::imwrite("/home/zsl/work/zsl/data/光场图像/result/nbi.png", BackGroundImage);
            if (saveSuccess) {
                std::cout << "Image saved successfully" << std::endl;
            }
            cv::waitKey(0); // 等待用户按键
        }
        void DepthSolver::sift_match(MLA_img &image, std::vector<MLA_img> &images) {
            //定义Sift的基本参数
            int numFeatures = 500;
            //创建detector存放到KeyPoints中
            cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(numFeatures);
            std::vector<cv::KeyPoint> keypoints, keypoints2;
            //	//计算特征点描述符,特征向量提取
            cv::Mat dstSIFT, dstSIFT2;
            cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
            //进行BFMatch暴力匹配
            cv::BFMatcher matcher(cv::NORM_L2);
            //定义匹配结果变量
            std::vector<cv::DMatch> matches;
            //最后的匹配结果
            std::vector<cv::DMatch> goodmatches;

            for (long unsigned int i = 0; i < images.size(); i++) {
                //特征点检测
                detector->detect(image.img, keypoints);
                detector->detect(images[i].img, keypoints2);
                if (keypoints.size() > 0 || keypoints2.size() > 0) {
                    //计算特征点描述符,特征向量提取
                    descriptor->compute(image.img, keypoints, dstSIFT);
                    descriptor->compute(images[i].img, keypoints2, dstSIFT2);
                    //实现描述符之间的匹配
                    matcher.match(dstSIFT, dstSIFT2, matches);
                    std::vector<cv::DMatch> match1;

                    //计算可选的阈值范围
                    std::vector<double> k_l = slope(image.Lcenter_point, images[i].Lcenter_point);

                    for (long unsigned int j = 0; j < matches.size(); ++j) {
                        //特征匹配对的距离
                        double b = sqrt(
                                pow((keypoints2[matches[j].trainIdx].pt.y - keypoints[matches[j].queryIdx].pt.y), 2) +
                                pow((keypoints2[matches[j].trainIdx].pt.x - keypoints[matches[j].queryIdx].pt.x), 2));

                        //视差范围
                        int d1 = -1, d2 = -1;
                        if (images[i].Base_line < 200) {
                            d1 = 30, d2 = 43;
                        }
                        if (images[i].Base_line > 200 && images[i].Base_line < 300) {
                            d1 = 52, d2 = 75;
                        }
                        if (images[i].Base_line > 300) {
                            d1 = 62, d2 = 86;
                        }

                        if (d1 < b && b < d2) {
                            //cout << "满足视差的b："<<b << endl;
                            if (j == 0) {
                                match1.push_back(matches[j]);
                            } else {
                                if (keypoints[matches[j].queryIdx].pt == keypoints[matches[j - 1].queryIdx].pt &&
                                    keypoints2[matches[j].trainIdx].pt == keypoints2[matches[j - 1].trainIdx].pt) {
                                } else {
                                    match1.push_back(matches[j]);
                                }
                            }
                        }

                    }
                    //按距离远近对匹配对排序
                    sort(match1.begin(), match1.end(), myCompare_LF);

                    double k = (images[i].Lcenter_point.y - image.Lcenter_point.y) /
                               (images[i].Lcenter_point.x - image.Lcenter_point.x);
                    //cout << "中心点的斜率:" << k << endl;

                    for (long unsigned int j = 0; j < match1.size(); ++j) {

                        //特征匹配对的斜率
                        double a = (images[i].Lcenter_point.y + keypoints2[match1[j].trainIdx].pt.y -
                                    keypoints[match1[j].queryIdx].pt.y - image.Lcenter_point.y) /
                                   (images[i].Lcenter_point.x + keypoints2[match1[j].trainIdx].pt.x -
                                    keypoints[match1[j].queryIdx].pt.x - image.Lcenter_point.x);
                        //cout << "满足视差的特征匹配对的斜率:" << a << endl;


                        //剔除不正确的匹配
                        if (abs(image.Lcenter_point.x - images[i].Lcenter_point.x) <= 3 && k_l[0] * k_l[1] < 0) {
                            if (a < k_l[0] || a > k_l[1]) {
                                images[i].other_point.push_back(keypoints[match1[j].queryIdx].pt);
                                images[i].mine_point.push_back(keypoints2[match1[j].trainIdx].pt);
                                //cout << keypoints[match1[j].queryIdx].pt << keypoints2[match1[j].trainIdx].pt << endl;
                                /*std::cout << sqrt(pow((keypoints2[match1[j].trainIdx].pt.y - keypoints[match1[j].queryIdx].pt.y), 2) +
                            pow((keypoints2[match1[j].trainIdx].pt.x - keypoints[match1[j].queryIdx].pt.x), 2)) << std::endl;*/
                                goodmatches.push_back(match1[j]);
                            }
                        } else {
                            if (k_l[0] < a && a < k_l[1]) {
                                cv::Point2f p1 = keypoints[match1[j].queryIdx].pt;
                                cv::Point2f p2 = keypoints2[match1[j].trainIdx].pt;
                                images[i].other_point.push_back(p1);
                                images[i].mine_point.push_back(p2);
                                //cout << p1 << p2 << endl;
                                /*std::cout << sqrt(pow((keypoints2[match1[j].trainIdx].pt.y - keypoints[match1[j].queryIdx].pt.y), 2) +
                            pow((keypoints2[match1[j].trainIdx].pt.x - keypoints[match1[j].queryIdx].pt.x), 2)) << std::endl;*/
                                goodmatches.push_back(match1[j]);
                            }
                        }

                    }
                    //cv::Mat result;
                    ////匹配特征点天蓝色，单一特征点颜色随机
                    //drawMatches(image.img, keypoints, images[i].img, keypoints2, goodmatches, result,
                    //	cv::Scalar(255, 255, 0), cv::Scalar::all(-1));
                    ////cout << "最终的最佳特征点匹配对数" << goodmatches.size() << endl;

                    ////std::cout << "查看图像中的特征点个数" << images[i].other_point.size() << std::endl;
                    //std::string imgname = "Result" + std::to_string(i);
                    //cv::imshow(imgname, result);

                    keypoints.clear();
                    keypoints2.clear();
                    matches.clear();
                    match1.clear();
                    goodmatches.clear();
                    k_l.clear();
                    //waitKey(0);
                }
            }
        }

        void DepthSolver::Sift_MatchFromTileKey(MLA_Problem &problem,
                                                QuadTreeProblemMap &problem_map) {
            // 定义Sift的基本参数
            int numFeatures = 500;
            // 创建detector存放到KeyPoints中
            //cv::SiftFeatureDetector siftDetector(30);
            cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(numFeatures);
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            // 计算特征点描述符,特征向量提取
            cv::Mat dstSIFT1, dstSIFT2;
            cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
            //进行BFMatch暴力匹配
            cv::BFMatcher matcher(cv::NORM_L2);
            //定义匹配结果变量
            std::vector<cv::DMatch> matches;
            //最后的匹配结果
            std::vector<cv::DMatch> goodmatches;

            // 提取参考图像的特征点及描述符
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrCurKey);
            if (itr == m_MLA_info_map.end()) {
                std::cout << "Sift_MatchFromNeibours: currentkey not find! " << ptrCurKey->GetTileX() << ", "
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }
            MLA_InfoPtr ptrMainInfo = itr->second;
            detector->detect(problem.m_Image_gray, keypoints1);
            descriptor->compute(problem.m_Image_gray, keypoints1, dstSIFT1);
            if (keypoints1.empty() || dstSIFT1.empty()) {
                std::cout << "Sift_MatchFromNeibours: main_img keypoints Empty!" << ptrCurKey->GetTileX() << ","
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }

            // 提取邻居图像的特征点和描述符
            for (int i = 0; i < problem.m_Res_Image_KeyVec.size(); ++i) {
                Res_image_Key &neib_Key = problem.m_Res_Image_KeyVec[i];
                QuadTreeTileInfoMap::iterator itr_neig = m_MLA_info_map.find(neib_Key.m_ptrKey);
                if (itr_neig == m_MLA_info_map.end()) {
                    std::cout << "Sift_MatchFromNeibours: currentkey not find! " << neib_Key.m_ptrKey->GetTileX()
                              << ", " << neib_Key.m_ptrKey->GetTileY() << std::endl;
                    continue;
                }
                MLA_InfoPtr ptrNeiInfo = itr_neig->second;

                QuadTreeProblemMap::iterator itrPN = problem_map.find(neib_Key.m_ptrKey);
                if (itrPN == problem_map.end()) {
                    std::cout << "Sift_MatchFromNeibours2: currentkey not find! " << neib_Key.m_ptrKey->GetTileX()
                              << ", " << neib_Key.m_ptrKey->GetTileY() << std::endl;
                    continue;
                }
                MLA_Problem &problem_neig = itrPN->second;
                detector->detect(problem_neig.m_Image_gray, keypoints2);
                descriptor->compute(problem_neig.m_Image_gray, keypoints2, dstSIFT2);
                if (keypoints2.empty() || dstSIFT2.empty()) {
                    continue;
                }

                // 匹配
                matcher.match(dstSIFT1, dstSIFT2, matches);
                std::vector<cv::DMatch> match1;

                // 计算可选的阈值范围
                std::vector<double> k_l = slope(ptrMainInfo->GetCenter(), ptrNeiInfo->GetCenter());
                for (int j = 0; j < matches.size(); ++j) {
                    int image1_index = matches[j].queryIdx;
                    int image2_index = matches[j].trainIdx;
                    cv::KeyPoint &p1 = keypoints1[image1_index];
                    cv::KeyPoint &p2 = keypoints2[image2_index];
                    // 特征匹配对的距离
                    double b = sqrt(pow((p2.pt.y - p1.pt.y), 2) + pow((p2.pt.x - p1.pt.x), 2));

                    // 视差范围 修改
                    int d1 = -1, d2 = -1;
                    if (problem.m_Res_Image_KeyVec[i].Base_line < m_Params.baseline * 1.1) {
                        d1 = 15, d2 = 40;// 28 43
                    }
                    if (problem.m_Res_Image_KeyVec[i].Base_line > m_Params.baseline * 1.1 &&
                        problem.m_Res_Image_KeyVec[i].Base_line < m_Params.baseline * 1.8) {
                        d1 = 40, d2 = 60;// 52 75
                    }
                    if (problem.m_Res_Image_KeyVec[i].Base_line > m_Params.baseline * 1.8) {
                        d1 = 48, d2 = 70;// 62 86
                    }

                    if (d1 < b && b < d2) {
                        //cout << "满足视差的b："<<b << endl;
                        if (j == 0) {
                            match1.push_back(matches[j]);
                        } else {
                            if (p1.pt == keypoints1[matches[j - 1].queryIdx].pt &&
                                p2.pt == keypoints2[matches[j - 1].trainIdx].pt) {
                            } else {
                                match1.push_back(matches[j]);
                            }
                        }
                    }
                }

                // 按距离远近对匹配对排序
                sort(match1.begin(), match1.end(), myCompare_LF);
                for (int j = 0; j < match1.size(); ++j) {
                    int image1_index = match1[j].queryIdx;
                    int image2_index = match1[j].trainIdx;
                    cv::KeyPoint &p1 = keypoints1[image1_index];
                    cv::KeyPoint &p2 = keypoints2[image2_index];

                    //特征匹配对的斜率
                    double a = (ptrNeiInfo->GetCenter().y + p2.pt.y - p1.pt.y - ptrMainInfo->GetCenter().y) /
                               (ptrNeiInfo->GetCenter().x + p2.pt.x - p1.pt.x - ptrMainInfo->GetCenter().x);
                    //cout << "满足视差的特征匹配对的斜率:" << a << endl;

                    // 剔除不正确的匹配
                    if (abs(ptrMainInfo->GetCenter().x - ptrNeiInfo->GetCenter().x) <= 3 && k_l[0] * k_l[1] < 0) {
                        if (a < k_l[0] || a > k_l[1]) {
                            problem.m_Res_Image_KeyVec[i].r_p.push_back(p1.pt);
                            problem.m_Res_Image_KeyVec[i].m_p.push_back(p2.pt);
                            goodmatches.push_back(match1[j]);
                        }
                    } else {
                        if (k_l[0] < a && a < k_l[1]) {
                            cv::Point2f p1 = keypoints1[match1[j].queryIdx].pt;
                            cv::Point2f p2 = keypoints2[match1[j].trainIdx].pt;
                            problem.m_Res_Image_KeyVec[i].r_p.push_back(p1);
                            problem.m_Res_Image_KeyVec[i].m_p.push_back(p2);
                            goodmatches.push_back(match1[j]);
                        }
                    }
                }

                cv::Mat result;
                // 匹配特征点天蓝色，单一特征点颜色随机
                drawMatches(problem.m_Image_gray, keypoints1, problem_neig.m_Image_gray, keypoints2,
                            goodmatches, result, cv::Scalar(255, 255, 0),
                            cv::Scalar::all(-1));

                // clear
                keypoints1.clear();
                keypoints2.clear();
                matches.clear();
                match1.clear();
                goodmatches.clear();
                k_l.clear();
            }
            std::cout << "current MLA Image: " << ptrCurKey->GetTileX() << ", " << ptrCurKey->GetTileY() << std::endl;
        }

        void DepthSolver::sift_match1(std::vector<cv::Mat> &img, std::vector<cv::Point2f> &center,
                                      MLA_Problem &problem) {
            //定义Sift的基本参数
            int numFeatures = 500;
            //创建detector存放到KeyPoints中
            //cv::SiftFeatureDetector siftDetector(30);
            cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(numFeatures);
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            //计算特征点描述符,特征向量提取
            cv::Mat dstSIFT1, dstSIFT2;
            cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
            //进行BFMatch暴力匹配
            cv::BFMatcher matcher(cv::NORM_L2);
            //定义匹配结果变量
            std::vector<cv::DMatch> matches;
            //最后的匹配结果
            std::vector<cv::DMatch> goodmatches;

            int m = problem.main_img;
            detector->detect(img[m], keypoints1);
            descriptor->compute(img[m], keypoints1, dstSIFT1);
            if (keypoints1.empty() || dstSIFT1.empty()) {
                std::cout << "sift_match1: main_img keypoints Empty!" << m << std::endl;
                return;
            }

            for (int i = 0; i < problem.res_img.size(); ++i) {
                int n = problem.res_img[i].num;

                detector->detect(img[n], keypoints2);
                descriptor->compute(img[n], keypoints2, dstSIFT2);
                if (keypoints2.empty() || dstSIFT2.empty()) {
                    continue;
                }

                // 匹配
                matcher.match(dstSIFT1, dstSIFT2, matches);
                std::vector<cv::DMatch> match1;

                // 计算可选的阈值范围
                std::vector<double> k_l = slope(center[m], center[n]);
                for (int j = 0; j < matches.size(); ++j) {
                    int image1_index = matches[j].queryIdx;
                    int image2_index = matches[j].trainIdx;
                    cv::KeyPoint &p1 = keypoints1[image1_index];
                    cv::KeyPoint &p2 = keypoints2[image2_index];
                    // 特征匹配对的距离
                    double b = sqrt(pow((p2.pt.y - p1.pt.y), 2) + pow((p2.pt.x - p1.pt.x), 2));

                    // 视差范围
                    int d1 = -1, d2 = -1;
                    if (problem.res_img[i].Base_line < 200) {
                        d1 = 28, d2 = 43;
                    }
                    if (problem.res_img[i].Base_line > 200 && problem.res_img[i].Base_line < 300) {
                        d1 = 52, d2 = 75;
                    }
                    if (problem.res_img[i].Base_line > 300) {
                        d1 = 62, d2 = 86;
                    }

                    if (d1 < b && b < d2) {
                        //cout << "满足视差的b："<<b << endl;
                        if (j == 0) {
                            match1.push_back(matches[j]);
                        } else {
                            if (keypoints1[matches[j].queryIdx].pt == keypoints1[matches[j - 1].queryIdx].pt &&
                                keypoints2[matches[j].trainIdx].pt == keypoints2[matches[j - 1].trainIdx].pt) {
                            } else {
                                match1.push_back(matches[j]);
                            }
                        }
                    }

                }
                //按距离远近对匹配对排序
                sort(match1.begin(), match1.end(), myCompare_LF);

                for (int j = 0; j < match1.size(); ++j) {
                    //特征匹配对的斜率
                    double a =
                            (center[n].y + keypoints2[match1[j].trainIdx].pt.y - keypoints1[match1[j].queryIdx].pt.y -
                             center[m].y) /
                            (center[n].x + keypoints2[match1[j].trainIdx].pt.x - keypoints1[match1[j].queryIdx].pt.x -
                             center[m].x);
                    //cout << "满足视差的特征匹配对的斜率:" << a << endl;

                    //剔除不正确的匹配
                    if (abs(center[m].x - center[n].x) <= 3 && k_l[0] * k_l[1] < 0) {
                        if (a < k_l[0] || a > k_l[1]) {
                            problem.res_img[i].r_p.push_back(keypoints1[match1[j].queryIdx].pt);
                            problem.res_img[i].m_p.push_back(keypoints2[match1[j].trainIdx].pt);
                            //cout << keypoints[match1[j].queryIdx].pt << keypoints2[match1[j].trainIdx].pt << endl;
                            /*std::cout << sqrt(pow((keypoints2[match1[j].trainIdx].pt.y - keypoints[match1[j].queryIdx].pt.y), 2) +
                        pow((keypoints2[match1[j].trainIdx].pt.x - keypoints[match1[j].queryIdx].pt.x), 2)) << std::endl;*/
                            goodmatches.push_back(match1[j]);
                        }
                    } else {
                        if (k_l[0] < a && a < k_l[1]) {
                            cv::Point2f p1 = keypoints1[match1[j].queryIdx].pt;
                            cv::Point2f p2 = keypoints2[match1[j].trainIdx].pt;
                            problem.res_img[i].r_p.push_back(p1);
                            problem.res_img[i].m_p.push_back(p2);
                            //cout << p1 << p2 << endl;
                            /*std::cout << sqrt(pow((keypoints2[match1[j].trainIdx].pt.y - keypoints[match1[j].queryIdx].pt.y), 2) +
                        pow((keypoints2[match1[j].trainIdx].pt.x - keypoints[match1[j].queryIdx].pt.x), 2)) << std::endl;*/
                            goodmatches.push_back(match1[j]);
                        }
                    }

                }

                cv::Mat result;
                //匹配特征点天蓝色，单一特征点颜色随机
                drawMatches(img[m], keypoints1, img[n], keypoints2, goodmatches, result,
                            cv::Scalar(255, 255, 0), cv::Scalar::all(-1));
                //cout << "最终的最佳特征点匹配对数" << goodmatches.size() << endl;
//                std::string imgname = "Result" + std::to_string(i);
//                cv::imshow(imgname, result);

                keypoints1.clear();
                keypoints2.clear();
                matches.clear();
                match1.clear();
                goodmatches.clear();
                k_l.clear();
            }
            std::cout << "图 ： " << m << std::endl;
            //cv::waitKey(0);
        }

        void DepthSolver::compute_NCC(MLA_img &image, std::vector<MLA_img> &images) {
            for (long unsigned int i = 0; i < images.size(); i++) {
                int x;
                cv::Mat mask1 = cv::Mat::zeros(image.img.size(), CV_8UC1); // 创建一个全黑的掩膜
                cv::Mat mask2 = cv::Mat::zeros(image.img.size(), CV_8UC1); // 创建一个全黑的掩膜
                //std::cout << "第"<<i<<"张影像： "<< images[i].other_point.size() << std::endl;
                for (long unsigned int j = 0; j < images[i].other_point.size(); j++) {
                    cv::Point2f p1 = images[i].other_point[j];
                    cv::Point2f p2 = images[i].mine_point[j];

                    x = std::min({floor(120 - p1.x), floor(p1.x - 0), floor(120 - p2.x), floor(p2.x - 0),
                                  floor(120 - p1.y), floor(p1.y - 0), floor(120 - p2.y), floor(p2.y - 0)});

                    if (x > 7) {
                        x = 7;
                    }

                    mask1 = image.img(cv::Rect(p1.x - x, p1.y - x, 2 * x + 1, 2 * x + 1));
                    mask2 = images[i].img(cv::Rect(p2.x - x, p2.y - x, 2 * x + 1, 2 * x + 1));

                    std::vector<uchar> vals1, vals2;

                    for (int r = 0; r < mask1.rows; r++) {
                        for (int c = 0; c <= mask1.rows; c++) {
                            vals1.push_back(mask1.ptr<uchar>(r)[c]);
                            vals2.push_back(mask2.ptr<uchar>(r)[c]);
                        }
                    }

                    float32 avg1 = cv::mean(vals1).val[0];

                    float32 avg2 = cv::mean(vals2).val[0];

                    float32 val12 = 0, val11 = 0, val22 = 0;
                    for (int i = 0; i < vals1.size(); i++) {
                        val12 += (vals1[i] - avg1) * (vals2[i] - avg2);
                        val11 += pow(vals1[i] - avg1, 2);
                        val22 += pow(vals2[i] - avg2, 2);
                    }

                    if (val11 == 0 || val12 == 0 || val22 == 0) {
                        images[i].SAD_grade.push_back(0);
                    } else {
                        float ncc = fabs(val12 / sqrt(val11 * val22));
                        images[i].SAD_grade.push_back(ncc);
                        //std::cout << ncc << std::endl;
                    }
                }

                mask1.release();
                mask2.release();
            }

        }

        void DepthSolver::compute_NCC1(std::vector<cv::Mat> &img, MLA_Problem &problem) {
            int m = problem.main_img;
            for (long unsigned int i = 0; i < problem.res_img.size(); i++) {
                int n = problem.res_img[i].num;
                int x;
                cv::Mat mask1 = cv::Mat::zeros(img[m].size(), CV_8UC1); // 创建一个全黑的掩膜
                cv::Mat mask2 = cv::Mat::zeros(img[m].size(), CV_8UC1); // 创建一个全黑的掩膜
                //std::cout << "第"<<i<<"张影像： "<< images[i].other_point.size() << std::endl;
                for (long unsigned int j = 0; j < problem.res_img[i].r_p.size(); j++) {
                    cv::Point2f p1 = problem.res_img[i].r_p[j];
                    cv::Point2f p2 = problem.res_img[i].m_p[j];

                    x = std::min({floor(120 - p1.x), floor(p1.x - 0), floor(120 - p2.x), floor(p2.x - 0),
                                  floor(120 - p1.y), floor(p1.y - 0), floor(120 - p2.y), floor(p2.y - 0)});
                    if (x > 5) {
                        x = 5;
                    }

                    mask1 = img[m](cv::Rect(p1.x - x, p1.y - x, 2 * x + 1, 2 * x + 1));
                    mask2 = img[n](cv::Rect(p2.x - x, p2.y - x, 2 * x + 1, 2 * x + 1));

                    std::vector<uchar> vals1, vals2;

                    for (int r = 0; r < mask1.rows; r++) {
                        for (int c = 0; c < mask1.cols; c++) {
                            vals1.push_back(mask1.ptr<uchar>(r)[c]);
                            vals2.push_back(mask2.ptr<uchar>(r)[c]);
                        }
                    }

                    float32 avg1 = cv::mean(vals1).val[0];
                    float32 avg2 = cv::mean(vals2).val[0];

                    // val12 存储两个向量差异乘积的和，val11 和 val22 分别存储两个向量各自差异平方的和。
                    float32 val12 = 0, val11 = 0, val22 = 0;
                    for (int i = 0; i < vals1.size(); i++) {
                        val12 += (vals1[i] - avg1) * (vals2[i] - avg2);
                        val11 += pow(vals1[i] - avg1, 2);
                        val22 += pow(vals2[i] - avg2, 2);
                    }

                    if (val11 == 0 || val12 == 0 || val22 == 0) {
                        problem.res_img[i].Ncc_grade.push_back(0);
                    } else {
                        float ncc = fabs(val12 / sqrt(val11 * val22));
                        problem.res_img[i].Ncc_grade.push_back(ncc);
                        //std::cout << ncc << std::endl;
                    }
                }
                //std::cout << "第"<<i<<"张影像： "<<problem.res_img[i].Ncc_grade.size() << std::endl;
                mask1.release();
                mask2.release();
            }
        }

        // 计算 Ncc_grade 值
        void DepthSolver::Compute_NCCFromTileKey(MLA_Problem &problem) {
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrCurKey);
            if (itr == m_MLA_info_map.end()) {
                std::cout << "Compute_NCCFromTileKey: currentkey not find! " << ptrCurKey->GetTileX() << ", "
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }
            MLA_InfoPtr ptrInfo_main = itr->second;

            for (long unsigned int i = 0; i < problem.m_Res_Image_KeyVec.size(); i++) {
                Res_image_Key &neib_Key = problem.m_Res_Image_KeyVec[i];
                LFMVS::QuadTreeTileKeyPtr ptrNeigKey = neib_Key.m_ptrKey;
                QuadTreeTileInfoMap::iterator itr_neib = m_MLA_info_map.find(ptrNeigKey);
                if (itr_neib == m_MLA_info_map.end()) {
                    std::cout << "Compute_NCCFromTileKey: neighbor key not find! " << ptrNeigKey->GetTileX()
                              << ", " << ptrNeigKey->GetTileY() << std::endl;
                    continue;
                }
                MLA_InfoPtr ptrNeiInfo = itr_neib->second;

                int x;
                cv::Mat mask1 = cv::Mat::zeros(problem.m_Image_gray.size(), CV_8UC1); // 创建一个全黑的掩膜
                cv::Mat mask2 = cv::Mat::zeros(problem.m_Image_gray.size(), CV_8UC1); // 创建一个全黑的掩膜
                //std::cout << "第"<<i<<"张影像： "<< images[i].other_point.size() << std::endl;
                for (long unsigned int j = 0; j < problem.m_Res_Image_KeyVec[i].r_p.size(); j++) {
                    cv::Point2f &p1 = problem.m_Res_Image_KeyVec[i].r_p[j];
                    cv::Point2f &p2 = problem.m_Res_Image_KeyVec[i].m_p[j];

                    x = std::min({floor(m_Params.mi_width_for_match - p1.x), floor(p1.x - 0),
                                  floor(m_Params.mi_width_for_match - p2.x), floor(p2.x - 0),
                                  floor(m_Params.mi_height_for_match - p1.y), floor(p1.y - 0),
                                  floor(m_Params.mi_height_for_match - p2.y), floor(p2.y - 0)});
                    if (x > 5) {
                        x = 5;
                    }

                    mask1 = problem.m_Image_gray(cv::Rect(p1.x - x, p1.y - x, 2 * x + 1, 2 * x + 1));
                    mask2 = problem.m_Image_gray(cv::Rect(p2.x - x, p2.y - x, 2 * x + 1, 2 * x + 1));

                    std::vector<uchar> vals1, vals2;
                    for (int r = 0; r < mask1.rows; r++) {
                        for (int c = 0; c < mask1.cols; c++) {
                            vals1.push_back(mask1.ptr<uchar>(r)[c]);
                            vals2.push_back(mask2.ptr<uchar>(r)[c]);
                        }
                    }
                    float32 avg1 = cv::mean(vals1).val[0];
                    float32 avg2 = cv::mean(vals2).val[0];

                    // val12 存储两个向量差异乘积的和，val11 和 val22 分别存储两个向量各自差异平方的和。
                    float32 val12 = 0, val11 = 0, val22 = 0;
                    for (int i = 0; i < vals1.size(); i++) {
                        val12 += (vals1[i] - avg1) * (vals2[i] - avg2);
                        val11 += pow(vals1[i] - avg1, 2);
                        val22 += pow(vals2[i] - avg2, 2);
                    }

                    if (val11 == 0 || val12 == 0 || val22 == 0) {
                        problem.m_Res_Image_KeyVec[i].Ncc_grade.push_back(0);
                    } else {
                        float ncc = fabs(val12 / sqrt(val11 * val22));
                        problem.m_Res_Image_KeyVec[i].Ncc_grade.push_back(ncc);
                        //std::cout << ncc << std::endl;
                    }
                }
                //std::cout << "第"<<i<<"张影像： "<<problem.res_img[i].Ncc_grade.size() << std::endl;
                mask1.release();
                mask2.release();
            }
        }

        void DepthSolver::CollectMLANeigImagesByPOSE(MLA_Problem &problem) {
            QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
            int32_t tile_X = ptrKey->GetTileY();
            int32_t tile_Y = ptrKey->GetTileY();

            for (int a = -2; a < 3; a++) {
                for (int b = -2; b < 3; b++) {
                    // 单数行=偶数行 微透镜剔除右上微透镜
                    if (abs(a) + abs(b) < 4 && tile_X % 2 == 0 && tile_X + a + 1 > 0 && tile_Y + b + 1 > 0) {
                        if ((abs(a) == 1 && b == 2) || (abs(a) == 0 && b == 0)) {
                        } else {
                            int32_t tile_now_x = tile_X + a;
                            int32_t tile_now_y = tile_Y + b;
                            QuadTreeTileKeyPtr ptrKey_even_row = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                                                                                                 tile_now_x,
                                                                                                 tile_now_y);
                            problem.m_NeigKeyPtrVec.push_back(ptrKey_even_row);
                        }
                    }
                    // 双数行=奇数行 微透镜剔除右左上微透镜
                    if (abs(a) + abs(b) < 4 && tile_X % 2 == 1 && tile_X + a + 1 > 0 && tile_Y + b + 1 > 0) {
                        if ((abs(a) == 1 && b == -2) || (abs(a) == 0 && b == 0)) {
                        } else {
                            int32_t tile_now_x = tile_X + a;
                            int32_t tile_now_y = tile_Y + b;
                            QuadTreeTileKeyPtr ptrKey_odd_row = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                                                                                                tile_now_x, tile_now_y);
                            problem.m_NeigKeyPtrVec.push_back(ptrKey_odd_row);
                        }
                    }
                }
            }
        }

        void DepthSolver::Get_len_img1(std::vector<cv::Mat> &img, std::vector<cv::Point2f> &center,
                                       MLA_Problem &problem, int i, int j) {
            for (int a = -2; a < 3; a++) {
                for (int b = -2; b < 3; b++) {
                    //单数行 微透镜剔除右上微透镜
                    if (abs(a) + abs(b) < 4 && i % 2 == 0 && i + a + 1 > 0 && j + b + 1 > 0) {
                        if ((abs(a) == 1 && b == 2) || (abs(a) == 0 && b == 0)) {
                        } else {
                            //- (i + a) / 2
                            int num = (i + a) * m_Params.mla_u_size + (j + b);
                            problem.number.push_back(num);
                        }
                    }
                    // 双数行 微透镜剔除右左上微透镜
                    if (abs(a) + abs(b) < 4 && i % 2 == 1 && i + a + 1 > 0 && j + b + 1 > 0) {
                        if ((abs(a) == 1 && b == -2) || (abs(a) == 0 && b == 0)) {
                        } else {
                            int num = (i + a) * m_Params.mla_u_size + (j + b);
                            problem.number.push_back(num);
                        }
                    }
                }
            }
        }

        //按基线长短从小到大排序
        void DepthSolver::Sort_len_img(MLA_img &main_img, std::vector<MLA_img> &MLa_img) {
            for (long unsigned int i = 0; i < MLa_img.size(); i++) {
                double length = sqrt(pow((MLa_img[i].Lcenter_point.y - main_img.Lcenter_point.y), 2) +
                                     pow((MLa_img[i].Lcenter_point.x - main_img.Lcenter_point.x), 2));
                MLa_img[i].Base_line = length;
            }
            sort(MLa_img.begin(), MLa_img.end(), Compare1_LF);
        }

        bool CompareR(Res_img &neibours_length, Res_img &neibours_length1) {
            return neibours_length.Base_line < neibours_length1.Base_line;
        }

        void DepthSolver::SortMLANeigImagesByLength(MLA_Problem &problem) {
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrCurKey);
            if (itr == m_MLA_info_map.end()) {
                std::cout << "SMLANIByLength: currentkey not find! " << ptrCurKey->GetTileX()
                          << ", " << ptrCurKey->GetTileY() << std::endl;
                return;
            }
            problem.CreateResTileKeysFromNeiKeyVec(itr->second->GetCenter(), m_MLA_info_map);
        }

        void DepthSolver::Sort_len_img1(std::vector<cv::Point2f> &center, MLA_Problem &problem) {
            std::vector<Res_img> res_img;
            for (long unsigned int i = 0; i < problem.number.size(); i++) {
                int m = problem.main_img;
                int n = problem.number[i];
                double length = sqrt(pow((center[n].y - center[m].y), 2) +
                                     pow((center[n].x - center[m].x), 2));
                Res_img rg(n, length);
                res_img.push_back(rg);
            }
            problem.number.clear();
            //sort(res_img.begin(), res_img.end(), CompareR);
            problem.res_img = res_img;
        }

        bool Compare_map(std::pair<int, float> x, std::pair<int, float> y) {
            return x.second > y.second;
        }

        // 计算NCC的均值和标准差
        std::vector<float> Compute_avg_std(std::vector<int> &num, std::vector<float> &NCC) {
            std::vector<float> result;
            int n = 0;
            float avg1 = 0.0;
            float avg2 = 0.0;
            float std1 = 0.0;
            float std2 = 0.0;

            // 平均值
            for (long unsigned int i = 0; i < num.size(); i++) {
                if (num[i] != 0) {
                    n++;
                    avg1 += float(num[i]);
                    avg2 += NCC[i];
                }
            }

            if (n == 0) {
                result.push_back(0.0);
                result.push_back(0.0);
            } else {
                result.push_back(avg1 / float(n));
                result.push_back(avg2 / float(n));
            }

            // 方差
            for (long unsigned int i = 0; i < num.size(); i++) {
                if (num[i] != 0) {
                    std1 += (float(num[i]) - result[0]) * (float(num[i]) - result[0]);
                    std2 += (NCC[i] - result[1]) * (NCC[i] - result[1]);
                }
            }

            if (n == 0) {
                result.push_back(0.0);
                result.push_back(0.0);
            } else {
                // 标准差
                result.push_back(sqrt(std1 / float(n)));
                result.push_back(sqrt(std2 / float(n)));
            }
            return result;
        }


        void DepthSolver::Select_img2(MLA_img &main_img, std::vector<MLA_img> &MLa_img, std::vector<MLA_img> &res_img) {
            int b1 = 0, b2 = 0;
            //int r1, r2;
            std::vector<std::pair<int, float>> Map;
            int n = 0;
            float NCC = 0;
            std::vector<int> PointNum;
            std::vector<float> Ncc;
            //对不同基线的进行操作
            for (long unsigned int i = 1; i < MLa_img.size(); i++) {
                if (MLa_img[i - 1].Base_line < 200 && MLa_img[i].Base_line > 200) {
                    b1 = i;
                }
                if (MLa_img[i - 1].Base_line > 200 && MLa_img[i - 1].Base_line < 300 && MLa_img[i].Base_line > 300) {
                    b2 = i;
                }
            }
            //对第一圈微透镜进行选择
            for (int i = 0; i < b1; i++) {
                if (MLa_img[i].SAD_grade.size() == 0) {
                    PointNum.push_back(0);
                    Ncc.push_back(0.0);
                } else {
                    for (int j = 0; j < MLa_img[i].SAD_grade.size(); j++) {
                        if (MLa_img[i].SAD_grade[j] > 0.8) {
                            n++;
                            NCC += MLa_img[i].SAD_grade[j];
                        }
                    }

                    if (n != 0) {
                        NCC = NCC / float(n);
                    }
                    PointNum.push_back(n);
                    Ncc.push_back(NCC);
                    NCC = 0, n = 0;
                }
            }
            std::vector<float> res = Compute_avg_std(PointNum, Ncc);
            //std::cout << "avg：" <<res[0]<<"std::"<<res[2] << std::endl;
            //std::cout << "avg：" << res[1] << "std::" << res[3] << std::endl;
            for (long unsigned int i = 0; i < PointNum.size(); i++) {
                //std::cout << "第n张图片：" << PointNum[i] << std::endl;
                if (PointNum[i] == 0) {
                    Map.push_back(std::pair<int, float>(i, -10.0));
                } else {
                    float sorce = 0.5 * ((PointNum[i] - res[0]) / res[2]) + 0.5 * ((Ncc[i] - res[1]) / res[3]);
                    Map.push_back(std::pair<int, float>(i, sorce));
                }
            }

            sort(Map.begin(), Map.end(), Compare_map);

            for (int i = 0; i < 3; i++) {
                res_img.push_back(MLa_img[Map[i].first]);
                //std::cout << "第n张图片："<<Map[i].second << std::endl;
            }
            res.clear();
            Map.clear();

            //对第二圈微透镜进行选择
            for (int i = b1; i < b2; i++) {
                if (MLa_img[i].SAD_grade.size() == 0) {
                    Map.push_back(std::pair<int, float>(1, 0.0));
                } else {
                    for (long unsigned int j = 0; j < MLa_img[i].SAD_grade.size(); j++) {

                        if (MLa_img[i].SAD_grade[j] > 0.8) {
                            n++;
                            NCC += MLa_img[i].SAD_grade[j];
                        }
                    }
                    if (n != 0) {
                        NCC = NCC / float(n);
                    }
                    PointNum.push_back(n);
                    Ncc.push_back(NCC);
                    NCC = 0, n = 0;
                }
            }
            res = Compute_avg_std(PointNum, Ncc);

            for (long unsigned int i = 0; i < PointNum.size(); i++) {
                if (PointNum[i] == 0) {
                    Map.push_back(std::pair<int, float>(i, -10.0));
                } else {
                    float sorce = 0.5 * ((PointNum[i] - res[0]) / res[2]) + 0.5 * ((Ncc[i] - res[1]) / res[3]);
                    Map.push_back(std::pair<int, float>(i, sorce));
                }
            }

            sort(Map.begin(), Map.end(), Compare_map);

            for (int i = 0; i < 3; i++) {
                res_img.push_back(MLa_img[Map[i].first]);
                //std::cout << "第n张图片："<<Map[i].first << std::endl;
            }
            res.clear();
            Map.clear();


            //对第三圈微透镜进行选择
            for (int i = b2; i < MLa_img.size(); i++) {
                if (MLa_img[i].SAD_grade.size() == 0) {
                    Map.push_back(std::pair<int, float>(1, 0.0));
                } else {
                    for (int j = 0; j < MLa_img[i].SAD_grade.size(); j++) {

                        if (MLa_img[i].SAD_grade[j] > 0.8) {
                            n++;
                            NCC += MLa_img[i].SAD_grade[j];
                        }
                    }
                    if (n != 0) {
                        NCC = NCC / float(n);
                    }
                    PointNum.push_back(n);
                    Ncc.push_back(NCC);
                    NCC = 0, n = 0;
                }
            }
            res = Compute_avg_std(PointNum, Ncc);

            for (long unsigned int i = 0; i < PointNum.size(); i++) {
                if (PointNum[i] == 0) {
                    Map.push_back(std::pair<int, float>(i, -10.0));
                } else {
                    float sorce = 0.5 * ((PointNum[i] - res[0]) / res[2]) + 0.5 * ((Ncc[i] - res[1]) / res[3]);
                    Map.push_back(std::pair<int, float>(i, sorce));
                }
            }

            sort(Map.begin(), Map.end(), Compare_map);

            for (int i = 0; i < 3; i++) {
                res_img.push_back(MLa_img[Map[i].first]);
                //std::cout << "第n张图片："<<Map[i].first << std::endl;
            }
            res.clear();
            Map.clear();
        }

        void DepthSolver::Select_NeighborsByNCC(MLA_Problem &problem) {
            // 进该函数前，需检查
            if (!problem.m_NeigKeyPtrVec.empty()) {
                std::cout << "problem.m_NeigKeyPtrVec not Empty!" << std::endl;
                problem.m_NeigKeyPtrVec.clear();
            }

            // 根据基线的长度将其分为三个级别
            float level1_threshold = m_Params.baseline * 1.1;
            float level2_threshold = m_Params.baseline * 1.8;
            for (uint i = 0; i < problem.m_Res_Image_KeyVec.size(); i++) {
                Res_image_Key &res_img = problem.m_Res_Image_KeyVec[i];
                if (res_img.Base_line < level1_threshold) {
                    res_img.m_iLevel = 1;
                } else if (res_img.Base_line <= level2_threshold) {
                    res_img.m_iLevel = 2;
                } else if (res_img.Base_line > level2_threshold) {
                    res_img.m_iLevel = 3;
                }
                res_img.ComputeValidNCCInfo();
            }

            // 分别对不同圈（1～3）的微透镜，计算Score，并选择作为深度估计的邻居图像
            for (int iLevel = 0; iLevel < 3; ++iLevel) {
                problem.Compute_avg_std_key(iLevel); // 计算NCC的均值和标准差
                problem.ComputeScoreByNCC(iLevel);
            }
        }

        void
        DepthSolver::Select_img1(std::vector<cv::Mat> &img, std::vector<cv::Point2f> &center, MLA_Problem &problem) {
            int b1 = 0, b2 = 0;
            //int r1, r2;
            std::vector<std::pair<int, float>> Map;
            int n = 0;
            float NCC1 = 0.0;
            std::vector<int> PointNum;
            std::vector<float> Ncc;
            //对不同基线的进行操作
            for (long unsigned int i = 1; i < problem.res_img.size(); i++) {
                if (problem.res_img[i - 1].Base_line < 200 && problem.res_img[i].Base_line > 200) {
                    b1 = i;
                }
                if (problem.res_img[i - 1].Base_line > 200 && problem.res_img[i - 1].Base_line < 300 &&
                    problem.res_img[i].Base_line > 300) {
                    b2 = i;
                }
            }
            //对第一圈微透镜进行选择
            for (int i = 0; i < b1; i++) {
                if (problem.res_img[i].Ncc_grade.size() == 0) {
                    PointNum.push_back(0);
                    Ncc.push_back(0.0);
                } else if (problem.res_img[i].Ncc_grade.size() > 0) {
                    for (int j = 0; j < problem.res_img[i].Ncc_grade.size(); j++) {
                        if (problem.res_img[i].Ncc_grade[j] > 0.8) {
                            n++;
                            NCC1 += problem.res_img[i].Ncc_grade[j];
                        }
                    }
                    if (n != 0) {
                        NCC1 = NCC1 / float(n);
                    }
                    PointNum.push_back(n);
                    Ncc.push_back(NCC1);
                    NCC1 = 0.0, n = 0;
                }
            }
            std::vector<float> res = Compute_avg_std(PointNum, Ncc);
//        std::cout << "avg：" <<res[0]<<"  std::"<<res[2] << std::endl;
//        std::cout << "avg：" << res[1] << "  std::" << res[3] << std::endl;
            for (long unsigned int i = 0; i < PointNum.size(); i++) {
                //std::cout << "第n张图片：" << PointNum[i] << std::endl;
                int Num_img = problem.res_img[i].num;
                if (PointNum[i] == 0) {
                    Map.push_back(std::pair<int, float>(Num_img, -10.0));
                    Ncc.push_back(0.0);
                } else {
                    float sorce = 0.5 * ((PointNum[i] - res[0]) / res[2]) + 0.5 * ((Ncc[i] - res[1]) / res[3]);
                    Map.push_back(std::pair<int, float>(Num_img, sorce));
                }
            }

            sort(Map.begin(), Map.end(), Compare_map);

            for (int i = 0; i < 3; i++) {
                problem.number.push_back(Map[i].first);
            }
            res.clear();
            Map.clear();
            PointNum.clear();
            Ncc.clear();

            //对第二圈微透镜进行选择
            for (int i = b1; i < b2; i++) {
                if (problem.res_img[i].Ncc_grade.size() == 0) {
                    PointNum.push_back(0);
                    Ncc.push_back(0.0);
                } else {
                    for (long unsigned int j = 0; j < problem.res_img[i].Ncc_grade.size(); j++) {
                        if (problem.res_img[i].Ncc_grade[j] > 0.8) {
                            n++;
                            NCC1 += problem.res_img[i].Ncc_grade[j];
                        }
                    }
                    if (n != 0) {
                        NCC1 = NCC1 / float(n);
                    }
                    PointNum.push_back(n);
                    Ncc.push_back(NCC1);
                    NCC1 = 0, n = 0;
                }
            }
            res = Compute_avg_std(PointNum, Ncc);

            for (long unsigned int i = 0; i < PointNum.size(); i++) {
                int Num_img = problem.res_img[b1 + i].num;
                if (PointNum[i] == 0) {
                    Map.push_back(std::pair<int, float>(Num_img, -10.0));
                } else {
                    float sorce = 0.5 * ((PointNum[i] - res[0]) / res[2]) + 0.5 * ((Ncc[i] - res[1]) / res[3]);
                    Map.push_back(std::pair<int, float>(Num_img, sorce));
                }
            }

            sort(Map.begin(), Map.end(), Compare_map);

            for (int i = 0; i < 3; i++) {
                problem.number.push_back(Map[i].first);
                //std::cout << "第n张图片："<<Map[i].first << std::endl;
            }
            res.clear();
            Map.clear();
            PointNum.clear();
            Ncc.clear();

            //对第三圈微透镜进行选择
            for (long unsigned int i = b2; i < problem.res_img.size(); i++) {
                if (problem.res_img[i].Ncc_grade.size() == 0) {
                    PointNum.push_back(0);
                    Ncc.push_back(0.0);
                } else {
                    for (long unsigned int j = 0; j < problem.res_img[i].Ncc_grade.size(); j++) {

                        if (problem.res_img[i].Ncc_grade[j] > 0.8) {
                            n++;
                            NCC1 += problem.res_img[i].Ncc_grade[j];
                        }
                    }
                    if (n != 0) {
                        NCC1 = NCC1 / float(n);
                    }
                    PointNum.push_back(n);
                    Ncc.push_back(NCC1);
                    NCC1 = 0, n = 0;
                }
            }
            res = Compute_avg_std(PointNum, Ncc);

            for (int i = 0; i < PointNum.size(); i++) {
                int Num_img = problem.res_img[b2 + i].num;
                if (PointNum[i] == 0) {
                    Map.push_back(std::pair<int, float>(Num_img, -10.0));
                } else {
                    float sorce = 0.5 * ((PointNum[i] - res[0]) / res[2]) + 0.5 * ((Ncc[i] - res[1]) / res[3]);
                    Map.push_back(std::pair<int, float>(Num_img, sorce));
                }
            }

            sort(Map.begin(), Map.end(), Compare_map);

            for (int i = 0; i < 3; i++) {
                problem.number.push_back(Map[i].first);
                //std::cout << "第n张图片："<<Map[i].first << std::endl;
            }
            res.clear();
            Map.clear();
            PointNum.clear();
            Ncc.clear();

        }

        void DepthSolver::Select_img(std::vector<cv::Mat> &img, std::vector<cv::Point2f> &center,
                                     MLA_Problem &problem, int i, int j) {
            int a1 = (i + 0) * m_Params.mla_u_size + (j - 1);
            int a2 = (i + 0) * m_Params.mla_u_size + (j + 1);
            int a3 = (i + 0) * m_Params.mla_u_size + (j + 2);
            int a4 = (i - 1) * m_Params.mla_u_size + (j);
            int a5 = (i - 1) * m_Params.mla_u_size + (j - 2);
            int a6 = (i + 1) * m_Params.mla_u_size + (j);
            problem.number.push_back(a1);
            problem.number.push_back(a2);
            problem.number.push_back(a3);
            problem.number.push_back(a4);
            problem.number.push_back(a5);
            problem.number.push_back(a6);
            for (long unsigned int k = 0; k < problem.number.size(); ++k) {
                if (problem.number[k] < 0 || problem.number[k] > m_Params.mla_u_size * m_Params.mla_v_size - 1) {
                    problem.number.erase(problem.number.begin() + k);
                }
            }
        }

        void DepthSolver::CollectNeighborKey(MLA_Problem &problem, QuadTreeTileKeyPtr ptrKey) {
            QuadTreeTileInfoMap::iterator itr = m_MLA_info_map.find(ptrKey);
            if (itr != m_MLA_info_map.end()) {
                MLA_InfoPtr ptrInfo = itr->second;
                if (ptrInfo->IsAbandonByArea() == false) {
                    problem.m_NeigKeyPtrVec.push_back(ptrKey);
                }
            }
        }

        void DepthSolver::CollectNeigFromFixedPosition(MLA_Problem &problem) {
            // 选择固定位置的邻居，其原则如下：
            // 邻居总数为6
            // 根据前微透镜图像所处的阵列的行列位置，分为一般和特殊两类
            // 其中，特殊类又分为4个边和4个角
            QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
            int32_t tile_X = ptrKey->GetTileX();
            int32_t tile_Y = ptrKey->GetTileY();

            if (tile_X > 1 && tile_X < m_Params.mla_u_size - 2 &&
                tile_Y > 0 && tile_Y < m_Params.mla_v_size - 1) {
                // 邻居：右
                QuadTreeTileKeyPtr ptrKey_Right = ptrKey->CreateNeighborKey(1, 0);
                CollectNeighborKey(problem, ptrKey_Right);
                // 邻居：左
                QuadTreeTileKeyPtr ptrKey_Left = ptrKey->CreateNeighborKey(-1, 0);
                CollectNeighborKey(problem, ptrKey_Left);
                // 邻居：右2
                QuadTreeTileKeyPtr ptrKey_Right2 = ptrKey->CreateNeighborKey(2, 0);
                CollectNeighborKey(problem, ptrKey_Right2);

                // 邻居：上
                QuadTreeTileKeyPtr ptrKey_Up = ptrKey->CreateNeighborKey(0, -1);
                CollectNeighborKey(problem, ptrKey_Up);

                // 邻居 : 上，左二
                QuadTreeTileKeyPtr ptrKey_Up_Left2 = ptrKey->CreateNeighborKey(-2, -1);
                CollectNeighborKey(problem, ptrKey_Up_Left2);

                // 邻居：下
                QuadTreeTileKeyPtr ptrKey_Down = ptrKey->CreateNeighborKey(0, 1);
                CollectNeighborKey(problem, ptrKey_Down);

                // 邻居 : 左二 lzd
                QuadTreeTileKeyPtr ptrKey_Left2 = ptrKey->CreateNeighborKey(-2, 0);
                CollectNeighborKey(problem, ptrKey_Left2);

                // 邻居 : 上，左一 lzd
                QuadTreeTileKeyPtr ptrKey_up_left = ptrKey->CreateNeighborKey(-1, -1);
                CollectNeighborKey(problem, ptrKey_up_left);

                // 邻居 : 上，右一 lzd
                QuadTreeTileKeyPtr ptrKey_up_right = ptrKey->CreateNeighborKey(1, -1);
                CollectNeighborKey(problem, ptrKey_up_right);

                // 邻居 : 下，左一 lzd
                QuadTreeTileKeyPtr ptrKey_down_left = ptrKey->CreateNeighborKey(-1, 1);
                CollectNeighborKey(problem, ptrKey_down_left);

                // 邻居 : 下，右一 lzd
                QuadTreeTileKeyPtr ptrKey_down_right = ptrKey->CreateNeighborKey(1, 1);
                CollectNeighborKey(problem, ptrKey_down_right);

                // 邻居 : 下，右二 lzd
                QuadTreeTileKeyPtr ptrKey_down_right2 = ptrKey->CreateNeighborKey(2, 1);
                CollectNeighborKey(problem, ptrKey_down_right2);
            } else {
                if (tile_X == 0 && tile_Y == 0) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：右下
                    QuadTreeTileKeyPtr ptrKey_right_down = ptrKey->CreateNeighborKey(1, 1);
                    CollectNeighborKey(problem, ptrKey_right_down);

                    // 邻居：右2 下2
                    QuadTreeTileKeyPtr ptrKey_right2_down2 = ptrKey->CreateNeighborKey(2, 2);
                    CollectNeighborKey(problem, ptrKey_right2_down2);

                    // 邻居： 下2
                    QuadTreeTileKeyPtr ptrKey_down2 = ptrKey->CreateNeighborKey(0, 2);
                    CollectNeighborKey(problem, ptrKey_down2);
                } else if ((tile_X == 1 && tile_Y == 0) || ((tile_X != m_Params.mla_u_size - 2 ||
                                                             tile_X != m_Params.mla_u_size - 1) && tile_Y == 0)) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：左下
                    QuadTreeTileKeyPtr ptrKey_left_down = ptrKey->CreateNeighborKey(-1, 1);
                    CollectNeighborKey(problem, ptrKey_left_down);

                    // 邻居：右下
                    QuadTreeTileKeyPtr ptrKey_right_down = ptrKey->CreateNeighborKey(1, 1);
                    CollectNeighborKey(problem, ptrKey_right_down);
                } else if (tile_X == 0 && tile_Y == m_Params.mla_v_size - 1) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居：右上
                    QuadTreeTileKeyPtr ptrKey_right_up = ptrKey->CreateNeighborKey(1, -1);
                    CollectNeighborKey(problem, ptrKey_right_up);

                    // 邻居：右2上2
                    QuadTreeTileKeyPtr ptrKey_right2_up2 = ptrKey->CreateNeighborKey(2, -2);
                    CollectNeighborKey(problem, ptrKey_right2_up2);

                    // 邻居：上2
                    QuadTreeTileKeyPtr ptrKey_up2 = ptrKey->CreateNeighborKey(0, -2);
                    CollectNeighborKey(problem, ptrKey_up2);
                } else if (tile_X == 1 && tile_Y == m_Params.mla_v_size - 1) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居：左上
                    QuadTreeTileKeyPtr ptrKey_left_up = ptrKey->CreateNeighborKey(-1, -1);
                    CollectNeighborKey(problem, ptrKey_left_up);

                    // 邻居：右上
                    QuadTreeTileKeyPtr ptrKey_right_up = ptrKey->CreateNeighborKey(1, -1);
                    CollectNeighborKey(problem, ptrKey_right_up);
                } else if (tile_X == m_Params.mla_u_size - 1 && tile_Y == m_Params.mla_v_size - 1) {
                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居: 上，左二
                    QuadTreeTileKeyPtr ptrKey_up_left2 = ptrKey->CreateNeighborKey(-2, -1);
                    CollectNeighborKey(problem, ptrKey_up_left2);

                    // 邻居：左2
                    QuadTreeTileKeyPtr ptrKey_left2 = ptrKey->CreateNeighborKey(-2, 0);
                    CollectNeighborKey(problem, ptrKey_left2);

                    // 邻居：上2
                    QuadTreeTileKeyPtr ptrKey_up2 = ptrKey->CreateNeighborKey(-2, 0);
                    CollectNeighborKey(problem, ptrKey_up2);

                    // 邻居：左上
                    QuadTreeTileKeyPtr ptrKey_left_up = ptrKey->CreateNeighborKey(-1, -1);
                    CollectNeighborKey(problem, ptrKey_left_up);
                } else if (tile_X == m_Params.mla_u_size - 2 && tile_Y == m_Params.mla_v_size - 1) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居 : 上，左二
                    QuadTreeTileKeyPtr ptrKey_left2 = ptrKey->CreateNeighborKey(-2, -1);
                    CollectNeighborKey(problem, ptrKey_left2);

                    // 邻居：左上
                    QuadTreeTileKeyPtr ptrKey_left_up = ptrKey->CreateNeighborKey(-1, -1);
                    CollectNeighborKey(problem, ptrKey_left_up);

                    // 邻居：右上
                    QuadTreeTileKeyPtr ptrKey_right_up = ptrKey->CreateNeighborKey(1, -1);
                    CollectNeighborKey(problem, ptrKey_right_up);
                } else if (tile_X == m_Params.mla_u_size - 1 && tile_Y == 0) {
                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：左2 下2
                    QuadTreeTileKeyPtr ptrKey_left2_down2 = ptrKey->CreateNeighborKey(-2, 2);
                    CollectNeighborKey(problem, ptrKey_left2_down2);

                    // 邻居：左2
                    QuadTreeTileKeyPtr ptrKey_left2 = ptrKey->CreateNeighborKey(-2, 0);
                    CollectNeighborKey(problem, ptrKey_left2);

                    // 邻居： 下2
                    QuadTreeTileKeyPtr ptrKey_down2 = ptrKey->CreateNeighborKey(0, 2);
                    CollectNeighborKey(problem, ptrKey_down2);

                    // 邻居：左下
                    QuadTreeTileKeyPtr ptrKey_left_down = ptrKey->CreateNeighborKey(-1, 1);
                    CollectNeighborKey(problem, ptrKey_left_down);
                } else if (tile_X == m_Params.mla_u_size - 2 && tile_Y == 0) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：左2
                    QuadTreeTileKeyPtr ptrKey_left2 = ptrKey->CreateNeighborKey(-2, 0);
                    CollectNeighborKey(problem, ptrKey_left2);

                    // 邻居：左下
                    QuadTreeTileKeyPtr ptrKey_left_down = ptrKey->CreateNeighborKey(-1, 1);
                    CollectNeighborKey(problem, ptrKey_left_down);

                    // 邻居：右下
                    QuadTreeTileKeyPtr ptrKey_right_down = ptrKey->CreateNeighborKey(1, 1);
                    CollectNeighborKey(problem, ptrKey_right_down);
                } else if (tile_X == 0) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：右上
                    QuadTreeTileKeyPtr ptrKey_right_up = ptrKey->CreateNeighborKey(1, -1);
                    CollectNeighborKey(problem, ptrKey_right_up);

                    // 邻居：右下
                    QuadTreeTileKeyPtr ptrKey_right_down = ptrKey->CreateNeighborKey(1, 1);
                    CollectNeighborKey(problem, ptrKey_right_down);
                } else if (tile_X == 1) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：左上
                    QuadTreeTileKeyPtr ptrKey_left_up = ptrKey->CreateNeighborKey(-1, -1);
                    CollectNeighborKey(problem, ptrKey_left_up);
                } else if (tile_Y == m_Params.mla_v_size - 1) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：右2
                    QuadTreeTileKeyPtr ptrKey_right2 = ptrKey->CreateNeighborKey(2, 0);
                    CollectNeighborKey(problem, ptrKey_right2);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居 : 上，左二
                    QuadTreeTileKeyPtr ptrKey_up_left2 = ptrKey->CreateNeighborKey(-2, -1);
                    CollectNeighborKey(problem, ptrKey_up_left2);

                    // 邻居：右上
                    QuadTreeTileKeyPtr ptrKey_up_right = ptrKey->CreateNeighborKey(1, -1);
                    CollectNeighborKey(problem, ptrKey_up_right);
                } else if (tile_X == m_Params.mla_u_size - 1) {
                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居 : 上，左二
                    QuadTreeTileKeyPtr ptrKey_up_left2 = ptrKey->CreateNeighborKey(-2, -1);
                    CollectNeighborKey(problem, ptrKey_up_left2);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：左上
                    QuadTreeTileKeyPtr ptrKey_left_up = ptrKey->CreateNeighborKey(-1, -1);
                    CollectNeighborKey(problem, ptrKey_left_up);

                    // 邻居：左下
                    QuadTreeTileKeyPtr ptrKey_left_down = ptrKey->CreateNeighborKey(-1, 1);
                    CollectNeighborKey(problem, ptrKey_left_down);
                } else if (tile_X == m_Params.mla_u_size - 2) {
                    // 邻居：右
                    QuadTreeTileKeyPtr ptrKey_right = ptrKey->CreateNeighborKey(1, 0);
                    CollectNeighborKey(problem, ptrKey_right);

                    // 邻居：左
                    QuadTreeTileKeyPtr ptrKey_left = ptrKey->CreateNeighborKey(-1, 0);
                    CollectNeighborKey(problem, ptrKey_left);

                    // 邻居：上
                    QuadTreeTileKeyPtr ptrKey_up = ptrKey->CreateNeighborKey(0, -1);
                    CollectNeighborKey(problem, ptrKey_up);

                    // 邻居 : 上，左二
                    QuadTreeTileKeyPtr ptrKey_up_left2 = ptrKey->CreateNeighborKey(-2, -1);
                    CollectNeighborKey(problem, ptrKey_up_left2);

                    // 邻居：下
                    QuadTreeTileKeyPtr ptrKey_down = ptrKey->CreateNeighborKey(0, 1);
                    CollectNeighborKey(problem, ptrKey_down);

                    // 邻居：右下
                    QuadTreeTileKeyPtr ptrKey_right_down = ptrKey->CreateNeighborKey(1, 1);
                    CollectNeighborKey(problem, ptrKey_right_down);
                }
            }
        }

        void DepthSolver::SelectNeighborsFromFeatures(QuadTreeTileKeyPtr ptrKey) {
            // 遍历所有帧
            for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
                 itr != m_MIA_problem_map_map.end(); itr++) {
                QuadTreeProblemMap &problem_map = itr->second;
                QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
                if (itrP == problem_map.end())
                    continue;

                MLA_Problem &problem = itrP->second;
                CollectMLANeigImagesByPOSE(problem);
                SortMLANeigImagesByLength(problem);
                Sift_MatchFromTileKey(problem, problem_map);
                Compute_NCCFromTileKey(problem);
                Select_NeighborsByNCC(problem); // 排序

                ConfirmProblemForEstimation(problem);
            }
        }

        void DepthSolver::SelectNeighborsFromFeaturesSequence(QuadTreeTileKeyPtr ptrKey, std::string &strName,
                                                              QuadTreeProblemMap &problem_map) {
            QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
            if (itrP == problem_map.end())
                return;

            MLA_Problem &problem = itrP->second;
            CollectMLANeigImagesByPOSE(problem);
            SortMLANeigImagesByLength(problem);
            Sift_MatchFromTileKey(problem, problem_map);
            Compute_NCCFromTileKey(problem);
            Select_NeighborsByNCC(problem); // 排序

            ConfirmProblemForEstimation(problem);
        }

        void DepthSolver::SelectNeighborsFromFixedPosition(QuadTreeTileKeyPtr ptrKey) {
            // 选择固定位置的邻居，其原则如下：
            // 邻居总数为6
            // 根据前微透镜图像所处的阵列的行列位置，分为一般和特殊两类
            // 其中，特殊类又分为4个边和4个角

            // 遍历所有帧
            LOG_INFO("Neighbors Old!");
            for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
                 itr != m_MIA_problem_map_map.end(); itr++) {
                QuadTreeProblemMap &problem_map = itr->second;
                QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
                if (itrP == problem_map.end())
                    continue;

                MLA_Problem &problem = itrP->second;
                CollectNeigFromFixedPosition(problem);

                // 排序
                SortMLANeigImagesByLength(problem);

                // 确定哪些problem不需要深度估计
                ConfirmProblemForEstimation(problem);
                problem.WriteNeighbosInfo_old();
            }
        }

        void DepthSolver::SelectNeighborsFromFixedPositionSequence(QuadTreeTileKeyPtr ptrKey,
                                                                   std::string &strName,
                                                                   QuadTreeProblemMap &problem_map) {
            // 选择固定位置的邻居，其原则如下：
            // 邻居总数为6
            // 根据前微透镜图像所处的阵列的行列位置，分为一般和特殊两类
            // 其中，特殊类又分为4个边和4个角

            // 遍历所有帧
            QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
            if (itrP == problem_map.end())
                return;
            MLA_Problem &problem = itrP->second;
            CollectNeigFromFixedPosition(problem);

            // 排序
            SortMLANeigImagesByLength(problem);
            // 确定哪些problem不需要深度估计
            ConfirmProblemForEstimation(problem);
        }

        void DepthSolver::ProcessProblem_LF(const std::vector<cv::Mat> &img,
                                            const std::vector<cv::Point2f> &center,
                                            const MLA_Problem &mlaProblem,
                                            std::vector<DisparityAndNormal> &DNS,
                                            float &Base, int i, int j) {
            cudaSetDevice(0);
            LF_ACMP lf_acmp(m_Params);
            //
            int n = i * m_Params.mla_u_size + j;
            int n1 = i * m_Params.mla_u_size + j - 1;
            std::vector<float4> Plane;
            std::vector<float> Cost;
            // 左侧微透镜
            if (j > 0 && DNS[n1].m_StereoStage == eSS_ACMH_Finished) {
                //遍历微图像中的像素，计算视差平面和代价
                for (int i = 0; i < m_Params.mi_height_for_match; ++i) {
                    for (int j = 0; j < m_Params.mi_width_for_match; ++j) {
                        int n2 = i * m_Params.mi_width_for_match + j;
                        Plane.push_back(DNS[n1].ph_cuda[n2]);
                        Cost.push_back(DNS[n1].c_cuda[n2]);
                    }
                }
            }
            //计算参考微透镜和邻居微透镜中心点坐标 定义视差范围等参数
            lf_acmp.InuputInitializationLF(img, center, mlaProblem, Base);
            //CUDA内存等参数设置
            lf_acmp.CudaSpaceInitializationLF(img, center, mlaProblem, Plane, Cost);
            lf_acmp.RunPatchMatchLF();

            const int width = img[0].cols;
            const int height = img[0].rows;

            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    int center = row * width + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
                    float cost = lf_acmp.GetCost(center);
                    DNS[n].d_cuda[center] = plane_hypothesis.w;
                    DNS[n].ph_cuda[center] = plane_hypothesis;
                    DNS[n].c_cuda[center] = cost;
                }
            }

            //lf_acmp.Delete_pc();
            DNS[n].m_StereoStage = eSS_ACMH_Finished;
        }

        void DepthSolver::ProcessProblem_planner_LF_TileKey(MLA_Problem &problem, QuadTreeProblemMap &problem_map,
                                                            QuadTreeDisNormalMap &dis_normal_map,
                                                            std::string &strFrameName,
                                                            bool geom_consistency) {
            if (m_top_device == -1) {
                std::cout << "PPLFT: Error! Find GPU device index is: " << m_top_device << std::endl;
                return;
            }
            cudaError_t err = cudaSetDevice(m_top_device);
            if (err != cudaSuccess) {
                std::cout << "PPLFT: Error! cudaSetDevice: " << err << std::endl;
                return;
            }

            // 当前微图像的key
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            LF_ACMP lf_acmp(m_Params);
            lf_acmp.SetTileKey(ptrCurKey->GetTileX(), ptrCurKey->GetTileY());
            std::vector<float4> planeVec;
            std::vector<float> costVec;

            // 左侧微图像
            QuadTreeBoundingbox box;
            int tile_X_LeftNeig = ptrCurKey->GetTileX() - 1;
            int tile_Y_LeftNeig = ptrCurKey->GetTileY();
            QuadTreeTileKeyPtr ptrLeftNeig_Key = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_X_LeftNeig,
                                                                                 tile_Y_LeftNeig);
            QuadTreeDisNormalMap::iterator itr_DN_left = dis_normal_map.find(ptrLeftNeig_Key);
            // if (itr_DN_left != dis_normal_map.end())
            // {
            //     DisparityAndNormalPtr ptrDis_Left = itr_DN_left->second;
            //     if(ptrDis_Left->m_StereoStage == eSS_ACMH_Finished ||
            //        ptrDis_Left->m_StereoStage == eSS_PlannerPrior_Finished)
            //     {
            //         // 遍历微图像中的像素，计算视差平面和代价
            //         for (int i = 0; i < g_MIA_match_height; ++i)
            //         {
            //             for (int j = 0; j < g_MIA_match_width; ++j)
            //             {
            //                 int n2 = i*g_MIA_match_width+j;
            //                 planeVec.push_back(ptrDis_Left->ph_cuda[n2]);
            //                 costVec.push_back(ptrDis_Left->c_cuda[n2]);
            //             }
            //         }
            //     }
            //     else
            //     {
            //         std::cout << "Left image stereo not finish: " << LeftNeig_Key.StrRemoveLOD() << std::endl;
            //     }
            // }
            // else
            // {
            //     std::cout << "Left neigbors not found: " << LeftNeig_Key.StrRemoveLOD() << std::endl;
            // }

            QuadTreeDisNormalMap::iterator itr_DN = dis_normal_map.find(ptrCurKey);
            if (itr_DN == dis_normal_map.end()) {
                std::cout << "Current Image not found: " << ptrCurKey->StrRemoveLOD().c_str() << std::endl;
                return;
            }

            // 计算参考微透镜和邻居微透镜中心点坐标，定义视差范围等参数
            lf_acmp.InuputInitialization_LF_TileKey(m_MLA_info_map, problem, problem_map, planeVec, costVec);
            lf_acmp.CudaSpaceInitialization_LF_TileKey();
            lf_acmp.RunPatchMatchLF();
            // 深度估计结果：cuda--->host
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            cv::Mat_<float> depths = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match,
                                                    CV_32FC1);

            cv::Mat neighinfo = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match, CV_8UC3);
            std::map<unsigned int, std::vector<int2>> neigviews_map;
            for (int row = 0; row < m_Params.mi_height_for_match; row++) {
                for (int col = 0; col < m_Params.mi_width_for_match; col++) {
                    int index = row * m_Params.mi_width_for_match + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                    float cost = lf_acmp.GetCost(index);
                    ptrDN->d_cuda[index] = plane_hypothesis.w;
                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->c_cuda[index] = cost;
                    depths(row, col) = plane_hypothesis.w;

                    unsigned int neig_viewBit = lf_acmp.GetSelected_viewIndexs(col, row);
                    //lf_acmp.SelectedViewIndexConvert(problem, neig_viewBit);
                    if (neig_viewBit > 0) {
                        std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.find(neig_viewBit);
                        if (itr == neigviews_map.end()) {
                            std::vector<int2> pixel_coords;
                            int2 p = make_int2(row, col);
                            pixel_coords.push_back(p);
                            neigviews_map[neig_viewBit] = pixel_coords;
                        } else {
                            std::vector<int2> &pixel_coords = itr->second;
                            int2 p = make_int2(row, col);
                            pixel_coords.push_back(p);
                        }
                    }
                }
            }

            unsigned int ix = 0;
            for (std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.begin();
                 itr != neigviews_map.end(); itr++, ix++) {
                std::vector<int2> &pixel_coords = itr->second;
                cv::Vec3b color = m_Colors[ix];

                for (int i = 0; i < pixel_coords.size(); i++) {
                    int2 &pixel_coord = pixel_coords[i];
                    neighinfo.at<cv::Vec3b>(pixel_coord.x, pixel_coord.y) = color;
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;

            // 处理写出和测试
            {
                std::string strMLAResultFolder = m_strRootPath + LF_DEPTH_INTRA_NAME + strFrameName;
                boost::filesystem::path dir_save_path(strMLAResultFolder);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                    }
                }
                strMLAResultFolder += LF_MVS_RESULT_DATA_NAME;
                {
                    boost::filesystem::path dir_save_path(strMLAResultFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                        }
                    }
                }

                // 写出微透镜图像的视差图
                static bool bWriteMLA_DisparityMap = true; // false
                if (bWriteMLA_DisparityMap) {
                    double d_factor = 5.0;
                    // 创建路径
                    std::string strMLADisFolder = strMLAResultFolder + LF_MLA_DISPARITYMAPS_NAME;
                    boost::filesystem::path dir_save_path(strMLADisFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
                        }
                    }

                    cv::Mat MLA_DisMap_gray = cv::Mat::zeros(m_Params.mi_width_for_match, m_Params.mi_height_for_match,
                                                             CV_8UC1);
                    for (int mla_row = 0; mla_row < m_Params.mi_height_for_match; ++mla_row) {
                        for (int mla_col = 0; mla_col < m_Params.mi_width_for_match; ++mla_col) {
                            int index = mla_row * m_Params.mi_width_for_match + mla_col;
                            float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                            MLA_DisMap_gray.at<uchar>(mla_row, mla_col) =
                                    (plane_hypothesis.w / m_Params.mi_width_for_match) * 255 * d_factor;
                        }
                    }
                    std::string strMLADisPath_key = strMLADisFolder + ptrCurKey->StrRemoveLOD();
                    cv::Mat disp_color;
                    applyColorMap(MLA_DisMap_gray, disp_color, cv::COLORMAP_JET);
                    //cv::imwrite(strMLADisPath_key + std::string("_Dis_gray.png"), MLA_DisMap_gray);
                    cv::imwrite(strMLADisPath_key + std::string("_color.png"), disp_color);
                    cv::imwrite(strMLADisPath_key + std::string("_neighborInfo.png"), neighinfo);
                }
            }


            // 平面先验优化深度值
            // {
            //     ptrDN->m_StereoStage = eSS_PlannerPrior_Begin;
            //     lf_acmp.SetPlanarPriorParams();
            //
            //     cv::Rect imageRC(0, 0, m_Params.mi_width_for_match, m_Params.mi_height_for_match);
            //
            //     // 获取高可信稀疏的对应关系
            //     std::vector<cv::Point> support2DPoints;
            //     lf_acmp.GetSupportPoints(support2DPoints); // LZD, 修改：5--->3
            //     const std::vector<Triangle> triangles = lf_acmp.DelaunayTriangulation(imageRC, support2DPoints);//未修改
            //
            //     bool bWrite_tri_Image = true; //
            //     if (bWrite_tri_Image)
            //     {
            //         cv::Mat refImage = lf_acmp.GetReferenceImage().clone();
            //         std::vector<cv::Mat> mbgr(3);
            //         mbgr[0] = refImage.clone();
            //         mbgr[1] = refImage.clone();
            //         mbgr[2] = refImage.clone();
            //         cv::Mat srcImage;
            //         cv::merge(mbgr, srcImage);
            //         for (const auto triangle : triangles)
            //         {
            //             if (imageRC.contains(triangle.pt1) &&
            //                 imageRC.contains(triangle.pt2) &&
            //                 imageRC.contains(triangle.pt3))
            //             {
            //                 cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
            //                 cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
            //                 cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            //             }
            //         }
            //
            //         // 创建路径
            //         std::string strMLAResultFolder  = m_strRootPath + LF_DEPTH_INTRA_NAME + strFrameName;
            //         std::string strMLADisFolder = strMLAResultFolder + LF_MVS_RESULT_DATA_NAME + LF_MLA_DISPARITYMAPS_PLANNER_NAME;
            //         boost::filesystem::path dir_save_path(strMLADisFolder);
            //         if (!boost::filesystem::exists(dir_save_path))
            //         {
            //             if (!boost::filesystem::create_directory(dir_save_path))
            //             {
            //                 std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
            //             }
            //         }
            //         std::string triangulation_path = strMLADisFolder + ptrCurKey->StrRemoveLOD()+".png";
            //         cv::imwrite(triangulation_path, srcImage);
            //     }
            //
            //     cv::Mat_<float> mask_tri = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match, CV_32FC1);
            //     std::vector<float4> planeParams_tri;
            //     planeParams_tri.clear();
            //     for (uint32_t idx = 0; idx < triangles.size(); idx++)
            //     {
            //         const Triangle& triangle = triangles.at(idx);
            //         if (imageRC.contains(triangle.pt1) &&
            //             imageRC.contains(triangle.pt2) &&
            //             imageRC.contains(triangle.pt3))
            //         {
            //             float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) +
            //                         pow(triangle.pt1.y - triangle.pt2.y, 2));
            //             float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) +
            //                         pow(triangle.pt1.y - triangle.pt3.y, 2));
            //             float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) +
            //                          pow(triangle.pt2.y - triangle.pt3.y, 2));
            //
            //             float max_edge_length = std::max(L01, std::max(L02, L12));
            //             float step = 1.0 / max_edge_length;
            //             for (float p = 0; p < 1.0; p += step)
            //             {
            //                 for (float q = 0; q < 1.0-p; q += step)
            //                 {
            //                     int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
            //                     int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
            //                     mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
            //                 }
            //             }
            //
            //             // 估计平面（以面法线表示）： estimate plane parameter
            //             float4 n4 = lf_acmp.GetPriorPlaneParams(triangle, depths); // 修改过
            //             planeParams_tri.push_back(n4);
            //         }
            //     }
            //
            //     cv::Mat_<float> prior_depths = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match, CV_32FC1);
            //     for (int col = 0; col < m_Params.mi_width_for_match; ++col)
            //     {
            //         for (int row = 0; row < m_Params.mi_height_for_match; ++row)
            //         {
            //             if (mask_tri(row, col) > 0)
            //             {
            //                 float d = lf_acmp.GetDepthFromPlaneParam_LF_Tilekey(planeParams_tri[mask_tri(row, col) - 1], col, row);
            //                 if (d <= lf_acmp.GetMaxDepth() && d >= lf_acmp.GetMinDepth())
            //                 {
            //                     prior_depths(row, col) = d;
            //                 }
            //                 else
            //                 {
            //                     mask_tri(row, col) = 0;
            //                 }
            //             }
            //         }
            //     }
            //     // std::string depth_path = result_folder + "/depths_prior.dmb";
            //     //  writeDepthDmb(depth_path, priordepths);
            //     lf_acmp.CudaPlanarPriorInitialization_LF_Tilekey(planeParams_tri, mask_tri);
            //     lf_acmp.RunPatchMatchLF_plane();
            //
            //     // 存储结果
            //     for (int col = 0; col < m_Params.mi_width_for_match; ++col)
            //     {
            //         for (int row = 0; row < m_Params.mi_height_for_match; ++row)
            //         {
            //             int center = row * m_Params.mi_width_for_match + col;
            //             float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
            //             float cost = lf_acmp.GetCost(center);
            //             ptrDN->ph_cuda[center] = plane_hypothesis;
            //             ptrDN->d_cuda[center] = plane_hypothesis.w;
            //             ptrDN->c_cuda[center] = cost;
            //         }
            //     }
            //     ptrDN->m_StereoStage = eSS_PlannerPrior_Finished;
            // }
            lf_acmp.ReleaseMemory(); // TODO: LZD 释放内存
        }

        void DepthSolver::ProcessProblem_HP_LF_TileKey(MLA_Problem &problem,
                                                       QuadTreeProblemMap &problem_map,
                                                       QuadTreeDisNormalMap &dis_normal_map,
                                                       std::string &strName,
                                                       bool geom_consistency) {
            if (m_top_device == -1) {
                std::cout << "PPHPLFT: Error! Find GPU device index is: " << m_top_device << std::endl;
                return;
            }
            cudaError_t err = cudaSetDevice(m_top_device);
            if (err != cudaSuccess) {
                std::cout << "PPHPLFT: Error! cudaSetDevice: " << err << std::endl;
                return;
            }

            // 当前微图像的key
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            LF_ACMP lf_acmp(m_Params);
            lf_acmp.SetTileKey(ptrCurKey->GetTileX(), ptrCurKey->GetTileY());
            std::vector<float4> planeVec;
            std::vector<float> costVec;

            // 左侧微透镜
            QuadTreeBoundingbox box;
            int tile_X_LeftNeig = ptrCurKey->GetTileX() - 1;
            int tile_Y_LeftNeig = ptrCurKey->GetTileY();
            QuadTreeTileKeyPtr ptrLeftNeig_Key = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_X_LeftNeig,
                                                                                 tile_Y_LeftNeig, box);
            QuadTreeDisNormalMap::iterator itr_DN_left = dis_normal_map.find(ptrLeftNeig_Key);
            // if (itr_DN_left != dis_normal_map.end())
            // {
            //     DisparityAndNormalPtr ptrDis_Left = itr_DN_left->second;
            //     if(ptrDis_Left->m_StereoStage == eSS_ACMH_Finished ||
            //        ptrDis_Left->m_StereoStage == eSS_PlannerPrior_Finished)
            //     {
            //         // 遍历微图像中的像素，计算视差平面和代价
            //         for (int i = 0; i < g_MIA_match_height; ++i)
            //         {
            //             for (int j = 0; j < g_MIA_match_width; ++j)
            //             {
            //                 int n2 = i*g_MIA_match_width+j;
            //                 planeVec.push_back(ptrDis_Left->ph_cuda[n2]);
            //                 costVec.push_back(ptrDis_Left->c_cuda[n2]);
            //             }
            //         }
            //     }
            //     else
            //     {
            //         std::cout << "Left image stereo not finish: " << LeftNeig_Key.StrRemoveLOD() << std::endl;
            //     }
            // }
            // else
            // {
            //     std::cout << "Left neigbors not found: " << LeftNeig_Key.StrRemoveLOD() << std::endl;
            // }

            QuadTreeDisNormalMap::iterator itr_DN = dis_normal_map.find(ptrCurKey);
            if (itr_DN == dis_normal_map.end()) {
                std::cout << "Current Image not found: " << ptrCurKey->StrRemoveLOD().c_str() << std::endl;
                return;
            }

            // 计算参考微透镜和邻居微透镜中心点坐标，定义视差范围等参数
            lf_acmp.InuputInitialization_LF_TileKey(m_MLA_info_map, problem, problem_map, planeVec, costVec);
            lf_acmp.CudaSpaceInitialization_LF_TileKey();
            lf_acmp.RunPatchMatchLF();
            // 深度估计结果
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            cv::Mat_<float> depths = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match,
                                                    CV_32FC1);

            cv::Mat neighinfo = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match, CV_8UC3);
            std::map<unsigned int, std::vector<int2>> neigviews_map;
            for (int row = 0; row < m_Params.mi_height_for_match; row++) {
                for (int col = 0; col < m_Params.mi_width_for_match; col++) {
                    int index = row * m_Params.mi_width_for_match + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                    float cost = lf_acmp.GetCost(index);
                    ptrDN->d_cuda[index] = plane_hypothesis.w;
                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->c_cuda[index] = cost;
                    depths(row, col) = plane_hypothesis.w;

                    unsigned int neig_viewBit = lf_acmp.GetSelected_viewIndexs(col, row);
                    //lf_acmp.SelectedViewIndexConvert(problem, neig_viewBit);
                    if (neig_viewBit > 0) {
                        std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.find(neig_viewBit);
                        if (itr == neigviews_map.end()) {
                            std::vector<int2> pixel_coords;
                            int2 p = make_int2(row, col);
                            pixel_coords.push_back(p);
                            neigviews_map[neig_viewBit] = pixel_coords;
                        } else {
                            std::vector<int2> &pixel_coords = itr->second;
                            int2 p = make_int2(row, col);
                            pixel_coords.push_back(p);
                        }
                    }
                }
            }

            unsigned int ix = 0;
            for (std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.begin();
                 itr != neigviews_map.end(); itr++, ix++) {
                std::vector<int2> &pixel_coords = itr->second;
                cv::Vec3b color = m_Colors[ix];

                for (int i = 0; i < pixel_coords.size(); i++) {
                    int2 &pixel_coord = pixel_coords[i];
                    neighinfo.at<cv::Vec3b>(pixel_coord.x, pixel_coord.y) = color;
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;

            // 处理写出和测试
            {
                std::string strMLAResultFolder = m_strRootPath + LF_DEPTH_INTRA_NAME + strName;
                boost::filesystem::path dir_save_path(strMLAResultFolder);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                    }
                }
                strMLAResultFolder += LF_MVS_RESULT_DATA_NAME;
                {
                    boost::filesystem::path dir_save_path(strMLAResultFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                        }
                    }
                }

                // 写出微透镜图像的视差图
                static bool bWriteMLA_DisparityMap = true;
                if (bWriteMLA_DisparityMap) {
                    double d_factor = 5.0;
                    // 创建路径
                    std::string strMLADisFolder = strMLAResultFolder + LF_MLA_DISPARITYMAPS_NAME;
                    boost::filesystem::path dir_save_path(strMLADisFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
                        }
                    }

                    cv::Mat MLA_DisMap_gray = cv::Mat::zeros(m_Params.mi_width_for_match, m_Params.mi_height_for_match,
                                                             CV_8UC1);
                    for (int mla_row = 0; mla_row < m_Params.mi_height_for_match; ++mla_row) {
                        for (int mla_col = 0; mla_col < m_Params.mi_width_for_match; ++mla_col) {
                            int index = mla_row * m_Params.mi_width_for_match + mla_col;
                            float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                            MLA_DisMap_gray.at<uchar>(mla_row, mla_col) =
                                    (plane_hypothesis.w / m_Params.mi_width_for_match) * 255 * d_factor;
                        }
                    }
                    std::string strMLADisPath_key = strMLADisFolder + ptrCurKey->StrRemoveLOD();
                    cv::Mat disp_color;
                    applyColorMap(MLA_DisMap_gray, disp_color, cv::COLORMAP_JET);
                    //cv::imwrite(strMLADisPath_key + std::string("_Dis_gray.png"), MLA_DisMap_gray);
                    cv::imwrite(strMLADisPath_key + std::string("_color.png"), disp_color);
                    cv::imwrite(strMLADisPath_key + std::string("_neighborInfo.png"), neighinfo);
                }
            }

            // 平面先验优化深度值
            {
                ptrDN->m_StereoStage = eSS_PlannerPrior_Begin;
                lf_acmp.SetPlanarPriorParams();

                cv::Rect imageRC(0, 0, m_Params.mi_width_for_match, m_Params.mi_height_for_match);

                // 获取高可信稀疏的对应关系
                std::vector<cv::Point> support2DPoints;
                lf_acmp.GetSupportPoints(support2DPoints); // LZD, 修改：5--->3
                const std::vector<Triangle> triangles = lf_acmp.DelaunayTriangulation(imageRC, support2DPoints);//未修改

                bool bWrite_tri_Image = false; //
                if (bWrite_tri_Image) {
                    cv::Mat refImage = lf_acmp.GetReferenceImage().clone();
                    std::vector<cv::Mat> mbgr(3);
                    mbgr[0] = refImage.clone();
                    mbgr[1] = refImage.clone();
                    mbgr[2] = refImage.clone();
                    cv::Mat srcImage;
                    cv::merge(mbgr, srcImage);
                    for (const auto triangle: triangles) {
                        if (imageRC.contains(triangle.pt1) &&
                            imageRC.contains(triangle.pt2) &&
                            imageRC.contains(triangle.pt3)) {
                            cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                            cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                            cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
                        }
                    }

                    // 创建路径
                    std::string strMLAResultFolder = m_strRootPath + LF_DEPTH_INTRA_NAME + strName;
                    std::string strMLADisFolder =
                            strMLAResultFolder + LF_MVS_RESULT_DATA_NAME + LF_MLA_DISPARITYMAPS_PLANNER_NAME;
                    boost::filesystem::path dir_save_path(strMLADisFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
                        }
                    }
                    std::string triangulation_path = strMLADisFolder + ptrCurKey->StrRemoveLOD() + ".png";
                    cv::imwrite(triangulation_path, srcImage);
                }

                cv::Mat_<float> mask_tri = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match,
                                                          CV_32FC1);
                std::vector<float4> planeParams_tri;
                planeParams_tri.clear();
                for (uint32_t idx = 0; idx < triangles.size(); idx++) {
                    const Triangle &triangle = triangles.at(idx);
                    if (imageRC.contains(triangle.pt1) &&
                        imageRC.contains(triangle.pt2) &&
                        imageRC.contains(triangle.pt3)) {
                        float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) +
                                         pow(triangle.pt1.y - triangle.pt2.y, 2));
                        float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) +
                                         pow(triangle.pt1.y - triangle.pt3.y, 2));
                        float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) +
                                         pow(triangle.pt2.y - triangle.pt3.y, 2));

                        float max_edge_length = std::max(L01, std::max(L02, L12));
                        float step = 1.0 / max_edge_length;
                        for (float p = 0; p < 1.0; p += step) {
                            for (float q = 0; q < 1.0 - p; q += step) {
                                int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                                int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                                mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                            }
                        }

                        // 估计平面（以面法线表示）： estimate plane parameter
                        float4 n4 = lf_acmp.GetPriorPlaneParams(triangle, depths); // 修改过
                        planeParams_tri.push_back(n4);
                    }
                }

                cv::Mat_<float> prior_depths = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match,
                                                              CV_32FC1);
                for (int col = 0; col < m_Params.mi_width_for_match; ++col) {
                    for (int row = 0; row < m_Params.mi_height_for_match; ++row) {
                        if (mask_tri(row, col) > 0) {
                            float d = lf_acmp.GetDepthFromPlaneParam_LF_Tilekey(planeParams_tri[mask_tri(row, col) - 1],
                                                                                col, row);
                            if (d <= lf_acmp.GetMaxDepth() && d >= lf_acmp.GetMinDepth()) {
                                prior_depths(row, col) = d;
                            } else {
                                mask_tri(row, col) = 0;
                            }
                        }
                    }
                }
                // std::string depth_path = result_folder + "/depths_prior.dmb";
                //  writeDepthDmb(depth_path, priordepths);
                lf_acmp.CudaPlanarPriorInitialization_LF_Tilekey(planeParams_tri, mask_tri);
                lf_acmp.RunPatchMatchLF_plane();

                // 存储结果
                std::map<unsigned int, std::vector<int2>> neigviews_planner_map;
                cv::Mat neighinfo_planner = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match,
                                                           CV_8UC3);
                for (int col = 0; col < m_Params.mi_width_for_match; ++col) {
                    for (int row = 0; row < m_Params.mi_height_for_match; ++row) {
                        int center = row * m_Params.mi_width_for_match + col;
                        float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
                        float cost = lf_acmp.GetCost(center);
                        ptrDN->ph_cuda[center] = plane_hypothesis;
                        ptrDN->d_cuda[center] = plane_hypothesis.w;
                        ptrDN->c_cuda[center] = cost;

                        unsigned int neig_viewBit = lf_acmp.GetSelected_viewIndexs(col, row);
                        //lf_acmp.SelectedViewIndexConvert(problem, neig_viewBit);
                        if (neig_viewBit > 0) {
                            std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_planner_map.find(
                                    neig_viewBit);
                            if (itr == neigviews_planner_map.end()) {
                                std::vector<int2> pixel_coords;
                                int2 p = make_int2(row, col);
                                pixel_coords.push_back(p);
                                neigviews_planner_map[neig_viewBit] = pixel_coords;
                            } else {
                                std::vector<int2> &pixel_coords = itr->second;
                                int2 p = make_int2(row, col);
                                pixel_coords.push_back(p);
                            }
                        }
                    }
                }

                unsigned int ix_planner = 0;
                for (std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_planner_map.begin();
                     itr != neigviews_planner_map.end(); itr++, ix_planner++) {
                    std::vector<int2> &pixel_coords = itr->second;
                    cv::Vec3b color = m_Colors[ix_planner];
                    for (int i = 0; i < pixel_coords.size(); i++) {
                        int2 &pixel_coord = pixel_coords[i];
                        neighinfo_planner.at<cv::Vec3b>(pixel_coord.x, pixel_coord.y) = color;
                    }
                }
                ptrDN->m_StereoStage = eSS_PlannerPrior_Finished;

                // 处理写出和测试
                {
                    std::string strMLAResultFolder = m_strRootPath + LF_DEPTH_INTRA_NAME + strName;
                    boost::filesystem::path dir_save_path(strMLAResultFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                        }
                    }
                    strMLAResultFolder += LF_MVS_RESULT_DATA_NAME;
                    {
                        boost::filesystem::path dir_save_path(strMLAResultFolder);
                        if (!boost::filesystem::exists(dir_save_path)) {
                            if (!boost::filesystem::create_directory(dir_save_path)) {
                                std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                            }
                        }
                    }

                    // 写出微透镜图像的视差图
                    static bool bWriteMLA_DisparityMap = true;
                    if (bWriteMLA_DisparityMap) {
                        double d_factor = 5.0;
                        // 创建路径
                        std::string strMLADisFolder = strMLAResultFolder + LF_MLA_DISPARITYMAPS_NAME;
                        boost::filesystem::path dir_save_path(strMLADisFolder);
                        if (!boost::filesystem::exists(dir_save_path)) {
                            if (!boost::filesystem::create_directory(dir_save_path)) {
                                std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
                            }
                        }

                        cv::Mat MLA_DisMap_gray = cv::Mat::zeros(m_Params.mi_height_for_match,
                                                                 m_Params.mi_width_for_match, CV_8UC1);
                        for (int mla_row = 0; mla_row < m_Params.mi_height_for_match; ++mla_row) {
                            for (int mla_col = 0; mla_col < m_Params.mi_width_for_match; ++mla_col) {
                                int index = mla_row * m_Params.mi_width_for_match + mla_col;
                                float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                                MLA_DisMap_gray.at<uchar>(mla_row, mla_col) =
                                        (plane_hypothesis.w / m_Params.mi_width_for_match) * 255 * d_factor;
                            }
                        }
                        std::string strMLADisPath_key = strMLADisFolder + ptrCurKey->StrRemoveLOD();
                        cv::Mat disp_color;
                        applyColorMap(MLA_DisMap_gray, disp_color, cv::COLORMAP_JET);
                        //cv::imwrite(strMLADisPath_key + std::string("_Dis_gray.png"), MLA_DisMap_gray);
                        cv::imwrite(strMLADisPath_key + std::string("_color_planner.png"), disp_color);
                        cv::imwrite(strMLADisPath_key + std::string("_neighborInfo_planner.png"), neighinfo_planner);
                    }
                }
            }

            lf_acmp.ReleaseMemory(); // 释放内存
        }

        void DepthSolver::ProcessProblem_LF_TileKey(MLA_Problem &problem, QuadTreeProblemMap &problem_map,
                                                    QuadTreeDisNormalMap &dis_normal_map, std::string &strName) {
            if (m_top_device == -1) {
                std::cout << "PPLFT: Error! Find GPU device index is: " << m_top_device << std::endl;
                return;
            }
            cudaError_t err = cudaSetDevice(m_top_device);
            if (err != cudaSuccess) {
                std::cout << "PPLFT: Error! cudaSetDevice: " << err << std::endl;
                return;
            }

            // 当前微图像的key
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            LF_ACMP lf_acmp(m_Params);
            lf_acmp.SetTileKey(ptrCurKey->GetTileX(), ptrCurKey->GetTileY());
            std::vector<float4> planeVec;
            std::vector<float> costVec;

            // 创建左侧微透镜key
            QuadTreeBoundingbox box;
            int tile_X_LeftNeig = ptrCurKey->GetTileX() - 1;
            int tile_Y_LeftNeig = ptrCurKey->GetTileY();
            QuadTreeTileKey LeftNeig_Key(TileKey_None, 0, tile_X_LeftNeig, tile_Y_LeftNeig, box);
            // auto itr_left = dis_normal_map.find(LeftNeig_Key);
            // if (itr_left != dis_normal_map.end())
            // {
            //     DisparityAndNormalPtr ptrDis_Left = itr_left->second;
            //     // 左侧微透镜
            //     if(ptrDis_Left->m_StereoStage == eSS_ACMH_Finished)
            //     {
            //         // 遍历微图像中的像素，计算视差平面和代价
            //         for (int i = 0; i < g_MIA_match_height; ++i)
            //         {
            //             for (int j = 0; j < g_MIA_match_width; ++j)
            //             {
            //                 int n2 = i*g_MIA_match_width+j;
            //                 planeVec.push_back(ptrDis_Left->ph_cuda[n2]);
            //                 costVec.push_back(ptrDis_Left->c_cuda[n2]);
            //             }
            //         }
            //     }
            //     else
            //     {
            //         std::cout << "Left image stereo not finish: " << LeftNeig_Key.StrRemoveLOD() << std::endl;
            //     }
            // }
            // else
            // {
            //     std::cout << "Left image not found: " << LeftNeig_Key.StrRemoveLOD() << std::endl;
            // }

            // 计算参考微透镜和邻居微透镜中心点坐标 定义视差范围等参数
            lf_acmp.InuputInitialization_LF_TileKey(m_MLA_info_map, problem, problem_map, planeVec, costVec);
            lf_acmp.CudaSpaceInitialization_LF_TileKey();
            lf_acmp.RunPatchMatchLF();

            // 获取参考微透镜的 DisparityAndNormal
            QuadTreeDisNormalMap::iterator itr_DN = dis_normal_map.find(ptrCurKey);
            if (itr_DN == dis_normal_map.end()) // 未找到
            {
                std::cout << "Current Image DN not found: " << ptrCurKey->StrRemoveLOD().c_str() << std::endl;
                lf_acmp.ReleaseMemory();
                return;
            }

            // 存储结果，及写出
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            for (int row = 0; row < m_Params.mi_height_for_match; ++row) {
                for (int col = 0; col < m_Params.mi_width_for_match; ++col) {
                    int index = row * m_Params.mi_width_for_match + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                    float cost = lf_acmp.GetCost(index);
                    ptrDN->d_cuda[index] = plane_hypothesis.w;

                    //std::cout<<plane_hypothesis.w<<std::endl;

                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->c_cuda[index] = cost;
                }
            }
            {
                std::string strMLAResultFolder = m_strRootPath + LF_DEPTH_INTRA_NAME + strName;
                boost::filesystem::path dir_save_path(strMLAResultFolder);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                    }
                }
                strMLAResultFolder += LF_MVS_RESULT_DATA_NAME;
                {
                    boost::filesystem::path dir_save_path(strMLAResultFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
                        }
                    }
                }

                // 写出微透镜图像的视差图
                static bool bWriteMLA_DisparityMap = false;
                if (bWriteMLA_DisparityMap) {
                    double d_factor = 5.0;
                    // 创建路径
                    std::string strMLADisFolder = strMLAResultFolder + LF_MLA_DISPARITYMAPS_NAME;
                    boost::filesystem::path dir_save_path(strMLADisFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
                        }
                    }

                    cv::Mat MLA_DisMap_gray = cv::Mat::zeros(m_Params.mi_height_for_match, m_Params.mi_width_for_match,
                                                             CV_8UC1);
                    for (int mla_row = 0; mla_row < m_Params.mi_height_for_match; ++mla_row) {
                        for (int mla_col = 0; mla_col < m_Params.mi_width_for_match; ++mla_col) {
                            int index = mla_row * m_Params.mi_width_for_match + mla_col;
                            float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(index);
                            MLA_DisMap_gray.at<uchar>(mla_row, mla_col) =
                                    (plane_hypothesis.w / m_Params.mi_width_for_match) * 255 * d_factor;
                        }
                    }

                    std::string strMLADisPath_key = strMLADisFolder + ptrCurKey->StrRemoveLOD();
                    cv::Mat disp_color;
                    applyColorMap(MLA_DisMap_gray, disp_color, cv::COLORMAP_JET);
                    //cv::imwrite(strMLADisPath_key + std::string("_Dis_gray.png"), MLA_DisMap_gray);
                    cv::imwrite(strMLADisPath_key + std::string("_color.png"), disp_color);
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;
            lf_acmp.ReleaseMemory();
        }

        void DepthSolver::ProcessProblem_planar(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,
                                                const MLA_Problem &mlaProblem, DisparityAndNormal &DN, float &Base,
                                                bool geom_consistency, bool planar_prior,
                                                bool multi_geometrty = false) {
            cudaSetDevice(0);
            if (DN.m_StereoStage == eSS_ACMH_Begin) {
                std::cout << "微透镜图像未密集匹配" << std::endl;
                return;
            }
            LF_ACMP lf_acmp(m_Params);
            lf_acmp.InuputInitialization_planarLF(img, center, mlaProblem, DN, Base);
            lf_acmp.CudaSpaceInitialization_planarLF(img, center, mlaProblem, DN);
            const int width = img[0].cols;
            const int height = img[0].rows;

            cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    int center = row * width + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
                    depths(row, col) = plane_hypothesis.w;
                }
            }

            if (planar_prior) {
                lf_acmp.SetPlanarPriorParams();

                const cv::Rect imageRC(0, 0, width, height);
                std::vector<cv::Point> support2DPoints;

                lf_acmp.GetSupportPoints(support2DPoints);//修改过 从五变3
                const auto triangles = lf_acmp.DelaunayTriangulation(imageRC, support2DPoints);//未修改
                cv::Mat refImage = lf_acmp.GetReferenceImage().clone();
                std::vector<cv::Mat> mbgr(3);
                mbgr[0] = refImage.clone();
                mbgr[1] = refImage.clone();
                mbgr[2] = refImage.clone();
                cv::Mat srcImage;
                cv::merge(mbgr, srcImage);
                for (const auto triangle: triangles) {
                    if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) &&
                        imageRC.contains(triangle.pt3)) {
                        cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                        cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                        cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
                    }
                }
                //cv::imshow("img", srcImage);
                //cv::waitKey(0);
                //std::string triangulation_path = "/home/wdy/Data/光场数据/result/三角化/triangulation.png";
                //cv::imwrite(triangulation_path, srcImage);

                cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
                std::vector<float4> planeParams_tri;
                planeParams_tri.clear();

                uint32_t idx = 0;
                for (const auto triangle: triangles) {
                    if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) &&
                        imageRC.contains(triangle.pt3)) {
                        float L01 = sqrt(
                                pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                        float L02 = sqrt(
                                pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                        float L12 = sqrt(
                                pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                        float max_edge_length = std::max(L01, std::max(L02, L12));
                        float step = 1.0 / max_edge_length;

                        for (float p = 0; p < 1.0; p += step) {
                            for (float q = 0; q < 1.0 - p; q += step) {
                                int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                                int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                                mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                            }
                        }

                        // estimate plane parameter
                        float4 n4 = lf_acmp.GetPriorPlaneParams(triangle, depths);//修改过
                        planeParams_tri.push_back(n4);
                        idx++;
                    }
                }

                cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
                for (int i = 0; i < width; ++i) {
                    for (int j = 0; j < height; ++j) {
                        if (mask_tri(j, i) > 0) {
                            //float d = lf_acmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                            float d = lf_acmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                            if (d <= lf_acmp.GetMaxDepth() && d >= lf_acmp.GetMinDepth()) {
                                priordepths(j, i) = d;
                                //std::cout << d << std::endl;
                            } else {
                                mask_tri(j, i) = 0;
                            }
                        }
                    }
                }
                // std::string depth_path = result_folder + "/depths_prior.dmb";
                //  writeDepthDmb(depth_path, priordepths);

                lf_acmp.CudaPlanarPriorInitialization_LF_Tilekey(planeParams_tri, mask_tri);
                lf_acmp.RunPatchMatchLF_plane();
            }
            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    int center = row * width + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
                    float cost = lf_acmp.GetCost(center);
                    if (cost < 0.3) {
                        DN.d_cuda[center] = plane_hypothesis.w;
                        //std::cout<<plane_hypothesis.w<<std::endl;
                    } else {
                        DN.d_cuda[center] = 0;
                    }

                    DN.d_cuda[center] = plane_hypothesis.w;
                    //DN.ph_cuda[center] = plane_hypothesis;
                    //DN.c_cuda[center] = cost;
                }
            }
            DN.m_StereoStage = eSS_ACMH_Finished;
        }

        void DepthSolver::ProcessProblem_planar_TileKey(MLA_Problem &problem,
                                                        QuadTreeProblemMap &problem_map,
                                                        QuadTreeDisNormalMap &dis_normal_map,
                                                        std::string &strName,
                                                        bool geom_consistency, bool planar_prior,
                                                        bool multi_geometrty = false) {
            if (m_top_device == -1) {
                std::cout << "PPPT: Error! Find GPU device index is: " << m_top_device << std::endl;
                return;
            }
            cudaError_t err = cudaSetDevice(m_top_device);
            if (err) {
                std::cout << "PPPT: CUDA Device is Error!" << std::endl;
                return;
            }

            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            QuadTreeTileInfoMap::iterator itrI = m_MLA_info_map.find(ptrCurKey);
            if (itrI == m_MLA_info_map.end()) {
                std::cout << "PPPT: current Key not found: " << ptrCurKey->GetTileX() << ", tile_y="
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }

            QuadTreeDisNormalMap::iterator itr_DN = dis_normal_map.find(ptrCurKey);
            if (itr_DN == dis_normal_map.end()) {
                std::cout << "PPPT: Current DNS Key not found: tile_x=" << ptrCurKey->GetTileX()
                          << ", tile_y=" << ptrCurKey->GetTileY() << std::endl;
                return;
            }

            DisparityAndNormalPtr ptrDN = itr_DN->second;
            if (ptrDN->m_StereoStage != eSS_ACMH_Finished) {
                std::cout << "MLA Image not process depth_estimation, tile_x= " << ptrCurKey->GetTileX() << ", tile_y="
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }

            // 开始执行平面先验的深度估计
            ptrDN->m_StereoStage = eSS_PlannerPrior_Begin;
            LF_ACMP lf_acmp(m_Params);
            lf_acmp.InuputInitialization_planarLF_TileKey(m_MLA_info_map, problem, problem_map, dis_normal_map);
            lf_acmp.CudaSpaceInitialization_planarLF_TileKey();

            const int width = m_Params.mi_width_for_match;
            const int height = m_Params.mi_height_for_match;
            cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    int center = row * width + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
                    depths(row, col) = plane_hypothesis.w;
                }
            }

            if (planar_prior) {
                lf_acmp.SetPlanarPriorParams();

                const cv::Rect imageRC(0, 0, width, height);
                std::vector<cv::Point> support2DPoints;

                lf_acmp.GetSupportPoints(support2DPoints); // 修改过 从五变3
                const auto triangles = lf_acmp.DelaunayTriangulation(imageRC, support2DPoints);//未修改
                cv::Mat refImage = lf_acmp.GetReferenceImage().clone();
                std::vector<cv::Mat> mbgr(3);
                mbgr[0] = refImage.clone();
                mbgr[1] = refImage.clone();
                mbgr[2] = refImage.clone();
                cv::Mat srcImage;
                cv::merge(mbgr, srcImage);
                for (const auto triangle: triangles) {
                    if (imageRC.contains(triangle.pt1) &&
                        imageRC.contains(triangle.pt2) &&
                        imageRC.contains(triangle.pt3)) {
                        cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                        cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                        cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
                    }
                }

                bool bWrite_dis = false;
                if (bWrite_dis) {
                    // 创建路径
                    std::string strMLAResultFolder = m_strRootPath + LF_DEPTH_INTRA_NAME + strName;
                    std::string strMLADisFolder =
                            strMLAResultFolder + LF_MVS_RESULT_DATA_NAME + LF_MLA_DISPARITYMAPS_PLANNER_NAME;
                    boost::filesystem::path dir_save_path(strMLADisFolder);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
                        }
                    }
                    std::string triangulation_path = strMLADisFolder + ptrCurKey->StrRemoveLOD() + ".png";
                    cv::imwrite(triangulation_path, srcImage);
                }

                cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
                std::vector<float4> planeParams_tri;
                planeParams_tri.clear();

                uint32_t idx = 0;
                for (const auto triangle: triangles) {
                    if (imageRC.contains(triangle.pt1) &&
                        imageRC.contains(triangle.pt2) &&
                        imageRC.contains(triangle.pt3)) {
                        float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) +
                                         pow(triangle.pt1.y - triangle.pt2.y, 2));
                        float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) +
                                         pow(triangle.pt1.y - triangle.pt3.y, 2));
                        float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) +
                                         pow(triangle.pt2.y - triangle.pt3.y, 2));

                        float max_edge_length = std::max(L01, std::max(L02, L12));
                        float step = 1.0 / max_edge_length;
                        for (float p = 0; p < 1.0; p += step) {
                            for (float q = 0; q < 1.0 - p; q += step) {
                                int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                                int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                                mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                            }
                        }

                        // estimate plane parameter
                        float4 n4 = lf_acmp.GetPriorPlaneParams(triangle, depths); // 修改过
                        planeParams_tri.push_back(n4);
                        idx++;
                    }
                }

                cv::Mat_<float> prior_depths = cv::Mat::zeros(height, width, CV_32FC1);
                for (int i = 0; i < width; ++i) {
                    for (int j = 0; j < height; ++j) {
                        if (mask_tri(j, i) > 0) {
                            //float d = lf_acmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                            float d = lf_acmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                            if (d <= lf_acmp.GetMaxDepth() && d >= lf_acmp.GetMinDepth()) {
                                prior_depths(j, i) = d;
                            } else {
                                mask_tri(j, i) = 0;
                            }
                        }
                    }
                }
                // std::string depth_path = result_folder + "/depths_prior.dmb";
                //  writeDepthDmb(depth_path, priordepths);
                lf_acmp.CudaPlanarPriorInitialization_LF_Tilekey(planeParams_tri, mask_tri);
                lf_acmp.RunPatchMatchLF_plane();
            }

            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    int center = row * width + col;
                    float4 plane_hypothesis = lf_acmp.GetPlaneHypothesis(center);
                    float cost = lf_acmp.GetCost(center);
                    //if (cost < 0.3)
                    {
                        ptrDN->ph_cuda[center] = plane_hypothesis;
                        ptrDN->d_cuda[center] = plane_hypothesis.w;
                        ptrDN->c_cuda[center] = cost;
                    }
                }
            }
            ptrDN->m_StereoStage = eSS_PlannerPrior_Finished;
        }

        void DepthSolver::ShowDMap(float *dis, cv::Mat &img) {
            for (int i = 0; i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                    //std::cout<<dis[i*img.cols+j]<<std::endl;
                    img.at<uchar>(i, j) = dis[i * img.cols + j] * 4;
                }
            }
            cv::imshow("视差", img);
            cv::Mat disp_color;
            applyColorMap(img, disp_color, cv::COLORMAP_JET);
            //cv::imwrite("E:\\LX\\DATA\\result\\test\\res.png", disp_color);
            cv::imshow("RGB视差", disp_color);
            cv::waitKey(0);
        }

        void DepthSolver::ShowDisMap(std::vector<DisparityAndNormal> &DNS1, cv::Mat &img, std::string &path,
                                     int half_length) {
            for (int i = 0; i < m_Params.mla_v_size; i++) {
                for (int j = 0; j < m_Params.mla_u_size; j++) {
                    if (DNS1[i * m_Params.mla_u_size + j].m_StereoStage == eSS_ACMH_Finished) {
                        for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                            for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                                float disp = DNS1[i * m_Params.mla_u_size + j].d_cuda[x * m_Params.mi_height_for_match +
                                                                                      y];
                                if (i % 2 == 0) {
                                    //img.at<uchar>(i * 111 + x, j * 111 + y ) = DNS1[i * 79 + j].dis[(x + 5) * 121 + y + 5] * 4;
                                    //img.at<uchar>(i * g_MIA_match_height + x, j * g_MIA_match_height + y ) = DNS1[i * g_MLA_column + j].d_cuda[x*g_MIA_match_height+y]*4;
                                    img.at<uchar>(i * m_Params.mi_width_for_match + x,
                                                  j * m_Params.mi_height_for_match + y) =
                                            disp * 255 / m_Params.baseline;
                                    //std::cout << i * g_MIA_match_height + x <<"," << j * g_MIA_match_height + y << ", disp: "<< disp << "color: "<< disp*255/g_MIA_fBase << std::endl;
                                } else {
                                    //img.at<uchar>(i * 111 + x, j * 111 + y + 55) = DNS1[i * 79 + j].dis[(x + 5) * 121 + y + 5] * 4;
                                    //img.at<uchar>(i * g_MIA_match_height + x, j * g_MIA_match_height + y+ half_length ) = DNS1[i * g_MLA_column + j].d_cuda[x*g_MIA_match_height+y]*4;
                                    img.at<uchar>(i * m_Params.mi_width_for_match + x,
                                                  j * m_Params.mi_height_for_match + y + half_length) =
                                            disp * 255 / m_Params.baseline;
                                    //std::cout << i * g_MIA_match_height + x <<"," << j * g_MIA_match_height + y + half_length << ", disp: "<< disp << "color: "<< disp*255/g_MIA_fBase << std::endl;
                                }
                            }
                        }
                    }
                }
            }

            //cv::imshow("视差", img);
            cv::Mat disp_color;
            applyColorMap(img, disp_color, cv::COLORMAP_JET);
            cv::imwrite(path + std::string("/res_dis.png"), img);
            cv::imwrite(path + std::string("/res_col.png"), disp_color);
            //cv::imshow("RGB视差", disp_color);
            //cv::waitKey(0);
        }

        void DepthSolver::Virtual_depth_map_TileKey_new() {
            struct vdinfo {
                vdinfo() {
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = 0;
                }

                float vd = 0;
                int count = 0;
                cv::Vec3i color;
            };

            std::vector<PointList> PointCloud;
            PointCloud.clear();

            for (QuadTreeProblemMapMap::iterator itrP = m_MIA_problem_map_map.begin();
                 itrP != m_MIA_problem_map_map.end(); itrP++) {
                std::string strName = itrP->first;
                QuadTreeProblemMap &problem_map = itrP->second;
                QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[itrP->first];

                cv::Mat virtualDepth_float_tmp = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_32FC1);

                cv::Mat virtualDepth = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);
                cv::Mat virtualDepth_realColor = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC3);

                std::vector<vdinfo> vd;
                vd.reserve(m_RawImage_Height * m_RawImage_Width);

                for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                    MLA_Problem &problem = itr->second;
                    QuadTreeDisNormalMap::iterator itr_dns = dis_normal_map.find(problem.m_ptrKey);
                    if (itr_dns == dis_normal_map.end()) {
                        std::cout << "VDMTN: cur_DNS not found," << problem.m_ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    DisparityAndNormalPtr ptrMLADis = itr_dns->second;

                    QuadTreeTileInfoMap::iterator itrC = m_MLA_info_map.find(problem.m_ptrKey);
                    if (itrC == m_MLA_info_map.end()) {
                        std::cout << "VDMTN: cur_info can not found, " << problem.m_ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    MLA_InfoPtr ptrInfo = itrC->second;

                    if (ptrMLADis->m_StereoStage == eSS_ACMH_Finished ||
                        ptrMLADis->m_StereoStage == eSS_PlannerPrior_Finished) {
                        for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                            for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                                float disp = ptrMLADis->d_cuda[y * m_Params.mi_width_for_match + x];
                                if (disp <= 0.0)
                                    continue;

                                //过滤条件
                                double dis1{0.1};
                                if (abs(disp - ptrMLADis->d_cuda[y * m_Params.mi_width_for_match + x + 1]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[y * m_Params.mi_width_for_match + x - 1]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[(y + 1) * m_Params.mi_width_for_match + x]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[(y - 1) * m_Params.mi_width_for_match + x]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[(y + 1) * m_Params.mi_width_for_match + x + 1]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[(y - 1) * m_Params.mi_width_for_match + x + 1]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[(y + 1) * m_Params.mi_width_for_match + x - 1]) > dis1)
                                    continue;
                                if (abs(disp - ptrMLADis->d_cuda[(y - 1) * m_Params.mi_width_for_match + x - 1]) > dis1)
                                    continue;
//                            //为使2600scene2的结果更好看另外增加的过滤条件
//                            if(disp>21)
//                                continue;
//                            if(disp<8)
//                                continue;

                                // 视差越小，物距越远，f_v越大
                                float f_v = m_Params.baseline / disp;
                                int x_mlaCenter = x - ((m_Params.mi_width_for_match - 1) * 0.5 + 1);
                                float Xv = x_mlaCenter * f_v + ptrInfo->GetCenter().x;
                                int y_mlaCenter = y - ((m_Params.mi_height_for_match - 1) * 0.5 + 1);
                                float Yv = y_mlaCenter * f_v + ptrInfo->GetCenter().y;

                                int vd_y = (int) round(Yv);
                                int vd_x = (int) round(Xv);


                                if (vd_x > 0 && vd_x < virtualDepth.cols &&
                                    vd_y > 0 && vd_y < virtualDepth.rows) {
                                    vdinfo &info = vd[vd_y * virtualDepth.cols + vd_x];
                                    info.vd += f_v;
                                    info.count++;
                                    info.color[0] += problem.m_Image_rgb.at<uchar3>(y, x).x;
                                    info.color[1] += problem.m_Image_rgb.at<uchar3>(y, x).y;
                                    info.color[2] += problem.m_Image_rgb.at<uchar3>(y, x).z;
                                }
                            }
                        }
                    }
                }
                //
                DisparityRange &dis_range = m_disparityRangeMap[strName];
                for (int y = 0; y < virtualDepth.rows; y++) {
                    for (int x = 0; x < virtualDepth.cols; x++) {
                        vdinfo &info = vd[y * virtualDepth.cols + x];
                        if (info.count > 0) {
                            float vd_value = info.vd / info.count;
                            virtualDepth_float_tmp.at<float32>(y, x) = vd_value;
                            if (dis_range.m_vd_min > vd_value)
                                dis_range.m_vd_min = vd_value;
                            if (dis_range.m_vd_max < vd_value)
                                dis_range.m_vd_max = vd_value;

                            virtualDepth_realColor.at<uchar3>(y, x).x = info.color[0] / info.count;
                            virtualDepth_realColor.at<uchar3>(y, x).y = info.color[1] / info.count;
                            virtualDepth_realColor.at<uchar3>(y, x).z = info.color[2] / info.count;
                        }
                    }
                }
                std::cout << "VDTN: vd_min = " << dis_range.m_vd_min << ", vd_max = " << dis_range.m_vd_max
                          << std::endl;

                for (int y = 0; y < virtualDepth.rows; y++) {
                    for (int x = 0; x < virtualDepth.cols; x++) {
                        vdinfo &info = vd[y * virtualDepth.cols + x];
                        if (info.count > 0) {
                            float vd_float = virtualDepth_float_tmp.at<float32>(y, x);
                            virtualDepth.at<uchar>(y, x) = (vd_float - dis_range.m_vd_min * 0.6) /
                                                           ((dis_range.m_vd_max - dis_range.m_vd_min) * 0.8) * 255;

                            PointList point3D;
                            {
                                // 2600w
                                float trans = 0.0037;//像素转毫米
                                float B = 1.14;//微透镜到传感器距离，单位为毫米
                                float bl0 = 105.31;//单位毫米
                                int fl = 105;//焦距。单位为mm

                                // 1e2
                                // float trans=0.0037;//像素转毫米
                                // float B=2.35;//微透镜到传感器距离，单位为毫米
                                // float bl0=309.23;//单位毫米
                                // int fl=300;//焦距。单位为mm

                                //将x,y转为世界坐标系
                                double tmp_x = (x - m_Params.mi_width_for_match * 0.5) * trans;
                                double tmp_y = (y - m_Params.mi_height_for_match * 0.5) * trans;


                                double tmp_z = info.vd / info.count * B + bl0;

                                //此处变量含义为真实深度，为方便程序运行没有改名
                                double real_d = fl / (tmp_z - fl) * tmp_z;

                                double real_x = -fl / (tmp_z - fl) * tmp_x;
                                double real_y = -fl / (tmp_z - fl) * tmp_y;

                                point3D.coord.x = real_x;
                                point3D.coord.y = real_y;
                                point3D.coord.z = real_d;
                                point3D.color.x = virtualDepth_realColor.at<uchar3>(y, x).x;
                                point3D.color.y = virtualDepth_realColor.at<uchar3>(y, x).y;
                                point3D.color.z = virtualDepth_realColor.at<uchar3>(y, x).z;
                                PointCloud.push_back(point3D);
                            }
                        }
                    }
                }

                std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
                {
                    boost::filesystem::path dir_save_path(strSavePath);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strSavePath << std::endl;
                        }
                    }
                }
                cv::Mat virtualDepth_color;
                // 从蓝色到红色渐变，中间经过绿色和黄色。此处，物距由小到大，对应的颜色为蓝色--绿色--黄色-红色。
                applyColorMap(virtualDepth, virtualDepth_color, cv::COLORMAP_JET);
                std::string strFullPath_gray = strSavePath;
                std::string strFullPath_color = strSavePath;
                std::string strFullPath_real_color = strSavePath;
                switch (m_eStereoType) {
                    case eST_ACMH: {
                        strFullPath_gray += "/VD_Base_gray.png";
                        strFullPath_color += "/VD_Base_color.png";
                        strFullPath_real_color += "/VD_Base_real_color.png";
                        if (m_bPlannar != false || m_bLRCheck != true) {
                            std::cout << "base: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_PlannerPrior: {
                        strFullPath_gray += "/VD_Plannar_gray.png";
                        strFullPath_color += "/VD_Plannar_color.png";
                        strFullPath_real_color += "/VD_Plannar_real_color.png";
                        if (m_bPlannar != true || m_bLRCheck != true) {
                            std::cout << "planner: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_Horizontal_Propagation: {
                        strFullPath_gray += "/VD_HorProga_gray.png";
                        strFullPath_color += "/VD_HorProga_color.png";
                        strFullPath_real_color += "/VD_HorProga_real_color.png";
                        if (m_bPlannar != true || m_bLRCheck != true) {
                            std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_BlurFeature: {
                        strFullPath_gray += "/VD_BlurFeature_gray.png";
                        strFullPath_color += "/VD_BlurFeature_color.png";
                        strFullPath_real_color += "/VD_BlurFeature_real_color.png";
                    }
                        break;
                    default:
                        break;
                }
                cv::imwrite(strFullPath_gray, virtualDepth);
                cv::imwrite(strFullPath_color, virtualDepth_color);
                cv::imwrite(strFullPath_real_color, virtualDepth_realColor);
                StoreColorPlyFileBinaryPointCloud(strSavePath + std::string("/VD.ply"), PointCloud);
            }
        }

        void DepthSolver::WriteDisMap_TileKey_new() {
            int half_length = int(m_Params.mi_width_for_match * 0.5);
            double d_factor = 5.0;

            for (QuadTreeProblemMapMap::iterator itrM = m_MIA_problem_map_map.begin();
                 itrM != m_MIA_problem_map_map.end(); itrM++) {
                std::string strName = itrM->first;
                QuadTreeProblemMap &problem_map = itrM->second;
                QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];

                cv::Mat disp_gray = cv::Mat::zeros(m_Params.mla_v_size * m_Params.mi_width_for_match,
                                                   m_Params.mla_u_size * m_Params.mi_height_for_match, CV_8UC1);

                for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                    MLA_Problem &problem = itr->second;
                    int col = problem.m_ptrKey->GetTileX();
                    int row = problem.m_ptrKey->GetTileY();

                    QuadTreeDisNormalMap::iterator itr_dns = dis_normal_map.find(problem.m_ptrKey);
                    if (itr_dns == dis_normal_map.end()) {
                        std::cout << "ShowDisMap Main Image DNS can not found! (X,Y) = " << col << "," << row
                                  << std::endl;
                        continue;
                    }

                    DisparityAndNormalPtr ptrDis = itr_dns->second;
                    if (ptrDis->m_StereoStage == eSS_ACMH_Finished) {
                        for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                            for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                                float disp = ptrDis->d_cuda[y * m_Params.mi_width_for_match + x];
                                int tmp_y = row * m_Params.mi_height_for_match + y;
                                if (row % 2 == 0) {
                                    int tmp_x = col * m_Params.mi_width_for_match + x;
                                    disp_gray.at<uchar>(tmp_y, tmp_x) =
                                            (disp / m_Params.mi_width_for_match) * 255 * d_factor;
                                } else {
                                    int tmp_x = col * m_Params.mi_width_for_match + x + half_length;
                                    disp_gray.at<uchar>(tmp_y, tmp_x) =
                                            (disp / m_Params.mi_width_for_match) * 255 * d_factor;
                                }
                            }
                        }
                    }
                }

                // 写出
                if (m_strSavePath.empty())
                    m_strSavePath = m_strRootPath + LF_DEPTH_INTRA_NAME + LF_MVS_RESULT_DATA_NAME + strName;
                cv::Mat disp_color;
                applyColorMap(disp_gray, disp_color, cv::COLORMAP_JET);
                if (m_bPlannar == true && m_bLRCheck == false) {
                    cv::imwrite(m_strSavePath + std::string("/Disparity_Plannar_gray.png"), disp_gray);
                    cv::imwrite(m_strSavePath + std::string("/Disparity_Plannar_color.png"), disp_color);
                } else if (m_bPlannar == false && m_bLRCheck == true) {
                    cv::imwrite(m_strSavePath + std::string("/Disparity_LRCheck_gray.png"), disp_gray);
                    cv::imwrite(m_strSavePath + std::string("/Disparity_LRCheck_color.png"), disp_color);
                } else if (m_bPlannar == true && m_bLRCheck == true) {
                    cv::imwrite(m_strSavePath + std::string("/Disparity_Plannar_LRCheck_gray.png"), disp_gray);
                    cv::imwrite(m_strSavePath + std::string("/Disparity_Plannar_LRCheck_color.png"), disp_color);
                } else {
                    cv::imwrite(m_strSavePath + std::string("/Disparity_gray.png"), disp_gray);
                    cv::imwrite(m_strSavePath + std::string("/Disparity_color.png"), disp_color);
                }
            }
        }

        void DepthSolver::WriteDisMap_TileKey_new_Accu() {
            double d_factor = 3.0;

            for (QuadTreeProblemMapMap::iterator itr = m_MIA_problem_map_map.begin();
                 itr != m_MIA_problem_map_map.end(); itr++) {
                std::string strName = itr->first;
                QuadTreeProblemMap &problem_map = itr->second;
                QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[itr->first];

                cv::Mat disp_gray = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);

                ///////////////////////////////////////////////////////////////////////////
                DisparityRange dis_range;
                // 查找最小、最大的视差
                for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                    MLA_Problem &problem = itr->second;
                    QuadTreeTileKeyPtr ptrKey = itr->first;
                    QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(problem.m_ptrKey);
                    if (itr_dis == dis_normal_map.end()) {
                        std::cout << "WDKNA: current disnormal can not foun，" << ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    DisparityAndNormalPtr ptrDN = itr_dis->second;
                    if (ptrDN->m_StereoStage == eSS_ACMH_Finished ||
                        ptrDN->m_StereoStage == eSS_PlannerPrior_Finished) {
                        for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                            for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                                float disparity = ptrDN->d_cuda[y * m_Params.mi_width_for_match + x];
                                if (disparity <= 0.0)
                                    continue;
                                if (dis_range.m_dis_min > disparity)
                                    dis_range.m_dis_min = disparity;
                                if (dis_range.m_dis_max < disparity)
                                    dis_range.m_dis_max = disparity;
                            }
                        }
                    }
                }
                m_disparityRangeMap[strName] = dis_range;
                std::cout << "WDKNA: dis_min = " << dis_range.m_dis_min << ", dis_max = " << dis_range.m_dis_max
                          << std::endl;
                ///////////////////////////////////////////////////////////////////////////

                for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                    MLA_Problem &problem = itr->second;
                    int col = problem.m_ptrKey->GetTileX();
                    int row = problem.m_ptrKey->GetTileY();

                    QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
                    if (itrInfo == m_MLA_info_map.end()) {
                        std::cout << "WDKNA: cur_info can not found，" << problem.m_ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    MLA_InfoPtr ptrInfo = itrInfo->second;

                    QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(problem.m_ptrKey);
                    if (itr_dis == dis_normal_map.end()) {
                        std::cout << "WDKNA: cur_disnormal can not found, " << problem.m_ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    DisparityAndNormalPtr ptrDN = itr_dis->second;

                    // 处理
                    if (ptrDN->m_StereoStage == eSS_ACMH_Finished ||
                        ptrDN->m_StereoStage == eSS_PlannerPrior_Finished) {
                        // 将中心点偏移到微图像的左上角
                        cv::Point2f ori_coord;
                        ori_coord.x = ptrInfo->GetCenter().x - (round(m_Params.mi_width_for_match * 0.5) - 1);
                        ori_coord.y = ptrInfo->GetCenter().y - (round(m_Params.mi_height_for_match * 0.5) - 1);
                        for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                            for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                                // 代价约束
                                float cost = ptrDN->c_cuda[y * m_Params.mi_width_for_match + x];
                                if (cost > 0.2)
                                    continue;

                                float disparity = ptrDN->d_cuda[y * m_Params.mi_width_for_match + x];
                                if (disparity <= 0.0)
                                    continue;

                                int tmp_y = tolower(ori_coord.y) + y;
                                int tmp_x = tolower(ori_coord.x) + x;
                                disp_gray.at<uchar>(tmp_y, tmp_x) = (disparity - dis_range.m_dis_min * 0.6) /
                                                                    ((dis_range.m_dis_max - dis_range.m_dis_min) *
                                                                     0.8) * 255 * d_factor;
                            }
                        }
                    }

                    // MLA_Tilekey: 微透镜编码
                    // 设置字体和颜色
                    int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
                    double fontScale = 0.4;                  // 字体大小
                    int thickness = 1;                        // 线条粗细
                    cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
                    // 文字内容
                    std::string text = problem.m_ptrKey->StrRemoveLOD();
                    // 文字位置，(x, y)为文字左下角的坐标
                    cv::Point textOrg(ptrInfo->GetCenter().x,
                                      ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
                    // 将文字写入图片
                    cv::putText(disp_gray, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);
                }

                // 写出
                std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
                {
                    boost::filesystem::path dir_save_path(strSavePath);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strSavePath << std::endl;
                        }
                    }
                }

                cv::Mat disp_color;
                // 从蓝色到红色渐变，中间经过绿色和黄色。此处，物距由小到大，对应的颜色为蓝色--绿色--黄色-红色。
                applyColorMap(disp_gray, disp_color, cv::COLORMAP_JET);
                std::string strFullPath_gray = strSavePath;
                std::string strFullPath_color = strSavePath;
                switch (m_eStereoType) {
                    case eST_ACMH: {
                        strFullPath_gray += "/Dis_Base_gray.png";
                        strFullPath_color += "/Dis_Base_color.png";
                        if (m_bPlannar != false || m_bLRCheck != true) {
                            std::cout << "base: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_PlannerPrior: {
                        strFullPath_gray += "/Dis_Plannar_gray.png";
                        strFullPath_color += "/Dis_Plannar_color.png";
                        if (m_bPlannar != true || m_bLRCheck != true) {
                            std::cout << "planner: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_Horizontal_Propagation: {
                        strFullPath_gray += "/Dis_HorProga_gray.png";
                        strFullPath_color += "/Dis_HorProga_color.png";
                        if (m_bPlannar != true || m_bLRCheck != true) {
                            std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_BlurFeature: {
                        strFullPath_gray += "/Dis_BlurFeature_gray.png";
                        strFullPath_color += "/Dis_BlurFeature_color.png";
                    }
                        break;
                    default:
                        break;
                }
                cv::imwrite(strFullPath_gray, disp_gray);
                cv::imwrite(strFullPath_color, disp_color);
            }
        }

        void DepthSolver::VisualizeDisparityWithMaskOverlay(const cv::Mat &dis_gray, cv::Mat &disp_color) {
            CV_Assert(dis_gray.type() == CV_8UC1);

            // 生成彩色图
            cv::Mat dis_gray_safe = dis_gray.isContinuous() ? dis_gray : dis_gray.clone();
            dis_gray_safe = dis_gray_safe.clone();
            cv::applyColorMap(dis_gray_safe, disp_color, cv::COLORMAP_JET);

            // 生成mask
            cv::Mat mask = (dis_gray == 128);
            CV_Assert(mask.type() == CV_8UC1);

            // 为无效区域着红色
            for (int y = 0; y < disp_color.rows; ++y) {
                for (int x = 0; x < disp_color.cols; ++x) {
                    if (mask.at<uchar>(y, x) != 0) {
                        // 红色叠加
                        disp_color.at<cv::Vec3b>(y, x)[0] = 255;   // Blue
                        disp_color.at<cv::Vec3b>(y, x)[1] = 255;   // Green
                        disp_color.at<cv::Vec3b>(y, x)[2] = 255;   // Red
                    }
                }
            }
        }

        //
        // ------------------ 公共小工具 ------------------
        static inline bool _is_valid_disp(float x) {
            // 你可以根据工程习惯调整：>0、非 NaN、非 Inf 等
            return std::isfinite(x) && x > 0.0f;
        }

        static inline float _median_inplace(std::vector<float> &v) {
            if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
            const size_t n = v.size();
            const size_t mid = n / 2;
            std::nth_element(v.begin(), v.begin() + mid, v.end());
            float m = v[mid];
            if ((n & 1) == 0) {
                // 偶数样本：取中间两者的平均
                auto it_left_max = std::max_element(v.begin(), v.begin() + mid);
                m = (m + *it_left_max) * 0.5f;
            }
            return m;
        }

        // 仅保留 [lo, hi] 内的数
        void DepthSolver::Filter_by_band(const std::vector<float> &in, float lo, float hi, std::vector<float> &out)
        {
            out.clear();
            out.reserve(in.size());
            for (float x: in)
            {
                if (x >= lo && x <= hi) out.push_back(x);
            }
        }

        void DepthSolver::ComputeGlobalBandByMAD(const std::vector<float> &disp_all,
                                                float k, float min_band,
                                                float &lo_out, float &hi_out)
    {
            if (disp_all.empty()) {
                lo_out = hi_out = 0.0f;
                return;
            }
            if (disp_all.size() < 8) {
                // 样本过少，不做剔除
                lo_out = *std::min_element(disp_all.begin(), disp_all.end());
                hi_out = *std::max_element(disp_all.begin(), disp_all.end());
                return;
            }

            std::vector<float> tmp = disp_all;
            float med = _median_inplace(tmp);

            tmp.clear();
            tmp.reserve(disp_all.size());
            for (float x: disp_all) tmp.push_back(std::fabs(x - med));
            float mad = _median_inplace(tmp);

            const float sigma = 1.4826f * mad;        // sigma ≈ 1.4826 * MAD
            float lo = med - k * sigma;
            float hi = med + k * sigma;

            if (hi - lo < std::max(min_band, 1e-4f)) {
                const float half = 0.5f * std::max(min_band, 1e-4f);
                lo = med - half;
                hi = med + half;
            }
            lo_out = lo;
            hi_out = hi;
        }

        void DepthSolver::BuildGlobalDispInlierMAD(const std::vector<float> &disp_all_raw,
                                                    std::vector<float> &disp_inlier_global,
                                                    float &lo_out, float &hi_out,
                                                    float k, float min_band)
    {
            // 先拣选有效值，避免 0/NaN/Inf 干扰
            std::vector<float> disp_all;
            disp_all.reserve(disp_all_raw.size());
            for (float x: disp_all_raw)
            {
                if (_is_valid_disp(x))
                {
                    disp_all.push_back(x);
                }
            }
            ComputeGlobalBandByMAD(disp_all, k, min_band, lo_out, hi_out);
            Filter_by_band(disp_all, lo_out, hi_out, disp_inlier_global);
        }

        // 仅用每个微图像中匹配的像素合并成一张大图，并进行着色
        void DepthSolver::WriteDisMapForMIA(std::string &strName, QuadTreeProblemMap &problem_map)
        {
            double d_factor = 3.0;
            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];

            // 搜集信息：最小+最大视差，有效匹配的微图像（行，列）
            DisparityRange dis_range;
            cv::Point2i cols_range_mla(DBL_MAX, -DBL_MAX); // <min, max>
            cv::Point2i rows_range_mla(DBL_MAX, -DBL_MAX);
            std::vector<float> disp_data_mla;
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
            {
                DisparityRange dis_range_mi;
                std::vector<float> disp_data_mi;
                MLA_Problem &problem = itr->second;
                QuadTreeTileKeyPtr ptrKey = itr->first;
                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;
                QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(problem.m_ptrKey);
                if (itr_dis == dis_normal_map.end())
                    continue;

                if (cols_range_mla.x > ptrKey->GetTileX()) {
                    cols_range_mla.x = ptrKey->GetTileX();
                }
                if (cols_range_mla.y < ptrKey->GetTileX()) {
                    cols_range_mla.y = ptrKey->GetTileX();
                }
                if (rows_range_mla.x > ptrKey->GetTileY()) {
                    rows_range_mla.x = ptrKey->GetTileY();
                }
                if (rows_range_mla.y < ptrKey->GetTileY()) {
                    rows_range_mla.y = ptrKey->GetTileY();
                }

                DisparityAndNormalPtr ptrDN = itr_dis->second;
                for (int row = 0; row < m_Params.mi_height_for_match; row++)
                {
                    for (int col = 0; col < m_Params.mi_width_for_match; col++)
                    {
                        float disparity = ptrDN->d_cuda[row * m_Params.mi_width_for_match + col];
                        if (disparity <= 0.0)
                            continue;
                        if (dis_range_mi.m_dis_min > disparity)
                            dis_range_mi.m_dis_min = disparity;
                        if (dis_range_mi.m_dis_max < disparity)
                            dis_range_mi.m_dis_max = disparity;
                        if (dis_range.m_dis_min > disparity)
                            dis_range.m_dis_min = disparity;
                        if (dis_range.m_dis_max < disparity)
                            dis_range.m_dis_max = disparity;

                        if (disparity > 0.1)
                        {
                            disp_data_mi.push_back(disparity);
                            disp_data_mla.push_back(disparity);
                        }
                    }
                }

                if (g_Debug_Static >= 1)
                {
                    KDE kde(disp_data_mi, 0.1);
                    std::pair<float, float> interval =
                        findHighDensityInterval(kde, dis_range_mi.m_dis_min,
                                                dis_range_mi.m_dis_max,
                                                disp_data_mi.size());

                    float md = findKDEMode(kde, dis_range_mi.m_dis_min, dis_range_mi.m_dis_max, disp_data_mi.size());
                    LOG_INFO(ptrKey->StrRemoveLOD().c_str(), ", dis(",
                            dis_range_mi.m_dis_min, ", ", dis_range_mi.m_dis_max,
                            "), density( ", interval.first, ", ", interval.second,"), mode= ", md);
                }
            }
            m_disparityRangeMap[strName] = dis_range;
            LOG_ERROR("mla: dis_range(", dis_range.m_dis_min, ", ", dis_range.m_dis_max, ")");

            std::vector<float> disp_inlier_global;
            float lo_global = 0.0f, hi_global = 0.0f;
            BuildGlobalDispInlierMAD(disp_data_mla, disp_inlier_global,
                                     lo_global, hi_global, /*k=*/3.0f, /*min_band=*/0.02f);
            LOG_ERROR("[GlobalInlier-MAD] N_all=", disp_data_mla.size(),
                      " N_inlier=", disp_inlier_global.size(),
                      " band=[", lo_global, ", ", hi_global, "]");
            float dis_min_clipped = lo_global;
            float dis_max_clipped = hi_global;
            LOG_ERROR("disparity clip range (MAD): (", dis_min_clipped, ", ", dis_max_clipped, ")");

            const int image_width = (cols_range_mla.y - cols_range_mla.x + 1) * m_Params.mi_width_for_match;
            const int image_heigth = (rows_range_mla.y - rows_range_mla.x + 1) * m_Params.mi_height_for_match;
            cv::Mat disp_gray = cv::Mat::zeros(image_heigth, image_width, CV_8UC1);
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
            {
                MLA_Problem &problem = itr->second;
                QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
                bool bValid_match = true;

                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrKey);
                if (itrInfo == m_MLA_info_map.end())
                    continue;
                MLA_InfoPtr ptrInfo = itrInfo->second;
                QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(ptrKey);
                if (itr_dis == dis_normal_map.end()) {
                    bValid_match = false;
                    continue;
                }
                DisparityAndNormalPtr ptrDN = itr_dis->second;

                // 微图像的左上角
                cv::Point2f leftUp_coord;
                leftUp_coord.x = m_Params.mi_width_for_match * ptrKey->GetTileX();
                leftUp_coord.y = m_Params.mi_height_for_match * ptrKey->GetTileY();
                const float disp_invalid = dis_range.m_dis_max;
                for (int row = 0; row < m_Params.mi_height_for_match; row++) {
                    for (int col = 0; col < m_Params.mi_width_for_match; col++) {
                        int tmp_y = leftUp_coord.y + row;
                        int tmp_x = leftUp_coord.x + col;
                        if (tmp_y >= image_heigth || tmp_x >= image_width) {
                            continue;
                        }
                        float disparity = ptrDN->d_cuda[row * m_Params.mi_width_for_match + col];
                        if (disparity <= 0.0f || std::isnan(disparity)) {
                            // 设置为中性灰色，突出“无视差”的区域（如匹配失败区域）
                            disp_gray.at<uchar>(tmp_y, tmp_x) = 20; // 128
                            continue;
                        }
                        if (disparity < dis_min_clipped || disparity > dis_max_clipped) {
                            disp_gray.at<uchar>(tmp_y, tmp_x) = 20; // 128
                            continue;
                        }

                        if (bValid_match) {
                            //disp_gray.at<uchar>(tmp_y, tmp_x) = (disparity/dis_range.m_dis_max)*255;
                            // 新的归一化方式，加入clip范围判断
                            float disp_clip = std::min(std::max(disparity, dis_min_clipped), dis_max_clipped);
                            float v = (disp_clip - dis_min_clipped) / (dis_max_clipped - dis_min_clipped + 1e-6f) *
                                      255.0f;
                            v = std::min(std::max(v, 0.0f), 255.0f);
                            disp_gray.at<uchar>(tmp_y, tmp_x) = static_cast<uchar>(v);
                        }
                        else {
                            disp_gray.at<uchar>(tmp_y, tmp_x) = 20;
                        }
                    }
                }
            }

            // 写出
            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }

            std::string strFullPath_gray = strSavePath;
            std::string strFullPath_color = strSavePath;
            switch (m_eStereoType) {
                case eST_ACMH: {
                    strFullPath_gray += "/Dis_Base_gray.png";
                    strFullPath_color +=  "/"+ strName + "_" + LF_DISPARITYHMAP_NAME +".png";
                    if (m_bPlannar != false || m_bLRCheck != true) {
                        std::cout << "base: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_PlannerPrior: {
                    strFullPath_gray += "/Dis_Plannar_gray.png";
                    strFullPath_color += "/"+ strName + "_" + LF_DISPARITYHMAP_NAME +".png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "planner: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_Horizontal_Propagation: {
                    strFullPath_gray += "/Dis_HorProga_gray.png";
                    strFullPath_color += "/"+ strName + "_" + LF_DISPARITYHMAP_NAME + ".png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_BlurFeature: {
                    strFullPath_gray += "/Dis_BlurFeature_gray.png";
                    strFullPath_color += "/"+ strName + "_" + LF_DISPARITYHMAP_NAME + ".png";
                }
                    break;
                default:
                    break;
            }
            cv::imwrite(strFullPath_gray, disp_gray);

            // 从蓝色到红色渐变，中间经过绿色和黄色。
            // 物距由大到小，虚拟深度由小到大，视差由小到大，
            // 对应的颜色为蓝色--绿色--黄色-红色。
            cv::Mat disp_color;
            VisualizeDisparityWithMaskOverlay(disp_gray, disp_color);
            //applyColorMap(disp_gray, disp_color, cv::COLORMAP_JET);
            cv::imwrite(strFullPath_color, disp_color);

            // 配置并一键执行
            if (g_Debug_Static >= 1)
            {
                std::vector<float> data;
                for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
                {
                    MLA_Problem &problem = itr->second;
                    QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
                    bool bValid_match = true;

                    QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrKey);
                    if (itrInfo == m_MLA_info_map.end())
                        continue;
                    MLA_InfoPtr ptrInfo = itrInfo->second;
                    QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(ptrKey);
                    if (itr_dis == dis_normal_map.end())
                    {
                        bValid_match = false;
                        continue;
                    }
                    DisparityAndNormalPtr ptrDN = itr_dis->second;

                    // 微图像的左上角
                    cv::Point2f leftUp_coord;
                    leftUp_coord.x = m_Params.mi_width_for_match * ptrKey->GetTileX();
                    leftUp_coord.y = m_Params.mi_height_for_match * ptrKey->GetTileY();
                    const float disp_invalid = dis_range.m_dis_max;
                    for (int row = 0; row < m_Params.mi_height_for_match; row++) {
                        for (int col = 0; col < m_Params.mi_width_for_match; col++) {
                            int tmp_y = leftUp_coord.y + row;
                            int tmp_x = leftUp_coord.x + col;
                            float disparity = ptrDN->d_cuda[row * m_Params.mi_width_for_match + col];
                            data.push_back(disparity);
                        }
                    }
                }
                bool exact = false;
                double hdi_p = 0.85;
                int imgW = 1400;
                int imgH = 480;
                bool show = false;
                std::string prefix = m_strSavePath + std::string("/Distribution_analysis");
                boost::filesystem::path dir_save_path(prefix);
                if (!boost::filesystem::exists(dir_save_path))
                    if (!boost::filesystem::create_directory(dir_save_path))
                        std::cout << "dir failed to create: " << prefix << std::endl;
                prefix += "/analysis_Dis_";

                FloatDistributionAnalyzer::Options opt;
                opt.exact_quantiles = exact;
                opt.exact_hdi = exact;
                opt.hdi_p = hdi_p;
                opt.image_width = imgW;
                opt.image_height = imgH;
                opt.output_prefix = prefix;
                opt.show_windows = show;
                FloatDistributionAnalyzer analyzer;
                AnalysisResult R; // 如需在代码中使用结果，可传出
                if (!analyzer.run(data, opt, &R)) {
                    std::cerr << "分析失败\n";
                }
                // 控制台打印结果（可选）
                AnalyzerConfig cfg;
                cfg.exact_quantiles = opt.exact_quantiles;
                cfg.exact_hdi = opt.exact_hdi;
                cfg.hdi_p = opt.hdi_p;
                print_result(R, cfg);
            }
        }

        void DepthSolver::WriteDisMap_TileKey_new_AccuSequence(std::string &strName, QuadTreeProblemMap &problem_map) {
            double d_factor = 3.0;

            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];
            cv::Mat disp_gray = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);
            cv::Mat disp_gray_new = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);

            ///////////////////////////////////////////////////////////////////////////
            DisparityRange dis_range;
            // 查找最小、最大的视差
            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr) {
                DisparityRange dis_range_mi;
                std::vector<float> disp_data;

                MLA_Problem &problem = itr->second;
                QuadTreeTileKeyPtr ptrKey = itr->first;
                QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(problem.m_ptrKey);
                if (itr_dis == dis_normal_map.end()) {
                    std::cout << "WDKNA: current disnormal can not foun，" << ptrKey->StrRemoveLOD().c_str()
                              << std::endl;
                    continue;
                }
                DisparityAndNormalPtr ptrDN = itr_dis->second;
                if (ptrDN->m_StereoStage == eSS_ACMH_Finished ||
                    ptrDN->m_StereoStage == eSS_PlannerPrior_Finished) {
                    for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                        for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                            float disparity = ptrDN->d_cuda[y * m_Params.mi_width_for_match + x];
                            if (disparity <= 0.0)
                                continue;
                            if (dis_range_mi.m_dis_min > disparity)
                                dis_range_mi.m_dis_min = disparity;
                            if (dis_range_mi.m_dis_max < disparity)
                                dis_range_mi.m_dis_max = disparity;
                            if (dis_range.m_dis_min > disparity)
                                dis_range.m_dis_min = disparity;
                            if (dis_range.m_dis_max < disparity)
                                dis_range.m_dis_max = disparity;
                            disp_data.push_back(disparity);
                        }
                    }
                }
                KDE kde(disp_data, 0.1);
                std::pair<float, float> interval = findHighDensityInterval(kde, dis_range_mi.m_dis_min,
                                                                           dis_range_mi.m_dis_max, disp_data.size());
                float md = findKDEMode(kde, dis_range_mi.m_dis_min, dis_range_mi.m_dis_max, disp_data.size());
                printf("%s, dis(%f, %f), density(%f, %f), mode=%f\n", ptrKey->StrRemoveLOD().c_str(),
                       dis_range_mi.m_dis_min, dis_range_mi.m_dis_max,
                       interval.first, interval.second, md);
            }
            m_disparityRangeMap[strName] = dis_range;
            std::cout << "WDKNA: dis_min = " << dis_range.m_dis_min << ", dis_max = " << dis_range.m_dis_max
                      << std::endl;
            ///////////////////////////////////////////////////////////////////////////

            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr) {
                MLA_Problem &problem = itr->second;
                int col = problem.m_ptrKey->GetTileX();
                int row = problem.m_ptrKey->GetTileY();

                QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrInfo == m_MLA_info_map.end()) {
                    std::cout << "WDKNA: cur_info can not found，" << problem.m_ptrKey->StrRemoveLOD().c_str()
                              << std::endl;
                    continue;
                }
                MLA_InfoPtr ptrInfo = itrInfo->second;

                QuadTreeDisNormalMap::iterator itr_dis = dis_normal_map.find(problem.m_ptrKey);
                if (itr_dis == dis_normal_map.end()) {
                    std::cout << "WDKNA: cur_disnormal can not found, " << problem.m_ptrKey->StrRemoveLOD().c_str()
                              << std::endl;
                    continue;
                }
                DisparityAndNormalPtr ptrDN = itr_dis->second;

                // 处理
                if (ptrDN->m_StereoStage == eSS_ACMH_Finished ||
                    ptrDN->m_StereoStage == eSS_PlannerPrior_Finished) {
                    // 将中心点偏移到微图像的左上角
                    cv::Point2f ori_coord;

                    ori_coord.x = ptrInfo->GetCenter().x - (round(m_Params.mi_width_for_match * 0.5) - 1);
                    ori_coord.y = ptrInfo->GetCenter().y - (round(m_Params.mi_height_for_match * 0.5) - 1);
                    for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                        for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                            // 代价约束
                            float cost = ptrDN->c_cuda[y * m_Params.mi_width_for_match + x];
                            if (cost > 0.2)
                                continue;

                            float disparity = ptrDN->d_cuda[y * m_Params.mi_width_for_match + x];
                            if (disparity <= 0.0)
                                continue;

                            int tmp_y = tolower(ori_coord.y) + y;
                            int tmp_x = tolower(ori_coord.x) + x;
                            disp_gray.at<uchar>(tmp_y, tmp_x) = (disparity - dis_range.m_dis_min * 0.6) /
                                                                ((dis_range.m_dis_max - dis_range.m_dis_min) * 0.8) *
                                                                255 * d_factor;
                            disp_gray_new.at<uchar>(tmp_y, tmp_x) = (disparity / m_Params.mi_width_for_match) * 255;
                        }
                    }
                }

                // MLA_Tilekey: 微透镜编码
                // 设置字体和颜色
                int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
                double fontScale = 0.4;                  // 字体大小
                int thickness = 1;                        // 线条粗细
                cv::Scalar color = cv::Scalar(255, 255, 255); // 字体颜色，BGR格式
                // 文字内容
                std::string text = problem.m_ptrKey->StrRemoveLOD();
                // 文字位置，(x, y)为文字左下角的坐标
                cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y - m_Params.mi_height_for_match * 0.5);
                // 将文字写入图片
                cv::putText(disp_gray, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);
                cv::putText(disp_gray_new, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);
            }

            // 写出
            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }

            cv::Mat disp_color;
            cv::Mat disp_color_new;
            // 从蓝色到红色渐变，中间经过绿色和黄色。
            // 物距由大到小，虚拟深度由小到大，视差由小到大，
            // 对应的颜色为蓝色--绿色--黄色-红色。
            applyColorMap(disp_gray, disp_color, cv::COLORMAP_JET);
            applyColorMap(disp_gray_new, disp_color_new, cv::COLORMAP_JET);
            std::string strFullPath_gray = strSavePath;
            std::string strFullPath_gray_new = strSavePath;
            std::string strFullPath_color = strSavePath;
            std::string strFullPath_color_new = strSavePath;
            switch (m_eStereoType) {
                case eST_ACMH: {
                    strFullPath_gray += "/Dis_Base_gray.png";
                    strFullPath_color += "/Dis_Base_color.png";
                    if (m_bPlannar != false || m_bLRCheck != true) {
                        std::cout << "base: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_PlannerPrior: {
                    strFullPath_gray += "/Dis_Plannar_gray.png";
                    strFullPath_color += "/Dis_Plannar_color.png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "planner: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_Horizontal_Propagation: {
                    strFullPath_gray += "/Dis_HorProga_gray.png";
                    strFullPath_color += "/Dis_HorProga_color.png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_BlurFeature: {
                    strFullPath_gray += "/Dis_BlurFeature_gray.png";
                    strFullPath_gray_new += "/Dis_BlurFeature_gray_new.png";
                    strFullPath_color += "/Dis_BlurFeature_color.png";
                    strFullPath_color_new += "/Dis_BlurFeature_color_new.png";
                }
                    break;
                default:
                    break;
            }
            cv::imwrite(strFullPath_gray, disp_gray);
            cv::imwrite(strFullPath_gray_new, disp_gray_new);
            cv::imwrite(strFullPath_color, disp_color);
            cv::imwrite(strFullPath_color_new, disp_color_new);
        }

        float DepthSolver::Sample(std::vector<float> dis, float x0, float y0) {
            const int lx((int) x0);
            const int ly((int) y0);
            const float x(x0 - lx), x1(float(1) - x);
            const float y(y0 - ly), y1(float(1) - y);
            return (dis[lx * 121 + ly] * y1 + dis[lx * 121 + ly + 1] * y) * x1 +
                   (dis[(lx + 1) * 121 + ly] * y1 + dis[(lx + 1) * 121 + ly + 1] * y) * x;
        }


        void DepthSolver::LRCheck(std::vector<DisparityAndNormal> &DNS, std::vector<cv::Point2f> &center) {
            //对于0值区域进行保留
            std::vector<std::pair<int, int>> Rc;
            for (int row = 0; row < m_Params.mla_v_size; row++) {
                for (int col = 0; col < m_Params.mla_u_size; ++col) {
                    int n = row * m_Params.mla_u_size + col;
                    if (DNS[n].m_StereoStage == eSS_ACMH_Finished) {
                        //std::cout << n << std::endl;
                        Rc.push_back(std::pair<int, int>(row, col - 1));
                        Rc.push_back(std::pair<int, int>(row, col + 1));
                        Rc.push_back(std::pair<int, int>(row - 1, col));
                        Rc.push_back(std::pair<int, int>(row + 1, col));
                        Rc.push_back(std::pair<int, int>(row + 1, col - 1));
                        Rc.push_back(std::pair<int, int>(row - 1, col - 1));
                        Rc.push_back(std::pair<int, int>(row + 1, col + 1));
                        Rc.push_back(std::pair<int, int>(row - 1, col + 1));
                        for (int i = 0; i < m_Params.mi_height_for_match; i++)//ROW
                        {
                            for (int j = 0; j < m_Params.mi_width_for_match; j++)//COL
                            {
                                int na = i * m_Params.mi_width_for_match + j;
                                auto &disp = DNS[n].d_cuda[na];
                                int num = 0;

                                for (long unsigned int s1 = 0; s1 < Rc.size(); s1++) {
                                    int n_row = Rc[s1].first;
                                    int n_col = Rc[s1].second;
                                    int n1 = n_row * m_Params.mla_u_size + n_col;
                                    float B = sqrt(
                                            pow(center[n].x - center[n1].x, 2) + pow(center[n].y - center[n1].y, 2));
                                    if (n_row < 0 || n_col < 0 || n_row > m_Params.mla_v_size - 1 ||
                                        n_col > m_Params.mla_u_size - 1
                                        || DNS[n_row * m_Params.mla_u_size + n_col].m_StereoStage == eSS_ACMH_Begin ||
                                        B > m_Params.baseline) {
                                        continue;
                                    }
                                    float32 d_x = (center[n].x - center[n1].x) / B;
                                    float32 d_y = (center[n].y - center[n1].y) / B;
                                    const auto col_y = lround(j + d_x * disp);
                                    const auto col_x = lround(i + d_y * disp);

                                    if (col_x >= 0 && col_x < m_Params.mi_width_for_match &&
                                        col_y >= 0 && col_y < m_Params.mi_height_for_match) {
                                        // 右影像上同名像素的视差值
                                        int nb = col_x * m_Params.mi_height_for_match + col_y;
                                        auto &disp_r = DNS[n1].d_cuda[nb];

                                        // 判断两个视差值是否一致（差值在阈值内为一致）
                                        if (abs(disp - disp_r) <= 0.5) {
                                            //	// 让视差值无效
                                            //DNS1[n].dis[i * 121 + j] = 0;
                                            num++;
                                        }
                                    }
                                }
                                if (num < 2) {
                                    DNS[n].d_cuda[na] = 0;
                                    DNS[n].ph_cuda[na].w = 0;
                                    DNS[n].c_cuda[na] = 2.0;
                                    //DNS1[n].costs.at<float>(i,j) = 2.0;
                                }
                            }
                        }
                        Rc.clear();
                    }
                }
            }
        }

        void DepthSolver::LRCheck_TileKey(MLA_Problem &problem, QuadTreeDisNormalMap &dis_normal_map) {
            QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;
            QuadTreeTileInfoMap::iterator itr_main = m_MLA_info_map.find(ptrCurKey);
            if (itr_main == m_MLA_info_map.end()) {
                std::cout << "LRCheck Main Image Key not found: " << ptrCurKey->GetTileX() << ", tile_y="
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }

            float main_center_x = itr_main->second->GetCenter().x;
            float main_center_y = itr_main->second->GetCenter().y;
            QuadTreeDisNormalMap::iterator itr_main_dns = dis_normal_map.find(ptrCurKey);
            if (itr_main_dns == dis_normal_map.end()) {
                std::cout << "LRC: Main_DNS not found: tile_x=" << ptrCurKey->GetTileX() << ", tile_y="
                          << ptrCurKey->GetTileY() << std::endl;
                return;
            }
            DisparityAndNormalPtr ptrDis_Main = itr_main_dns->second;

            // step 1: 寻找有效的邻居
            std::vector<std::pair<int, int>> Rc;
            QuadTreeTileKeyPtrVec valid_neighbor_KeyVec;
            if (ptrDis_Main->m_StereoStage != eSS_ACMH_Finished) {
                return;
            }

            int tile_x = ptrCurKey->GetTileX();
            int tile_y = ptrCurKey->GetTileY();
            // 邻居
            Rc.push_back(std::pair<int, int>(tile_y, tile_x - 1)); // 左
            Rc.push_back(std::pair<int, int>(tile_y, tile_x + 1)); // 右
            Rc.push_back(std::pair<int, int>(tile_y - 1, tile_x)); // 上
            Rc.push_back(std::pair<int, int>(tile_y + 1, tile_x)); // 下
            Rc.push_back(std::pair<int, int>(tile_y + 1, tile_x - 1)); // 左下
            Rc.push_back(std::pair<int, int>(tile_y - 1, tile_x - 1)); // 左上
            Rc.push_back(std::pair<int, int>(tile_y + 1, tile_x + 1)); // 右下
            Rc.push_back(std::pair<int, int>(tile_y - 1, tile_x + 1)); // 右上

            for (uint neig_index = 0; neig_index < Rc.size(); neig_index++) {
                int tile_Y_neig = Rc[neig_index].first;
                int tile_X_neig = Rc[neig_index].second;

                QuadTreeTileKeyPtr ptrNeigKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_X_neig,
                                                                                tile_Y_neig);
                QuadTreeTileInfoMap::iterator itr_neig = m_MLA_info_map.find(ptrNeigKey);
                if (itr_neig == m_MLA_info_map.end()) {
                    std::cout << "LRC: Neig_info not found: tile_x= " << tile_X_neig << ", tile_y= "
                              << ptrNeigKey->GetTileY() << std::endl;
                    continue;
                }
                QuadTreeDisNormalMap::iterator itr_neig_dns = dis_normal_map.find(ptrNeigKey);
                if (itr_neig_dns == dis_normal_map.end()) {
                    std::cout << "LRC: Neig_dns not found: tile_x= " << tile_X_neig << ", tile_y= "
                              << ptrNeigKey->GetTileY() << std::endl;
                    continue;
                }

                MLA_InfoPtr ptrInfo_neig = itr_neig->second;
                DisparityAndNormalPtr ptrDN_neig = itr_neig_dns->second;
                float neig_center_x = ptrInfo_neig->GetCenter().x;
                float neig_center_y = ptrInfo_neig->GetCenter().y;

                float distance = sqrt(pow(main_center_x - neig_center_x, 2) + pow(main_center_y - neig_center_y, 2));
                if (ptrDN_neig->m_StereoStage == eSS_ACMH_Begin || distance > m_Params.baseline) {
                    continue;
                }
                valid_neighbor_KeyVec.push_back(ptrNeigKey);
            }

            // step 2: 遍历当前微图像中像素的深度值，做左右一致性检测
            for (int row = 0; row < m_Params.mi_height_for_match; row++) {
                for (int col = 0; col < m_Params.mi_width_for_match; col++) {
                    int d_index = row * m_Params.mi_width_for_match + col;
                    float &disp = ptrDis_Main->d_cuda[d_index];
                    int num = 0;

                    for (uint neig_index = 0; neig_index < valid_neighbor_KeyVec.size(); neig_index++) {
                        QuadTreeTileKeyPtr ptrNeigKey = valid_neighbor_KeyVec.at(neig_index);
                        MLA_InfoPtr ptrInfo_neig = m_MLA_info_map[ptrNeigKey];
                        float neig_center_x = ptrInfo_neig->GetCenter().x;
                        float neig_center_y = ptrInfo_neig->GetCenter().y;

                        DisparityAndNormalPtr ptrDN_neig = dis_normal_map[ptrNeigKey];

                        float distance = sqrt(
                                pow(main_center_x - neig_center_x, 2) + pow(main_center_y - neig_center_y, 2));
                        float32 d_x = (main_center_x - neig_center_x) / distance;
                        float32 d_y = (main_center_y - neig_center_y) / distance;
                        int right_x = lround(col + d_x * disp);
                        int right_y = lround(row + d_y * disp);
                        if (right_y >= 0 && right_y < m_Params.mi_height_for_match &&
                            right_x >= 0 && right_x < m_Params.mi_width_for_match) {
                            // 右影像上同名像素的视差值
                            int nb = right_y * m_Params.mi_width_for_match + right_x;
                            float &disp_r = ptrDN_neig->d_cuda[nb];
                            // 判断两个视差值是否一致（差值在阈值内为一致）
                            if (abs(disp - disp_r) <= 1.0) {
                                num++;
                            }
                        }
                    }

                    // 当前像素的视差偏差的统计结果
                    if (num < 2) {
                        ptrDis_Main->d_cuda[d_index] = -1.0;
                        ptrDis_Main->ph_cuda[d_index].w = 0.0;
                        ptrDis_Main->c_cuda[d_index] = 2.0;
                    }
                }
            }
        }

        void DepthSolver::LRCheck2(std::vector<DisparityAndNormal> &DNS, std::vector<DisparityAndNormal> &DNS1) {
            for (long unsigned int n = 0; n < DNS.size(); n++) {
                if (n != DNS.size() - 1) {
                    if (DNS[n].m_StereoStage == eSS_ACMH_Finished &&
                        DNS[n + 1].m_StereoStage == eSS_ACMH_Finished &&
                        DNS[n].row == DNS[n + 1].row) {
                        for (int i = 0; i < 121; i++) {
                            for (int j = 0; j < 121; j++) {
                                auto &disp = DNS[n].dis[i * 121 + j];
                                const auto col_right = lround(j - disp);

                                if (col_right >= 0 + 5 && col_right < 121 - 5) {
                                    // 右影像上同名像素的视差值
                                    auto &disp_r = DNS[n + 1].dis[i * 121 + col_right];

                                    // 判断两个视差值是否一致（差值在阈值内为一致）
                                    // 在本代码里，左右视图的视差值符号相反
                                    if (abs(disp - disp_r) > 0.5) {
                                        // 让视差值无效
                                        DNS1[n].dis[i * 121 + j] = 0;
                                    }
                                }
                            }
                        }
                    }
                }

                if (n != 0) {
                    if (DNS[n].m_StereoStage == eSS_ACMH_Finished &&
                        DNS[n - 1].m_StereoStage == eSS_ACMH_Finished &&
                        DNS[n].row == DNS[n - 1].row) {
                        for (int i = 0; i < 121; i++) {
                            for (int j = 0; j < 121; j++) {
                                auto &disp = DNS[n].dis[i * 121 + j];
                                const auto col_left = lround(j + disp);

                                if (col_left >= 0 + 5 && col_left < 121 - 5) {
                                    // 左影像上同名像素的视差值
                                    auto &disp_l = DNS[n - 1].dis[i * 121 + col_left];

                                    // 判断两个视差值是否一致（差值在阈值内为一致）
                                    if (abs(disp - disp_l) > 0.5) {
                                        // 让视差值无效
                                        DNS1[n].dis[i * 121 + j] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }


        void DepthSolver::Virtual_depth(std::vector<DisparityAndNormal> &DNS1, cv::Mat &VD_res, float &Base) {
            for (int i = 0; i < m_Params.mla_v_size; i++) {
                for (int j = 0; j < m_Params.mla_u_size; j++) {
                    if (DNS1[i * m_Params.mla_u_size + j].m_StereoStage == eSS_ACMH_Finished) {
                        for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                            for (int y = 0; y < m_Params.mi_height_for_match; y++) {

                                if (i % 2 == 0) {
                                    if (DNS1[i * m_Params.mla_u_size + j].d_cuda[x * m_Params.mi_height_for_match +
                                                                                 y] != 0) {
                                        //std::cout<< Base / DNS1[i * COL + j].d_cuda[x*WIDTH+y] <<std::endl;
                                        VD_res.at<uchar>(i * m_Params.mi_width_for_match + x,
                                                         j * m_Params.mi_height_for_match + y) = Base / DNS1[i *
                                                                                                             m_Params.mla_u_size +
                                                                                                             j].d_cuda[
                                                x * m_Params.mi_height_for_match + y] * 6;
                                        //VD_res.ptr(i * 111 + x)[j * 111 + y + 60] = B / DNS1[i * 79 + j].dis[(x + 5) * 121 + y + 5] * 50;
                                    } else {
                                        VD_res.at<uchar>(i * m_Params.mi_width_for_match + x,
                                                         j * m_Params.mi_height_for_match + y) = 0;
                                    }
                                } else {
                                    if (DNS1[i * m_Params.mla_u_size + j].d_cuda[x * m_Params.mi_height_for_match +
                                                                                 y] != 0) {
                                        VD_res.at<uchar>(i * m_Params.mi_width_for_match + x,
                                                         j * m_Params.mi_height_for_match + y +
                                                         m_Params.mi_height_for_match * 0.5) = Base / DNS1[i *
                                                                                                           m_Params.mla_u_size +
                                                                                                           j].d_cuda[
                                                x * m_Params.mi_height_for_match + y] * 6;
                                    } else {
                                        VD_res.at<uchar>(i * m_Params.mi_width_for_match + x,
                                                         j * m_Params.mi_height_for_match + y +
                                                         m_Params.mi_height_for_match * 0.5) = 0;
                                    }

                                }
                            }
                        }
                    }
                }
            }
            //cv::imwrite("E:\\LX\\DATA\\result\\test_select\\VD.png",VD_res);

        }

        void DepthSolver::Reshape_img(cv::Mat &img, std::vector<std::vector<cv::Point2f>> &center_1) {
            cv::Mat result = cv::Mat::zeros(6882, 8820, CV_8UC3);
            cv::Mat mask = cv::Mat::zeros(111, 111, CV_8UC3);

            for (int i = 0; i < 62; i++) {
                for (int j = 0; j < 79; j++) {
                    mask = img(cv::Rect(center_1[i][j].x - 55, center_1[i][j].y - 55, 111, 111));
                    for (int x = 0; x < 111; x++) {
                        for (int y = 0; y < 111; y++) {
                            if (i % 2 == 0) {
                                result.at<cv::Vec3b>(i * 111 + x, j * 111 + y) = mask.at<cv::Vec3b>(x, y);

                            } else {
                                result.at<cv::Vec3b>(i * 111 + x, j * 111 + y + 55) = mask.at<cv::Vec3b>(x, y);
                            }
                        }
                    }
                    mask.release();
                }
            }

            cv::imwrite("E:\\LX\\DATA\\result\\test_coll\\gd1.0\\img_rp.png", result);
        }

        void DepthSolver::Virtual_depth_map(cv::Mat &img, cv::Mat &result,
                                            std::vector<std::vector<cv::Point2f>> &center_inner, std::string &path,
                                            float Base) {
            for (int i = 0; i < img.rows; i++) {
                int H = i / m_Params.mi_height_for_match;
                for (int j = 0; j < img.cols; j++) {
                    //判断奇偶行
                    int T = m_Params.mi_width_for_match / 2;
                    int t = H % 2 == 0 ? 0 : T;
                    //每个像素
                    if (img.at<uchar>(i, j) != 0) {
                        int L = (j - t) / m_Params.mi_width_for_match;
                        int y = i - H * m_Params.mi_height_for_match;
                        int x = (j - t) - L * m_Params.mi_width_for_match;
                        //原始光场图像坐标点
                        int xr = x - T;
                        int yr = y - T;
                        double v = double(img.at<uchar>(i, j)) / 20.0;
                        //虚像空间中 投影坐标点
                        int xv = xr * v + center_inner[H][L].x;
                        int yv = yr * v + center_inner[H][L].y;

                        if (xv < m_Params.mla_u_size * Base && xv > 0 && yv > 0 && yv < m_Params.mla_v_size * Base) {
                            result.at<uchar>(yv, xv) = Base * 20 / img.at<uchar>(i, j) * 10;
                        }
                    }

                }
            }
            cv::Mat disp_color;
            applyColorMap(result, disp_color, cv::COLORMAP_JET);
            cv::imwrite(path + std::string("/VD_col.png"), disp_color);
            cv::imwrite(path + std::string("/VD.png"), result);
        }

        void DepthSolver::Focus_AIF_VD(QuadTreeProblemMapMap::iterator& itrFrame)
        {
            struct vdinfo {
                //以向量形式堆积颜色和虚拟深度，后面带m的为中位数，以中位数代替最终结果，以去除噪声影响
                std::vector<uchar> color0;
                uchar colorm0;
                std::vector<uchar> color1;
                uchar colorm1;
                std::vector<uchar> color2;
                uchar colorm2;
                std::vector<float> vd;
                float vdm;
            };

            std::vector<PointList> PointCloud;
            PointCloud.clear();

            std::string strName = itrFrame->first;
            QuadTreeProblemMap &problem_map = itrFrame->second;
            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[itrFrame->first];

            cv::Mat virtualDepth_float_tmp = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_32FC1);
            cv::Mat virtualDepth = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);
            cv::Mat virtualDepth_realColor = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC3);

            std::vector<vdinfo> vd;
            vd.reserve(m_RawImage_Height * m_RawImage_Width);

            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;
                QuadTreeDisNormalMap::iterator itr_dns = dis_normal_map.find(problem.m_ptrKey);
                if (itr_dns == dis_normal_map.end()) {
                    std::cout << "VDMTN: cur_DNS not found," << problem.m_ptrKey->StrRemoveLOD().c_str()
                              << std::endl;
                    continue;
                }
                DisparityAndNormalPtr ptrMLADis = itr_dns->second;

                QuadTreeTileInfoMap::iterator itrC = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrC == m_MLA_info_map.end()) {
                    std::cout << "VDMTN: cur_info can not found, " << problem.m_ptrKey->StrRemoveLOD().c_str()
                              << std::endl;
                    continue;
                }
                MLA_InfoPtr ptrInfo = itrC->second;

                if (ptrMLADis->m_StereoStage == eSS_ACMH_Finished ||
                    ptrMLADis->m_StereoStage == eSS_PlannerPrior_Finished) {
                    for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                        for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                            float disp = ptrMLADis->d_cuda[y * m_Params.mi_width_for_match + x];
                            if (disp <= 0.0)
                                disp = 1;
//                            //为使2600scene2的结果更好看另外增加的过滤条件
//                            if(disp>21)
//                                continue;
//                            if(disp<8)
//                                continue;

                            // 视差越小，物距越小，f_v越大
                            float mla_Base =
                                    m_Params.baseline * LFMVS::g_bl0 / (LFMVS::g_bl0 + LFMVS::g_B);//使用微透镜基线
                            float f_v = mla_Base / disp;
                            int x_mlaCenter = x - ((m_Params.mi_width_for_match - 1) * 0.5 + 1);
                            // 投影
                            float Xv = x_mlaCenter * f_v + ptrInfo->GetCenter().x;
                            int y_mlaCenter = y - ((m_Params.mi_height_for_match - 1) * 0.5 + 1);
                            float Yv = y_mlaCenter * f_v + ptrInfo->GetCenter().y;

                            int vd_y = (int) round(Yv);
                            int vd_x = (int) round(Xv);


                            if (vd_x > 0 && vd_x < virtualDepth.cols &&
                                vd_y > 0 && vd_y < virtualDepth.rows) {
                                vdinfo &info = vd[vd_y * virtualDepth.cols + vd_x];
                                info.vd.push_back(f_v);
                                info.color0.push_back(problem.m_Image_rgb.at<uchar3>(y, x).x);
                                info.color1.push_back(problem.m_Image_rgb.at<uchar3>(y, x).y);
                                info.color2.push_back(problem.m_Image_rgb.at<uchar3>(y, x).z);
                                //格式修改完成,接下来去除离群值
                            }
                        }
                    }
                }
            }
            //选出其中的中值,表示去除离群值后的均值
            for (int y = 0; y < virtualDepth.rows; y++) {
                for (int x = 0; x < virtualDepth.cols; x++) {
                    vdinfo &info = vd[y * virtualDepth.cols + x];
                    //从小到大排序
                    if (info.vd.size() > 0) {
                        std::sort(info.vd.begin(), info.vd.end());
                        std::sort(info.color0.begin(), info.color0.end());
                        std::sort(info.color1.begin(), info.color1.end());
                        std::sort(info.color2.begin(), info.color2.end());
                        info.vdm =
                                (info.vd[int(info.vd.size() / 2 + 1)] + info.vd[int((info.vd.size() + 1) / 2)]) / 2;
                        info.colorm0 = (info.color0[int(info.color0.size() / 2)] +
                                        info.color0[int((info.color0.size() + 1) / 2) - 1]) / 2;
                        info.colorm1 = (info.color1[int(info.color1.size() / 2)] +
                                        info.color1[int((info.color1.size() + 1) / 2) - 1]) / 2;
                        info.colorm2 = (info.color2[int(info.color2.size() / 2)] +
                                        info.color2[int((info.color2.size() + 1) / 2) - 1]) / 2;

                        //改回均值看看效果，均值效果正常，问题就在这一步
//                        info.colorm1=0;
//                        info.colorm1=0;
//                        info.colorm2=0;
//
//
//                        for(int i=0;i<info.vd.size();i++)
//                        {
//                            info.colorm0+=info.color0[i];
//                            info.colorm1+=info.color1[i];
//                            info.colorm2+=info.color2[i];
//                        }
//                        info.colorm0/=info.color0.size();
//                        info.colorm1/=info.color0.size();
//                        info.colorm2/=info.color0.size();
                    }
                }
            }

            //
            DisparityRange &dis_range = m_disparityRangeMap[strName];

            for (int y = 0; y < virtualDepth.rows; y++) {
                for (int x = 0; x < virtualDepth.cols; x++) {
                    vdinfo &info = vd[y * virtualDepth.cols + x];
                    if (info.vd.size() > 0) {
                        float vd_value = info.vdm;
                        virtualDepth_float_tmp.at<float32>(y, x) = vd_value;
                        if (dis_range.m_vd_min > vd_value)
                            dis_range.m_vd_min = vd_value;
                        if (dis_range.m_vd_max < vd_value)
                            dis_range.m_vd_max = vd_value;
                        virtualDepth_realColor.at<uchar3>(y, x).x = info.colorm0;
                        virtualDepth_realColor.at<uchar3>(y, x).y = info.colorm1;
                        virtualDepth_realColor.at<uchar3>(y, x).z = info.colorm2;

                    } else//修改以填满所有像素
                    {
                        bool bContinue = true;
                        int t = 4;
                        int k = y * virtualDepth.cols + x;
                        while (bContinue) {

                            switch (t % 4) {
                                case 0:
                                    k = y * virtualDepth.cols + x + t / 4;
                                    break;
                                case 1:
                                    k = y * virtualDepth.cols + x - t / 4;
                                    break;
                                case 2:
                                    k = (y + t / 4) * virtualDepth.cols + x - t / 4;
                                    break;
                                case 3:
                                    k = (y - t / 4) * virtualDepth.cols + x;
                                    break;
                            }
                            t++;
                            if (k >= virtualDepth.rows * virtualDepth.cols || k < 0)
                                break;
                            vdinfo &info1 = vd[k];
                            if (info1.vd.size() > 0) {

                                virtualDepth_realColor.at<uchar3>(y, x).x = info1.colorm0;
                                virtualDepth_realColor.at<uchar3>(y, x).y = info1.colorm1;
                                virtualDepth_realColor.at<uchar3>(y, x).z = info1.colorm2;
                                bContinue = false;
                            }
                        }
                    }
                }
            }
            //现打算去噪
            //直接模糊处理效果不好
            //cv::blur(virtualDepth_realColor,virtualDepth_realColor,cv::Size(2,2),cv::Point(-1,-1),4);
            //若像素和周围均值相差不大则不动，相差大则改为周围像素均值，均值会受到其他噪声影响
            //试试中值滤波，原理和众数收集一致
            for (int i = 0; i < 10; i++)
                cv::medianBlur(virtualDepth_realColor, virtualDepth_realColor, 5);


            std::cout << "VDTN: vd_min = " << dis_range.m_vd_min << ", vd_max = " << dis_range.m_vd_max
                      << std::endl;

            for (int y = 0; y < virtualDepth.rows; y++) {
                for (int x = 0; x < virtualDepth.cols; x++) {
                    vdinfo &info = vd[y * virtualDepth.cols + x];
                    if (info.vd.size() > 0) {
                        float vd_float = virtualDepth_float_tmp.at<float32>(y, x);
                        virtualDepth.at<uchar>(y, x) = (vd_float - dis_range.m_vd_min * 0.6) /
                                                       ((dis_range.m_vd_max - dis_range.m_vd_min) * 0.8) * 255;

                        PointList point3D;
                        {
                            //将x,y转为世界坐标系
                            double tmp_x = (x - m_Params.mi_width_for_match * 0.5) * m_Params.sensor_pixel_size;
                            double tmp_y = (y - m_Params.mi_height_for_match * 0.5) * m_Params.sensor_pixel_size;
                            double tmp_z = info.vdm * LFMVS::g_B + LFMVS::g_bl0;

                            //std::cout<<"info.vdm:"<<info.vdm<<"-tmp_z:"<<tmp_z<<std::endl;

                            //此处变量含义为真实深度，为方便程序运行没有改名
                            double real_d =
                                    m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) *
                                    tmp_z;

                            double real_x =
                                    -m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) *
                                    tmp_x;
                            double real_y =
                                    -m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) *
                                    tmp_y;

                            point3D.coord.x = real_x;
                            point3D.coord.y = real_y;
                            point3D.coord.z = real_d;
                            point3D.color.x = virtualDepth_realColor.at<uchar3>(y, x).x;
                            point3D.color.y = virtualDepth_realColor.at<uchar3>(y, x).y;
                            point3D.color.z = virtualDepth_realColor.at<uchar3>(y, x).z;

                            if (abs(point3D.coord.z) < 10000 && abs(info.vdm) > 0.0001)
                                PointCloud.push_back(point3D);
                        }
                    }
                }
            }

            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
            cv::Mat virtualDepth_color;
            // 从蓝色到红色渐变，中间经过绿色和黄色。此处，物距由小到大，对应的颜色为蓝色--绿色--黄色-红色。
            applyColorMap(virtualDepth, virtualDepth_color, cv::COLORMAP_JET);
            std::string strFullPath_gray = strSavePath;
            std::string strFullPath_color = strSavePath;
            std::string strFullPath_real_color = strSavePath;
            switch (m_eStereoType) {
                case eST_ACMH: {
                    strFullPath_gray += "/VD_Base_gray.png";
                    strFullPath_color += "/VD_Base_color.png";
                    strFullPath_real_color += "/" + strName + "_" + LF_FOUCSIMAGE_NAME + ".png";
                    if (m_bPlannar != false || m_bLRCheck != true) {
                        std::cout << "base: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_PlannerPrior: {
                    strFullPath_gray += "/VD_Plannar_gray.png";
                    strFullPath_color += "/VD_Plannar_color.png";
                    strFullPath_real_color += "/" + strName + "_" + LF_FOUCSIMAGE_NAME + ".png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "planner: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_Horizontal_Propagation: {
                    strFullPath_gray += "/VD_HorProga_gray.png";
                    strFullPath_color += "/VD_HorProga_color.png";
                    strFullPath_real_color += "/" + strName + "_" + LF_FOUCSIMAGE_NAME + ".png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_BlurFeature: {
                    strFullPath_gray += "/VD_BlurFeature_gray.png";
                    strFullPath_color += "/VD_BlurFeature_color.png";
                    strFullPath_real_color += "/" + strName + "_" + LF_FOUCSIMAGE_NAME + ".png";
                }
                    break;
                default:
                    break;
            }
            cv::imwrite(strFullPath_gray, virtualDepth);
            cv::imwrite(strFullPath_color, virtualDepth_color);
            cv::imwrite(strFullPath_real_color, virtualDepth_realColor);
            StoreColorPlyFileBinaryPointCloud(strSavePath + std::string("/VD.ply"), PointCloud);
        }

        void DepthSolver::Virtual_depth_map_TileKey() {
            struct vdinfo {
                //以向量形式堆积颜色和虚拟深度，后面带m的为中位数，以中位数代替最终结果，以去除噪声影响
                std::vector<uchar> color0;
                uchar colorm0;
                std::vector<uchar> color1;
                uchar colorm1;
                std::vector<uchar> color2;
                uchar colorm2;
                std::vector<float> vd;
                float vdm;
            };

            std::vector<PointList> PointCloud;
            PointCloud.clear();

            for (QuadTreeProblemMapMap::iterator itrP = m_MIA_problem_map_map.begin();
                 itrP != m_MIA_problem_map_map.end(); itrP++) {
                std::string strName = itrP->first;
                QuadTreeProblemMap &problem_map = itrP->second;
                QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[itrP->first];

                cv::Mat virtualDepth_float_tmp = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_32FC1);

                cv::Mat virtualDepth = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);
                cv::Mat virtualDepth_realColor = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC3);

                std::vector<vdinfo> vd;
                vd.reserve(m_RawImage_Height * m_RawImage_Width);

                for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                    MLA_Problem &problem = itr->second;
                    QuadTreeDisNormalMap::iterator itr_dns = dis_normal_map.find(problem.m_ptrKey);
                    if (itr_dns == dis_normal_map.end()) {
                        std::cout << "VDMTN: cur_DNS not found," << problem.m_ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    DisparityAndNormalPtr ptrMLADis = itr_dns->second;

                    QuadTreeTileInfoMap::iterator itrC = m_MLA_info_map.find(problem.m_ptrKey);
                    if (itrC == m_MLA_info_map.end()) {
                        std::cout << "VDMTN: cur_info can not found, " << problem.m_ptrKey->StrRemoveLOD().c_str()
                                  << std::endl;
                        continue;
                    }
                    MLA_InfoPtr ptrInfo = itrC->second;

                    if (ptrMLADis->m_StereoStage == eSS_ACMH_Finished ||
                        ptrMLADis->m_StereoStage == eSS_PlannerPrior_Finished) {
                        for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                            for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                                float disp = ptrMLADis->d_cuda[y * m_Params.mi_width_for_match + x];
                                if (disp <= 0.0)
                                    disp = 1;
//                            //为使2600scene2的结果更好看另外增加的过滤条件
//                            if(disp>21)
//                                continue;
//                            if(disp<8)
//                                continue;

                                // 视差越小，物距越小，f_v越大
                                float mla_Base =
                                        m_Params.baseline * LFMVS::g_bl0 / (LFMVS::g_bl0 + LFMVS::g_B);//使用微透镜基线
                                float f_v = mla_Base / disp;
                                int x_mlaCenter = x - ((m_Params.mi_width_for_match - 1) * 0.5 + 1);
                                float Xv = x_mlaCenter * f_v + ptrInfo->GetCenter().x;
                                int y_mlaCenter = y - ((m_Params.mi_height_for_match - 1) * 0.5 + 1);
                                float Yv = y_mlaCenter * f_v + ptrInfo->GetCenter().y;

                                int vd_y = (int) round(Yv);
                                int vd_x = (int) round(Xv);


                                if (vd_x > 0 && vd_x < virtualDepth.cols &&
                                    vd_y > 0 && vd_y < virtualDepth.rows) {
                                    vdinfo &info = vd[vd_y * virtualDepth.cols + vd_x];
                                    info.vd.push_back(f_v);
                                    info.color0.push_back(problem.m_Image_rgb.at<uchar3>(y, x).x);
                                    info.color1.push_back(problem.m_Image_rgb.at<uchar3>(y, x).y);
                                    info.color2.push_back(problem.m_Image_rgb.at<uchar3>(y, x).z);
                                    //格式修改完成,接下来去除离群值
                                }
                            }
                        }
                    }
                }
                //选出其中的中值,表示去除离群值后的均值
                for (int y = 0; y < virtualDepth.rows; y++) {
                    for (int x = 0; x < virtualDepth.cols; x++) {
                        vdinfo &info = vd[y * virtualDepth.cols + x];
                        //从小到大排序
                        if (info.vd.size() > 0) {
                            std::sort(info.vd.begin(), info.vd.end());
                            std::sort(info.color0.begin(), info.color0.end());
                            std::sort(info.color1.begin(), info.color1.end());
                            std::sort(info.color2.begin(), info.color2.end());
                            info.vdm =
                                    (info.vd[int(info.vd.size() / 2 + 1)] + info.vd[int((info.vd.size() + 1) / 2)]) / 2;
                            info.colorm0 = (info.color0[int(info.color0.size() / 2)] +
                                            info.color0[int((info.color0.size() + 1) / 2) - 1]) / 2;
                            info.colorm1 = (info.color1[int(info.color1.size() / 2)] +
                                            info.color1[int((info.color1.size() + 1) / 2) - 1]) / 2;
                            info.colorm2 = (info.color2[int(info.color2.size() / 2)] +
                                            info.color2[int((info.color2.size() + 1) / 2) - 1]) / 2;

                            //改回均值看看效果，均值效果正常，问题就在这一步
//                        info.colorm1=0;
//                        info.colorm1=0;
//                        info.colorm2=0;
//
//
//                        for(int i=0;i<info.vd.size();i++)
//                        {
//                            info.colorm0+=info.color0[i];
//                            info.colorm1+=info.color1[i];
//                            info.colorm2+=info.color2[i];
//                        }
//                        info.colorm0/=info.color0.size();
//                        info.colorm1/=info.color0.size();
//                        info.colorm2/=info.color0.size();
                        }
                    }
                }

                //
                DisparityRange &dis_range = m_disparityRangeMap[strName];

                for (int y = 0; y < virtualDepth.rows; y++) {
                    for (int x = 0; x < virtualDepth.cols; x++) {
                        vdinfo &info = vd[y * virtualDepth.cols + x];
                        if (info.vd.size() > 0) {
                            float vd_value = info.vdm;
                            virtualDepth_float_tmp.at<float32>(y, x) = vd_value;
                            if (dis_range.m_vd_min > vd_value)
                                dis_range.m_vd_min = vd_value;
                            if (dis_range.m_vd_max < vd_value)
                                dis_range.m_vd_max = vd_value;
                            virtualDepth_realColor.at<uchar3>(y, x).x = info.colorm0;
                            virtualDepth_realColor.at<uchar3>(y, x).y = info.colorm1;
                            virtualDepth_realColor.at<uchar3>(y, x).z = info.colorm2;

                        } else//修改以填满所有像素
                        {
                            bool bContinue = true;
                            int t = 4;
                            int k = y * virtualDepth.cols + x;
                            while (bContinue) {

                                switch (t % 4) {
                                    case 0:
                                        k = y * virtualDepth.cols + x + t / 4;
                                        break;
                                    case 1:
                                        k = y * virtualDepth.cols + x - t / 4;
                                        break;
                                    case 2:
                                        k = (y + t / 4) * virtualDepth.cols + x - t / 4;
                                        break;
                                    case 3:
                                        k = (y - t / 4) * virtualDepth.cols + x;
                                        break;
                                }
                                t++;
                                if (k >= virtualDepth.rows * virtualDepth.cols || k < 0)
                                    break;
                                vdinfo &info1 = vd[k];
                                if (info1.vd.size() > 0) {

                                    virtualDepth_realColor.at<uchar3>(y, x).x = info1.colorm0;
                                    virtualDepth_realColor.at<uchar3>(y, x).y = info1.colorm1;
                                    virtualDepth_realColor.at<uchar3>(y, x).z = info1.colorm2;
                                    bContinue = false;
                                }
                            }
                        }
                    }
                }
                //现打算去噪
                //直接模糊处理效果不好
                //cv::blur(virtualDepth_realColor,virtualDepth_realColor,cv::Size(2,2),cv::Point(-1,-1),4);
                //若像素和周围均值相差不大则不动，相差大则改为周围像素均值，均值会受到其他噪声影响
                //试试中值滤波，原理和众数收集一致
                for (int i = 0; i < 10; i++)
                    cv::medianBlur(virtualDepth_realColor, virtualDepth_realColor, 5);


                std::cout << "VDTN: vd_min = " << dis_range.m_vd_min << ", vd_max = " << dis_range.m_vd_max
                          << std::endl;

                for (int y = 0; y < virtualDepth.rows; y++) {
                    for (int x = 0; x < virtualDepth.cols; x++) {
                        vdinfo &info = vd[y * virtualDepth.cols + x];
                        if (info.vd.size() > 0) {
                            float vd_float = virtualDepth_float_tmp.at<float32>(y, x);
                            virtualDepth.at<uchar>(y, x) = (vd_float - dis_range.m_vd_min * 0.6) /
                                                           ((dis_range.m_vd_max - dis_range.m_vd_min) * 0.8) * 255;

                            PointList point3D;
                            {
                                //将x,y转为世界坐标系
                                double tmp_x = (x - m_Params.mi_width_for_match * 0.5) * m_Params.sensor_pixel_size;
                                double tmp_y = (y - m_Params.mi_height_for_match * 0.5) * m_Params.sensor_pixel_size;
                                double tmp_z = info.vdm * LFMVS::g_B + LFMVS::g_bl0;

                                //std::cout<<"info.vdm:"<<info.vdm<<"-tmp_z:"<<tmp_z<<std::endl;

                                //此处变量含义为真实深度，为方便程序运行没有改名
                                double real_d =
                                        m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) *
                                        tmp_z;

                                double real_x =
                                        -m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) *
                                        tmp_x;
                                double real_y =
                                        -m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) *
                                        tmp_y;

                                point3D.coord.x = real_x;
                                point3D.coord.y = real_y;
                                point3D.coord.z = real_d;
                                point3D.color.x = virtualDepth_realColor.at<uchar3>(y, x).x;
                                point3D.color.y = virtualDepth_realColor.at<uchar3>(y, x).y;
                                point3D.color.z = virtualDepth_realColor.at<uchar3>(y, x).z;

                                if (abs(point3D.coord.z) < 10000 && abs(info.vdm) > 0.0001)
                                    PointCloud.push_back(point3D);
                            }
                        }
                    }
                }

                std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
                {
                    boost::filesystem::path dir_save_path(strSavePath);
                    if (!boost::filesystem::exists(dir_save_path)) {
                        if (!boost::filesystem::create_directory(dir_save_path)) {
                            std::cout << "dir failed to create: " << strSavePath << std::endl;
                        }
                    }
                }
                cv::Mat virtualDepth_color;
                // 从蓝色到红色渐变，中间经过绿色和黄色。此处，物距由小到大，对应的颜色为蓝色--绿色--黄色-红色。
                applyColorMap(virtualDepth, virtualDepth_color, cv::COLORMAP_JET);
                std::string strFullPath_gray = strSavePath;
                std::string strFullPath_color = strSavePath;
                std::string strFullPath_real_color = strSavePath;
                switch (m_eStereoType) {
                    case eST_ACMH: {
                        strFullPath_gray += "/VD_Base_gray.png";
                        strFullPath_color += "/VD_Base_color.png";
                        strFullPath_real_color += "/VD_Base_real_color.png";
                        if (m_bPlannar != false || m_bLRCheck != true) {
                            std::cout << "base: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_PlannerPrior: {
                        strFullPath_gray += "/VD_Plannar_gray.png";
                        strFullPath_color += "/VD_Plannar_color.png";
                        strFullPath_real_color += "/VD_Plannar_real_color.png";
                        if (m_bPlannar != true || m_bLRCheck != true) {
                            std::cout << "planner: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_Horizontal_Propagation: {
                        strFullPath_gray += "/VD_HorProga_gray.png";
                        strFullPath_color += "/VD_HorProga_color.png";
                        strFullPath_real_color += "/VD_HorProga_real_color.png";
                        if (m_bPlannar != true || m_bLRCheck != true) {
                            std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                        }
                    }
                        break;
                    case eST_BlurFeature: {
                        strFullPath_gray += "/VD_BlurFeature_gray.png";
                        strFullPath_color += "/VD_BlurFeature_color.png";
                        strFullPath_real_color += "/VD_BlurFeature_real_color.png";
                    }
                        break;
                    default:
                        break;
                }
                cv::imwrite(strFullPath_gray, virtualDepth);
                cv::imwrite(strFullPath_color, virtualDepth_color);
                cv::imwrite(strFullPath_real_color, virtualDepth_realColor);
                StoreColorPlyFileBinaryPointCloud(strSavePath + std::string("/VD.ply"), PointCloud);
            }
        }

        void DepthSolver::Virtual_depth_map_TileKeySequence(std::string &strName, QuadTreeProblemMap &problem_map) {
            struct vdinfo {
                //以向量形式堆积颜色和虚拟深度，后面带m的为中位数，以中位数代替最终结果，以去除噪声影响
                std::vector<uchar> color0;
                uchar colorm0;
                std::vector<uchar> color1;
                uchar colorm1;
                std::vector<uchar> color2;
                uchar colorm2;
                std::vector<float> vd;
                float vdm;
            };

            std::vector<PointList> PointCloud;
            PointCloud.clear();

            QuadTreeDisNormalMap &dis_normal_map = m_MIA_dispNormal_map_map[strName];
            cv::Mat virtualDepth_float_tmp = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_32FC1);

            cv::Mat virtualDepth = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC1);
            cv::Mat virtualDepth_realColor = cv::Mat::zeros(m_RawImage_Height, m_RawImage_Width, CV_8UC3);

            std::vector<vdinfo> vd;
            vd.reserve(m_RawImage_Height * m_RawImage_Width);

            for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++) {
                MLA_Problem &problem = itr->second;
                QuadTreeDisNormalMap::iterator itr_dns = dis_normal_map.find(problem.m_ptrKey);
                if (itr_dns == dis_normal_map.end()) {
                    std::cout << "VDMTN: cur_DNS not found," << problem.m_ptrKey->StrRemoveLOD().c_str() << std::endl;
                    continue;
                }
                DisparityAndNormalPtr ptrMLADis = itr_dns->second;

                QuadTreeTileInfoMap::iterator itrC = m_MLA_info_map.find(problem.m_ptrKey);
                if (itrC == m_MLA_info_map.end()) {
                    std::cout << "VDMTN: cur_info can not found, " << problem.m_ptrKey->StrRemoveLOD().c_str()
                              << std::endl;
                    continue;
                }
                MLA_InfoPtr ptrInfo = itrC->second;

                if (ptrMLADis->m_StereoStage == eSS_ACMH_Finished ||
                    ptrMLADis->m_StereoStage == eSS_PlannerPrior_Finished) {
                    for (int y = 0; y < m_Params.mi_height_for_match; y++) {
                        for (int x = 0; x < m_Params.mi_width_for_match; x++) {
                            float disp = ptrMLADis->d_cuda[y * m_Params.mi_width_for_match + x];
                            if (disp <= 0.0)
                                disp = 1;


//                            //为使2600scene2的结果更好看另外增加的过滤条件
//                            if(disp>21)
//                                continue;
//                            if(disp<8)
//                                continue;

                            // 视差越小，物距越远，f_v越大
                            float mla_Base = m_Params.baseline * LFMVS::g_bl0 / (LFMVS::g_bl0 + LFMVS::g_B);//使用微透镜基线
                            float f_v = mla_Base / disp;
                            int x_mlaCenter = x - ((m_Params.mi_width_for_match - 1) * 0.5 + 1);
                            float Xv = x_mlaCenter * f_v + ptrInfo->GetCenter().x;
                            int y_mlaCenter = y - ((m_Params.mi_height_for_match - 1) * 0.5 + 1);
                            float Yv = y_mlaCenter * f_v + ptrInfo->GetCenter().y;

                            int vd_y = (int) round(Yv);
                            int vd_x = (int) round(Xv);


                            if (vd_x > 0 && vd_x < virtualDepth.cols &&
                                vd_y > 0 && vd_y < virtualDepth.rows) {
                                vdinfo &info = vd[vd_y * virtualDepth.cols + vd_x];
                                info.vd.push_back(f_v);
                                info.color0.push_back(problem.m_Image_rgb.at<uchar3>(y, x).x);
                                info.color1.push_back(problem.m_Image_rgb.at<uchar3>(y, x).y);
                                info.color2.push_back(problem.m_Image_rgb.at<uchar3>(y, x).z);
                                //格式修改完成,接下来去除离群值
                            }
                        }
                    }
                }
            }
            //选出其中的中值,表示去除离群值后的均值
            for (int y = 0; y < virtualDepth.rows; y++) {
                for (int x = 0; x < virtualDepth.cols; x++) {
                    vdinfo &info = vd[y * virtualDepth.cols + x];
                    //从小到大排序
                    if (info.vd.size() > 0) {
                        std::sort(info.vd.begin(), info.vd.end());
                        std::sort(info.color0.begin(), info.color0.end());
                        std::sort(info.color1.begin(), info.color1.end());
                        std::sort(info.color2.begin(), info.color2.end());
                        info.vdm = (info.vd[int(info.vd.size() / 2 + 1)] + info.vd[int((info.vd.size() + 1) / 2)]) / 2;
                        info.colorm0 = (info.color0[int(info.color0.size() / 2)] +
                                        info.color0[int((info.color0.size() + 1) / 2) - 1]) / 2;
                        info.colorm1 = (info.color1[int(info.color1.size() / 2)] +
                                        info.color1[int((info.color1.size() + 1) / 2) - 1]) / 2;
                        info.colorm2 = (info.color2[int(info.color2.size() / 2)] +
                                        info.color2[int((info.color2.size() + 1) / 2) - 1]) / 2;

                        //改回均值看看效果，均值效果正常，问题就在这一步
//                        info.colorm1=0;
//                        info.colorm1=0;
//                        info.colorm2=0;
//
//
//                        for(int i=0;i<info.vd.size();i++)
//                        {
//                            info.colorm0+=info.color0[i];
//                            info.colorm1+=info.color1[i];
//                            info.colorm2+=info.color2[i];
//                        }
//                        info.colorm0/=info.color0.size();
//                        info.colorm1/=info.color0.size();
//                        info.colorm2/=info.color0.size();
                    }


                }
            }

            //
            DisparityRange &dis_range = m_disparityRangeMap[strName];

            for (int y = 0; y < virtualDepth.rows; y++) {
                for (int x = 0; x < virtualDepth.cols; x++) {
                    vdinfo &info = vd[y * virtualDepth.cols + x];
                    if (info.vd.size() > 0) {
                        float vd_value = info.vdm;
                        virtualDepth_float_tmp.at<float32>(y, x) = vd_value;
                        if (dis_range.m_vd_min > vd_value)
                            dis_range.m_vd_min = vd_value;
                        if (dis_range.m_vd_max < vd_value)
                            dis_range.m_vd_max = vd_value;
                        virtualDepth_realColor.at<uchar3>(y, x).x = info.colorm0;
                        virtualDepth_realColor.at<uchar3>(y, x).y = info.colorm1;
                        virtualDepth_realColor.at<uchar3>(y, x).z = info.colorm2;

                    } else//修改以填满所有像素
                    {
                        bool bContinue = true;
                        int t = 4;
                        int k = y * virtualDepth.cols + x;
                        while (bContinue) {

                            switch (t % 4) {
                                case 0:
                                    k = y * virtualDepth.cols + x + t / 4;
                                    break;
                                case 1:
                                    k = y * virtualDepth.cols + x - t / 4;
                                    break;
                                case 2:
                                    k = (y + t / 4) * virtualDepth.cols + x - t / 4;
                                    break;
                                case 3:
                                    k = (y - t / 4) * virtualDepth.cols + x;
                                    break;
                            }
                            t++;
                            if (k >= virtualDepth.rows * virtualDepth.cols || k < 0)
                                break;
                            vdinfo &info1 = vd[k];
                            if (info1.vd.size() > 0) {

                                virtualDepth_realColor.at<uchar3>(y, x).x = info1.colorm0;
                                virtualDepth_realColor.at<uchar3>(y, x).y = info1.colorm1;
                                virtualDepth_realColor.at<uchar3>(y, x).z = info1.colorm2;
                                bContinue = false;
                            }
                        }
                    }
                }
            }
            //现打算去噪
            //直接模糊处理效果不好
            //cv::blur(virtualDepth_realColor,virtualDepth_realColor,cv::Size(2,2),cv::Point(-1,-1),4);
            //若像素和周围均值相差不大则不动，相差大则改为周围像素均值，均值会受到其他噪声影响
            //试试中值滤波，原理和众数收集一致
            for (int i = 0; i < 10; i++)
                cv::medianBlur(virtualDepth_realColor, virtualDepth_realColor, 5);

            std::cout << "VDTN: vd_min = " << dis_range.m_vd_min << ", vd_max = " << dis_range.m_vd_max << std::endl;
            if (dis_range.m_vd_min < 0.0) {
                dis_range.m_vd_min = 0.0;
            }
            std::cout << "VDTN: vd_min = " << dis_range.m_vd_min << ", vd_max = " << dis_range.m_vd_max << std::endl;

            for (int y = 0; y < virtualDepth.rows; y++) {
                for (int x = 0; x < virtualDepth.cols; x++) {
                    vdinfo &info = vd[y * virtualDepth.cols + x];
                    if (info.vd.size() > 0) {
                        float vd_float = virtualDepth_float_tmp.at<float32>(y, x);
                        virtualDepth.at<uchar>(y, x) = (vd_float - dis_range.m_vd_min * 0.6) /
                                                       ((dis_range.m_vd_max - dis_range.m_vd_min) * 0.6) * 255;

                        PointList point3D;
                        {
                            //将x,y转为世界坐标系
                            double tmp_x = (x - m_Params.mi_width_for_match * 0.5) * m_Params.sensor_pixel_size;
                            double tmp_y = (y - m_Params.mi_height_for_match * 0.5) * m_Params.sensor_pixel_size;
                            double tmp_z = info.vdm * LFMVS::g_B + LFMVS::g_bl0;

                            //此处变量含义为真实深度，为方便程序运行没有改名
                            double real_d =
                                    m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) * tmp_z;

                            double real_x =
                                    -m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) * tmp_x;
                            double real_y =
                                    -m_Params.mainlen_flocal_length / (tmp_z - m_Params.mainlen_flocal_length) * tmp_y;

                            point3D.coord.x = real_x;
                            point3D.coord.y = real_y;
                            point3D.coord.z = real_d;
                            point3D.color.x = virtualDepth_realColor.at<uchar3>(y, x).x;
                            point3D.color.y = virtualDepth_realColor.at<uchar3>(y, x).y;
                            point3D.color.z = virtualDepth_realColor.at<uchar3>(y, x).z;
                            PointCloud.push_back(point3D);
                        }
                    }
                }
            }

            std::string strSavePath = m_strSavePath + strName + LF_MVS_RESULT_DATA_NAME;
            {
                boost::filesystem::path dir_save_path(strSavePath);
                if (!boost::filesystem::exists(dir_save_path)) {
                    if (!boost::filesystem::create_directory(dir_save_path)) {
                        std::cout << "dir failed to create: " << strSavePath << std::endl;
                    }
                }
            }
            cv::Mat virtualDepth_color;
            // 从蓝色到红色渐变，中间经过绿色和黄色。此处，物距由小到大，对应的颜色为蓝色--绿色--黄色-红色。
            applyColorMap(virtualDepth, virtualDepth_color, cv::COLORMAP_JET);
            std::string strFullPath_gray = strSavePath;
            std::string strFullPath_color = strSavePath;
            std::string strFullPath_real_color = strSavePath;
            switch (m_eStereoType) {
                case eST_ACMH: {
                    strFullPath_gray += "/VD_Base_gray.png";
                    strFullPath_color += "/VD_Base_color.png";
                    //strFullPath_real_color += "/VD_Base_real_color.png";
                    strFullPath_real_color = m_strSavePath + strName + "_VD_Base_real_color.png";
                    if (m_bPlannar != false || m_bLRCheck != true) {
                        std::cout << "base: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_PlannerPrior: {
                    strFullPath_gray += "/VD_Plannar_gray.png";
                    strFullPath_color += "/VD_Plannar_color.png";
                    //strFullPath_real_color += "/VD_Plannar_real_color.png";
                    strFullPath_real_color = m_strSavePath + strName + "_VD_Plannar_real_color.png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "planner: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_Horizontal_Propagation: {
                    strFullPath_gray += "/VD_HorProga_gray.png";
                    strFullPath_color += "/VD_HorProga_color.png";
                    //strFullPath_real_color += "/VD_HorProga_real_color.png";
                    strFullPath_real_color = m_strSavePath + strName + "VD_HorProga_real_color.png";
                    if (m_bPlannar != true || m_bLRCheck != true) {
                        std::cout << "Horizontal_Propagation: error switch!" << std::endl;
                    }
                }
                    break;
                case eST_BlurFeature: {
                    strFullPath_gray += "/VD_BlurFeature_gray.png";
                    strFullPath_color += "/VD_BlurFeature_color.png";
                    strFullPath_real_color += "/VD_BlurFeature_real_color.png";
                    strFullPath_real_color = m_strSavePath + strName + "_VD_BlurFeature_real_color.png";
                }
                    break;
                default:
                    break;
            }
            cv::imwrite(strFullPath_gray, virtualDepth);
            cv::imwrite(strFullPath_color, virtualDepth_color);
            cv::imwrite(strFullPath_real_color, virtualDepth_realColor);
            StoreColorPlyFileBinaryPointCloud(strSavePath + std::string("/VD.ply"), PointCloud);
        }
    }
