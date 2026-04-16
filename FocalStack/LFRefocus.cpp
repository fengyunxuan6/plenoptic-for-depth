/********************************************************************
file base:      LFRefocus.h
author:         LZD
created:        2025/03/12
purpose:        数字重聚焦、焦点堆栈、焦平面等生成
*********************************************************************/
#include <random>
#include "LFRefocus.h"

#include "boost/filesystem.hpp"
#include "functional"
#include "unordered_map"

#include "Util/Distribution_analyzer.h"
#include "Util/Logger.h"
#include "CommonMVS/OctreeFusionVoxel.h"

// 需在标准命名空间下定义哈希，所以暂时放在这里
namespace std
{
    // 定义哈希函数，统计重复出现元素的频次
    template<> struct hash<LFMVS::pixel_point>
    {
        size_t operator() (const LFMVS::pixel_point& c_pixel_point)  const
        {
            size_t h=0;
            h ^=hash<int>()(c_pixel_point.x) + 0x9e3779b9 + (h<<6) +(h>>2);
            h ^=hash<int>()(c_pixel_point.y) + 0x9e3779b9 + (h<<6) +(h>>2);
            h ^=hash<int>()(c_pixel_point.num) + 0x9e3779b9 + (h<<6) +(h>>2);
            return h;
        }
    };
}

namespace LFMVS
{
    LFRefocus::LFRefocus(DepthSolver* pDepthSolver)
        : m_ptrDepthSolver(pDepthSolver)
    {

    }

    LFRefocus::~LFRefocus()
    {

    }

    void LFRefocus::AIFImageCompositeForMIA(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("LFRefocus: AIFImageCompositeForMIA, Begin");
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problem_map = itrFrame->second;
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        QuadTreeTileInfoMap& m_MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        const int image_height = m_ptrDepthSolver->GetRawImageHeight();
        const int image_width = m_ptrDepthSolver->GetRawImageWidth();

        m_vIDepth = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_VirtualDepthMap = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_vIDepth_Reciprocal = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_vIDepthGray = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_rIDepth_mm = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_rIDepth_m = cv::Mat::zeros(image_height, image_width, CV_32F);

        //存放计算后的虚像
        m_AIF_Color = cv::Mat::zeros(image_height, image_width, CV_8UC3);
        cv::Mat m_virtualImageHF = cv::Mat::zeros(image_height, image_width, CV_8UC3);  // 存储孔洞填充虚拟图像
        cv::Mat m_vIDenose = cv::Mat::zeros(image_height, image_width, CV_8UC3);    // 存储去噪后虚拟图像

        // Step1: 遍历微图像，按序获取视差值，并计算虚拟深度值
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++)
        {
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage)
                continue;
            QuadTreeTileKeyPtr ptrKey = itr->first;

            // TODO:
            QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrKey);
            if (itrInfo == m_MLA_info_map.end())
            {
                std::cout << "Current Image m_MLA_info_map not found: " << std::endl;
                continue;
            }
            MLA_InfoPtr MLA_InfoPtr = itrInfo->second;
            cv::Point2f srcCenterLocation = MLA_InfoPtr->GetCenter();

            QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
            QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
            if (itrDis == disNormalMapMap.end())
            {
                std::cout << "Current Image disNormalMapMap not found: " << std::endl;
                continue;
            }
            QuadTreeDisNormalMap& disNormals_map = itrDis->second;
            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                std::cout << "Current Image not found: "  << ptrKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            std::pair<int, int> vIPointCoord;
            std::pair<int, int> pointLocation;
            int mlaImageRows = problem.m_Image_gray.rows;
            int mlaImageCols = problem.m_Image_gray.cols;

            for(int y = 0; y < mlaImageRows; y++)
            {
                for (int x = 0; x < mlaImageCols; x++)
                {
                    float virtualDepth = 0;
                    float disp = 0;
                    int index = y * mlaImageCols + x;
                    disp = ptrDN->d_cuda[index];
                    //  float mla_Base=params.baseline*LFMVS::g_bl0 / (LFMVS::g_bl0 + LFMVS::g_B); // 使用微透镜基线
                    float mla_Base=params.baseline; // 使用微透镜基线
                    if(disp > 0)
                    {
                        virtualDepth = mla_Base / disp;
                        float vd_gpu = ptrDN->disp_v_cuda[index].w;
                        // std::cout<<"(virtualDepth,vd_gpu)="<<"("<<virtualDepth<<","<<vd_gpu<<")"<<std::endl;
                        float centerX = (mlaImageCols - 1) * 0.5f;
                        float centerY = (mlaImageRows - 1) * 0.5f;
                        float offsetX = x - centerX;
                        float offsetY = y - centerY;
                        m_virtualImageX = (int)round(offsetX * virtualDepth + srcCenterLocation.x);   // 将坐标映射到虚拟深度图上
                        m_virtualImageY = (int)round(offsetY * virtualDepth + srcCenterLocation.y);
                        if((m_virtualImageX <= m_AIF_Color.cols && m_virtualImageX >= 0)
                           && (m_virtualImageY <= m_AIF_Color.rows && m_virtualImageY >= 0))
                        {
                            m_vIPoint.virtualImageX = m_virtualImageX;
                            m_vIPoint.virtualImageY = m_virtualImageY;
                            m_vIPoint.srcImageKey = ptrKey;
                            m_vIPoint.srcImageX = x;
                            m_vIPoint.srcImageY = y;
                            m_vIPoint.virtualDepth = virtualDepth;
                            vIPointCoord = {m_virtualImageY, m_virtualImageX};
                            // 虚拟深度图上,可能会计算出多个同名点对应同一个虚像点
                            m_coVIPointMap[vIPointCoord].push_back(m_vIPoint);
                        };
                    }
                }
            }
        }

        // Step2: 去除重复值：使用评分函数，从微图像同名点中确定唯一的虚像点
        for(int y = 0; y < m_AIF_Color.rows; y++)
        {
            for (int x = 0; x < m_AIF_Color.cols; x++)
            {
                CoVIPointMap::iterator cVIPMItr = m_coVIPointMap.find({y, x});
                if(cVIPMItr == m_coVIPointMap.end())
                    continue;
                m_vIPointVec = cVIPMItr->second;
                // 如果虚拟图像中有坐标同名点，选出确定的像素点
                if(m_vIPointVec.size() > 1)
                {
                    // 独立成函数，使用模糊度、纹理值、基线(待考虑) 赋权排序,选出唯一的虚拟图像像素点
                    FilterMatchingPoints(m_vIPointVec, problem_map);
                    m_vIPointMap[{y, x}] = m_vIPointVec[0];
                }
                else if(m_vIPointVec.size() == 1)
                {
                    m_vIPointMap[{y, x}] = m_vIPointVec[0];
                }
                else
                {
                    continue;
                }
            }
        }

        // Step3: 生成初始的虚拟深度图和全聚焦图像
        GenerateVirtualImage(m_vIPointMap, problem_map,params);

        // 测试用-生成彩色虚拟深度图
        double minValue,maxValue;
        cv::minMaxLoc(m_vIDepth, &minValue, &maxValue);
        std::vector<float> vd_data_mla;
        for (int row=0; row < image_height; row++)
        {
            for (int col=0; col < image_width; col++)
            {
                float vd = m_vIDepth.at<float>(row, col);
                vd_data_mla.push_back(vd);
            }
        }
        std::vector<float> vd_inlier_global;
        float lo_global = 0.0f, hi_global = 0.0f;
        m_ptrDepthSolver->BuildGlobalDispInlierMAD(vd_data_mla, vd_inlier_global,
            lo_global, hi_global, 3.0f, 0.02f);
        LOG_ERROR("LF-Refocus: min_vd= ", lo_global, "max_vd= ", hi_global);
        for (int row=0; row < image_height; row++)
        {
            for (int col=0; col < image_width; col++)
            {
                if (m_vIDepth.at<float>(row, col) < lo_global)
                {
                    m_vIDepth.at<float>(row, col) = lo_global;
                }
                else if (m_vIDepth.at<float>(row, col) > hi_global)
                {
                    m_vIDepth.at<float>(row, col) = hi_global;
                }
            }
        }
        // 使用对比度增强的归一化方法
        cv::normalize(m_vIDepth, m_vIDepth, 0, 255, cv::NORM_MINMAX);
        
        // 应用伽马校正增强对比度
        cv::Mat enhanced_depth;
        m_vIDepth.convertTo(enhanced_depth, CV_32F);
        
        // 伽马校正参数
        double gamma = 0.7; // 降低gamma值以增强对比度
        cv::pow(enhanced_depth/255.0, gamma, enhanced_depth);
        
        // 转换回8位并应用颜色映射
        enhanced_depth.convertTo(m_vIDepthGray, CV_8UC1, 255.0);
        cv::applyColorMap(m_vIDepthGray, m_vIDepth, cv::COLORMAP_JET);

        // Step4: 孔洞填充：利用原始微图像中的邻域像素进行插值，然后填充到全聚焦图像
        m_virtualImageHF = m_AIF_Color.clone();
        HoleFilling(m_virtualImageHF, m_vIPointMap, problem_map);

        // 该去噪方法不适用
        m_vIDenose = m_AIF_Color.clone();
        for(int i = 0;i < 5;i++)
        {
            cv::medianBlur(m_vIDenose, m_vIDenose, 5);
        }

        // Step5: 准备路径，并写出
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strFrameName + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
                }
            }
        }

        std::string virtualImgFullPath = m_strSavePath + std::string("/m_AIF_Color.png");
        cv::imwrite(virtualImgFullPath, m_AIF_Color);

        std::string virtualImgHfFullPath = m_strSavePath + std::string("/m_virtualImage_HF.png");
        cv::imwrite(virtualImgHfFullPath, m_virtualImageHF);

        std::string virtualDepthImgFullPath = m_strSavePath + std::string("/m_vIDepth.png");

        cv::imwrite(virtualDepthImgFullPath, m_vIDepth);

        std::string virtualDepthGrayImgFullPath = m_strSavePath + std::string("/m_vIDepthGray.png");
        cv::imwrite(virtualDepthGrayImgFullPath, m_vIDepthGray);

        // 为m_vIDepth生成二值化黑白图
        cv::Mat m_vIDepth_Binary;
        cv::threshold(m_vIDepthGray, m_vIDepth_Binary, 128, 255, cv::THRESH_BINARY);
        // 使用形态学操作平滑边界
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(m_vIDepth_Binary, m_vIDepth_Binary, cv::MORPH_CLOSE, kernel);
        std::string virtualDepthBinaryImgFullPath = m_strSavePath + std::string("/m_vIDepth_binary.png");
        cv::imwrite(virtualDepthBinaryImgFullPath, m_vIDepth_Binary);

        std::string virtualImageDenoseFullPath = m_strSavePath + std::string("/m_vIDenose.png");
        cv::imwrite(virtualImageDenoseFullPath, m_vIDenose);

        std::string m_vIDepth_RawFullPath = m_strSavePath + std::string("/m_VirtualDepthMap.tiff");
        cv::imwrite(m_vIDepth_RawFullPath, m_VirtualDepthMap);

        std::string m_vIDepth_ReciprocalFullPath = m_strSavePath + std::string("/m_vIDepth_Reciprocal.png");
        cv::imwrite(m_vIDepth_ReciprocalFullPath, m_vIDepth_Reciprocal);

        std::string realImgFullPath = m_strSavePath + std::string("/m_rIDepth_mm.png");
        cv::imwrite(realImgFullPath, m_rIDepth_mm);

        std::string realImgFullPath_tif = m_strSavePath + std::string("/m_rIDepth_mm.tiff");
        cv::imwrite(realImgFullPath_tif, m_rIDepth_mm);

        std::string realImgFullPath_m = m_strSavePath + std::string("/m_rIDepth_m.png");
        cv::imwrite(realImgFullPath_m, m_rIDepth_m);

        std::cout<<"params.mainlen_flocal_length="<<params.mainlen_flocal_length<<std::endl;
        LOG_ERROR("LFRefocus: AIFImageCompositeForMIA, End");
    }

    void LFRefocus::AIFImageCompositeForMIA_Weights(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("LFRefocus: AIFImageCompositeForMIA_Weights, Begin");
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problem_map = itrFrame->second;
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        QuadTreeTileInfoMap& m_MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        const int image_height = m_ptrDepthSolver->GetRawImageHeight();
        const int image_width = m_ptrDepthSolver->GetRawImageWidth();

        m_vIDepth = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_VirtualDepthMap = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_vIDepth_Reciprocal = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_vIDepthGray = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_rIDepth_mm = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_rIDepth_m = cv::Mat::zeros(image_height, image_width, CV_32F);

        //存放计算后的虚像
        m_AIF_Color = cv::Mat::zeros(image_height, image_width, CV_8UC3);
        cv::Mat m_virtualImageHF = cv::Mat::zeros(image_height, image_width, CV_8UC3);  // 存储孔洞填充虚拟图像
        cv::Mat m_vIDenose = cv::Mat::zeros(image_height, image_width, CV_8UC3);    // 存储去噪后虚拟图像

        // Step1: 遍历微图像，按序获取视差值，并计算虚拟深度值
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++)
        {
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage)
                continue;
            QuadTreeTileKeyPtr ptrKey = itr->first;

            // TODO:
            QuadTreeTileInfoMap::iterator itrInfo = m_MLA_info_map.find(ptrKey);
            if (itrInfo == m_MLA_info_map.end())
            {
                std::cout << "Current Image m_MLA_info_map not found: " << std::endl;
                continue;
            }
            MLA_InfoPtr MLA_InfoPtr = itrInfo->second;
            cv::Point2f srcCenterLocation = MLA_InfoPtr->GetCenter();

            QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
            QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
            if (itrDis == disNormalMapMap.end())
            {
                std::cout << "Current Image disNormalMapMap not found: " << std::endl;
                continue;
            }
            QuadTreeDisNormalMap& disNormals_map = itrDis->second;
            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                std::cout << "Current Image not found: "  << ptrKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            std::pair<int, int> vIPointCoord;
            std::pair<int, int> pointLocation;
            int mlaImageRows = problem.m_Image_gray.rows;
            int mlaImageCols = problem.m_Image_gray.cols;

            for(int y = 0; y < mlaImageRows; y++)
            {
                for (int x = 0; x < mlaImageCols; x++)
                {
                    float virtualDepth = 0;
                    float disp = 0;
                    int index = y * mlaImageCols + x;
                    disp = ptrDN->d_cuda[index];
                    //  float mla_Base=params.baseline*LFMVS::g_bl0 / (LFMVS::g_bl0 + LFMVS::g_B); // 使用微透镜基线
                    float mla_Base=params.baseline; // 使用微透镜基线
                    if(disp > 0)
                    {
                        virtualDepth = mla_Base / disp;
                        float vd_gpu = ptrDN->disp_v_cuda[index].w;
                        std::cout<<"(virtualDepth,vd_gpu)="<<"("<<virtualDepth<<","<<vd_gpu<<")"<<std::endl;
                        float centerX = (mlaImageCols - 1) * 0.5f;
                        float centerY = (mlaImageRows - 1) * 0.5f;
                        float offsetX = x - centerX;
                        float offsetY = y - centerY;
                        m_virtualImageX = (int)round(offsetX * virtualDepth + srcCenterLocation.x);   // 将坐标映射到虚拟深度图上
                        m_virtualImageY = (int)round(offsetY * virtualDepth + srcCenterLocation.y);
                        if((m_virtualImageX <= m_AIF_Color.cols && m_virtualImageX >= 0)
                           && (m_virtualImageY <= m_AIF_Color.rows && m_virtualImageY >= 0))
                        {
                            m_vIPoint.virtualImageX = m_virtualImageX;
                            m_vIPoint.virtualImageY = m_virtualImageY;
                            m_vIPoint.srcImageKey = ptrKey;
                            m_vIPoint.srcImageX = x;
                            m_vIPoint.srcImageY = y;
                            m_vIPoint.virtualDepth = virtualDepth;
                            vIPointCoord = {m_virtualImageY, m_virtualImageX};
                            // 虚拟深度图上,可能会计算出多个同名点对应同一个虚像点
                            m_coVIPointMap[vIPointCoord].push_back(m_vIPoint);
                        };
                    }
                }
            }
        }

        // Step2: 去除重复值：使用评分函数，从微图像同名点中确定唯一的虚像点
        // TODO 待修改
        ImageDenoising(problem_map);

        // Step3: 生成初始的虚拟深度图和全聚焦图像
        GenerateVIByColorConsist(m_vIPointMap, problem_map,params);

        // 测试用-生成彩色虚拟深度图
        double minValue,maxValue;
        cv::minMaxLoc(m_vIDepth, &minValue, &maxValue);
        
        // 使用对比度增强的归一化方法
        cv::normalize(m_vIDepth, m_vIDepth, 0, 255, cv::NORM_MINMAX);
        
        // 应用伽马校正增强对比度
        cv::Mat enhanced_depth;
        m_vIDepth.convertTo(enhanced_depth, CV_32F);
        
        // 伽马校正参数
        double gamma = 0.7; // 降低gamma值以增强对比度
        cv::pow(enhanced_depth/255.0, gamma, enhanced_depth);
        
        // 转换回8位并应用颜色映射
        enhanced_depth.convertTo(m_vIDepthGray, CV_8UC1, 255.0);
        cv::applyColorMap(m_vIDepthGray, m_vIDepth, cv::COLORMAP_JET);

        // Step4: 孔洞填充：利用原始微图像中的邻域像素进行插值，然后填充到全聚焦图像
        m_virtualImageHF = m_AIF_Color.clone();
        // HoleFilling(m_virtualImageHF, m_vIPointMap, problem_map);

        // 该去噪方法不适用
        m_vIDenose = m_AIF_Color.clone();
        for(int i = 0;i < 5;i++)
        {
            cv::medianBlur(m_vIDenose, m_vIDenose, 5);
        }

        // Step5: 准备路径，并写出
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strFrameName + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
                }
            }
        }

        std::string virtualImgFullPath = m_strSavePath + std::string("/m_AIF_Color.png");
        cv::imwrite(virtualImgFullPath, m_AIF_Color);

        std::string virtualImgHfFullPath = m_strSavePath + std::string("/m_virtualImage_HF.png");
        cv::imwrite(virtualImgHfFullPath, m_virtualImageHF);

        std::string virtualDepthImgFullPath = m_strSavePath + std::string("/m_vIDepth.png");
        cv::imwrite(virtualDepthImgFullPath, m_vIDepth);

        std::string virtualDepthGrayImgFullPath = m_strSavePath + std::string("/m_vIDepthGray.png");
        cv::imwrite(virtualDepthGrayImgFullPath, m_vIDepthGray);

        // 为m_vIDepth生成二值化黑白图
        cv::Mat m_vIDepth_Binary;
        cv::threshold(m_vIDepthGray, m_vIDepth_Binary, 128, 255, cv::THRESH_BINARY);
        // 使用形态学操作平滑边界
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(m_vIDepth_Binary, m_vIDepth_Binary, cv::MORPH_CLOSE, kernel);
        std::string virtualDepthBinaryImgFullPath = m_strSavePath + std::string("/m_vIDepth_binary.png");
        cv::imwrite(virtualDepthBinaryImgFullPath, m_vIDepth_Binary);

        std::string virtualImageDenoseFullPath = m_strSavePath + std::string("/m_vIDenose.png");
        cv::imwrite(virtualImageDenoseFullPath, m_vIDenose);

        std::string m_vIDepth_RawFullPath = m_strSavePath + std::string("/m_VirtualDepthMap.png");
        cv::imwrite(m_vIDepth_RawFullPath, m_VirtualDepthMap);

        std::string m_vIDepth_ReciprocalFullPath = m_strSavePath + std::string("/m_vIDepth_Reciprocal.png");
        cv::imwrite(m_vIDepth_ReciprocalFullPath, m_vIDepth_Reciprocal);

        std::string realImgFullPath = m_strSavePath + std::string("/m_rIDepth_mm.png");
        cv::imwrite(realImgFullPath, m_rIDepth_mm);

        std::string realImgFullPath_m = m_strSavePath + std::string("/m_rIDepth_m.png");
        cv::imwrite(realImgFullPath_m, m_rIDepth_m);

        std::cout<<"params.mainlen_flocal_length="<<params.mainlen_flocal_length<<std::endl;

        LOG_ERROR("LFRefocus: AIFImageCompositeForMIA_Weights, End");
    }

    void LFRefocus::FuseVirtualDepth_BackProject_OctreeVoxel_Copy(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("LFRefocus: FuseVirtualDepth_BackProject, Begin");
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problem_map = itrFrame->second;
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        QuadTreeTileInfoMap& mla_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        const int image_height = m_ptrDepthSolver->GetRawImageHeight();
        const int image_width = m_ptrDepthSolver->GetRawImageWidth();

        // 日晷
        int x = 1819;
        int y = 1153;
        int x0 = 2819;
        int y0 = 2153;

        // 墙面
//        int x = 3801;
//        int y = 140;
//        int x0 = 4001;
//        int y0 = 1140;

        // 标定板2
//        int x = 338;
//        int y = 1151;
//        int x0 = 911;
//        int y0 = 1536;

        // 标定板1
//        int x = 1969;
//        int y = 3605;
//        int x0 = 2547;
//        int y0 = 3985;

        m_VirtualDepthMap = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_AIF_Color = cv::Mat::zeros(image_height, image_width, CV_8UC3);

        std::vector<PointList> point_lists_Vec;
        float vdMin =  std::numeric_limits<float>::infinity();
        float vdMax = -std::numeric_limits<float>::infinity();

        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("Current disNormalMapMap not found: " , strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        std::string  xmlPath = m_ptrDepthSolver->GetRootPath() + "/behaviorModelParams.xml";
        std::array<double, 3> behaviorModelParams;
        LoadCoeffsFromXml(xmlPath, behaviorModelParams);

        cv::Mat realDepthMap_BM;
        realDepthMap_BM = cv::Mat::zeros(image_height, image_width, CV_32F);
        // Step0:分层随机采样100个点，计算V的范围
        const int step = 4;           // 每隔4个微图像取1个
        int mi_idx = 0;
        std::vector<float> realDepthSamples;
        std::vector<float> virtualDepthSamples;
        float vd_min = std::numeric_limits<float>::infinity();
        float vd_min_rd = std::numeric_limits<float>::infinity();
        float vd_max = -std::numeric_limits<float>::infinity();
        float vd_max_rd = -std::numeric_limits<float>::infinity();

        //  Step0: 初始化八叉数
        SEACAVE::TOctreeFusionVoxel OctreeVoxel;
        Eigen::Vector3f center(image_width/2,image_width/2,image_width/2);
//        Eigen::Vector3f size(image_width,image_width,image_width); // 盒子边长
        float radius = image_width/2;
        SEACAVE::TAABB<float,3> aabb(center,radius);
        int expectedVoxelNum = 216000000000;
        float desiredVoxelSize = 0.5;
        OctreeVoxel.IniOctree(aabb, expectedVoxelNum, desiredVoxelSize);

        // Step1: 为每个有效估计视差的微图像建立一个mask图
        QuadTreeImageMap masks;
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
        {
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage /*|| problem.m_bNeedMatch==false*/)
            {
                continue;
            }
            QuadTreeTileKeyPtr ptrKey = itr->first;

            QuadTreeTileInfoMap::iterator itrInfo = mla_info_map.find(ptrKey);
            if (itrInfo == mla_info_map.end())
            {
                LOG_ERROR("Current MLA_info not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }
            QuadTreeDisNormalMap& disNormals_map = itrDis->second;
            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                LOG_ERROR("Current disNormal not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }
            int mi_rows = problem.m_Image_gray.rows;
            int mi_cols = problem.m_Image_gray.cols;
            cv::Mat mask = cv::Mat::zeros(mi_rows, mi_cols, CV_8UC1);
            masks[ptrKey] = mask;
        }

        int counter = 0;
        // step2: 开始融合
        std::vector<PointList> PointCloud;
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
        {
            counter = counter + 4;
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage /*|| problem.m_bNeedMatch==false*/)
                continue;
            QuadTreeTileKeyPtr ptrKey = itr->first;
            //LOG_ERROR("Fuse virtual_depth, key: ", ptrKey->StrRemoveLOD().c_str());

            QuadTreeTileInfoMap::iterator itrInfo = mla_info_map.find(ptrKey);
            if (itrInfo == mla_info_map.end())
                continue;
            MLA_InfoPtr mla_InfoPtr = itrInfo->second;

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
                continue;
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            cv::Point2f center_at_mla = mla_InfoPtr->GetCenter();

            // 开始处理
            int num_ngb = problem.m_NeighsSortVecForMatch.size();
            std::vector<int2> used_list(num_ngb, make_int2(-1, -1));

            int mi_rows = problem.m_Image_gray.rows;
            int mi_cols = problem.m_Image_gray.cols;
            cv::Vec2f center_mi((mi_cols-1)*0.5f+1.0, (mi_rows-1)*0.5f+1.0);
            for(int r = 0; r < mi_rows; r++)
            {
                for (int c = 0; c < mi_cols; c++)
                {
                    cv::Mat& mask = masks[ptrKey];
                    if (mask.at<uchar>(r, c) == 1)
                        continue;
                    int index = r * mi_cols + c;
                    float disp_ref = ptrDN->d_cuda[index];

                    // TODO: 临时
                    if (disp_ref < 1.0)
                        continue;

                    float4 normal_ref_tmp = ptrDN->ph_cuda[index];
                    cv::Vec3f normal_ref = cv::Vec3f(normal_ref_tmp.x, normal_ref_tmp.y, normal_ref_tmp.z);
                    cv::Vec3f consistent_normal = normal_ref;

                    // 计算虚像点
                    float virtualDepth_ref = ComputeVirtualDepth(disp_ref, params);
                    if (fabs(virtualDepth_ref-g_Invalid_image) < 1.0e-6)
                    {
                        continue;
                    }
                    cv::Vec3f virtualPoint3D_ref = ComputeVirtual3D(virtualDepth_ref, c, r, mi_cols, mi_rows, center_at_mla);
                    cv::Vec3f consistent_Point = virtualPoint3D_ref;
                    uchar consistent_Color[3] = {problem.m_Image_rgb.at<uchar3>(r, c).x,
                                                 problem.m_Image_rgb.at<uchar3>(r, c).y,
                                                 problem.m_Image_rgb.at<uchar3>(r, c).z};

                    // 遍历邻居
                    int num_consistent = 0;
                    for (int j = 0; j < num_ngb; j++)
                    {
                        QuadTreeTileKeyPtr ptrKey_src = problem.m_NeighsSortVecForMatch[j];


                        QuadTreeProblemMap::iterator itrQP = problem_map.find(ptrKey_src);
                        if (itrQP == problem_map.end())
                            continue;
                        MLA_Problem& problem_src = itrQP->second;

                        QuadTreeDisNormalMap::iterator itr_DN_src = disNormals_map.find(ptrKey_src);
                        if (itr_DN_src == disNormals_map.end())
                            continue;
                        DisparityAndNormalPtr ptrDN_src = itr_DN_src->second;
                        QuadTreeTileInfoMap::iterator itrInfo_src = mla_info_map.find(ptrKey_src);
                        if (itrInfo_src == mla_info_map.end())
                            continue;
                        MLA_InfoPtr ptrInfo_src = itrInfo_src->second;

                        cv::Point2f center_src_at_mla = ptrInfo_src->GetCenter();
                        cv::Vec2f center_mi_src((mi_cols-1)*0.5f+1, (mi_rows-1)*0.5f+1);

                        // 反投影
                        cv::Vec2f coord_src_mi = BackProject2MI(virtualPoint3D_ref, center_src_at_mla, center_mi_src);
                        int src_r = int(coord_src_mi[1] + 0.5f);
                        int src_c = int(coord_src_mi[0] + 0.5f);
                        if (src_c >= 0 && src_c < mi_cols && src_r >= 0 && src_r < mi_rows)
                        {
                            if (masks[ptrKey_src].at<uchar>(src_r, src_c) == 1)
                                continue;

                            int index_src = src_r * mi_cols + src_c;
                            float4 normal_src_tmp = ptrDN_src->ph_cuda[index_src];
                            cv::Vec3f normal_src = cv::Vec3f(normal_src_tmp.x, normal_src_tmp.y, normal_src_tmp.z);

                            float disp_src = ptrDN_src->d_cuda[index_src];
                            float virtualDepth_src = ComputeVirtualDepth(disp_src, params);
                            if (fabs(virtualDepth_src -g_Invalid_image)<1e-6)
                            {
                                continue;
                            }
                            cv::Vec3f virtual_Point_src = ComputeVirtual3D(virtualDepth_src, src_c,
                                                                           src_r, mi_cols, mi_rows, center_src_at_mla);

                            // 反投影到参考微图像
                            cv::Vec2f coord_mi_reproject = BackProject2MI(virtual_Point_src, center_at_mla, center_mi);
                            float reproj_error = sqrt(pow(c - coord_mi_reproject[0], 2) + pow(r - coord_mi_reproject[1], 2));
                            float relative_virtualDepth_diff = fabs(virtualDepth_src - virtualDepth_ref) / virtualDepth_ref;
                            if (reproj_error < 1.0f && relative_virtualDepth_diff < 0.01f)
                            {
                                consistent_Point[0] += virtual_Point_src[0];
                                consistent_Point[1] += virtual_Point_src[1];
                                consistent_Point[2] += virtual_Point_src[2];
                                consistent_normal += normal_src;

                                consistent_Color[0] += problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c)[0];
                                consistent_Color[1] += problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c)[1];
                                consistent_Color[2] += problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c)[2];

                                used_list[j].x = src_c;
                                used_list[j].y = src_r;
                                num_consistent++;
                            }
                        }
                    }

                    if (num_consistent >= 2)
                    {
                        consistent_Point[0] /= (num_consistent + 1.0f);
                        consistent_Point[1] /= (num_consistent + 1.0f);
                        consistent_Point[2] /= (num_consistent + 1.0f);
                        consistent_normal /= (num_consistent + 1.0f);
                        consistent_Color[0] /= (num_consistent + 1.0f);
                        consistent_Color[1] /= (num_consistent + 1.0f);
                        consistent_Color[2] /= (num_consistent + 1.0f);

                        PointList point3D;
                        point3D.coord = make_float3(consistent_Point[0], consistent_Point[1], consistent_Point[2]);
                        point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                        point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);

                        if (point3D.coord.y >= 0 && point3D.coord.y < image_height
                            && point3D.coord.x >= 0 && point3D.coord.x < image_width)
                        {
                            m_VirtualDepthMap.at<float>(static_cast<int>(point3D.coord.y),
                                                        static_cast<int>(point3D.coord.x)) = point3D.coord.z;
                            // 分段使用分段行为模型
                            float realDepth_BM = ConvertVdToRdSegment(point3D.coord.z);
//                            float realDepth_BM = ConvertVdToRd(point3D.coord.z, behaviorModelParams);
//                            if (counter < 1000)
//                            {
//                                std::cout<<"virtualDepth_s="<<point3D.coord.z<<"---"<<"realDepth="<<realDepth_BM<<std::endl;
//                            }

                            if (point3D.coord.x >= x && point3D.coord.x < x0 &&
                                    point3D.coord.y >= y && point3D.coord.y < y0)
                            {
//                                LOG_WARN("behaviorModelParams",behaviorModelParams[0],behaviorModelParams[1],behaviorModelParams[2],"virtualDepth_ref：",virtualDepth_ref,"---","realDepth_BM：",realDepth_BM);
                                LOG_WARN("location:(",point3D.coord.x,"---",point3D.coord.y,")---virtualDepth_ref：",virtualDepth_ref,"---","realDepth_BM：",realDepth_BM);
                            }

                            realDepthMap_BM.at<float>(static_cast<int>(point3D.coord.y),
                                                      static_cast<int>(point3D.coord.x)) = realDepth_BM;
                            PointCloud.push_back(point3D);

                            // 插入体素
                            SEACAVE::TOctreeFusionVoxel::F_POINT_TYPE pointCoord;
//                            pointCoord << point3D.coord.x, point3D.coord.y, point3D.coord.z*500;


                            pointCoord << point3D.coord.x,(image_height-1.0) -point3D.coord.y, point3D.coord.z*300;
                            OctreeVoxel.InsertForFusion_Bucket(pointCoord);

                            m_AIF_Color.at<cv::Vec3b>(static_cast<int>(point3D.coord.y),
                                                      static_cast<int>(point3D.coord.x)) =
                                    cv::Vec3b(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                        }

                        for (int j = 0; j < num_ngb; ++j)
                        {
                            if (used_list[j].x == -1)
                                continue;
                            QuadTreeTileKeyPtr ptrKey_src = problem.m_NeighsSortVecForMatch[j];
                            masks[ptrKey_src].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                        }
                    }
                }
            }
        }

        // 保存结果
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strFrameName + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
                }
            }
        }
        std::string vIDepth_RawFullPath = m_strSavePath + "/" + LF_VIRTUALDEPTHMAP_NAME + std::string(".tiff");
        cv::imwrite(vIDepth_RawFullPath, m_VirtualDepthMap);
        std::string vIDepth_VirFullPath = m_strSavePath + LF_ALLINFOUCSIMAGE_NAME + std::string(".png");
        cv::imwrite(vIDepth_VirFullPath, m_AIF_Color);
        std::string ply_path = m_strSavePath + "/vitrual_3dmodel.ply";
        StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);


        cv::Mat tmp = realDepthMap_BM.clone();
        cv::patchNaNs(tmp, 0.0f);                      // 把 NaN 置 0，避免污染 min/max

        cv::Mat mask = (tmp > 0);                      // 只把 >0 当有效（按你约定调整）
        double minV, maxV;
        cv::minMaxLoc(tmp, &minV, &maxV, nullptr, nullptr, mask);

        cv::Mat depth8u;
        tmp.convertTo(depth8u, CV_8U, 255.0/(maxV-minV), -minV*255.0/(maxV-minV));
        depth8u.setTo(0, ~mask);                       // 无效像素置 0（黑）
        std::cout << "valid=" << cv::countNonZero(mask)
                  << " minV=" << minV << " maxV=" << maxV
                  << " range=" << (maxV-minV) << std::endl;
        cv::Mat realDepthMap_BM_color;
        cv::Mat realDepthMap_BM_color_HOT;
//        cv::normalize(realDepthMap_BM, depth8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth8u, realDepthMap_BM_color, cv::COLORMAP_JET);
        cv::applyColorMap(depth8u, realDepthMap_BM_color_HOT, cv::COLORMAP_HOT);
        cv::imwrite(m_strSavePath + "/real_depth_map_BM.png", realDepthMap_BM);
        cv::imwrite(m_strSavePath + "/real_depth_map_BM_color.png", realDepthMap_BM_color);
        cv::imwrite(m_strSavePath + "/realDepthMap_BM_color_HOT.png", realDepthMap_BM_color_HOT);

        OctreeVoxel.CollectLeafs(false);
        std::string octreePath = m_strSavePath + "init_octreeSurface3.obj";
//        OctreeVoxel.ExportInitOctreeObj_RootPlus8Children(aabb, octreePath, true, 0.98);
//        OctreeVoxel.ExportOccupiedLeafVoxelsObj(octreePath, false, 1);
        OctreeVoxel.ExportOccupiedLeafSurfaceObj(octreePath, OctreeVoxel.occupiedLeaves, false);

        // 全聚焦图像的填充
        //FillVirtualDepthMap();

        // 统计
        std::vector<float> vd_data_mla;
        for (int row=0; row < image_height; row++)
        {
            for (int col=0; col < image_width; col++)
            {
                float vd = m_VirtualDepthMap.at<float>(row, col);
                vd_data_mla.push_back(vd);
            }
        }
        std::vector<float> vd_inlier_global;
        float lo_global = 0.0f, hi_global = 0.0f;
        m_ptrDepthSolver->BuildGlobalDispInlierMAD(vd_data_mla, vd_inlier_global,
                                                   lo_global, hi_global, 3.0f, 0.02f);
        LOG_ERROR("LF-Refocus-backproject: min_vd= ", lo_global, "max_vd= ", hi_global);
        for (int row=0; row < image_height; row++)
        {
            for (int col=0; col < image_width; col++)
            {
                if (m_VirtualDepthMap.at<float>(row, col) < lo_global)
                {
                    m_VirtualDepthMap.at<float>(row, col) = lo_global;
                }
                else if (m_VirtualDepthMap.at<float>(row, col) > hi_global)
                {
                    m_VirtualDepthMap.at<float>(row, col) = hi_global;
                }
            }
        }
        cv::normalize(m_VirtualDepthMap, m_VirtualDepthMap, 0, 255, cv::NORM_MINMAX);
        m_VirtualDepthMap.convertTo(m_vIDepthGray, CV_8U);
        cv::applyColorMap(m_vIDepthGray, m_VirtualDepthMap, cv::COLORMAP_JET);
        std::string vIDepth_RawFullPath_g = m_strSavePath + std::string("/VD_Raw.tiff");
        cv::imwrite(vIDepth_RawFullPath_g, m_vIDepthGray);
        std::string vIDepth_RawFullPath_c = m_strSavePath + std::string("/VD_Raw_color.png");
        cv::imwrite(vIDepth_RawFullPath_c, m_VirtualDepthMap);

        // 配置并一键执行
        bool bAnalyse = false;
        if (bAnalyse)
        {
            std::vector<float> data;
            data.reserve(PointCloud.size());
            for (int i=0; i<PointCloud.size(); i++)
            {
                float3& point = PointCloud.at(i).coord;
                data.push_back(point.z);
            }

            bool exact=false;
            double hdi_p=0.85;
            int imgW=1400;
            int imgH=480;
            bool show=false;

            std::string prefix = m_strSavePath + std::string("/Distribution_analysis");
            boost::filesystem::path dir_save_path(prefix);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << prefix << std::endl;
                }
            }
            prefix += "/analysis";

            FloatDistributionAnalyzer::Options opt;
            opt.exact_quantiles = exact;
            opt.exact_hdi = exact;
            opt.hdi_p = hdi_p;
            opt.image_width = imgW;
            opt.image_height = imgH;
            opt.output_prefix = prefix;
            opt.show_windows = show;

            FloatDistributionAnalyzer analyzer;

            // 决定 HDI 求法（是否精确）
            AnalyzerConfig cfg_f;
            cfg_f.exact_hdi = true;   // 精确最窄区间（排序法）；大数据可设 false 更快
            cfg_f.hdi_p = 0.85;       // 仅影响打印；函数参数会再传一次 p
            AnalysisResult R_filtered;
            analyzer.save_histogram_hdi_filtered(data, hdi_p, imgW, imgH, cfg_f,
                                                 prefix+"hdi85_hist.png", &R_filtered);

            AnalysisResult R; // 如需在代码中使用结果，可传出
            if (!analyzer.run(data, opt, &R))
            {
                std::cerr << "分析失败\n";
            }

            cv::Mat img = analyzer.draw_histogram_hdi_filtered(data,  hdi_p,
                                                               imgW, imgH, cfg_f, &R_filtered);
            cv::imwrite(prefix+"hdi85_hist_new.png", img);
            // 控制台打印结果（可选）
            AnalyzerConfig cfg; cfg.exact_quantiles = opt.exact_quantiles; cfg.exact_hdi = opt.exact_hdi; cfg.hdi_p = opt.hdi_p;
            print_result(R, cfg);
        }

        std::vector<PointList> PointCloud_Object;
        Virtual2ObjectDepth(PointCloud_Object, PointCloud);
        std::string object_ply_path = m_strSavePath + "/object_3dmodel.ply";
        StoreColorPlyFileBinaryPointCloud (object_ply_path, PointCloud_Object);
        LOG_ERROR("LFRefocus: FuseVirtualDepth_BackProject, End");
    }

    void LFRefocus::FuseVirtualDepth_BackProject_OctreeVoxel(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("LFRefocus: FuseVirtualDepth_BackProject, Begin");
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problem_map = itrFrame->second;
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        QuadTreeTileInfoMap& mla_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        const int image_height = m_ptrDepthSolver->GetRawImageHeight();
        const int image_width = m_ptrDepthSolver->GetRawImageWidth();

        m_VirtualDepthMap = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_AIF_Color = cv::Mat::zeros(image_height, image_width, CV_8UC3);

        std::vector<PointList> point_lists_Vec;
        float vdMin =  std::numeric_limits<float>::infinity();
        float vdMax = -std::numeric_limits<float>::infinity();

        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("Current disNormalMapMap not found: " , strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        std::string  xmlPath = m_ptrDepthSolver->GetRootPath() + "/behaviorModelParams.xml";
        std::array<double, 3> behaviorModelParams;
        LoadCoeffsFromXml(xmlPath, behaviorModelParams);

        cv::Mat realDepthMap_BM;
        realDepthMap_BM = cv::Mat::zeros(image_height, image_width, CV_32F);
        // Step0:分层随机采样100个点，计算V的范围
        const int step = 4;           // 每隔4个微图像取1个
        int mi_idx = 0;
        std::vector<float> realDepthSamples;
        std::vector<float> virtualDepthSamples;
        float vd_min = std::numeric_limits<float>::infinity();
        float vd_min_rd = std::numeric_limits<float>::infinity();
        float vd_max = -std::numeric_limits<float>::infinity();
        float vd_max_rd = -std::numeric_limits<float>::infinity();

//        for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr, ++mi_idx)
//        {
//            if (mi_idx % step != 0)
//                continue;
//            MLA_Problem& problem = itr->second;
//            if (problem.m_bGarbage /*|| problem.m_bNeedMatch==false*/)
//            {
//                continue;
//            }
//            QuadTreeTileKeyPtr ptrKey = itr->first;
//
//            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
//            if (itr_DN == disNormals_map.end())
//                continue;
//            DisparityAndNormalPtr ptrDN = itr_DN->second;
//
//            int mi_rows = problem.m_Image_gray.rows;
//            int mi_cols = problem.m_Image_gray.cols;
//            int index = (mi_rows/2) * mi_cols + mi_cols/2;
//            float disp_ref = ptrDN->d_cuda[index];
//            // 计算虚像点
//            float virtualDepth_s = ComputeVirtualDepth(disp_ref, params);
//            if (fabs(virtualDepth_s-g_Invalid_image) < 1.0e-6)
//            {
//                continue;
//            }
//
//            float realDepth = ConvertVdToRd(virtualDepth_s,behaviorModelParams);
//
//
//            if(realDepth < 0)
//                continue;
//            std::cout<<"virtualDepth_s="<<virtualDepth_s<<"---"<<"realDepth="<<realDepth<<std::endl;
////            std::cout<<"realDepth="<<realDepth<<std::endl;
//            virtualDepthSamples.push_back(virtualDepth_s);
//            realDepthSamples.push_back(realDepth);
//            vd_min = std::min(vd_min, virtualDepth_s);
//            vd_max = std::max(vd_max, virtualDepth_s);
//            vd_min_rd = std::min(vd_min_rd, virtualDepth_s);
//            vd_max_rd = std::max(vd_max_rd, virtualDepth_s);
//        }

        //  Step0: 初始化八叉数
        SEACAVE::TOctreeFusionVoxel OctreeVoxel;
        Eigen::Vector3f center(image_width/2,image_width/2,image_width/2);
//        Eigen::Vector3f size(image_width,image_width,image_width); // 盒子边长
        float radius = image_width/2;
        SEACAVE::TAABB<float,3> aabb(center,radius);
        int expectedVoxelNum = 216000000000;
        float desiredVoxelSize = 0.5;
        OctreeVoxel.IniOctree(aabb, expectedVoxelNum, desiredVoxelSize);



        // Step1: 为每个有效估计视差的微图像建立一个mask图
        QuadTreeImageMap masks;
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
        {
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage /*|| problem.m_bNeedMatch==false*/)
            {
                continue;
            }
            QuadTreeTileKeyPtr ptrKey = itr->first;

            QuadTreeTileInfoMap::iterator itrInfo = mla_info_map.find(ptrKey);
            if (itrInfo == mla_info_map.end())
            {
                LOG_ERROR("Current MLA_info not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }
            QuadTreeDisNormalMap& disNormals_map = itrDis->second;
            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                LOG_ERROR("Current disNormal not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }
            int mi_rows = problem.m_Image_gray.rows;
            int mi_cols = problem.m_Image_gray.cols;
            cv::Mat mask = cv::Mat::zeros(mi_rows, mi_cols, CV_8UC1);
            masks[ptrKey] = mask;
        }

        int counter = 0;
        // step2: 开始融合
        std::vector<PointList> PointCloud;
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
        {
            counter = counter + 4;
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage /*|| problem.m_bNeedMatch==false*/)
                continue;
            QuadTreeTileKeyPtr ptrKey = itr->first;
            //LOG_ERROR("Fuse virtual_depth, key: ", ptrKey->StrRemoveLOD().c_str());

            QuadTreeTileInfoMap::iterator itrInfo = mla_info_map.find(ptrKey);
            if (itrInfo == mla_info_map.end())
                continue;
            MLA_InfoPtr mla_InfoPtr = itrInfo->second;

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
                continue;
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            cv::Point2f center_at_mla = mla_InfoPtr->GetCenter();

            // 开始处理
            int num_ngb = problem.m_NeighsSortVecForMatch.size();
            std::vector<int2> used_list(num_ngb, make_int2(-1, -1));

            int mi_rows = problem.m_Image_gray.rows;
            int mi_cols = problem.m_Image_gray.cols;
            cv::Vec2f center_mi((mi_cols-1)*0.5f+1.0, (mi_rows-1)*0.5f+1.0);
            for(int r = 0; r < mi_rows; r++)
            {
                for (int c = 0; c < mi_cols; c++)
                {
                    cv::Mat& mask = masks[ptrKey];
                    if (mask.at<uchar>(r, c) == 1)
                        continue;
                    int index = r * mi_cols + c;
                    float disp_ref = ptrDN->d_cuda[index];

                    // TODO: 临时
                    if (disp_ref < 1.0)
                        continue;

                    float4 normal_ref_tmp = ptrDN->ph_cuda[index];
                    cv::Vec3f normal_ref = cv::Vec3f(normal_ref_tmp.x, normal_ref_tmp.y, normal_ref_tmp.z);
                    cv::Vec3f consistent_normal = normal_ref;

                    // 计算虚像点
                    float virtualDepth_ref = ComputeVirtualDepth(disp_ref, params);
                    if (fabs(virtualDepth_ref-g_Invalid_image) < 1.0e-6)
                    {
                        continue;
                    }
                    cv::Vec3f virtualPoint3D_ref = ComputeVirtual3D(virtualDepth_ref, c, r, mi_cols, mi_rows, center_at_mla);
                    cv::Vec3f consistent_Point = virtualPoint3D_ref;
                    uchar consistent_Color[3] = {problem.m_Image_rgb.at<uchar3>(r, c).x,
                                                 problem.m_Image_rgb.at<uchar3>(r, c).y,
                                                 problem.m_Image_rgb.at<uchar3>(r, c).z};

                    // 遍历邻居
                    int num_consistent = 0;
                    for (int j = 0; j < num_ngb; j++)
                    {
                        QuadTreeTileKeyPtr ptrKey_src = problem.m_NeighsSortVecForMatch[j];


                        QuadTreeProblemMap::iterator itrQP = problem_map.find(ptrKey_src);
                        if (itrQP == problem_map.end())
                            continue;
                        MLA_Problem& problem_src = itrQP->second;

                        QuadTreeDisNormalMap::iterator itr_DN_src = disNormals_map.find(ptrKey_src);
                        if (itr_DN_src == disNormals_map.end())
                            continue;
                        DisparityAndNormalPtr ptrDN_src = itr_DN_src->second;
                        QuadTreeTileInfoMap::iterator itrInfo_src = mla_info_map.find(ptrKey_src);
                        if (itrInfo_src == mla_info_map.end())
                            continue;
                        MLA_InfoPtr ptrInfo_src = itrInfo_src->second;

                        cv::Point2f center_src_at_mla = ptrInfo_src->GetCenter();
                        cv::Vec2f center_mi_src((mi_cols-1)*0.5f+1, (mi_rows-1)*0.5f+1);

                        // 反投影
                        cv::Vec2f coord_src_mi = BackProject2MI(virtualPoint3D_ref, center_src_at_mla, center_mi_src);
                        int src_r = int(coord_src_mi[1] + 0.5f);
                        int src_c = int(coord_src_mi[0] + 0.5f);
                        if (src_c >= 0 && src_c < mi_cols && src_r >= 0 && src_r < mi_rows)
                        {
                            if (masks[ptrKey_src].at<uchar>(src_r, src_c) == 1)
                                continue;

                            int index_src = src_r * mi_cols + src_c;
                            float4 normal_src_tmp = ptrDN_src->ph_cuda[index_src];
                            cv::Vec3f normal_src = cv::Vec3f(normal_src_tmp.x, normal_src_tmp.y, normal_src_tmp.z);

                            float disp_src = ptrDN_src->d_cuda[index_src];
                            float virtualDepth_src = ComputeVirtualDepth(disp_src, params);
                            if (fabs(virtualDepth_src -g_Invalid_image)<1e-6)
                            {
                                continue;
                            }
                            cv::Vec3f virtual_Point_src = ComputeVirtual3D(virtualDepth_src, src_c,
                                                                           src_r, mi_cols, mi_rows, center_src_at_mla);

                            // 反投影到参考微图像
                            cv::Vec2f coord_mi_reproject = BackProject2MI(virtual_Point_src, center_at_mla, center_mi);
                            float reproj_error = sqrt(pow(c - coord_mi_reproject[0], 2) + pow(r - coord_mi_reproject[1], 2));
                            float relative_virtualDepth_diff = fabs(virtualDepth_src - virtualDepth_ref) / virtualDepth_ref;
                            if (reproj_error < 1.0f && relative_virtualDepth_diff < 0.01f)
                            {
                                consistent_Point[0] += virtual_Point_src[0];
                                consistent_Point[1] += virtual_Point_src[1];
                                consistent_Point[2] += virtual_Point_src[2];
                                consistent_normal += normal_src;

                                consistent_Color[0] += problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c)[0];
                                consistent_Color[1] += problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c)[1];
                                consistent_Color[2] += problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c)[2];

                                used_list[j].x = src_c;
                                used_list[j].y = src_r;
                                num_consistent++;
                            }
                        }
                    }

                    if (num_consistent >= 2)
                    {
                        consistent_Point[0] /= (num_consistent + 1.0f);
                        consistent_Point[1] /= (num_consistent + 1.0f);
                        consistent_Point[2] /= (num_consistent + 1.0f);
                        consistent_normal /= (num_consistent + 1.0f);
                        consistent_Color[0] /= (num_consistent + 1.0f);
                        consistent_Color[1] /= (num_consistent + 1.0f);
                        consistent_Color[2] /= (num_consistent + 1.0f);

                        PointList point3D;
                        point3D.coord = make_float3(consistent_Point[0], consistent_Point[1], consistent_Point[2]);
                        point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                        point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);

                        // TODO XYY ：Collect 3D points
//                        point_lists_Vec.push_back(point3D);
//                        float temp = point3D.coord.z;
//                        if (std::isfinite(temp)) {
//                            vdMin = std::min(vdMin, temp);
//                            vdMax = std::max(vdMax, temp);
//                        }

                        if (point3D.coord.y >= 0 && point3D.coord.y < image_height
                            && point3D.coord.x >= 0 && point3D.coord.x < image_width)
                        {
                            m_VirtualDepthMap.at<float>(static_cast<int>(point3D.coord.y),
                                                        static_cast<int>(point3D.coord.x)) = point3D.coord.z;
                            float realDepth_BM = ConvertVdToRd(point3D.coord.z, behaviorModelParams);
//                            float tmp = point3D.coord.z * LFMVS::g_B + LFMVS::g_bl0;
//                            float realDepth_BM = (params.mainlen_flocal_length * tmp) / (tmp - params.mainlen_flocal_length);
                            if (counter < 1000)
                            {
                                std::cout<<"virtualDepth_s="<<point3D.coord.z<<"---"<<"realDepth="<<realDepth_BM<<std::endl;
                            }
                            realDepthMap_BM.at<float>(static_cast<int>(point3D.coord.y),
                                                      static_cast<int>(point3D.coord.x)) = realDepth_BM;
                            PointCloud.push_back(point3D);

                            // 插入体素
                            SEACAVE::TOctreeFusionVoxel::F_POINT_TYPE pointCoord;
//                            pointCoord << point3D.coord.x, point3D.coord.y, point3D.coord.z*500;


                            pointCoord << point3D.coord.x,(image_height-1.0) -point3D.coord.y, point3D.coord.z*300;
                            OctreeVoxel.InsertForFusion_Bucket(pointCoord);

                            m_AIF_Color.at<cv::Vec3b>(static_cast<int>(point3D.coord.y),
                                                      static_cast<int>(point3D.coord.x)) =
                                    cv::Vec3b(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                        }

                        for (int j = 0; j < num_ngb; ++j)
                        {
                            if (used_list[j].x == -1)
                                continue;
                            QuadTreeTileKeyPtr ptrKey_src = problem.m_NeighsSortVecForMatch[j];
                            masks[ptrKey_src].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                        }
                    }
                }
            }
        }

        // 保存结果
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strFrameName + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
                }
            }
        }
        std::string vIDepth_RawFullPath = m_strSavePath + "/" + LF_VIRTUALDEPTHMAP_NAME + std::string(".tiff");
        cv::imwrite(vIDepth_RawFullPath, m_VirtualDepthMap);
        std::string vIDepth_VirFullPath = m_strSavePath + LF_ALLINFOUCSIMAGE_NAME + std::string(".png");
        cv::imwrite(vIDepth_VirFullPath, m_AIF_Color);
        std::string ply_path = m_strSavePath + "/vitrual_3dmodel.ply";
        StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);


        cv::Mat tmp = realDepthMap_BM.clone();
        cv::patchNaNs(tmp, 0.0f);                      // 把 NaN 置 0，避免污染 min/max

        cv::Mat mask = (tmp > 0);                      // 只把 >0 当有效（按你约定调整）
        double minV, maxV;
        cv::minMaxLoc(tmp, &minV, &maxV, nullptr, nullptr, mask);

        cv::Mat depth8u;
        tmp.convertTo(depth8u, CV_8U, 255.0/(maxV-minV), -minV*255.0/(maxV-minV));
        depth8u.setTo(0, ~mask);                       // 无效像素置 0（黑）
        std::cout << "valid=" << cv::countNonZero(mask)
                  << " minV=" << minV << " maxV=" << maxV
                  << " range=" << (maxV-minV) << std::endl;
        cv::Mat realDepthMap_BM_color;
        cv::Mat realDepthMap_BM_color_HOT;
//        cv::normalize(realDepthMap_BM, depth8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth8u, realDepthMap_BM_color, cv::COLORMAP_JET);
        cv::applyColorMap(depth8u, realDepthMap_BM_color_HOT, cv::COLORMAP_HOT);
        cv::imwrite(m_strSavePath + "/real_depth_map_BM.png", realDepthMap_BM);
        cv::imwrite(m_strSavePath + "/real_depth_map_BM_color.png", realDepthMap_BM_color);
        cv::imwrite(m_strSavePath + "/realDepthMap_BM_color_HOT.png", realDepthMap_BM_color_HOT);

        OctreeVoxel.CollectLeafs(false);
        std::string octreePath = m_strSavePath + "init_octreeSurface3.obj";
//        OctreeVoxel.ExportInitOctreeObj_RootPlus8Children(aabb, octreePath, true, 0.98);
//        OctreeVoxel.ExportOccupiedLeafVoxelsObj(octreePath, false, 1);
        OctreeVoxel.ExportOccupiedLeafSurfaceObj(octreePath, OctreeVoxel.occupiedLeaves, false);

        // 全聚焦图像的填充
        //FillVirtualDepthMap();

        // 统计
        std::vector<float> vd_data_mla;
        for (int row=0; row < image_height; row++)
        {
            for (int col=0; col < image_width; col++)
            {
                float vd = m_VirtualDepthMap.at<float>(row, col);
                vd_data_mla.push_back(vd);
            }
        }
        std::vector<float> vd_inlier_global;
        float lo_global = 0.0f, hi_global = 0.0f;
        m_ptrDepthSolver->BuildGlobalDispInlierMAD(vd_data_mla, vd_inlier_global,
                                                   lo_global, hi_global, 3.0f, 0.02f);
        LOG_ERROR("LF-Refocus-backproject: min_vd= ", lo_global, "max_vd= ", hi_global);
        for (int row=0; row < image_height; row++)
        {
            for (int col=0; col < image_width; col++)
            {
                if (m_VirtualDepthMap.at<float>(row, col) < lo_global)
                {
                    m_VirtualDepthMap.at<float>(row, col) = lo_global;
                }
                else if (m_VirtualDepthMap.at<float>(row, col) > hi_global)
                {
                    m_VirtualDepthMap.at<float>(row, col) = hi_global;
                }
            }
        }
        std::string vIDepth_RawFullPath_g = m_strSavePath + std::string("/VD_Raw.tiff");
        cv::imwrite(vIDepth_RawFullPath_g, m_VirtualDepthMap);
        cv::normalize(m_VirtualDepthMap, m_VirtualDepthMap, 0, 255, cv::NORM_MINMAX);
        m_VirtualDepthMap.convertTo(m_vIDepthGray, CV_8U);
        cv::applyColorMap(m_vIDepthGray, m_VirtualDepthMap, cv::COLORMAP_JET);
//        std::string vIDepth_RawFullPath_g = m_strSavePath + std::string("/VD_Raw.tiff");
//        cv::imwrite(vIDepth_RawFullPath_g, m_vIDepthGray);
        std::string vIDepth_RawFullPath_c = m_strSavePath + std::string("/VD_Raw_color.png");
        cv::imwrite(vIDepth_RawFullPath_c, m_VirtualDepthMap);

        // 配置并一键执行
//        bool bAnalyse = false;
        bool bAnalyse = true;
        if (bAnalyse)
        {
            std::vector<float> data;
            data.reserve(PointCloud.size());
            for (int i=0; i<PointCloud.size(); i++)
            {
                float3& point = PointCloud.at(i).coord;
                data.push_back(point.z);
            }

            bool exact=false;
            double hdi_p=0.85;
            int imgW=1400;
            int imgH=480;
            bool show=false;

            std::string prefix = m_strSavePath + std::string("/Distribution_analysis");
            boost::filesystem::path dir_save_path(prefix);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << prefix << std::endl;
                }
            }
            prefix += "/analysis";

            FloatDistributionAnalyzer::Options opt;
            opt.exact_quantiles = exact;
            opt.exact_hdi = exact;
            opt.hdi_p = hdi_p;
            opt.image_width = imgW;
            opt.image_height = imgH;
            opt.output_prefix = prefix;
            opt.show_windows = show;

            FloatDistributionAnalyzer analyzer;

            // 决定 HDI 求法（是否精确）
            AnalyzerConfig cfg_f;
            cfg_f.exact_hdi = true;   // 精确最窄区间（排序法）；大数据可设 false 更快
            cfg_f.hdi_p = 0.85;       // 仅影响打印；函数参数会再传一次 p
            AnalysisResult R_filtered;
            analyzer.save_histogram_hdi_filtered(data, hdi_p, imgW, imgH, cfg_f,
                                                 prefix+"hdi85_hist.png", &R_filtered);

            AnalysisResult R; // 如需在代码中使用结果，可传出
            if (!analyzer.run(data, opt, &R))
            {
                std::cerr << "分析失败\n";
            }

            cv::Mat img = analyzer.draw_histogram_hdi_filtered(data,  hdi_p,
                                                               imgW, imgH, cfg_f, &R_filtered);
            cv::imwrite(prefix+"hdi85_hist_new.png", img);
            // 控制台打印结果（可选）
            AnalyzerConfig cfg; cfg.exact_quantiles = opt.exact_quantiles; cfg.exact_hdi = opt.exact_hdi; cfg.hdi_p = opt.hdi_p;
            print_result(R, cfg);
        }

        std::vector<PointList> PointCloud_Object;
        Virtual2ObjectDepth(PointCloud_Object, PointCloud);
        std::string object_ply_path = m_strSavePath + "/object_3dmodel.ply";
        StoreColorPlyFileBinaryPointCloud (object_ply_path, PointCloud_Object);
        LOG_ERROR("LFRefocus: FuseVirtualDepth_BackProject, End");
    }

    void LFRefocus::FuseVirtualDepth_BackProject(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("LFRefocus: FuseVirtualDepth_BackProject, Begin");
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problem_map = itrFrame->second;
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        QuadTreeTileInfoMap& mla_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        const int image_height = m_ptrDepthSolver->GetRawImageHeight();
        const int image_width = m_ptrDepthSolver->GetRawImageWidth();

        m_VirtualDepthMap = cv::Mat::zeros(image_height, image_width, CV_32F);
        m_AIF_Color = cv::Mat::zeros(image_height, image_width, CV_8UC3);
        cv::Mat VD_Confidence = cv::Mat::zeros(image_height, image_width, CV_8UC1);

        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("Current disNormalMapMap not found: ", strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        // Step1: 为每个有效估计视差的微图像建立一个mask图
        QuadTreeImageMap masks;
        for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
        {
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage)
                continue;

            QuadTreeTileKeyPtr ptrKey = itr->first;

            QuadTreeTileInfoMap::iterator itrInfo = mla_info_map.find(ptrKey);
            if (itrInfo == mla_info_map.end())
            {
                LOG_ERROR("Current MLA_info not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                LOG_ERROR("Current disNormal not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }

            int mi_rows = problem.m_Image_gray.rows;
            int mi_cols = problem.m_Image_gray.cols;
            masks[ptrKey] = cv::Mat::zeros(mi_rows, mi_cols, CV_8UC1);
        }

        // Step2: 融合虚拟深度
        std::vector<PointList> PointCloud;
        PointCloud.reserve(image_width * image_height / 8);

        for (QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); ++itr)
        {
            MLA_Problem& problem = itr->second;
            if (problem.m_bGarbage)
                continue;

            QuadTreeTileKeyPtr ptrKey = itr->first;

            QuadTreeTileInfoMap::iterator itrInfo = mla_info_map.find(ptrKey);
            if (itrInfo == mla_info_map.end())
                continue;
            MLA_InfoPtr mla_InfoPtr = itrInfo->second;

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
                continue;
            DisparityAndNormalPtr ptrDN = itr_DN->second;

            cv::Point2f center_at_mla = mla_InfoPtr->GetCenter();

            int num_ngb = static_cast<int>(problem.m_NeighsSortVecForMatch.size());
            std::vector<int2> used_list(num_ngb, make_int2(-1, -1));

            int mi_rows = problem.m_Image_gray.rows;
            int mi_cols = problem.m_Image_gray.cols;
            cv::Vec2f center_mi((mi_cols - 1) * 0.5f + 1.0f, (mi_rows - 1) * 0.5f + 1.0f);

            for (int r = 0; r < mi_rows; ++r)
            {
                for (int c = 0; c < mi_cols; ++c)
                {
                    cv::Mat& mask = masks[ptrKey];
                    if (mask.at<uchar>(r, c) == 1)
                        continue;

                    int index = r * mi_cols + c;
                    float disp_ref = ptrDN->d_cuda[index];

                    // 保持与现有流程一致：仅融合有效视差
                    if (disp_ref < 1.0f)
                        continue;

                    float4 normal_ref_tmp = ptrDN->ph_cuda[index];
                    cv::Vec3f consistent_normal(normal_ref_tmp.x, normal_ref_tmp.y, normal_ref_tmp.z);

                    float virtualDepth_ref = ComputeVirtualDepth(disp_ref, params);
                    if (fabs(virtualDepth_ref - g_Invalid_image) < 1.0e-6f)
                        continue;

                    cv::Vec3f virtualPoint3D_ref =
                        ComputeVirtual3D(virtualDepth_ref, c, r, mi_cols, mi_rows, center_at_mla);
                    cv::Vec3f consistent_Point = virtualPoint3D_ref;

                    // 注意：这里必须用 float 累加，不能继续用 uchar，避免跨邻居累加时溢出
                    cv::Vec3f consistent_Color(
                        static_cast<float>(problem.m_Image_rgb.at<uchar3>(r, c).x),
                        static_cast<float>(problem.m_Image_rgb.at<uchar3>(r, c).y),
                        static_cast<float>(problem.m_Image_rgb.at<uchar3>(r, c).z));

                    std::fill(used_list.begin(), used_list.end(), make_int2(-1, -1));

                    int num_consistent = 0;
                    for (int j = 0; j < num_ngb; ++j)
                    {
                        QuadTreeTileKeyPtr ptrKey_src = problem.m_NeighsSortVecForMatch[j];

                        QuadTreeProblemMap::iterator itrQP = problem_map.find(ptrKey_src);
                        if (itrQP == problem_map.end())
                            continue;
                        MLA_Problem& problem_src = itrQP->second;

                        QuadTreeDisNormalMap::iterator itr_DN_src = disNormals_map.find(ptrKey_src);
                        if (itr_DN_src == disNormals_map.end())
                            continue;
                        DisparityAndNormalPtr ptrDN_src = itr_DN_src->second;

                        QuadTreeTileInfoMap::iterator itrInfo_src = mla_info_map.find(ptrKey_src);
                        if (itrInfo_src == mla_info_map.end())
                            continue;
                        MLA_InfoPtr ptrInfo_src = itrInfo_src->second;

                        cv::Point2f center_src_at_mla = ptrInfo_src->GetCenter();
                        cv::Vec2f center_mi_src((mi_cols - 1) * 0.5f + 1.0f, (mi_rows - 1) * 0.5f + 1.0f);

                        cv::Vec2f coord_src_mi = BackProject2MI(virtualPoint3D_ref, center_src_at_mla, center_mi_src);
                        int src_r = static_cast<int>(coord_src_mi[1] + 0.5f);
                        int src_c = static_cast<int>(coord_src_mi[0] + 0.5f);

                        if (src_c < 0 || src_c >= mi_cols || src_r < 0 || src_r >= mi_rows)
                            continue;

                        if (masks[ptrKey_src].at<uchar>(src_r, src_c) == 1)
                            continue;

                        int index_src = src_r * mi_cols + src_c;

                        float disp_src = ptrDN_src->d_cuda[index_src];
                        float virtualDepth_src = ComputeVirtualDepth(disp_src, params);
                        if (fabs(virtualDepth_src - g_Invalid_image) < 1.0e-6f)
                            continue;

                        float4 normal_src_tmp = ptrDN_src->ph_cuda[index_src];
                        cv::Vec3f normal_src(normal_src_tmp.x, normal_src_tmp.y, normal_src_tmp.z);

                        cv::Vec3f virtual_Point_src =
                            ComputeVirtual3D(virtualDepth_src, src_c, src_r, mi_cols, mi_rows, center_src_at_mla);

                        cv::Vec2f coord_mi_reproject = BackProject2MI(virtual_Point_src, center_at_mla, center_mi);
                        float reproj_error =
                            std::sqrt((c - coord_mi_reproject[0]) * (c - coord_mi_reproject[0]) +
                                      (r - coord_mi_reproject[1]) * (r - coord_mi_reproject[1]));
                        float relative_virtualDepth_diff =
                            std::fabs(virtualDepth_src - virtualDepth_ref) / std::max(virtualDepth_ref, 1.0e-6f);

                        if (reproj_error < 1.0f && relative_virtualDepth_diff < 0.05f)
                        {
                            consistent_Point += virtual_Point_src;
                            consistent_normal += normal_src;
                            cv::Vec3b src_color = problem_src.m_Image_rgb.at<cv::Vec3b>(src_r, src_c);
                            consistent_Color += cv::Vec3f(
                                static_cast<float>(src_color[0]),
                                static_cast<float>(src_color[1]),
                                static_cast<float>(src_color[2]));

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;
                            ++num_consistent;
                        }
                    }

                    if (num_consistent >= 2)
                    {
                        const float denom = num_consistent + 1.0f;
                        consistent_Point /= denom;
                        consistent_normal /= denom;
                        consistent_Color /= denom;

                        PointList point3D;
                        point3D.coord = make_float3(consistent_Point[0], consistent_Point[1], consistent_Point[2]);
                        point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                        point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);

                        const int vy = static_cast<int>(point3D.coord.y);
                        const int vx = static_cast<int>(point3D.coord.x);

                        if (vy >= 0 && vy < image_height && vx >= 0 && vx < image_width)
                        {
                            m_VirtualDepthMap.at<float>(vy, vx) = point3D.coord.z;
                            PointCloud.push_back(point3D);

                            // 置信度：按支持视图数做一个简单编码
                            VD_Confidence.at<uchar>(vy, vx) =
                                static_cast<uchar>(std::min(255, num_consistent * 80));

                            m_AIF_Color.at<cv::Vec3b>(vy, vx) = cv::Vec3b(
                                cv::saturate_cast<uchar>(consistent_Color[0]),
                                cv::saturate_cast<uchar>(consistent_Color[1]),
                                cv::saturate_cast<uchar>(consistent_Color[2]));
                        }

                        for (int j = 0; j < num_ngb; ++j)
                        {
                            if (used_list[j].x == -1)
                                continue;
                            QuadTreeTileKeyPtr ptrKey_src = problem.m_NeighsSortVecForMatch[j];
                            masks[ptrKey_src].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                        }
                    }
                }
            }
        }

        // Step3: 保存基础结果
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strFrameName + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
            }
        }

        std::string strVDImagePath = m_strSavePath + "/" + LF_VIRTUALDEPTHMAP_NAME + std::string(".tiff");
        cv::imwrite(strVDImagePath, m_VirtualDepthMap);

        std::string strAIFImagePath = m_strSavePath + LF_ALLINFOUCSIMAGE_NAME + std::string(".png");
        cv::imwrite(strAIFImagePath, m_AIF_Color);

        // 原始 float 虚拟深度图（调试/定量使用）
        std::string strVDRawFloatPath = m_strSavePath + std::string("/VD_Raw_float.tiff");
        cv::imwrite(strVDRawFloatPath, m_VirtualDepthMap);

        // Step4: 统计有效虚拟深度范围（只统计有效像素）
        std::vector<float> vd_data_valid;
        vd_data_valid.reserve(image_height * image_width / 8);

        for (int row = 0; row < image_height; ++row)
        {
            const float* ptr = m_VirtualDepthMap.ptr<float>(row);
            for (int col = 0; col < image_width; ++col)
            {
                float vd = ptr[col];
                if (std::isfinite(vd) && vd > 0.0f)
                    vd_data_valid.push_back(vd);
            }
        }

        float lo_global = 0.0f, hi_global = 0.0f;
        std::vector<float> vd_inlier_global;
        if (!vd_data_valid.empty())
        {
            m_ptrDepthSolver->BuildGlobalDispInlierMAD(
                vd_data_valid, vd_inlier_global, lo_global, hi_global, 3.0f, 0.02f);
            LOG_ERROR("LF-Refocus-backproject: min_vd= ", lo_global, " max_vd= ", hi_global);
        }
        else
        {
            LOG_ERROR("LF-Refocus-backproject: no valid virtual depth points.");
            lo_global = 0.0f;
            hi_global = 1.0f;
        }

        // Step5: 生成灰度图和伪彩色图
        // 核心修正：
        // 1) 不再把 m_VirtualDepthMap 直接覆盖成颜色图；
        // 2) 只对有效像素做归一化；
        // 3) 无效背景单独设为黑色，避免 JET 把 0 映射成整片蓝色。
        cv::Mat VD_Raw = m_VirtualDepthMap.clone();
        cv::Mat validMask = (VD_Raw > 0.0f);

        double minV = 0.0, maxV = 0.0;
        cv::minMaxLoc(VD_Raw, &minV, &maxV, nullptr, nullptr, validMask);

        float lo_vis = std::max(static_cast<float>(minV), lo_global);
        float hi_vis = std::min(static_cast<float>(maxV), hi_global);

        if (!(hi_vis > lo_vis + 1.0e-6f))
        {
            lo_vis = static_cast<float>(minV);
            hi_vis = static_cast<float>(maxV);
            if (!(hi_vis > lo_vis + 1.0e-6f))
                hi_vis = lo_vis + 1.0f;
        }

        cv::Mat VD_Gray = cv::Mat::zeros(image_height, image_width, CV_8UC1);
        const float gamma = 0.8f; // 轻度增强对比度，避免有效点都挤在同一小色带

        for (int row = 0; row < image_height; ++row)
        {
            const float* src_ptr = VD_Raw.ptr<float>(row);
            const uchar* mask_ptr = validMask.ptr<uchar>(row);
            uchar* gray_ptr = VD_Gray.ptr<uchar>(row);

            for (int col = 0; col < image_width; ++col)
            {
                if (!mask_ptr[col])
                    continue;

                float v = src_ptr[col];
                v = std::min(std::max(v, lo_vis), hi_vis);
                float norm_v = (v - lo_vis) / (hi_vis - lo_vis);
                norm_v = std::pow(norm_v, gamma);
                gray_ptr[col] = cv::saturate_cast<uchar>(norm_v * 255.0f);
            }
        }

        std::string strVDGrayPath = m_strSavePath + std::string("/VD_Raw_gray.png");
        cv::imwrite(strVDGrayPath, VD_Gray);

        cv::Mat VD_Color_Output;
        cv::applyColorMap(VD_Gray, VD_Color_Output, cv::COLORMAP_JET);

        // 无效背景统一设为黑色；若你想保留其他背景色，可在这里修改
        VD_Color_Output.setTo(cv::Scalar(0, 0, 0), ~validMask);

        std::string strVDColorPath = m_strSavePath + std::string("/VD_Raw_color.png");
        cv::imwrite(strVDColorPath, VD_Color_Output);

        // Step6: 置信图
        std::string strVDConfPath = m_strSavePath + strFrameName + "_" +
            LF_VIRTUALDEPTH_CONFMAP_RAW_NAME + std::string(".png");
        cv::imwrite(strVDConfPath, VD_Confidence);

        cv::Mat VD_Confidence_Binary;
        cv::threshold(VD_Confidence, VD_Confidence_Binary, 50, 255, cv::THRESH_BINARY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(VD_Confidence_Binary, VD_Confidence_Binary, cv::MORPH_CLOSE, kernel);

        std::string strVDConfBinaryPath = m_strSavePath + strFrameName + "_" +
            LF_VIRTUALDEPTH_CONFMAP_NAME + ".png";
        cv::imwrite(strVDConfBinaryPath, VD_Confidence_Binary);

        // Step7: 分布分析
        bool bAnalyse = false;
        if (bAnalyse && !PointCloud.empty())
        {
            std::vector<float> data;
            data.reserve(PointCloud.size());
            for (size_t i = 0; i < PointCloud.size(); ++i)
                data.push_back(PointCloud[i].coord.z);

            bool exact = false;
            double hdi_p = 0.85;
            int imgW = 1400;
            int imgH = 480;
            bool show = false;

            std::string prefix = m_strSavePath + std::string("/Distribution_analysis");
            boost::filesystem::path dir_save_path(prefix);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                    std::cout << "dir failed to create: " << prefix << std::endl;
            }
            prefix += "/analysis";

            FloatDistributionAnalyzer::Options opt;
            opt.exact_quantiles = exact;
            opt.exact_hdi = exact;
            opt.hdi_p = hdi_p;
            opt.image_width = imgW;
            opt.image_height = imgH;
            opt.output_prefix = prefix;
            opt.show_windows = show;

            FloatDistributionAnalyzer analyzer;

            AnalyzerConfig cfg_f;
            cfg_f.exact_hdi = true;
            cfg_f.hdi_p = 0.85;
            AnalysisResult R_filtered;
            analyzer.save_histogram_hdi_filtered(
                data, hdi_p, imgW, imgH, cfg_f, prefix + "hdi85_hist.png", &R_filtered);

            AnalysisResult R;
            if (!analyzer.run(data, opt, &R))
                std::cerr << "分析失败\n";

            cv::Mat img = analyzer.draw_histogram_hdi_filtered(
                data, hdi_p, imgW, imgH, cfg_f, &R_filtered);
            cv::imwrite(prefix + "hdi85_hist_new.png", img);

            AnalyzerConfig cfg;
            cfg.exact_quantiles = opt.exact_quantiles;
            cfg.exact_hdi = opt.exact_hdi;
            cfg.hdi_p = opt.hdi_p;
            print_result(R, cfg);
        }

        std::vector<PointList> PointCloud_Object;
        Virtual2ObjectDepth(PointCloud_Object, PointCloud);
        std::string object_ply_path = m_strSavePath + "/object_3dmodel.ply";
        StoreColorPlyFileBinaryPointCloud(object_ply_path, PointCloud_Object);

        LOG_ERROR("LFRefocus: FuseVirtualDepth_BackProject, End");
    }

    float LFRefocus::ComputeVirtualDepth(const double disparity, const LightFieldParams& params)
    {
        if (disparity <= 0)
            return g_Invalid_image;
        //double vir_d = g_bl0*(params.baseline/disparity); //
        //float mla_Base=m_Params.baseline*LFMVS::g_bl0/(LFMVS::g_bl0+LFMVS::g_B);//使用微透镜基线
        //float f_v = mla_Base / disp;

        float mla_Base = params.baseline*g_bl0/(g_bl0+g_B); // 根据zeller公式，应使用微透镜基线，而不是微图像的基线
        float vir_d = mla_Base / disparity;
        return vir_d;
    }

    cv::Vec3f LFRefocus::ComputeVirtual3D(float virtualDepth, int p_col, int p_row, int mi_cols, int mi_rows,
                            cv::Point2f center_at_mla)
    {
        //int x_mlaCenter = x - ((m_Params.mi_width_for_match-1)*0.5 + 1);
        //float Xv = x_mlaCenter*f_v +ptrInfo->GetCenter().x;
        //int y_mlaCenter = y - ((m_Params.mi_height_for_match-1)*0.5 + 1);
        //float Yv = y_mlaCenter*f_v +ptrInfo->GetCenter().y;


        cv::Vec3f virtual_Point3D;
        cv::Vec2f center_mi((mi_cols-1)*0.5f+ 1, (mi_rows-1)*0.5f+ 1);
        cv::Vec2f offset_coord (p_col-center_mi[0], p_row-center_mi[1]);

        virtual_Point3D[0] = offset_coord[0]*virtualDepth + center_at_mla.x;
        virtual_Point3D[1] = offset_coord[1]*virtualDepth + center_at_mla.y;
        virtual_Point3D[2] = virtualDepth;
        return virtual_Point3D;
    }

    cv::Vec2f LFRefocus::BackProject2MI(cv::Vec3f virtual_Point, cv::Point2f center_at_mla, cv::Vec2f center_mi_src)
    {
        cv::Vec2f coord_src_mi_2D;
        coord_src_mi_2D[0] = ((virtual_Point[0] - center_at_mla.x)/virtual_Point[2])+center_mi_src[0];
        coord_src_mi_2D[1] = ((virtual_Point[1] - center_at_mla.y)/virtual_Point[2])+center_mi_src[1];
        return coord_src_mi_2D;
    }

    void LFRefocus::Virtual2ObjectDepth(std::vector<PointList>& PointCloud_Object,
        const std::vector<PointList>& PointCloud_Virtual)
    {
        const LightFieldParams& params = m_ptrDepthSolver->GetParams();
        PointCloud_Object.reserve(PointCloud_Virtual.size());

        const float inv_main_focal_length = 1.0/params.mainlen_flocal_length;
        const int pc_size = PointCloud_Virtual.size();
        for (int i = 0; i < pc_size; i++)
        {
            const PointList& pc_virtual = PointCloud_Virtual.at(i);
            const float3& coord_virtual = pc_virtual.coord;
            float sec_part = 1.0/(g_bl0+coord_virtual.z*g_B);
            float object_depth = 1.0/(inv_main_focal_length-sec_part);
            PointList pc_object;
            pc_object.coord.x = coord_virtual.x;
            pc_object.coord.y = coord_virtual.y;
            pc_object.coord.z = object_depth;
            pc_object.color = pc_virtual.color;
            pc_object.normal = pc_virtual.normal;
            PointCloud_Object.push_back(pc_object);
        }
    }



    void LFRefocus::ImageDenoising(QuadTreeProblemMap& problem_map)
    {
        for(int y = 0; y < m_AIF_Color.rows; y++)
        {
            for (int x = 0; x < m_AIF_Color.cols; x++)
            {
                CoVIPointMap::iterator cVIPMItr = m_coVIPointMap.find({y, x});
                if(cVIPMItr == m_coVIPointMap.end())
                    continue;
                m_vIPointVec = cVIPMItr->second;
                // 如果虚拟图像中有坐标同名点，选出确定的像素点
                if(m_vIPointVec.size() > 1)
                {
                    // 独立成函数，使用模糊度、纹理值、基线(待考虑) 赋权排序,选出唯一的虚拟图像像素点
                    VIPointProperty fusedVIPoint = DenoisByColorConsist(m_vIPointVec, problem_map);
                    m_vIPointMap[{y, x}] = fusedVIPoint;
                }
                    /*else if(m_vIPointVec.size() == 1)
                    {
                        m_vIPointMap[{y, x}] = m_vIPointVec[0];
                    }*/
                else
                {
                    continue;
                }
            }
        }
    }

    VIPointProperty LFRefocus::DenoisByColorConsist(std::vector<VIPointProperty>& vIPointVec, QuadTreeProblemMap& problem_map)
    {
        cv::Vec3b rgbPoint;
        cv::Mat lab;
        cv::Vec3b lab_Result;
        std::vector<float> L,A,B;
        float L_Median,A_Median,B_Median;
        float threshold = 300.0;
        float totalScore = 0;
        VIPointProperty VIPointMedia;
        cv::Vec3f fusedRgb(0,0,0);
        float fusedVd = 0;
        cv::Vec3b fusedRgbPoint;


        // 返回一个 VIPointProperty
        for(int i = 0; i < vIPointVec.size(); i++)
        {
            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPointVec[i].srcImageKey);
            MLA_Problem& problem = Itr->second;
            // 把像素点 的相似度和模糊度值归一化到0-1（若该方法效果不好则进行更换）
            //TODO 理清楚
            float similarityScore = problem.m_Image_Richness[vIPointVec[i].srcImageX][vIPointVec[i].srcImageY] / 255;
            float blureScore = problem.m_Image_Blureness[vIPointVec[i].srcImageX][vIPointVec[i].srcImageY] / 255;
            float a=0.70;
            float b=0.30;
            vIPointVec[i].score = similarityScore * a + blureScore * b;

            rgbPoint = problem.m_Image_rgb.at<cv::Vec3b>(vIPointVec[i].srcImageY, vIPointVec[i].srcImageX);
            cv::Mat rgb(1,1,CV_8UC3,rgbPoint);
            cv::cvtColor(rgb,lab,cv::COLOR_BGR2Lab);
            lab_Result = lab.at<cv::Vec3b>(0.0);
            //  m_labVec.emplace_back(lab_Result[0],lab_Result[1],lab_Result[2]);
            vIPointVec[i].lab = cv::Vec3f(lab_Result[0],lab_Result[1],lab_Result[2]);
            L.push_back(lab_Result[0]);
            A.push_back(lab_Result[1]);
            B.push_back(lab_Result[2]);
        }

        std::sort(L.begin(),L.end());
        std::sort(A.begin(),A.end());
        std::sort(B.begin(),B.end());

        if(L.size() % 2 ==0)
        {
            L_Median = (L[L.size() / 2] + L[L.size() / 2 - 1]) / 2.0f;
            A_Median = (A[A.size() / 2] + A[A.size() / 2 - 1]) / 2.0f;
            B_Median = (B[B.size() / 2] + B[B.size() / 2 - 1]) / 2.0f;
        }
        else
        {
            L_Median = L[L.size() / 2];
            A_Median = A[L.size() / 2];
            B_Median = B[L.size() / 2];
        }

        for(int i = 0; i < vIPointVec.size(); i++)
        {
            cv::Vec3f lab = vIPointVec[i].lab;
            float dis = cv::norm(lab,cv::Vec3f(L_Median,A_Median,B_Median));
            if(dis > threshold)
            {
                vIPointVec.erase(vIPointVec.begin() + i);
                std::cout<<"vIPointVec.erase:"<<dis<<std::endl;
            }
            else
            {
                totalScore += vIPointVec[i].score;
                i++;
            }
        }

        for(int i = 0; i < vIPointVec.size(); i++)
        {
            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPointVec[i].srcImageKey);
            MLA_Problem& problem = Itr->second;
            cv::Vec3b rgbPoint = problem.m_Image_rgb.at<cv::Vec3b>(vIPointVec[i].srcImageY, vIPointVec[i].srcImageX);
            cv::Vec3f f_Rgb(rgbPoint[0],rgbPoint[1],rgbPoint[2]);
            float weight = vIPointVec[i].score / totalScore;
            fusedRgb += f_Rgb * weight;
            fusedVd += vIPointVec[i].virtualDepth * weight;

            VIPointMedia.virtualImageX = vIPointVec[i].virtualImageX;
            VIPointMedia.virtualImageY = vIPointVec[i].virtualImageY;
        }
        fusedRgbPoint[0] = cv::saturate_cast<uchar>(fusedRgb[0]),
        fusedRgbPoint[1] = cv::saturate_cast<uchar>(fusedRgb[1]),
        fusedRgbPoint[2] = cv::saturate_cast<uchar>(fusedRgb[2]);
        VIPointMedia.rgb_Media = fusedRgbPoint;
        VIPointMedia.vd_Media = fusedVd;


        return VIPointMedia;
        /*std::sort(vIPointVec.begin(), vIPointVec.end(),
                  [](const auto& a, const auto& b) {
                      return a.score > b.score;
                  });*/

    }


    void LFRefocus::FilterMatchingPoints(std::vector<VIPointProperty>& vIPointVec, QuadTreeProblemMap& problem_map)
    {
        // 返回一个 VIPointProperty
        for(int i = 0; i < vIPointVec.size(); i++)
        {
            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPointVec[i].srcImageKey);
            MLA_Problem& problem = Itr->second;
            // 把像素点 的纹理值和模糊度值归一化到0-1（若该方法效果不好则进行更换）
            float similarityScore = problem.m_Image_Richness[vIPointVec[i].srcImageX][vIPointVec[i].srcImageY] / 255;
            float blureScore = problem.m_Image_Blureness[vIPointVec[i].srcImageX][vIPointVec[i].srcImageY] / 255;
            float a=0.90;
            float b=0.10;
            vIPointVec[i].score = similarityScore * a + blureScore * b;
        }
        std::sort(vIPointVec.begin(), vIPointVec.end(),
                  [](const auto& a, const auto& b) {
                      return a.score > b.score;
                  });
    }

    void LFRefocus::GenerateVirtualImage(VIPointMap& vIPointMap, QuadTreeProblemMap& problem_map,LightFieldParams& params)
    {
        for(VIPointMap::iterator vPItr = vIPointMap.begin(); vPItr != vIPointMap.end(); vPItr++)
        {
            VIPointProperty& vIPoint = vPItr->second;
            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPoint.srcImageKey);
            MLA_Problem& problem = Itr->second;

            // 真实深度
            float tmp = vIPoint.virtualDepth *LFMVS::g_B + LFMVS::g_bl0;
            float real_d = (params.mainlen_flocal_length * tmp) / (tmp - params.mainlen_flocal_length);
            /*float mainlen_flocal_length = 105;          // 单位：毫米
            float real_d = (mainlen_flocal_length * tmp) / (tmp - mainlen_flocal_length);*/

            m_rIDepth_m.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = real_d / 1000;   // 单位转换为m
            m_rIDepth_mm.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = real_d;   // 单位为mm
            m_vIDepth.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = 1 / vIPoint.virtualDepth;
            m_VirtualDepthMap.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = vIPoint.virtualDepth;
            m_vIDepth_Reciprocal.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = 1 / vIPoint.virtualDepth;
            m_AIF_Color.at<cv::Vec3b>(vIPoint.virtualImageY, vIPoint.virtualImageX) =
                    problem.m_Image_rgb.at<cv::Vec3b>(vIPoint.srcImageY, vIPoint.srcImageX);

            //std::cout<<"real_d="<<real_d<<std::endl;
            // std::cout<<"("<<m_vIPoint.m_virtualImageX<<","<<m_vIPoint.m_virtualImageY<<")"<<"--"<<"("<<m_vIPoint.srcImageX<<","<<m_vIPoint.srcImageY<<")"<<std::endl;
        }
    }

    void LFRefocus::GenerateVIByColorConsist(VIPointMap& vIPointMap, QuadTreeProblemMap& problem_map,LightFieldParams& params)
    {
        for(VIPointMap::iterator vPItr = vIPointMap.begin(); vPItr != vIPointMap.end(); vPItr++)
        {
            VIPointProperty& vIPoint = vPItr->second;
            /*QuadTreeProblemMap::iterator Itr = problem_map.find(vIPoint.srcImageKey);
            MLA_Problem& problem = Itr->second;*/

            // 真实深度
            float tmp = vIPoint.vd_Media *LFMVS::g_B + LFMVS::g_bl0;
            float real_d = (params.mainlen_flocal_length * tmp) / (tmp - params.mainlen_flocal_length);
            /*float mainlen_flocal_length = 105;          // 单位：毫米
            float real_d = (mainlen_flocal_length * tmp) / (tmp - mainlen_flocal_length);*/
            if(real_d > 0)
            {
                m_rIDepth_m.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = real_d / 1000;   // 单位转换为m
                m_rIDepth_mm.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = real_d;   // 单位为mm
            }
            if(vIPoint.vd_Media > 0 && vIPoint.rgb_Media != cv::Vec3b(0,0,0))
            {
                m_vIDepth.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = 1 / vIPoint.vd_Media;
                m_VirtualDepthMap.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = vIPoint.vd_Media;
                m_vIDepth_Reciprocal.at<float>(vIPoint.virtualImageY, vIPoint.virtualImageX) = 1 / vIPoint.vd_Media;
                m_AIF_Color.at<cv::Vec3b>(vIPoint.virtualImageY, vIPoint.virtualImageX) =
                        vIPoint.rgb_Media;
            }

            // std::cout<<"real_d="<<real_d<<std::endl;
            // std::cout<<"("<<m_vIPoint.m_virtualImageX<<","<<m_vIPoint.m_virtualImageY<<")"<<"--"<<"("<<m_vIPoint.srcImageX<<","<<m_vIPoint.srcImageY<<")"<<std::endl;
        }
    }

    void LFRefocus::HoleFilling(cv::Mat& virtualImage, VIPointMap& vIPointMap, QuadTreeProblemMap& problem_map)
    {
        int counter = 0;
        bool bstop = true;
        float targetFillRatio = 0.98;
        cv::Mat mask;
        cv::inRange(virtualImage, cv::Scalar(1, 1, 1), cv::Scalar(255, 255, 255), mask);
        int pixelCount = cv::countNonZero(mask);
        while (bstop)
        {
            cv::Vec3b pixel;
            int filledCounter = 0;
            for(int y = 0; y < virtualImage.rows; y++)
            {
                for(int x = 0; x < virtualImage.cols; x++)
                {
                    pixel = virtualImage.at<cv::Vec3b>(y, x);
                    cv::Vec3b holePixel(0, 0, 0);
                    if (pixel == holePixel)
                    {
                        VIPointProperty filledPoint;
                        bool filled = false;
                        if(x+1 < virtualImage.cols && virtualImage.at<cv::Vec3b>(y, x + 1) != holePixel)
                        {
                            VIPointMap::iterator vPItr = vIPointMap.find({y, x + 1});
                            if(vPItr == vIPointMap.end())
                                continue;
                            VIPointProperty& vIPoint = vPItr->second;
                            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPoint.srcImageKey);
                            MLA_Problem& problem = Itr->second;
                            if(vIPoint.srcImageX - 1 >= 0)
                            {
                                virtualImage.at<cv::Vec3b>(y, x) = problem.m_Image_rgb.at<cv::Vec3b>(vIPoint.srcImageY, vIPoint.srcImageX - 1);
                                filledPoint.srcImageX = vIPoint.srcImageX - 1;
                                filledPoint.srcImageY = vIPoint.srcImageY;
                                filled = true;
                            }
                            else
                            {
                                continue;
                            }
                        }
                        else if(x-1 >=0 && virtualImage.at<cv::Vec3b>(y, x - 1) != holePixel)
                        {
                            VIPointMap::iterator vPItr = vIPointMap.find({y, x - 1});
                            if(vPItr == vIPointMap.end())
                                continue;
                            VIPointProperty& vIPoint = vPItr->second;
                            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPoint.srcImageKey);
                            MLA_Problem& problem = Itr->second;
                            if(vIPoint.srcImageX + 1 < problem.m_Image_rgb.cols)
                            {
                                virtualImage.at<cv::Vec3b>(y, x) = problem.m_Image_rgb.at<cv::Vec3b>(vIPoint.srcImageY, vIPoint.srcImageX + 1);
                                filledPoint.srcImageX = vIPoint.srcImageX + 1;
                                filledPoint.srcImageY = vIPoint.srcImageY;
                                filled = true;
                            }
                            else
                            {
                                continue;
                            }
                        }
                        else if(y+1 < virtualImage.rows && virtualImage.at<cv::Vec3b>(y + 1, x) != holePixel)
                        {
                            VIPointMap::iterator vPItr = vIPointMap.find({y + 1, x});
                            if(vPItr == vIPointMap.end())
                                continue;
                            VIPointProperty& vIPoint = vPItr->second;
                            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPoint.srcImageKey);
                            MLA_Problem& problem = Itr->second;
                            if(vIPoint.srcImageY - 1 >= 0)
                            {
                                virtualImage.at<cv::Vec3b>(y, x) = problem.m_Image_rgb.at<cv::Vec3b>(vIPoint.srcImageY - 1, vIPoint.srcImageX);
                                filledPoint.srcImageX = vIPoint.srcImageX;
                                filledPoint.srcImageY = vIPoint.srcImageY - 1;
                                filled = true;
                            }
                            else
                            {
                                continue;
                            }
                        }
                        else if(y-1 >= 0 && virtualImage.at<cv::Vec3b>(y - 1, x) != holePixel)
                        {
                            VIPointMap::iterator vPItr = vIPointMap.find({y - 1, x});
                            if(vPItr == vIPointMap.end())
                                continue;
                            VIPointProperty& vIPoint = vPItr->second;
                            QuadTreeProblemMap::iterator Itr = problem_map.find(vIPoint.srcImageKey);
                            MLA_Problem& problem = Itr->second;
                            if(vIPoint.srcImageY + 1 < problem.m_Image_rgb.rows)
                            {
                                virtualImage.at<cv::Vec3b>(y, x) = problem.m_Image_rgb.at<cv::Vec3b>(vIPoint.srcImageY + 1, vIPoint.srcImageX);
                                filledPoint.srcImageX = vIPoint.srcImageX;
                                filledPoint.srcImageY = vIPoint.srcImageY + 1;
                                filled = true;
                            }
                            else
                            {
                                continue;
                            }
                        }
                        else
                        {
                            continue;
                        }
                        if(filled)
                        {
                            counter ++;
                            filledCounter ++;
                            pixelCount ++;
                            filledPoint.srcImageKey = m_vIPoint.srcImageKey;
                            vIPointMap[{y, x}] = filledPoint;
                        }
                    }
                    else
                    {
                        continue;

                    }
                }
            }
            float FillRatio = static_cast<float>(pixelCount) / static_cast<float>(virtualImage.rows * virtualImage.cols) ;
            if(FillRatio >= targetFillRatio)
            {
                bstop = false;
                std::cout<<"FillRatio >= targetFillRatio，FillRatio="<<FillRatio<<std::endl;
            }
            else if(filledCounter==0)
            {
                bstop = false;
                std::cout<<"filledCounter==0，counter="<<counter<<std::endl;
            } else if(counter > 30000)
            {
                bstop = false;
                std::cout<<"counter="<<counter<<std::endl;
            }
        }
    }

    bool LFRefocus::LoadCoeffsFromXml(const std::string& xmlPath,std::array<double, 3>& behaviorModelParams)
    {
        cv::FileStorage fs(xmlPath, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;

        cv::FileNode n1 = fs["c0"], n2 = fs["c1"], n3 = fs["c2"];
        if (n1.empty() || n2.empty() || n3.empty()) return false;
        behaviorModelParams[0] = (double)n1;
        behaviorModelParams[1] = (double)n2;
        behaviorModelParams[2] = (double)n3;
        return true;
    }

    float LFRefocus::ConvertVdToRd(float v_depth, std::array<double, 3>& behaviorModelParams)
    {
        double c0 = behaviorModelParams[0];
        double c1 = behaviorModelParams[1];
        double c2 = behaviorModelParams[2];

        double v = static_cast<double>(v_depth);
        double denom = 1.0 - v * c0;

        // 避免除 0 / 数值爆炸
        if (std::fabs(denom) < 1.0e-12)
            return 0.0f;

        const double z = (v * c1 + c2) / denom;

        if (!std::isfinite(z) || z < 0.0)
            return 0.0f;

        return static_cast<float>(z);
    }

    float LFRefocus::ConvertVdToRdSegment(float v_depth)
    {
        struct BehaviorModelSegment
        {
            double depth_min;
            double depth_max;
            std::array<double,3> params;
        };
        std::vector<BehaviorModelSegment> behaviorModels;
        BehaviorModelSegment  seg;

        std::string m_strRootPath = m_ptrDepthSolver->GetRootPath();
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME;
        std::string BehaviorModelFile = strCalibPath + "behaviorModelParamsSegment.xml";
        cv::FileStorage fs(BehaviorModelFile, cv::FileStorage::READ);
        cv::FileNode segments = fs["BehaviorModelSegments"];

        for (auto it = segments.begin(); it != segments.end(); ++it)
        {
            (*it)["DepthMin"] >> seg.depth_min;
            (*it)["DepthMax"] >> seg.depth_max;

            cv::FileNode param = (*it)["Param"];

            param["c0"] >> seg.params[0];
            param["c1"] >> seg.params[1];
            param["c2"] >> seg.params[2];

            behaviorModels.push_back(seg);
        }

        double z = 0.0;
        // 分段使用行为模型
        for (const auto& seg : behaviorModels)
        {
            if (v_depth >= seg.depth_min && v_depth < seg.depth_max)
            {
                double c0 = seg.params[0];
                double c1 = seg.params[1];
                double c2 = seg.params[2];

                double v = static_cast<double>(v_depth);
                // 避免除 0 / 数值爆炸
                double denom = 1.0 - v * c0;
                if (std::fabs(denom) < 1.0e-12)
                    return 0.0f;
                z = (v * c1 + c2) / denom;
                break;
            }
        }

        if (!std::isfinite(z) || z < 0.0)
            return 0.0f;

        return static_cast<float>(z);
    }

    /*void LFRefocus::CollectPointNeigInfo(int64_t p_index, Vec3f& p_coord, NeighborCorrInfoMap& neigInfoMap)
    {
        PointPacket packet;
        packet.m_Point3D = p_coord;
        packet.m_NeigInfoMap = neigInfoMap;
        m_PointPackMap[p_index] = packet;
    }*/

  /*  bool LFRefocus::WritePointAndNeigInfo()
    {
       if (m_PointPackMap.empty())
          return false;

        std::string PointBin_FullPath = m_strSavePath + std::string("/pointsBin/");
        PointBin_FullPath = PointBin_FullPath + "Point3d_neigInfo.bin";

        char buf[256];
        std::ofstream fin(PointBin_FullPath);
        // total points
        sprintf(buf, "total %d\n", m_PointPackMap.size());
        fin.write(buf, strlen(buf));

        PointPacketMap::iterator itr = m_PointPackMap.begin();
        for ( ; itr != m_PointPackMap.end(); itr++)
        {
            PointPacket& pointPacket = itr->second;
            // write point
            Vec3f& point = pointPacket.m_Point3D;
            sprintf(buf, "%lf %lf %lf,", point[0], point[1], point[2]);
            fin.write(buf, strlen(buf));
            // traverse and write neighbor info
            NeighborCorrInfoMap& neigInfoMap = pointPacket.m_NeigInfoMap;
            NeighborCorrInfoMap::iterator itrN = neigInfoMap.begin();
            for ( ; itrN != neigInfoMap.end(); itrN++)
            {
                QuadTreeTileKeyPtr ptrNeigKey = itrN->first;
                NeighborCorrespondingInfo& negInfo = itrN->second;
                sprintf(buf, "%s %i %i %lf,", ptrNeigKey->StrRemoveLOD().c_str(), negInfo.m_Point2d[0],
                    negInfo.m_Point2d[1], negInfo.m_Contribute);
                fin.write(buf, strlen(buf));
            }
            sprintf(buf, "\n");
            fin.write(buf, strlen(buf));
        }
        fin.close();
        m_PointPackMap.clear();
        return true;
    }*/

}



#include "LFRefocus.h"

