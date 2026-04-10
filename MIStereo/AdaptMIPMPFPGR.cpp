/********************************************************************
file base:      AdaptMIPMPFPGR.h
author:         LZD
created:        2025/07/12
purpose:        微图像的视差计算
*********************************************************************/

#include "AdaptMIPMPFPGR.h"

namespace LFMVS
{
    AdaptMIPMPFPGR::AdaptMIPMPFPGR(LightFieldParams& params)
        :AdaptMIPM(params)
    {
        neighbor_PGR_host = nullptr;
        neighbor_PGR_cuda = nullptr;
        proxy_plane_host = nullptr;
        proxy_plane_cuda = nullptr;
    }

    AdaptMIPMPFPGR::~AdaptMIPMPFPGR()
    {
        if (neighbor_PGR_host != nullptr)
        {
            delete[] neighbor_PGR_host;
        }
        if (neighbor_PGR_cuda != nullptr)
        {
            cudaFree(neighbor_PGR_cuda);
        }

        if (proxy_plane_host != nullptr)
        {
            delete[] proxy_plane_host;
        }
        if (proxy_plane_cuda != nullptr)
        {
            cudaFree(proxy_plane_cuda);
        }
    }

    void AdaptMIPMPFPGR::Initialize(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                        QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                        std::vector<float>& costVec)
    {
        // reset
        images_MI.clear();
        blur_images_MI.clear();

        // 邻居数量
        int neighsCount = problem.m_NeighsSortVecForMatch.size();
        // TODO: LZD test
        if (neighsCount > 15)
        {
            neighsCount = 15;
        }

        cv::Mat image_float;
        // 找到参考图像索引
        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey; // 当前微图像的key
        QuadTreeTileInfoMap::iterator itr_MLA = MLA_info_map.find(ptrKey);
        if (itr_MLA == MLA_info_map.end())
        {
            std::cout<<"InLF_TileKey: current key not Find! " << ptrKey->GetTileX() << ", " <<ptrKey->GetTileY() << std::endl;
            return;
        }
        problem.m_Image_gray.convertTo(image_float, CV_32FC1);
        images_MI.push_back(image_float); // 存入当前微图像的灰度图
        cv::Mat blur_image_float;
        problem.m_Image_Blureness.convertTo(blur_image_float, CV_32FC1);
        blur_images_MI.push_back(blur_image_float);

        // 存储当前微透镜图像的灰度图及其中心点坐标
        centerPointS_MI = new float2[neighsCount+1];
        float2 center_point_MI;
        MLA_InfoPtr ptrInfo = itr_MLA->second;
        center_point_MI.x = ptrInfo->GetCenter().x;
        center_point_MI.y = ptrInfo->GetCenter().y;
        centerPointS_MI[0] = center_point_MI;

        plane_hypotheses_host = new float4[image_float.rows * image_float.cols];
        const int mi_width = m_ParamsCUDA.mi_width_for_match;
        const int mi_height = m_ParamsCUDA.mia_height_for_match;
        if (planeVec.size() == mi_width*mi_height)
        {
            for (int row = 0; row < mi_height; ++row)
            {
                for (int col = 0; col < mi_width; ++col)
                {
                    float depth_value = planeVec[row*mi_width+col].w;
                    float4 f_plane = planeVec[row*mi_width+col];
                    // 判断随机视差 选择平面
                    const auto x = lround(col - depth_value);
                    if (x >= 0 && costVec[row*mi_width+col] < 0.3)
                    {
                        plane_hypotheses_host[row*mi_width+col] = f_plane;
                    }
                    else
                    {
                        plane_hypotheses_host[row*mi_width+col].w = 0;
                    }
                }
            }
        }
        else
        {
            for (int row = 0; row < mi_height; ++row)
            {
                for (int col = 0; col < mi_width; ++col)
                {
                    plane_hypotheses_host[row*mi_width+col].w = 0;
                }
            }
        }
        costs_host = new float[image_float.rows * image_float.cols];
        neighbor_patchFill_host = new int3[image_float.rows * image_float.cols];
        neighbor_PGR_host = new int3[image_float.rows * image_float.cols];
        disp_baseline_host = new float4[image_float.rows * image_float.cols];

        selected_views_host = new unsigned int[image_float.rows * image_float.cols];
        rand_states_host = new curandState[image_float.rows * image_float.cols];

        // TODO: LZD 初始化邻居：灰度图及其中心点坐标
        // TODO：根据模糊程度、纹理丰富性、基线等因子，确定匹配的优先顺序
        for (size_t i = 0; i < neighsCount; ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_NeighsSortVecForMatch[i];
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeigKey);
            if (itrNP == problem_map.end())
            {
                std::cout<<"InLF_TileKey: neig image not Find! " << ptrNeigKey->GetTileX()
<< ", " << ptrNeigKey->GetTileY() << std::endl;
                continue;
            }

            cv::Mat image_float;
            itrNP->second.m_Image_gray.convertTo(image_float, CV_32FC1);
            images_MI.push_back(image_float); // 压入到数据队列
            cv::Mat blur_image_float;
            itrNP->second.m_Image_Blureness.convertTo(blur_image_float, CV_32FC1);
            blur_images_MI.push_back(blur_image_float);

            // std::string strTmp = "/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult";
            // std::stringstream ss;
            // ss << strTmp <<"/"<<i<<".png";
            // strTmp = ss.str();
            // cv::imwrite(strTmp, itrNP->second.m_Image_Blureness_Bianry);

            QuadTreeTileInfoMap::iterator itrNP_MLA = MLA_info_map.find(ptrNeigKey);
            if (itrNP_MLA == MLA_info_map.end())
            {
                std::cout<<"InLF_TileKey: neig key not Find! " << ptrNeigKey->GetTileX()
                << ", " << ptrNeigKey->GetTileY() << std::endl;
                continue;
            }

            MLA_InfoPtr ptrNeig_MLA = itrNP_MLA->second;
            float2 neig_image_point;
            neig_image_point.x = ptrNeig_MLA->GetCenter().x;
            neig_image_point.y = ptrNeig_MLA->GetCenter().y;
            centerPointS_MI[i+1] = neig_image_point; // 邻居MI的中心点坐标
        }

        TestWriteMIBeforeGPU();

        // TODO: LZD CPU--->GPU
        InitializeGPU_collect();
    }

    void AdaptMIPMPFPGR::InitializeGPU_collect()
    {
        InitGPUParamsFromCPUParams();

        if (images_MI.empty() || blur_images_MI.empty())
        {
            std::cout<<"AdaptMIPM, InitializeGPU: images_MI is EMPTY!"<<std::endl;
            return;
        }

        int image_rows = images_MI[0].rows; // 行
        int image_cols = images_MI[0].cols; // 列
        if (image_rows == 0 || image_cols == 0)
        {
            std::cout<<"reference image size is EMPTY!"<<std::endl;
            return;
        }

        // 初始化
        for (int i = 0; i < params.num_images; ++i)
        {
            CreateGrayImageObject(i);
            CreateBlurImageObject(i);
        }
        cudaError_t error_toc = cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        if (error_toc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_toc" << std::endl;
            return;
        }
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        cudaError_t error_cc = cudaMalloc((void**)&centers_cuda, sizeof(float2) * (params.num_images));
        if (error_cc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_cc" << std::endl;
            return;
        }
        cudaMemcpy(centers_cuda, centerPointS_MI, params.num_images * sizeof(float2), cudaMemcpyHostToDevice);

        cudaError_t error_phc = cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (image_rows * image_cols));
        if (error_phc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_phc" << std::endl;
            return;
        }
        cudaMalloc((void**)&costs_cuda, sizeof(float)*(image_rows*image_cols));
        cudaMalloc((void**)&neighbor_patchFill_cuda, sizeof(int3)*(image_rows*image_cols));
        cudaMalloc((void**)&neighbor_PGR_cuda, sizeof(int3)*(image_rows*image_cols));
        cudaMalloc((void**)&disp_baseline_cuda, sizeof(float4)*(image_rows*image_cols));

        cudaError_t error_rsc = cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (image_rows * image_cols));
        if (error_rsc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_rsc" << std::endl;
            return;
        }
        cudaError_t error_svc = cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (image_rows * image_cols));
        if (error_svc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_svc" << std::endl;
            return;
        }
        cudaError_t error_dc = cudaMalloc((void**)&depths_cuda, sizeof(float) * (image_rows * image_cols)); // Updated by Qingshan 2020-01-15
        if (error_dc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_dc" << std::endl;
        }
    }

    void AdaptMIPMPFPGR::Initialize_SecondStage(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                        QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                        std::vector<float>& costVec,
                        Proxy_DisPlane* proxy_Plane,
                        StereoResultInfo* pStereoResult)
    {
        // reset
        images_MI.clear();
        blur_images_MI.clear();

        // 邻居数量
        int neighsCount = problem.m_NeighsSortVecForMatch.size();
        // TODO: LZD test
        if (neighsCount > 15)
        {
            neighsCount = 15;
        }

        cv::Mat image_float;
        // 找到参考图像索引
        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey; // 当前微图像的key
        QuadTreeTileInfoMap::iterator itr_MLA = MLA_info_map.find(ptrKey);
        if (itr_MLA == MLA_info_map.end())
        {
            std::cout<<"InLF_TileKey: current key not Find! " << ptrKey->StrRemoveLOD()<< std::endl;
            return;
        }

        //plane_hypotheses_host = new float4[image_float.rows * image_float.cols];
        const int mi_width = m_ParamsCUDA.mi_width_for_match;
        const int mi_height = m_ParamsCUDA.mia_height_for_match;

        /////////////////////////////////////////////////////////////////////
        proxy_plane_host = new Proxy_DisPlane[mi_width*mi_height];
        // cpu内存复制
        memcpy(proxy_plane_host, proxy_Plane, (mi_width*mi_height)*sizeof(Proxy_DisPlane));

        plane_hypotheses_host = new float4[mi_width*mi_height];
        // cpu内存复制
        memcpy(plane_hypotheses_host, pStereoResult->plane_hypotheses_host, (mi_width*mi_height)*sizeof(float4));

        costs_host = new float[mi_width*mi_height];
        // cpu内存复制
        memcpy(costs_host , pStereoResult->costs_host, (mi_width*mi_height)*sizeof(float));

        rand_states_host = new curandState[mi_width*mi_height];
        // cpu内存复制
        memcpy(rand_states_host , pStereoResult->rand_states_host,(mi_width*mi_height)*sizeof(curandState));

        neighbor_patchFill_host = new int3[mi_width*mi_height];
        // cpu内存复制
        memcpy(neighbor_patchFill_host , pStereoResult->neighbor_patchFill_host,(mi_width*mi_height)*sizeof(int3));

        disp_baseline_host = new float4[mi_width*mi_height];
        // cpu内存复制
        memcpy(disp_baseline_host , pStereoResult->disp_baseline_host,(mi_width*mi_height)*sizeof(float4));

        selected_views_host = new unsigned int[mi_width*mi_height];
        // cpu内存复制
        memcpy(selected_views_host , pStereoResult->selected_views_host,(mi_width*mi_height)*sizeof(unsigned int));
        /////////////////////////////////////////////////////////////////////

        // 存储当前微透镜图像：灰度图及其中心点坐标
        problem.m_Image_gray.convertTo(image_float, CV_32FC1);
        images_MI.push_back(image_float); // 存入当前微图像的灰度图
        cv::Mat blur_image_float;
        problem.m_Image_Blureness.convertTo(blur_image_float, CV_32FC1);
        blur_images_MI.push_back(blur_image_float);
        centerPointS_MI = new float2[neighsCount+1];
        float2 center_point_MI;
        MLA_InfoPtr ptrInfo = itr_MLA->second;
        center_point_MI.x = ptrInfo->GetCenter().x;
        center_point_MI.y = ptrInfo->GetCenter().y;
        centerPointS_MI[0] = center_point_MI;
        // 初始化邻居：灰度图及其中心点坐标
        for (size_t i = 0; i < neighsCount; ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_NeighsSortVecForMatch[i];
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeigKey);
            if (itrNP == problem_map.end())
            {
                std::cout<<"InLF_TileKey: neig image not Find! " << ptrNeigKey->StrRemoveLOD()<< std::endl;
                continue;
            }

            cv::Mat image_float_neig;
            itrNP->second.m_Image_gray.convertTo(image_float_neig, CV_32FC1);
            images_MI.push_back(image_float_neig); // 压入到数据队列
            cv::Mat blur_image_float_neig;
            itrNP->second.m_Image_Blureness.convertTo(blur_image_float_neig, CV_32FC1);
            blur_images_MI.push_back(blur_image_float_neig);

            QuadTreeTileInfoMap::iterator itrNP_MLA = MLA_info_map.find(ptrNeigKey);
            if (itrNP_MLA == MLA_info_map.end())
            {
                std::cout<<"InLF_TileKey: neig key not Find! " << ptrNeigKey->StrRemoveLOD()<< std::endl;
                continue;
            }

            MLA_InfoPtr ptrNeig_MLA = itrNP_MLA->second;
            float2 neig_image_point;
            neig_image_point.x = ptrNeig_MLA->GetCenter().x;
            neig_image_point.y = ptrNeig_MLA->GetCenter().y;
            centerPointS_MI[i+1] = neig_image_point; // 邻居MI的中心点坐标
        }
        TestWriteMIBeforeGPU();
        // TODO: LZD CPU--->GPU
        InitializeGPU_new();
    }

    int3 AdaptMIPMPFPGR::GetNeighborPGR(const int index)
    {
        return neighbor_PGR_host[index];
    }

    void AdaptMIPMPFPGR::InitializeGPU_new()
    {
        InitGPUParamsFromCPUParams();

        if (images_MI.empty() || blur_images_MI.empty())
        {
            std::cout<<"AdaptMIPMPFPGR, InitializeGPU_new: images_MI is EMPTY!"<<std::endl;
            return;
        }

        int image_rows = images_MI[0].rows; // 行
        int image_cols = images_MI[0].cols; // 列
        if (image_rows == 0 || image_cols == 0)
        {
            std::cout<<"reference image size is EMPTY!"<<std::endl;
            return;
        }

        // 图像数据和中心点坐标
        for (int i = 0; i < params.num_images; ++i)
        {
            CreateGrayImageObject(i);
            CreateBlurImageObject(i);
        }
        cudaError_t error_toc = cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        if (error_toc != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_toc" << std::endl;
            return;
        }
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
        cudaError_t error_cc = cudaMalloc((void**)&centers_cuda, sizeof(float2) * (params.num_images));
        if (error_cc != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_cc" << std::endl;
            return;
        }
        cudaMemcpy(centers_cuda, centerPointS_MI, params.num_images * sizeof(float2), cudaMemcpyHostToDevice);

        ///////////////////////////////////////////////////////////////////////////
        cudaError_t error_phc = cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4)*(image_rows*image_cols));
        if (error_phc != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_phc" << std::endl;
            return;
        }
        // cpu-gpu
        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        cudaError_t error_cost = cudaMalloc((void**)&costs_cuda, sizeof(float)*(image_rows*image_cols));
        if (error_cost != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_cost" << std::endl;
            return;
        }
        // cpu-gpu
        cudaMemcpy(costs_cuda, costs_host, sizeof(float)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        cudaError_t error_rsc = cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (image_rows * image_cols));
        if (error_rsc != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_rsc" << std::endl;
            return;
        }
        // cpu-gpu
        cudaMemcpy(rand_states_cuda, rand_states_host, sizeof(curandState)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        cudaError_t error_proxy_patch = cudaMalloc((void**)&neighbor_patchFill_cuda, sizeof(int3)*(image_rows*image_cols));
        if (error_proxy_patch != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_proxy_patch" << std::endl;
            return;
        }
        // cpu-gpu
        cudaMemcpy(neighbor_patchFill_cuda, neighbor_patchFill_host, sizeof(int3)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        cudaError_t er_disp_baseline = cudaMalloc((void**)&disp_baseline_cuda, sizeof(float4)*(image_rows*image_cols));
        if (er_disp_baseline != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: er_disp_baseline" << std::endl;
            return;
        }
        cudaMemcpy(disp_baseline_cuda, disp_baseline_host, sizeof(float4)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        cudaError_t error_svc = cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int)*(image_rows*image_cols));
        if (error_svc != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_svc" << std::endl;
            return;
        }
        // cpu-gpu
        cudaMemcpy(selected_views_cuda, selected_views_host, sizeof(unsigned int)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        cudaError_t error_proxy_p = cudaMalloc((void**)&proxy_plane_cuda, sizeof(Proxy_DisPlane)*(image_rows*image_cols));
        if (error_proxy_p != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_proxy_p" << std::endl;
            return;
        }
        // cpu-gpu
        cudaMemcpy(proxy_plane_cuda, proxy_plane_host, sizeof(Proxy_DisPlane)*(image_rows*image_cols), cudaMemcpyHostToDevice);

        // other
        cudaMalloc((void**)&neighbor_PGR_cuda, sizeof(int3)*(image_rows*image_cols));
        cudaError_t error_dc = cudaMalloc((void**)&depths_cuda, sizeof(float)*(image_rows*image_cols)); // Updated by Qingshan 2020-01-15
        if (error_dc != cudaSuccess)
        {
            std::cout<<"InitializeGPU_new: error_dc" << std::endl;
        }
    }

    void AdaptMIPMPFPGR::TestWritePF_PGRInfo()
    {
        for (int i=0; i < params.MLA_Mask_Width_Cuda; i++)
        {
            for (int j=0; j < params.MLA_Mask_Height_Cuda; j++)
            {
                int center = j*params.MLA_Mask_Width_Cuda+i;
                int3 patchFill = neighbor_patchFill_host[center];
                int3 pgr = neighbor_PGR_host[center];
                // printf("index=%d, pF:(%i, %i,%i),pgr:(%i, %i,%i)\n", center,
                //     patchFill.x, patchFill.y, patchFill.z, pgr.x, pgr.y, pgr.z);
            }
        }
    }

    void AdaptMIPMPFPGR::TestWriteNeighbour()
    {
        int width = params.MLA_Mask_Width_Cuda+5;
        int height = params.MLA_Mask_Height_Cuda+5;
        cv::Mat image = cv::Mat::zeros(width, height, CV_32F);
        for (int row = 0; row < params.MLA_Mask_Height_Cuda; row++)
        {
            for (int col = 0; col < params.MLA_Mask_Width_Cuda; col++)
            {
                int center = row*params.MLA_Mask_Width_Cuda + col;
                int3 patchFill = neighbor_patchFill_host[center];
                if(patchFill.x<=0)
                {
                    image.at<float>(row, col) = 0.0f;
                }
                else
                {
                    image.at<float>(row, col) = 255.0f;
                }

            }
        }
        cv::imwrite("/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/MLA_DMaps/tt.png",
            image);
    }

    void AdaptMIPMPFPGR::TestWriteNeighbour_color(MLA_Problem& problem, QuadTreeProblemMap& problem_map)
    {
        cv::Mat ref_image = problem.m_Image_rgb;
        cv::imwrite("/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/MLA_DMaps/tt-ref.png",
    ref_image);
        std::vector<cv::Mat>  image_color_vector;
        for (size_t i = 0; i < problem.m_NeighsSortVecForMatch.size(); ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_NeighsSortVecForMatch[i];
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeigKey);
            if (itrNP == problem_map.end())
            {
                std::cout<<"InLF_TileKey: neig image not Find! " << ptrNeigKey->StrRemoveLOD()<< std::endl;
                continue;
            }
            image_color_vector.push_back(itrNP->second.m_Image_rgb);
        }
        int radius = params.patch_size / 2; // 5
        int width = params.MLA_Mask_Width_Cuda+5;
        int height = params.MLA_Mask_Height_Cuda+5;
        cv::Mat image = cv::Mat::zeros(width, height, CV_8UC3);
        for (int row = 2; row < params.MLA_Mask_Height_Cuda+5; row++)
        {
            for (int col = 2; col < params.MLA_Mask_Width_Cuda+5; col++)
            {
                int center = (row-2)*params.MLA_Mask_Width_Cuda + col-2;

                if (col > 2 && col<params.MLA_Mask_Width_Cuda+2 &&
                    row > 2 && row<params.MLA_Mask_Height_Cuda+2)
                {
                    image.at<cv::Vec3b>(row, col) = ref_image.at<cv::Vec3b>(row-2, col-2);
                }
                else if (col-2 == 0 ) // && row%3==0
                {
                    int3 patchFill = neighbor_patchFill_host[center];
                    if(patchFill.x > 0 && patchFill.x<image_color_vector.size())
                    {

                        //printf("patch: %d, (x,y)= %d,%d\n", patchFill.x, patchFill.y, patchFill.z);
                        cv::Mat neig_image = image_color_vector[patchFill.x-1];
                        const int2 proxy_src_p = make_int2(patchFill.y, patchFill.z);
                        int2 proxy_src_pt = make_int2(proxy_src_p.x, proxy_src_p.y);
                        cv::Vec3b neig_pixel = neig_image.at<cv::Vec3b>(proxy_src_pt.y, proxy_src_pt.x);
                        image.at<cv::Vec3b>(row, col) = neig_pixel;
                        //
                        // for (int i = -radius; i < radius + 1; i += params.radius_increment) // x
                        // {
                        //     for (int j = -radius; j < radius + 1; j += params.radius_increment) // y
                        //     {
                        //         int2 pt = make_int2(col-2+i, row-2+j);
                        //         if (pt.x >= params.MLA_Mask_Width_Cuda || pt.x < 0.0f
                        //             || pt.y >= params.MLA_Mask_Height_Cuda || pt.y < 0.0f)
                        //         {
                        //             int dx = pt.x-col+2;
                        //             int dy = pt.y-row+2;
                        //             int2 proxy_src_pt = make_int2(proxy_src_p.x+dx, proxy_src_p.y+dy);
                        //             cv::Vec3b neig_pixel = neig_image.at<cv::Vec3b>(proxy_src_pt.y, proxy_src_pt.x);
                        //             image.at<cv::Vec3b>(row, col) = neig_pixel;
                        //         }
                        //     }
                        // }
                    }
                }
            }
        }
        std::string strName = "/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/MLA_DMaps/";
        strName += problem.m_ptrKey->StrRemoveLOD();
        cv::imwrite(strName+"tt-color.png",image);
    }
}
