/********************************************************************
file base:      AdaptMIPMAlg.cpp
author:         LZD
created:        2025/06/07
purpose:        微图像的视差计算
*********************************************************************/
#include "AdaptMIPM.h"

#include <sstream>

#include "CudaUtil.h"
#include "Common/PTotEstimator.h"
#include "MIStereo/AdaptMIPMUtil.cuh" // 为SetHexPerCallPTot()
#include "Util/Logger.h"

namespace LFMVS
{
    AdaptMIPM::AdaptMIPM(LightFieldParams& params)
    : m_ParamsCUDA(params), m_bReleased(false)
    {
    }

    AdaptMIPM::~AdaptMIPM()
    {
        if (!m_bReleased) {
            delete[] plane_hypotheses_host;
            delete[] costs_host;
            delete[] neighbor_patchFill_host;
            delete[] disp_baseline_host;

            for (int i = 0; i < params.num_images; ++i)
            {
                cudaDestroyTextureObject(texture_objects_host.images[i]);
                cudaFreeArray(cuArray[i]);
                cudaDestroyTextureObject(texture_objects_host.blur_images[i]);
                cudaFreeArray(cu_blur_Array[i]);
            }

            cudaFree(cameras_cuda);
            cudaFree(texture_objects_cuda);
            cudaFree(centers_cuda);
            cudaFree(tilekeys_cuda);

            if (d_nei_lin)
                cudaFree(d_nei_lin);
            if (d_kSteps)
                cudaFree(d_kSteps);
            if (d_pTot)
                cudaFree(d_pTot);

            cudaFree(plane_hypotheses_cuda);
            cudaFree(costs_cuda);
            cudaFree(neighbor_patchFill_cuda);
            cudaFree(rand_states_cuda);
            cudaFree(disp_baseline_cuda);

            cudaFree(selected_views_cuda);
            cudaFree(depths_cuda);

            //if (params.geom_consistency)
            //{
                //for (int i = 0; i < params.num_images; ++i)
                //{
                    //cudaDestroyTextureObject(texture_depths_host.images[i]);
                    //cudaFreeArray(cuDepthArray[i]);
               // }
                //cudaFree(texture_depths_cuda);
            //}

            if (params.planar_prior)
            {
                delete[] prior_planes_host;
                delete[] plane_masks_host;

                cudaFree(prior_planes_cuda);
                cudaFree(plane_masks_cuda);
            }
        }
    }

    void AdaptMIPM::ReleaseMemory()
    {
        if (m_bReleased) {
            return;
        }
        
        m_bReleased = true;
        
        images_MI.clear();
        blur_images_MI.clear();
        delete [] centerPointS_MI;
        delete [] tileKeyS_MI;
        //depths.clear();
        //cameras.clear();

        delete[] plane_hypotheses_host;
        delete[] costs_host;
        delete[] neighbor_patchFill_host;
        delete[] disp_baseline_host;

        for (int i = 0; i < params.num_images; ++i)
        {
            cudaDestroyTextureObject(texture_objects_host.images[i]);
            cudaFreeArray(cuArray[i]);
            cudaDestroyTextureObject(texture_objects_host.blur_images[i]);
            cudaFreeArray(cu_blur_Array[i]);
        }

        cudaFree(cameras_cuda);
        cudaFree(texture_objects_cuda);
        cudaFree(centers_cuda);
        cudaFree(tilekeys_cuda);

        if (d_nei_lin)
            cudaFree(d_nei_lin);
        if (d_kSteps)
            cudaFree(d_kSteps);
        if (d_pTot)
            cudaFree(d_pTot);

        cudaFree(plane_hypotheses_cuda);
        cudaFree(costs_cuda);
        cudaFree(neighbor_patchFill_cuda);
        cudaFree(rand_states_cuda);
        cudaFree(disp_baseline_cuda);

        cudaFree(selected_views_cuda);
        cudaFree(depths_cuda);

        if (params.planar_prior)
        {
            delete[] prior_planes_host;
            delete[] plane_masks_host;

            cudaFree(prior_planes_cuda);
            cudaFree(plane_masks_cuda);
        }
    }

    void AdaptMIPM::SetTileKey(int tile_x, int tile_y)
    {
        m_tile_x = tile_x;
        m_tile_y = tile_y;
    }

    void AdaptMIPM::Initialize(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                        QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                        std::vector<float>& costVec)
    {
        // reset
        images_MI.clear();
        blur_images_MI.clear();

        // 邻居数量
        int neighsCount = problem.m_NeighsSortVecForMatch.size();
        // TODO: LZD test
        if (neighsCount > 8) // 14
        {
            neighsCount = 8;
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
        tileKeyS_MI = new int2[neighsCount+1];
        float2 center_point_MI;
        MLA_InfoPtr ptrInfo = itr_MLA->second;
        center_point_MI.x = ptrInfo->GetCenter().x;
        center_point_MI.y = ptrInfo->GetCenter().y;
        centerPointS_MI[0] = center_point_MI;
        int2 ref_key;
        ref_key.x = ptrKey->GetTileX();
        ref_key.y = ptrKey->GetTileY();
        tileKeyS_MI[0] = ref_key;

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
        disp_baseline_host = new float4[image_float.rows * image_float.cols];

        selected_views_host = new unsigned int[image_float.rows * image_float.cols];

        // TODO: LZD 初始化邻居：灰度图及其中心点坐标
        // TODO：根据模糊程度、纹理丰富性、基线等因子，确定匹配的优先顺序
        for (size_t i = 0; i < neighsCount; ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_NeighsSortVecForMatch[i];
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeigKey);
            if (itrNP == problem_map.end())
            {
                LOG_ERROR("InLF_TileKey: neig image not Find! ", ptrNeigKey->StrRemoveLOD().c_str());
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
                std::cout<<"InLF_TileKey: neig key not Find! " << ptrNeigKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }

            MLA_InfoPtr ptrNeig_MLA = itrNP_MLA->second;
            float2 neig_image_point;
            neig_image_point.x = ptrNeig_MLA->GetCenter().x;
            neig_image_point.y = ptrNeig_MLA->GetCenter().y;
            centerPointS_MI[i+1] = neig_image_point; // 邻居MI的中心点坐标
            int2 neig_key;
            neig_key.x = ptrNeigKey->GetTileX();
            neig_key.y = ptrNeigKey->GetTileY();
            tileKeyS_MI[i+1] = neig_key;
        }

        ///////////////////////////////////////////
        if (0)
        {
            // 1) 组织“本参考的 slice”（参考 + 所有邻域中心）
            std::vector<float2> slice_centers;
            slice_centers.reserve(neighsCount + 1);
            slice_centers.push_back(centerPointS_MI[0]);
            std::vector<int2> neighKeys;
            neighKeys.reserve(neighsCount);
            for (size_t i=0; i<neighsCount; ++i){
                slice_centers.push_back(centerPointS_MI[i+1]);
                neighKeys.push_back(tileKeyS_MI[i+1]);
            }

            // 2) 用 PTotEstimator 直接构建三张小表（nei_lin/k/pTot）
            PTotEstimator est(params.MLA_Mask_Width_Cuda, params.MLA_Mask_Height_Cuda, params.Base);
            std::vector<int>   nei_lin;
            std::vector<int>   kSteps;
            std::vector<float> pTot;

            est.buildPerCallTablesFromSlice(/*tilekey_ref*/ tileKeyS_MI[0],
                                            /*neighbors  */ neighKeys,
                                            /*slice      */ slice_centers,
                                            /*out*/ nei_lin, kSteps, pTot);

            // 3) 下发 CUDA（每次参考一小份）
            CUDA_SAFE_CALL(cudaMalloc(&d_nei_lin,  nei_lin.size()*sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&d_kSteps,   kSteps.size()*sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&d_pTot,     pTot.size()*sizeof(float)));

            CUDA_SAFE_CALL(cudaMemcpy(d_nei_lin, nei_lin.data(), nei_lin.size()*sizeof(int),   cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_kSteps,  kSteps.data(),  kSteps.size()*sizeof(int),    cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_pTot,    pTot.data(),    pTot.size()*sizeof(float),    cudaMemcpyHostToDevice));

            const int ref_lin = est.lin(tileKeyS_MI[0]);
            SetHexPerCallPTot(d_nei_lin, d_pTot, d_kSteps, (int)nei_lin.size(), ref_lin);
        }
        ///////////////////////////////////////////

        TestWriteMIBeforeGPU();

        // TODO: LZD CPU--->GPU
        InitializeGPU();
    }

    void AdaptMIPM::Initialize_old(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                        QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                        std::vector<float>& costVec)
    {
        // reset
        images_MI.clear();
        blur_images_MI.clear();

        // 邻居数量
        int neighsCount = problem.m_Res_Image_KeyVec.size();

        // TODO: LZD test
        // if (neighsCount > 8)
        // {
        //     neighsCount = 8;
        // }

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
        disp_baseline_host = new float4[image_float.rows * image_float.cols];

        selected_views_host = new unsigned int[image_float.rows * image_float.cols];

        // TODO: LZD 初始化邻居：灰度图及其中心点坐标
        // TODO：根据模糊程度、纹理丰富性、基线等因子，确定匹配的优先顺序
        for (size_t i = 0; i < neighsCount; ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_Res_Image_KeyVec[i].m_ptrKey;
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeigKey);
            if (itrNP == problem_map.end())
            {
                std::cout<<"InLF_TileKey: neig image not Find! " << ptrNeigKey->StrRemoveLOD() << std::endl;
                continue;
            }

            cv::Mat image_float;
            itrNP->second.m_Image_gray.convertTo(image_float, CV_32FC1);
            images_MI.push_back(image_float); // 压入到数据队列
            cv::Mat blur_image_float;
            itrNP->second.m_Image_Blureness.convertTo(blur_image_float, CV_32FC1);
            blur_images_MI.push_back(blur_image_float);

            QuadTreeTileInfoMap::iterator itrNP_MLA = MLA_info_map.find(ptrNeigKey);
            if (itrNP_MLA == MLA_info_map.end())
            {
                std::cout<<"InLF_TileKey: neig MLA not Find! " << ptrNeigKey->StrRemoveLOD()<< std::endl;
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
        InitializeGPU();
    }

    void AdaptMIPM::TestWriteMIBeforeGPU()
    {
        return;
        for (int i=0; i<images_MI.size(); i++)
        {
            cv::Mat img = images_MI.at(i);
            std::stringstream ss;
            ss<<"/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/"<<i<<".png";
            std::string strPath = ss.str();
            cv::imwrite(strPath, img);
        }
    }

    float4 AdaptMIPM::GetPlaneHypothesis(const int index)
    {
        return plane_hypotheses_host[index];
    }

    float4* AdaptMIPM::GetPlaneHypothesisVector()
    {
        return plane_hypotheses_host;
    }

    float AdaptMIPM::GetCost(const int index)
    {
        return costs_host[index];
    }

    float* AdaptMIPM::GetCostVector()
    {
        return costs_host;
    }

    curandState* AdaptMIPM::GetRandStateVector()
    {
        return rand_states_host;
    }

    int3* AdaptMIPM::GetNeighborPatchFillVector()
    {
        return neighbor_patchFill_host;
    }

    int3 AdaptMIPM::GetNeighborPatch(const int index)
    {
        return neighbor_patchFill_host[index];
    }

    unsigned int* AdaptMIPM::GetSelected_view_vector()
    {
        return selected_views_host;
    }

    float4 AdaptMIPM::GetDisparityBaseline(const int index)
    {
        return disp_baseline_host[index];
    }

    float4* AdaptMIPM::GetDisparityBaselineVector()
    {
        return disp_baseline_host;
    }

    void AdaptMIPM::SetPlanarPriorParams()
    {
        params.planar_prior = true;
    }

    void AdaptMIPM::GetSupportPoints(std::vector<cv::Point>& support2DPoints)
    {
        support2DPoints.clear();
        const int step_size = 8;
        const int mi_width = m_ParamsCUDA.mi_width_for_match;
        const int mi_height = m_ParamsCUDA.mia_height_for_match;
        for (int col = 0; col < mi_width; col += step_size)
        {
            for (int row = 0; row < mi_height; row += step_size)
            {
                float min_cost = 2.0f;
                cv::Point temp_point;
                int c_bound = std::min(mi_width, col + step_size);
                int r_bound = std::min(mi_height, row + step_size);
                for (int c = col; c < c_bound; ++c)
                {
                    for (int r = row; r < r_bound; ++r)
                    {
                        int center = r * mi_width + c;
                        if (GetCost(center) < 2.0f && min_cost > GetCost(center))
                        {
                            temp_point = cv::Point(c, r);
                            min_cost = GetCost(center);
                        }
                    }
                }
                if (min_cost < 0.1f)
                {
                    support2DPoints.push_back(temp_point);
                }
            }
        }
    }

    std::vector<Triangle> AdaptMIPM::DelaunayTriangulation(const cv::Rect boundRC,
        const std::vector<cv::Point>& points)
    {
        if (points.empty())
            return std::vector<Triangle>();

        std::vector<Triangle> results;

        std::vector<cv::Vec6f> temp_results;
        cv::Subdiv2D subdiv2d(boundRC);
        for (const auto point : points)
        {
            subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
        }
        subdiv2d.getTriangleList(temp_results);

        for (const auto temp_vec : temp_results)
        {
            cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
            cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
            cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
            results.push_back(Triangle(pt1, pt2, pt3));
        }
        return results;
    }

    unsigned int AdaptMIPM::GetSelected_viewIndexs(const int pt_col, const int pt_row)
    {
        // pt_col和pt_row均从0开始起算
        int width = images_MI[0].cols;
        const int pt_oneDimen = pt_row * width + pt_col; // 将当前像素的二维数组表现形式转换为一维数组
        return selected_views_host[pt_oneDimen];
    }

    cv::Mat AdaptMIPM::GetReferenceImage()
    {
        return images_MI[0];
    }

    float3 AdaptMIPM::Get3DPointonRefCamLF(const int x, const int y, const float depth)
    {
        float3 pointX;
        // Reprojection
        pointX.x = x;
        pointX.y = y;
        pointX.z = depth;

        return pointX;
    }

    void AdaptMIPM::CreateGrayImageObject(int image_index)
    {
        int image_rows = images_MI[image_index].rows; // 行
        int image_cols = images_MI[image_index].cols; // 列
        cv::Mat& gray_image = images_MI[image_index];

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
                                            cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[image_index], &channelDesc, image_cols, image_rows);
        cudaMemcpy2DToArray(cuArray[image_index], 0, 0, gray_image.ptr<float>(),
                gray_image.step[0], image_cols*sizeof(float), image_rows,
                cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[image_index];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap; //  cudaAddressModeBorder
        texDesc.addressMode[1] = cudaAddressModeWrap; // cudaAddressModeBorder
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[image_index]), &resDesc, &texDesc, NULL);
    }

    void AdaptMIPM::CreateBlurImageObject(int image_index)
    {
        int image_rows = blur_images_MI[image_index].rows; // 行
        int image_cols = blur_images_MI[image_index].cols; // 列
        cv::Mat& blur_image = blur_images_MI[image_index];

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
                                            cudaChannelFormatKindFloat);
        cudaMallocArray(&cu_blur_Array[image_index], &channelDesc, image_cols, image_rows);
        cudaMemcpy2DToArray(cu_blur_Array[image_index], 0, 0, blur_image.ptr<float>(),
                blur_image.step[0], image_cols*sizeof(float), image_rows,
                cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cu_blur_Array[image_index];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap; //  cudaAddressModeBorder
        texDesc.addressMode[1] = cudaAddressModeWrap; // cudaAddressModeBorder
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.blur_images[image_index]),
                                &resDesc, &texDesc, NULL);
    }

    float4 AdaptMIPM::GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths)
    {
        float3 ptX1 = Get3DPointonRefCamLF(triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x));
        float3 ptX2 = Get3DPointonRefCamLF(triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x));
        float3 ptX3 = Get3DPointonRefCamLF(triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x));

        float x1 = ptX1.x;
        float y1 = ptX1.y;
        float z1 = ptX1.z;
        float x2 = ptX2.x;
        float y2 = ptX2.y;
        float z2 = ptX2.z;
        float x3 = ptX3.x;
        float y3 = ptX3.y;
        float z3 = ptX3.z;
        float4 n4;
        n4.x = (y3 - y1)*(z3 - z1) - (z2 -z1)*(y3 - y1);
        n4.y = (x3 - x1)*(z2 - z1) - (x2 - x1)*(z3 - z1);
        n4.z = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
        n4.w = -(n4.x * x1 + n4.y * y1 + n4.z * z1);

        return n4;
    }

    float AdaptMIPM::GetDepthFromPlaneParam_LF_Tilekey(const float4 plane_hypothesis,
        const int x, const int y)
    {
        return -(plane_hypothesis.x*x +plane_hypothesis.y*y +plane_hypothesis.w)/plane_hypothesis.z;
    }

    float AdaptMIPM::GetMinDepth()
    {
        return params.depth_min;
    }

    float AdaptMIPM::GetMaxDepth()
    {
        return params.depth_max;
    }

    void AdaptMIPM::CudaPlanarPriorInitialization_LF_Tilekey(const std::vector<float4>& planeParams,
        const cv::Mat_<float>& masks)
    {
        const int mi_width = m_ParamsCUDA.mi_width_for_match;
        const int mi_height = m_ParamsCUDA.mia_height_for_match;

        prior_planes_host = new float4[mi_height*mi_width];
        cudaMalloc((void**)&prior_planes_cuda, sizeof(float4)*(mi_height*mi_width));

        plane_masks_host = new unsigned int[mi_height*mi_width];
        cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int)*(mi_height*mi_width));

        for (int col = 0; col < mi_width; ++col)
        {
            for (int row = 0; row < mi_height; ++row)
            {
                int i_center = row * mi_width + col;
                plane_masks_host[i_center] = (unsigned int)masks(row, col);
                if (masks(row, col) > 0)
                {
                    prior_planes_host[i_center].x = planeParams[masks(row, col) - 1].x;
                    prior_planes_host[i_center].y = planeParams[masks(row, col) - 1].y;
                    prior_planes_host[i_center].z = planeParams[masks(row, col) - 1].z;
                    prior_planes_host[i_center].w = GetDepthFromPlaneParam_LF_Tilekey(planeParams[masks(row, col) - 1], col, row);;
                }
            }
        }
        cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (mi_height*mi_width), cudaMemcpyHostToDevice);
        cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (mi_height*mi_width), cudaMemcpyHostToDevice);
    }

    void AdaptMIPM::InitializeGPU()
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
        cudaError_t error_tk = cudaMalloc((void**)&tilekeys_cuda, sizeof(int2) * (params.num_images));
        if (error_tk != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_tk" << std::endl;
            return;
        }
        cudaMemcpy(tilekeys_cuda, tileKeyS_MI, params.num_images * sizeof(int2), cudaMemcpyHostToDevice);

        cudaError_t error_phc = cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (image_rows * image_cols));
        if (error_phc != cudaSuccess)
        {
            std::cout<<"InitializeGPU: error_phc" << std::endl;
            return;
        }
        cudaMalloc((void**)&costs_cuda, sizeof(float)*(image_rows*image_cols));
        cudaMalloc((void**)&neighbor_patchFill_cuda, sizeof(int3)*(image_rows*image_cols));
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

    void AdaptMIPM::InitGPUParamsFromCPUParams()
    {
        params.num_images = (int)images_MI.size();

        params.patch_Bound_size = params.patch_size*0.5;

        params.depth_min = 0.0; // 0.0
        params.depth_max = m_ParamsCUDA.baseline*0.5; //  40.0
        params.disparity_min = 5.0; // 0.0
        params.disparity_max = m_ParamsCUDA.baseline*0.5; //
        params.Base = m_ParamsCUDA.baseline; //
        params.MLA_Mask_Width_Cuda = m_ParamsCUDA.mi_width_for_match;
        params.MLA_Mask_Height_Cuda = m_ParamsCUDA.mia_height_for_match;

        params.base_height_ratio = 0.2;
        params.base_height_sigma = 0.05;

        //std::cout<< "mla_width=" <<params.MLA_Mask_Width_Cuda << "mla_height= "  <<params.MLA_Mask_Height_Cuda <<std::endl;
    }
}
