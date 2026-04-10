/********************************************************************
file base:      LFACMP.cpp
author:         LZD
created:        2024/06/12
purpose:
*********************************************************************/
#include "LFACMP.h"
#include <atomic>

// 声明全局退出标志
extern std::atomic<bool> g_program_exiting;

namespace LFMVS
{
    LF_ACMP::LF_ACMP(LightFieldParams& params)
        :m_ParamsCUDA(params)
    {
        num_images = 0;
        cameras_cuda = NULL;
        texture_objects_cuda = NULL;
        centers_cuda = NULL;
        plane_hypotheses_cuda = NULL;
        rand_states_cuda = NULL;
        costs_cuda = NULL;
        selected_views_cuda = NULL;
        depths_cuda = NULL;
        prior_planes_cuda = NULL;
        plane_masks_cuda = NULL;
        COL1 = NULL;
        ROW1 = NULL;
        WIDTH1 = NULL;
        HIGHT1 = NULL;

        m_tile_x = -2;
        m_tile_y = -2;
    }

    LF_ACMP::~LF_ACMP()
    {
        // 如果程序正在退出，跳过复杂的资源释放过程
        // 使用try-catch确保即使g_program_exiting已被析构也不会导致崩溃
        try {
            if (g_program_exiting.load()) {
                return;
            }
        } catch (...) {
            // 如果访问g_program_exiting失败，假设程序正在退出并跳过资源释放
            return;
        }
        
        try {
            // 安全释放内存，检查指针是否为空
            if (plane_hypotheses_host) {
                delete[] plane_hypotheses_host;
                plane_hypotheses_host = nullptr;
            }
            
            if (costs_host) {
                delete[] costs_host;
                costs_host = nullptr;
            }

            // 只有当texture_objects_host已初始化时才销毁纹理对象
            if (cameras_cuda) {  // 使用cameras_cuda作为初始化标志
                for (int i = 0; i < num_images && i < MAX_IMAGES; ++i) {
                    if (cuArray[i]) {
                        try {
                            cudaDestroyTextureObject(texture_objects_host.images[i]);
                        } catch (...) {
                            // 忽略可能的CUDA错误
                        }
                        try {
                            cudaFreeArray(cuArray[i]);
                        } catch (...) {
                            // 忽略可能的CUDA错误
                        }
                        cuArray[i] = nullptr;
                    }
                }
                
                if (texture_objects_cuda) {
                    try {
                        cudaFree(texture_objects_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    texture_objects_cuda = nullptr;
                }
                
                if (cameras_cuda) {
                    try {
                        cudaFree(cameras_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    cameras_cuda = nullptr;
                }
                
                if (centers_cuda) {
                    try {
                        cudaFree(centers_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    centers_cuda = nullptr;
                }
                
                if (plane_hypotheses_cuda) {
                    try {
                        cudaFree(plane_hypotheses_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    plane_hypotheses_cuda = nullptr;
                }
                
                if (costs_cuda) {
                    try {
                        cudaFree(costs_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    costs_cuda = nullptr;
                }
                
                if (rand_states_cuda) {
                    try {
                        cudaFree(rand_states_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    rand_states_cuda = nullptr;
                }
                
                if (selected_views_cuda) {
                    try {
                        cudaFree(selected_views_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    selected_views_cuda = nullptr;
                }
                
                if (depths_cuda) {
                    try {
                        cudaFree(depths_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    depths_cuda = nullptr;
                }

                if (params.geom_consistency) {
                    for (int i = 0; i < num_images && i < MAX_IMAGES; ++i) {
                        if (cuDepthArray[i]) {
                            try {
                                cudaDestroyTextureObject(texture_depths_host.images[i]);
                            } catch (...) {
                                // 忽略可能的CUDA错误
                            }
                            try {
                                cudaFreeArray(cuDepthArray[i]);
                            } catch (...) {
                                // 忽略可能的CUDA错误
                            }
                            cuDepthArray[i] = nullptr;
                        }
                    }
                    if (texture_depths_cuda) {
                        try {
                            cudaFree(texture_depths_cuda);
                        } catch (...) {
                            // 忽略可能的CUDA错误
                        }
                        texture_depths_cuda = nullptr;
                    }
                }
            }

            if (params.planar_prior) {
                if (prior_planes_host) {
                    delete[] prior_planes_host;
                    prior_planes_host = nullptr;
                }
                
                if (plane_masks_host) {
                    delete[] plane_masks_host;
                    plane_masks_host = nullptr;
                }

                if (prior_planes_cuda) {
                    try {
                        cudaFree(prior_planes_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    prior_planes_cuda = nullptr;
                }
                
                if (plane_masks_cuda) {
                    try {
                        cudaFree(plane_masks_cuda);
                    } catch (...) {
                        // 忽略可能的CUDA错误
                    }
                    plane_masks_cuda = nullptr;
                }
            }
            
            // 清理其他可能分配的内存
            if (cp) {
                delete[] cp;
                cp = nullptr;
            }
            
            if (selected_views_host) {
                delete[] selected_views_host;
                selected_views_host = nullptr;
            }
            
            if (COL1) {
                try {
                    cudaFree(COL1);
                } catch (...) {
                    // 忽略可能的CUDA错误
                }
                COL1 = nullptr;
            }
            
            if (ROW1) {
                try {
                    cudaFree(ROW1);
                } catch (...) {
                    // 忽略可能的CUDA错误
                }
                ROW1 = nullptr;
            }
            
            if (WIDTH1) {
                try {
                    cudaFree(WIDTH1);
                } catch (...) {
                    // 忽略可能的CUDA错误
                }
                WIDTH1 = nullptr;
            }
            
            if (HIGHT1) {
                try {
                    cudaFree(HIGHT1);
                } catch (...) {
                    // 忽略可能的CUDA错误
                }
                HIGHT1 = nullptr;
            }
        } catch (...) {
            // 忽略析构过程中可能出现的任何CUDA错误
        }
        
        // 重置参数
        try {
            params.geom_consistency = false;
            params.planar_prior = false;
            num_images = 0;
        } catch (...) {
            // 忽略参数重置过程中可能出现的任何错误
        }
    }

    void LF_ACMP::Delete_pc()
    {
        delete[] plane_hypotheses_host;
        delete[] costs_host;
    }

    void LF_ACMP::ReleaseMemory()
    {
        images.clear();
        delete [] cp;
        depths.clear();
        cameras.clear();
    }

    void LF_ACMP::SetGeomConsistencyParams(bool multi_geometry=false)
    {
        params.geom_consistency = true;
        params.max_iterations = 2;
        if (multi_geometry)
        {
            params.multi_geometry = true;
        }
    }

    void LF_ACMP::SetPlanarPriorParams()
    {
        params.planar_prior = true;
    }

    void LF_ACMP::InuputInitializationLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,const MLA_Problem & mlaProblem,float &Base)
    {
        images.clear();
        //pcenters.clear();
        int n = mlaProblem.number.size();

        cv::Mat image_float;
        int m = mlaProblem.main_img;
        //转为float
        img[m].convertTo(image_float, CV_32FC1);
        images.push_back(image_float);
        // 存储中心点坐标
        cp = new float2[ n+1 ];
        float2 p0;
        p0.x = center[m].x;
        p0.y = center[m].y;
        cp[0] = p0;

        for (size_t i = 0; i < n; ++i)
        {
            //r为邻居索引
            int r = mlaProblem.number[i];
            cv::Mat image_float;
            img[r].convertTo(image_float, CV_32FC1);
            images.push_back(image_float);
            //存储邻居中心点坐标
            float2 p;
            p.x = center[r].x;
            p.y = center[r].y;
            cp[i+1] = p;
        }

        params.num_images = (int)images.size();
        //std::cout << "num images: " << params.num_images << std::endl;
        params.depth_min = 5.0; // 0.5
        params.depth_max = 40.0; // g_MIA_fBase
        params.disparity_min = 5.0; // 0.5
        params.disparity_max = 40.0; // g_MIA_fBase
        params.Base = Base;
        params.MLA_Mask_Width_Cuda = m_ParamsCUDA.mi_width_for_match;
        params.MLA_Mask_Height_Cuda = m_ParamsCUDA.mia_height_for_match;
    }

    void LF_ACMP::InuputInitialization_LF_TileKey(QuadTreeTileInfoMap& MLA_info_map,
        MLA_Problem& problem, QuadTreeProblemMap& problem_map,
        std::vector<float4>& planeVec, std::vector<float>& costVec)
    {
        // reset
        images.clear();

        // 邻居数量
        int neig_count = problem.m_Res_Image_KeyVec.size();
        cv::Mat image_float;

        // 找到参考图像索引
        QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey; // 当前微图像的key
        QuadTreeTileInfoMap::iterator itr_map = MLA_info_map.find(ptrCurKey);
        if (itr_map == MLA_info_map.end())
        {
            std::cout<<"InLF_TileKey: current key not Find! " << ptrCurKey->GetTileX() << ", " <<ptrCurKey->GetTileY() << std::endl;
            return;
        }

        problem.m_Image_gray.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);

        // 存储当前微透镜图像的灰度图及其中心点坐标
        cp = new float2[neig_count+1];
        float2 main_image_point;
        MLA_InfoPtr ptrCurInfo = itr_map->second;
        main_image_point.x = ptrCurInfo->GetCenter().x;
        main_image_point.y = ptrCurInfo->GetCenter().y;
        cp[0] = main_image_point;

        plane_hypotheses_host = new float4[image_float.rows * image_float.cols];
        if (planeVec.size() == m_ParamsCUDA.mi_width_for_match*m_ParamsCUDA.mia_height_for_match)
        {
            for (int i = 0; i < m_ParamsCUDA.mia_height_for_match; ++i)
            {
                for (int j = 0; j < m_ParamsCUDA.mi_width_for_match; ++j)
                {
                    float d = planeVec[i*m_ParamsCUDA.mi_width_for_match+j].w;
                    float4 plane = planeVec[i*m_ParamsCUDA.mi_width_for_match+j];
                    // 判断随机视差 选择平面
                    const auto x = lround(j - d);
                    if (x >= 0 && costVec[i*m_ParamsCUDA.mi_width_for_match+j] < 0.3)
                    {
                        plane_hypotheses_host[i*m_ParamsCUDA.mi_width_for_match+j] = plane;
                    }
                    else
                    {
                        plane_hypotheses_host[i*m_ParamsCUDA.mi_width_for_match+j].w = 0;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < m_ParamsCUDA.mia_height_for_match; ++i)
            {
                for (int j = 0; j < m_ParamsCUDA.mi_width_for_match; ++j)
                {
                    plane_hypotheses_host[i*m_ParamsCUDA.mi_width_for_match+j].w = 0;
                }
            }
        }
        costs_host = new float[image_float.rows * image_float.cols];
        selected_views_host = new unsigned int[image_float.rows * image_float.cols];

        // 初始化邻居：灰度图及其中心点坐标
        for (size_t i = 0; i < neig_count; ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_Res_Image_KeyVec[i].m_ptrKey;

            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeigKey);
            if (itrNP == problem_map.end())
                continue;
            cv::Mat image_float;
            itrNP->second.m_Image_gray.convertTo(image_float, CV_32FC1);
            images.push_back(image_float);

            QuadTreeTileInfoMap::iterator neig_itr = MLA_info_map.find(ptrNeigKey);
            if (neig_itr == MLA_info_map.end())
            {
                std::cout<<"InLF_TileKey: neig key not Find! " << ptrNeigKey->GetTileX()
                << ", " << ptrNeigKey->GetTileY() << std::endl;
                continue;
            }

            MLA_InfoPtr ptrNeig_info = neig_itr->second;
            float2 neig_image_point;
            neig_image_point.x = ptrNeig_info->GetCenter().x;
            neig_image_point.y = ptrNeig_info->GetCenter().y;
            cp[i+1] = neig_image_point;
        }

        num_images = (int)images.size();
        params.num_images = (int)images.size();
        //std::cout << "num images: " << params.num_images << std::endl;
        params.depth_min = 5.0; // 0.5
        params.depth_max = 40.0; // g_MIA_fBase
        params.disparity_min = 5.0; // 0.5
        params.disparity_max = 40.0; // g_MIA_fBase

        params.Base = m_ParamsCUDA.baseline; //

        params.MLA_Mask_Width_Cuda = m_ParamsCUDA.mi_width_for_match;
        params.MLA_Mask_Height_Cuda = m_ParamsCUDA.mia_height_for_match;
    }

    void LF_ACMP::InuputInitialization_planarLF(const std::vector<cv::Mat> &img,
                                                const std::vector<cv::Point2f> &center,
                                                const MLA_Problem & mlaProblem,
                                                const DisparityAndNormal & DN,float &Base)
    {
        images.clear();
        //pcenters.clear();
        int n = mlaProblem.number.size();
        plane_hypotheses_host = DN.ph_cuda;
        costs_host = DN.c_cuda;

        cv::Mat image_float;
        int m = mlaProblem.main_img;
        img[m].convertTo(image_float, CV_32FC1);
        images.push_back(image_float);
        cp = new float2[ n+1 ];
        float2 p0;
        p0.x = center[m].x;
        p0.y = center[m].y;
        cp[0] = p0;

        for (size_t i = 0; i < n; ++i)
        {
            int r = mlaProblem.number[i];
            cv::Mat image_float;
            img[r].convertTo(image_float, CV_32FC1);
            images.push_back(image_float);
            float2 p;
            p.x = center[r].x;
            p.y = center[r].y;
            cp[i+1] = p;
            //pcenters.push_back(center[r]);
        }

        params.num_images = (int)images.size();
        //std::cout << "num images: " << params.num_images << std::endl;
        params.disparity_min = 20;
        params.disparity_max = 30;
        params.Base = Base;
    }

    void LF_ACMP::InuputInitialization_planarLF_TileKey(QuadTreeTileInfoMap& MLA_info_map,
                                                        MLA_Problem& problem,
                                                        QuadTreeProblemMap& problem_map,
                                                        QuadTreeDisNormalMap& MLA_DisNormalMap)
    {
        images.clear();

        int neig_count = problem.m_Res_Image_KeyVec.size();

        // 找到参考图像索引
        QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey; // 当前微图像的key
        QuadTreeTileInfoMap::iterator itr_map = MLA_info_map.find(ptrCurKey);
        if (itr_map == MLA_info_map.end())
        {
            std::cout<<"InLF_TileKey: current key not Find! " << ptrCurKey->GetTileX() << ", " <<ptrCurKey->GetTileY() << std::endl;
            return;
        }

        // 找到参考图像dns索引
        MLA_InfoPtr ptrCurInfo = itr_map->second;
        QuadTreeDisNormalMap::iterator itr_main_dns = MLA_DisNormalMap.find(ptrCurKey);
        if (itr_main_dns == MLA_DisNormalMap.end())
        {
            std::cout<<"InLF_TileKey: current key dns not Find! " << ptrCurKey->GetTileX() << ", " <<ptrCurKey->GetTileY() << std::endl;
            return;
        }

        DisparityAndNormalPtr ptrDisNormal_Main = itr_main_dns->second;
        plane_hypotheses_host = ptrDisNormal_Main->ph_cuda;
        costs_host = ptrDisNormal_Main->c_cuda;

        QuadTreeProblemMap::iterator itrPM = problem_map.find(ptrCurKey);
        if (itrPM == problem_map.end())
        {
            std::cout<<"InLF_TileKey2: current key dns not Find! " << ptrCurKey->GetTileX() << ", " <<ptrCurKey->GetTileY() << std::endl;
            return;
        }
        cv::Mat image_float;
        itrPM->second.m_Image_gray.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);

        cp = new float2[neig_count+1];
        float2 p0;
        p0.x = ptrCurInfo->GetCenter().x;
        p0.y = ptrCurInfo->GetCenter().y;
        cp[0] = p0;

        for (size_t i = 0; i < neig_count; ++i)
        {
            QuadTreeTileKeyPtr ptrNeigKey = problem.m_Res_Image_KeyVec[i].m_ptrKey;
            QuadTreeTileInfoMap::iterator neig_itr= MLA_info_map.find(ptrNeigKey);
            if (neig_itr == MLA_info_map.end())
            {
                std::cout<<"InLF_TileKey: neig key not Find! " << ptrNeigKey->GetTileX()
                         << ", " << ptrNeigKey->GetTileY() << std::endl;
                continue;
            }

            QuadTreeProblemMap::iterator itrPM = problem_map.find(ptrNeigKey);
            if (itrPM == problem_map.end())
            {
                std::cout<<"InLF_TileKey3: current key dns not Find! " << ptrNeigKey->GetTileX() << ", " <<ptrNeigKey->GetTileY() << std::endl;
                return;
            }
            cv::Mat image_float;
            itrPM->second.m_Image_gray.convertTo(image_float, CV_32FC1);
            images.push_back(image_float);

            MLA_InfoPtr ptrNeig_info = neig_itr->second;
            float2 neig_image_point;
            neig_image_point.x = ptrNeig_info->GetCenter().x;
            neig_image_point.y = ptrNeig_info->GetCenter().y;
            cp[i+1] = neig_image_point;
        }

        params.num_images = (int)images.size();
        //std::cout << "num images: " << params.num_images << std::endl;
        params.depth_min = 5.0; // 0.5
        params.depth_max = 40.0; // g_MIA_fBase
        params.disparity_min = 5.0; // 0.5
        params.disparity_max = 40.0; // g_MIA_fBase

        params.Base = m_ParamsCUDA.baseline; //
    }

    void LF_ACMP::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem)
    {
        num_images = (int)images.size();

        for (int i = 0; i < num_images; ++i)
        {
            int rows = images[i].rows;
            int cols = images[i].cols;

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
        cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

        plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
        cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

        costs_host = new float[cameras[0].height * cameras[0].width];
        cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

        cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
        cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

        cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width)); // Updated by Qingshan 2020-01-15

        if (params.geom_consistency) {
            for (int i = 0; i < num_images; ++i) {
                int rows = depths[i].rows;
                int cols = depths[i].cols;

                cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
                cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
                cudaMemcpy2DToArray (cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

                struct cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(cudaResourceDesc));
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = cuDepthArray[i];

                struct cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(cudaTextureDesc));
                texDesc.addressMode[0] = cudaAddressModeWrap;
                texDesc.addressMode[1] = cudaAddressModeWrap;
                texDesc.filterMode = cudaFilterModeLinear;
                texDesc.readMode  = cudaReadModeElementType;
                texDesc.normalizedCoords = 0;

                cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
            }
            cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
            cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

            std::stringstream result_path;
            result_path << dense_folder << "/ACMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
            std::string result_folder = result_path.str();
            std::string suffix = "/depths.dmb";
            if (params.multi_geometry) {
                suffix = "/depths_geom.dmb";
            }
            std::string depth_path = result_folder + suffix;
            std::string normal_path = result_folder + "/normals.dmb";
            std::string cost_path = result_folder + "/costs.dmb";
            cv::Mat_<float> ref_depth;
            cv::Mat_<cv::Vec3f> ref_normal;
            cv::Mat_<float> ref_cost;
            readDepthDmb(depth_path, ref_depth);
            depths.push_back(ref_depth);
            readNormalDmb(normal_path, ref_normal);
            readDepthDmb(cost_path, ref_cost);
            int width = ref_depth.cols;
            int height = ref_depth.rows;
            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < height; ++row) {
                    int center = row * width + col;
                    float4 plane_hypothesis;
                    plane_hypothesis.x = ref_normal(row, col)[0];
                    plane_hypothesis.y = ref_normal(row, col)[1];
                    plane_hypothesis.z = ref_normal(row, col)[2];
                    plane_hypothesis.w = ref_depth(row, col);
                    plane_hypotheses_host[center] = plane_hypothesis;
                    costs_host[center] = ref_cost(row, col);
                }
            }

            cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
            cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
        }
    }

    void LF_ACMP::CudaSpaceInitializationLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,
                                            const MLA_Problem & mlaProblem,
                                            std::vector<float4> Plane,std::vector<float> Cost)
    {
        num_images = (int)images.size();
        int rows = images[0].rows;//行
        int cols = images[0].cols;//列

        for (int i = 0; i < num_images; ++i)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        centers_cuda = new float2[ num_images];
        cudaMalloc((void**)&centers_cuda, sizeof(float2) * (num_images));
        cudaMemcpy(centers_cuda, cp, num_images * sizeof(float2), cudaMemcpyHostToDevice);

        plane_hypotheses_host = new float4[rows * cols];

        if (Plane.size() == m_ParamsCUDA.mi_width_for_match*m_ParamsCUDA.mia_height_for_match)
        {
            for (int i = 0; i < m_ParamsCUDA.mia_height_for_match; ++i)
            {
                for (int j = 0; j < m_ParamsCUDA.mi_width_for_match; ++j)
                {
                    float d = Plane[i*m_ParamsCUDA.mi_width_for_match+j].w;
                    float4 plane = Plane[i*m_ParamsCUDA.mi_width_for_match+j];
                    // 判断随机视差 选择平面
                    const auto x = lround(j - d);
                    if (x >= 0 && Cost[i*m_ParamsCUDA.mi_width_for_match+j]<0.3)
                    {
                        plane_hypotheses_host[i*m_ParamsCUDA.mi_width_for_match+j] = plane;
                    }
                    else
                    {
                        plane_hypotheses_host[i*m_ParamsCUDA.mi_width_for_match+j].w = 0;
                    }
                }
            }
        }
        cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (rows * cols));

        costs_host = new float[rows * cols];
        costs_cuda = new float [rows * cols];
        cudaMalloc((void**)&costs_cuda, sizeof(float) * (rows * cols));

        cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (rows * cols));

        cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (rows * cols));

        cudaMalloc((void**)&depths_cuda, sizeof(float) * (rows * cols)); // Updated by Qingshan 2020-01-15

    }

    void LF_ACMP::CudaSpaceInitialization_LF_TileKey()
    {
        if (images.empty())
        {
            std::cout<<"CudaSpaceInitLF_TileKey: images is EMPTY!"<<std::endl;
            return;
        }

        int image_rows = images[0].rows; // 行
        int image_cols = images[0].cols; // 列
        if (image_rows == 0 || image_cols == 0)
        {
            std::cout<<"reference image size is EMPTY!"<<std::endl;
            return;
        }

        // 初始化
        for (int i = 0; i < num_images; ++i)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuArray[i], &channelDesc, image_cols, image_rows);
            cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(),
                    images[i].step[0], image_cols*sizeof(float), image_rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaError_t error_toc = cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        if (error_toc != cudaSuccess)
        {
            std::cout<<"CudaSpaceInitLF_TileKey: error_toc" << std::endl;
            return;
        }
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        cudaError_t error_cc = cudaMalloc((void**)&centers_cuda, sizeof(float2) * (num_images));
        if (error_cc != cudaSuccess)
        {
            std::cout<<"CudaSpaceInitLF_TileKey: error_cc" << std::endl;
            return;
        }
        cudaMemcpy(centers_cuda, cp, num_images * sizeof(float2), cudaMemcpyHostToDevice);

        cudaError_t error_phc = cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (image_rows * image_cols));
        if (error_phc != cudaSuccess)
        {
            std::cout<<"CudaSpaceInitLF_TileKey: error_phc" << std::endl;
            return;
        }
        cudaMalloc((void**)&costs_cuda, sizeof(float) * (image_rows * image_cols));

        cudaError_t error_rsc = cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (image_rows * image_cols));
        if (error_rsc != cudaSuccess)
        {
            std::cout<<"CudaSpaceInitLF_TileKey: error_rsc" << std::endl;
            return;
        }
        cudaError_t error_svc = cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (image_rows * image_cols));
        if (error_svc != cudaSuccess)
        {
            std::cout<<"CudaSpaceInitLF_TileKey: error_svc" << std::endl;
            return;
        }
        cudaError_t error_dc = cudaMalloc((void**)&depths_cuda, sizeof(float) * (image_rows * image_cols)); // Updated by Qingshan 2020-01-15
        if (error_dc != cudaSuccess)
        {
            std::cout<<"CudaSpaceInitLF_TileKey: error_dc" << std::endl;
            return;
        }


        if (images.empty())
        {
            std::cout<<"111-CudaSpaceInitLF_TileKey: images is EMPTY!"<<std::endl;
        }
    }

    void LF_ACMP::CudaSpaceInitialization_planarLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,const MLA_Problem & mlaProblem,const DisparityAndNormal & DN)
    {
        num_images = (int)images.size();
        int rows = images[0].rows;//行
        int cols = images[0].cols;//列

        for (int i = 0; i < num_images; ++i)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        centers_cuda = new float2[ num_images];
        cudaMalloc((void**)&centers_cuda, sizeof(float2) * (num_images));
        cudaMemcpy(centers_cuda, cp, num_images * sizeof(float2), cudaMemcpyHostToDevice);

        //plane_hypotheses_host = new float4[rows * cols];
        cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (rows * cols));
        cudaMemcpy(plane_hypotheses_cuda,plane_hypotheses_host , sizeof(float4) * (rows * cols), cudaMemcpyHostToDevice);

        //costs_host = new float[rows * cols];
        cudaMalloc((void**)&costs_cuda, sizeof(float) * (rows * cols));
        cudaMemcpy(costs_cuda,costs_host , sizeof(float) * (rows * cols), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (rows * cols));

        cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (rows * cols));

        cudaMalloc((void**)&depths_cuda, sizeof(float) * (rows * cols)); // Updated by Qingshan 2020-01-15
        //cudaMemcpy(depths_cuda,DN.d_cuda , sizeof(float) * (rows * cols), cudaMemcpyHostToDevice);

    }

    void LF_ACMP::CudaSpaceInitialization_planarLF_TileKey()
    {
        num_images = (int)images.size();
        int rows = images[0].rows;//行
        int cols = images[0].cols;//列

        for (int i = 0; i < num_images; ++i)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        centers_cuda = new float2[ num_images];
        cudaMalloc((void**)&centers_cuda, sizeof(float2) * (num_images));
        cudaMemcpy(centers_cuda, cp, num_images * sizeof(float2), cudaMemcpyHostToDevice);

        //plane_hypotheses_host = new float4[rows * cols];
        cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (rows * cols));
        cudaMemcpy(plane_hypotheses_cuda,plane_hypotheses_host , sizeof(float4) * (rows * cols), cudaMemcpyHostToDevice);

        //costs_host = new float[rows * cols];
        cudaMalloc((void**)&costs_cuda, sizeof(float) * (rows * cols));
        cudaMemcpy(costs_cuda,costs_host , sizeof(float) * (rows * cols), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (rows * cols));

        cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (rows * cols));

        cudaMalloc((void**)&depths_cuda, sizeof(float) * (rows * cols)); // Updated by Qingshan 2020-01-15
        //cudaMemcpy(depths_cuda,DN.d_cuda , sizeof(float) * (rows * cols), cudaMemcpyHostToDevice);
    }

    void LF_ACMP::CudaPlanarPriorInitialization(const std::vector<float4>& PlaneParams, const cv::Mat_<float>& masks)
    {
        prior_planes_host = new float4[cameras[0].height * cameras[0].width];
        cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

        plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
        cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

        for (int i = 0; i < cameras[0].width; ++i)
        {
            for (int j = 0; j < cameras[0].height; ++j)
            {
                int center = j * cameras[0].width + i;
                plane_masks_host[center] = (unsigned int)masks(j, i);
                if (masks(j, i) > 0)
                {
                    prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
                }
            }
        }
        cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
        cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    }

    void LF_ACMP::CudaPlanarPriorInitialization_LF_Tilekey(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks)
    {
        prior_planes_host = new float4[m_ParamsCUDA.mia_height_for_match * m_ParamsCUDA.mi_width_for_match];
        cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (m_ParamsCUDA.mia_height_for_match * m_ParamsCUDA.mi_width_for_match));

        plane_masks_host = new unsigned int[m_ParamsCUDA.mia_height_for_match * m_ParamsCUDA.mi_width_for_match];
        cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * (m_ParamsCUDA.mia_height_for_match * m_ParamsCUDA.mi_width_for_match));

        for (int i = 0; i < m_ParamsCUDA.mi_width_for_match; ++i)
        {
            for (int j = 0; j < m_ParamsCUDA.mia_height_for_match; ++j)
            {
                int center = j * m_ParamsCUDA.mi_width_for_match + i;
                plane_masks_host[center] = (unsigned int)masks(j, i);
                if (masks(j, i) > 0)
                {
                    prior_planes_host[center].x = PlaneParams[masks(j, i) - 1].x;
                    prior_planes_host[center].y = PlaneParams[masks(j, i) - 1].y;
                    prior_planes_host[center].z = PlaneParams[masks(j, i) - 1].z;
                    prior_planes_host[center].w = GetDepthFromPlaneParam_LF_Tilekey(PlaneParams[masks(j, i) - 1], i, j);;
                }
            }
        }
        cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (m_ParamsCUDA.mia_height_for_match * m_ParamsCUDA.mi_width_for_match), cudaMemcpyHostToDevice);
        cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (m_ParamsCUDA.mia_height_for_match * m_ParamsCUDA.mi_width_for_match), cudaMemcpyHostToDevice);
    }

    int LF_ACMP::GetReferenceImageWidth()
    {
        return cameras[0].width;
    }

    int LF_ACMP::GetReferenceImageHeight()
    {
        return cameras[0].height;
    }

    cv::Mat LF_ACMP::GetReferenceImage()
    {
        return images[0];
    }

    float4 LF_ACMP::GetPlaneHypothesis(const int index)
    {
        return plane_hypotheses_host[index];
    }

    unsigned int LF_ACMP::GetSelected_viewIndexs(const int pt_col, const int pt_row)
    {
        // pt_col和pt_row均从0开始起算
        int width = images[0].cols;
        const int pt_oneDimen = pt_row * width + pt_col; // 将当前像素的二维数组表现形式转换为一维数组
        return selected_views_host[pt_oneDimen];
    }

    // 邻居索引号的转换二进制转换为十进制
    void LF_ACMP::SelectedViewIndexConvert(MLA_Problem& problem, unsigned int neig_viewBit)
    {
        std::vector<Res_image_Key> selected_pixel_neigs;
        uint8 view_value = 0;
        for (int i = 0; i < params.num_images - 1; ++i)
        {
            // i为局部索引
            if (((neig_viewBit >> i) & 1) == 1)
            {
                selected_pixel_neigs.push_back(problem.m_Res_Image_KeyVec[i]);
                view_value += i;
            }
        }
    }

    float LF_ACMP::GetCost(const int index)
    {
        return costs_host[index];
    }

    float LF_ACMP::GetMinDepth()
    {
        return params.depth_min;
    }

    float LF_ACMP::GetMaxDepth()
    {
        return params.depth_max;
    }

    void LF_ACMP::GetSupportPoints(std::vector<cv::Point>& support2DPoints)
    {
        support2DPoints.clear();
        const int step_size = 8;
        const int width = m_ParamsCUDA.mi_width_for_match;
        const int height = m_ParamsCUDA.mia_height_for_match;
        for (int col = 0; col < width; col += step_size)
        {
            for (int row = 0; row < height; row += step_size)
            {
                float min_cost = 2.0f;
                cv::Point temp_point;
                int c_bound = std::min(width, col + step_size);
                int r_bound = std::min(height, row + step_size);
                for (int c = col; c < c_bound; ++c)
                {
                    for (int r = row; r < r_bound; ++r)
                    {
                        int center = r * width + c;
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

    std::vector<Triangle> LF_ACMP::DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points)
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

    float3 LF_ACMP::Get3DPointonRefCamLF(const int x, const int y, const float depth)
    {
        float3 pointX;
        // Reprojection
        pointX.x = x;
        pointX.y = y;
        pointX.z = depth;

        return pointX;
    }

    float4 LF_ACMP::GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths)
    {
        //cv::Mat A(3, 4, CV_32FC1);
        //cv::Mat B(4, 1, CV_32FC1);

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

    //float3 DisparityPlane(const int2 p, const float4 plane_hypothesis)
    //{
    //    float3 plane;
    //    plane.x = -plane_hypothesis.x / plane_hypothesis.z;
    //    plane.y = -plane_hypothesis.y / plane_hypothesis.z;
    //    plane.z = (plane_hypothesis.x*p.x +plane_hypothesis.y*p.y +plane_hypothesis.z *plane_hypothesis.w)/plane_hypothesis.z;
    //    return plane;
    //}
    //
    //float to_disparity(const int2 p,float3 d_plane)
    //{
    //    return d_plane.x*p.x +d_plane.y*p.y + d_plane.z*1.0;
    //}

    float LF_ACMP::GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y)
    {
        return -plane_hypothesis.w * cameras[0].K[0] / ((x-cameras[0].K[2]) * plane_hypothesis.x + (cameras[0].K[0] / cameras[0].K[4]) * (y-cameras[0].K[5]) * plane_hypothesis.y + cameras[0].K[0]*plane_hypothesis.z);
    }

    float LF_ACMP::GetDepthFromPlaneParam_LF_Tilekey(const float4 plane_hypothesis, const int x, const int y)
    {
        return -(plane_hypothesis.x*x +plane_hypothesis.y*y +plane_hypothesis.w)/plane_hypothesis.z;
    }
}
