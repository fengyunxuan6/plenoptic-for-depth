/********************************************************************
file base:      MIStereoMatch.cpp
author:         LZD
created:        2025/06/04
purpose:        微图像的视差计算
*********************************************************************/
#include "MIStereoMatch.h"
#include "AdaptMIPMFrame.h"
#include "AdaptMIPMFrame.cpp"

#include "Util/Logger.h"
#include "boost/filesystem.hpp"

#include "CheckModule/MICycleCheck.h"
#include "CheckModule/MIDisparityFilter.h"

namespace LFMVS
{
    MIStereoMatch::MIStereoMatch(DepthSolver* pDepthSolver)
        : m_ptrDepthSolver(pDepthSolver)
    {
    }

    MIStereoMatch::~MIStereoMatch()
    {
        m_ptrDepthSolver = NULL;
    }

    void MIStereoMatch::TestWriteDisparityImage(std::string& strFrameName, QuadTreeTileKeyPtr ptrKey,
        AdaptMIPM& adapt_MIPM)
    {
        // 只写指定的微图像的深度估计结果。如果全部写出，太消耗时间
        // if (!(ptrKey->GetTileX() == 23 && ptrKey->GetTileY()==10))
        // {
        //     return;
        // }

        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        std::string& strRootPath = m_ptrDepthSolver->GetRootPath();
        int mi_width = params.mi_width_for_match;
        int mi_height = params.mi_height_for_match;
        std::vector<cv::Vec3b>& colorList = m_ptrDepthSolver->GetColorList();

        std::string strMLAResultFolder  = strRootPath + LF_DEPTH_INTRA_NAME + strFrameName;
        boost::filesystem::path dir_save_path(strMLAResultFolder);
        if (!boost::filesystem::exists(dir_save_path))
        {
            if (!boost::filesystem::create_directory(dir_save_path))
            {
                std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
            }
        }

        strMLAResultFolder += LF_MVS_RESULT_DATA_NAME;
        boost::filesystem::path dir_save_path_MLA(strMLAResultFolder);
        if (!boost::filesystem::exists(dir_save_path_MLA))
        {
            if (!boost::filesystem::create_directory(dir_save_path_MLA))
            {
                std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
            }
        }

        // 写出微透镜图像的视差图
        double d_factor = 5.0;
        // 创建路径
        std::string strMLADisFolder = strMLAResultFolder + LF_MLA_DISPARITYMAPS_NAME;
        boost::filesystem::path dir_save_path_Dis(strMLADisFolder);
        if (!boost::filesystem::exists(dir_save_path_Dis))
        {
            if (!boost::filesystem::create_directory(dir_save_path_Dis))
            {
                std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
            }
        }

        cv::Mat MLA_DisMap_gray= cv::Mat::zeros(mi_width, mi_height, CV_8UC1);
        std::map<unsigned int, std::vector<int2>> neigviews_map;
        for (int mla_row = 0; mla_row < mi_height; ++mla_row)
        {
            for (int mla_col = 0; mla_col < mi_width; ++mla_col)
            {
                int index = mla_row * mi_width + mla_col;
                float4 plane_hypothesis = adapt_MIPM.GetPlaneHypothesis(index);
                float4 disp_baseline = adapt_MIPM.GetDisparityBaseline(index);
                MLA_DisMap_gray.at<uchar>(mla_row, mla_col) = (plane_hypothesis.w/mi_width)*255*d_factor;

                float cost = adapt_MIPM.GetCost(index);
                unsigned int neig_viewBit = adapt_MIPM.GetSelected_viewIndexs(mla_col, mla_row);
                if (neig_viewBit > 0)
                {
                    std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.find(neig_viewBit);
                    if (itr == neigviews_map.end())
                    {
                        std::vector<int2> pixel_coords;
                        int2 p = make_int2(mla_row, mla_col);
                        pixel_coords.push_back(p);
                        neigviews_map[neig_viewBit] = pixel_coords;
                    }
                    else
                    {
                        std::vector<int2>& pixel_coords = itr->second;
                        int2 p = make_int2(mla_row, mla_col);
                        pixel_coords.push_back(p);
                    }
                }
            }
        }
        std::string strMLADisPath_key = strMLADisFolder + ptrKey->StrRemoveLOD();
        cv::Mat disp_color;
        applyColorMap(MLA_DisMap_gray, disp_color, cv::COLORMAP_JET);
        cv::imwrite(strMLADisPath_key + std::string("_color.png"), disp_color);


        // neighinfo
        // cv::Mat neighinfo = cv::Mat::zeros(mi_height, mi_width, CV_8UC3);
        // unsigned int ix = 0;
        // for (std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.begin();
        //     itr != neigviews_map.end(); itr++, ix++)
        // {
        //     std::vector<int2>& pixel_coords = itr->second;
        //     cv::Vec3b color = colorList[ix];
        //
        //     for (int i = 0; i < pixel_coords.size(); i++)
        //     {
        //         int2& pixel_coord = pixel_coords[i];
        //         neighinfo.at<cv::Vec3b>(pixel_coord.x, pixel_coord.y) = color;
        //     }
        // }
        //
        // cv::imwrite(strMLADisPath_key + std::string("_neighborInfo.png"), neighinfo);
    }

    void MIStereoMatch::TestWriteDisparityImage_PRG(std::string& strFrameName, QuadTreeTileKeyPtr ptrKey,
        AdaptMIPM& adapt_MIPM)
    {
        // 只写指定的微图像的深度估计结果。如果全部写出，太消耗时间
        if (!(ptrKey->GetTileX() == 23 && ptrKey->GetTileY()==10))
        {
            return;
        }

        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();
        std::string& strRootPath = m_ptrDepthSolver->GetRootPath();
        int mi_width = params.mi_width_for_match;
        int mi_height = params.mi_height_for_match;
        std::vector<cv::Vec3b>& colorList = m_ptrDepthSolver->GetColorList();

        std::string strMLAResultFolder  = strRootPath + LF_DEPTH_INTRA_NAME + strFrameName;
        boost::filesystem::path dir_save_path(strMLAResultFolder);
        if (!boost::filesystem::exists(dir_save_path))
        {
            if (!boost::filesystem::create_directory(dir_save_path))
            {
                std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
            }
        }

        strMLAResultFolder += LF_MVS_RESULT_DATA_NAME;
        boost::filesystem::path dir_save_path_MLA(strMLAResultFolder);
        if (!boost::filesystem::exists(dir_save_path_MLA))
        {
            if (!boost::filesystem::create_directory(dir_save_path_MLA))
            {
                std::cout << "dir failed to create: " << strMLAResultFolder << std::endl;
            }
        }

        // 写出微透镜图像的视差图
        double d_factor = 5.0;
        // 创建路径
        std::string strMLADisFolder = strMLAResultFolder + LF_MLA_DISPARITYMAPS_NAME;
        boost::filesystem::path dir_save_path_Dis(strMLADisFolder);
        if (!boost::filesystem::exists(dir_save_path_Dis))
        {
            if (!boost::filesystem::create_directory(dir_save_path_Dis))
            {
                std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
            }
        }

        cv::Mat MLA_DisMap_gray= cv::Mat::zeros(mi_width, mi_height, CV_8UC1);
        std::map<unsigned int, std::vector<int2>> neigviews_map;
        for (int mla_row = 0; mla_row < mi_height; ++mla_row)
        {
            for (int mla_col = 0; mla_col < mi_width; ++mla_col)
            {
                int index = mla_row * mi_width + mla_col;
                float4 plane_hypothesis = adapt_MIPM.GetPlaneHypothesis(index);
                float4 disp_baseline = adapt_MIPM.GetDisparityBaseline(index);
                MLA_DisMap_gray.at<uchar>(mla_row, mla_col) = (plane_hypothesis.w/mi_width)*255*d_factor;

                float cost = adapt_MIPM.GetCost(index);
                unsigned int neig_viewBit = adapt_MIPM.GetSelected_viewIndexs(mla_col, mla_row);
                if (neig_viewBit > 0)
                {
                    std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.find(neig_viewBit);
                    if (itr == neigviews_map.end())
                    {
                        std::vector<int2> pixel_coords;
                        int2 p = make_int2(mla_row, mla_col);
                        pixel_coords.push_back(p);
                        neigviews_map[neig_viewBit] = pixel_coords;
                    }
                    else
                    {
                        std::vector<int2>& pixel_coords = itr->second;
                        int2 p = make_int2(mla_row, mla_col);
                        pixel_coords.push_back(p);
                    }
                }
            }
        }
        std::string strMLADisPath_key = strMLADisFolder + ptrKey->StrRemoveLOD();
        cv::Mat disp_color;
        applyColorMap(MLA_DisMap_gray, disp_color, cv::COLORMAP_JET);
        cv::imwrite(strMLADisPath_key + std::string("_color_PRG.png"), disp_color);

        // neighinfo
        cv::Mat neighinfo = cv::Mat::zeros(mi_height, mi_width, CV_8UC3);
        unsigned int ix = 0;
        for (std::map<unsigned int, std::vector<int2>>::iterator itr = neigviews_map.begin();
            itr != neigviews_map.end(); itr++, ix++)
        {
            std::vector<int2>& pixel_coords = itr->second;
            cv::Vec3b color = colorList[ix];

            for (int i = 0; i < pixel_coords.size(); i++)
            {
                int2& pixel_coord = pixel_coords[i];
                neighinfo.at<cv::Vec3b>(pixel_coord.x, pixel_coord.y) = color;
            }
        }

        cv::imwrite(strMLADisPath_key + std::string("_neighborInfo_PRG.png"), neighinfo);
    }

    void MIStereoMatch::StereoMatchingForMIA(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("MISM: StereoMatchingForMIA, Begin");

        // 微透镜阵列相关参数（硬件）
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();

        std::string& strRootPath = m_ptrDepthSolver->GetRootPath();
        int mi_width = params.mi_width_for_match;
        int mi_height = params.mi_height_for_match;

        // step2: 视觉匹配
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problems_map = itrFrame->second;
        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("DisNormalMap not find = ", strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        int garbge = 0;
        // 当前帧中，逐个微图像遍历进行视差匹配
// #pragma omp parallel for
        for(QuadTreeProblemMap::iterator itrP = problems_map.begin(); itrP != problems_map.end(); ++itrP)
        {
            MLA_Problem& problem = itrP->second;
            if (problem.m_bGarbage || problem.m_bNeedMatch==false)
            {
                garbge++;
                continue;
            }

            // step1: 判断是否有可用于匹配的gpu
            const int top_gpu_device = m_ptrDepthSolver->GetTopGPUDevice();
            if (top_gpu_device == -1)
            {
                std::cout<<"PPLFT: Error! Find GPU device index is: " << top_gpu_device << std::endl;
                return;
            }
            cudaError_t err = cudaSetDevice(top_gpu_device);
            if (err != cudaSuccess)
            {
                std::cout<<"PPLFT: Error! cudaSetDevice: " << err << std::endl;
                return;
            }

            QuadTreeTileKeyPtr ptrKey = itrP->first;

            // test
            // if (!(ptrKey->GetTileX() == 5 && ptrKey->GetTileY()==15))
            // {
            //     continue;
            // }

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                std::cout << "Current Image not found: "  << ptrKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }

            // TODO: 要重写LFACMP.
            std::vector<float4> planeVec;
            std::vector<float> costVec;
            AdaptMIPM adapt_MIPM(params);
            adapt_MIPM.SetTileKey(ptrKey->GetTileX(), ptrKey->GetTileY());
            adapt_MIPM.Initialize(MLA_info_map, problem, problems_map, planeVec, costVec);
            adapt_MIPM.RunPatchMatchCUDAForMI();

            // 深度估计结果：cuda--->host
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            cv::Mat_<float> depths = cv::Mat::zeros(mi_height, mi_width, CV_32FC1);
            cv::Mat neighinfo = cv::Mat::zeros(mi_height, mi_width, CV_8UC3);
            for (int row = 0; row < mi_height; row++)
            {
                for (int col = 0; col < mi_width; col++)
                {
                    int index = row*mi_width + col;
                    float4 plane_hypothesis = adapt_MIPM.GetPlaneHypothesis(index);
                    float cost = adapt_MIPM.GetCost(index);
                    unsigned int neig_viewBit = adapt_MIPM.GetSelected_viewIndexs(col, row);
                    float4 disp_baseline = adapt_MIPM.GetDisparityBaseline(index);

                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->d_cuda[index] = plane_hypothesis.w ; // disp_baseline.w
                    depths(row, col) = plane_hypothesis.w; // disp_baseline.w
                    // std::cout<<"d_real: "<<disp_baseline.y<<std::endl;
                    ptrDN->c_cuda[index] = cost;
                    ptrDN->selected_views[index] = neig_viewBit;
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;
            // 处理写出和测试
            static bool bTestWriteDisImage = true;
            if (bTestWriteDisImage)
            {
                TestWriteDisparityImage(strFrameName, ptrKey, adapt_MIPM);
            }

            // 平面先验优化深度值
            // {
            //     ptrDN->m_StereoStage = eSS_PlannerPrior_Begin;
            //     adapt_MIPM.SetPlanarPriorParams();
            //
            //     cv::Rect imageRC(0, 0, params.mi_width_for_match, params.mi_height_for_match);
            //
            //     // 获取高可信稀疏的对应关系
            //     std::vector<cv::Point> support2DPoints;
            //     adapt_MIPM.GetSupportPoints(support2DPoints); // LZD, 修改：5--->3
            //     const std::vector<Triangle> triangles = adapt_MIPM.DelaunayTriangulation(imageRC, support2DPoints);//未修改
            //
            //     bool bWrite_tri_Image = true; //
            //     if (bWrite_tri_Image)
            //     {
            //         cv::Mat refImage = adapt_MIPM.GetReferenceImage().clone();
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
            //         std::string strMLAResultFolder  = strRootPath + LF_DEPTH_INTRA_NAME + strFrameName;
            //         std::string strMLADisFolder = strMLAResultFolder + LF_MVS_RESULT_DATA_NAME + LF_MLA_DISPARITYMAPS_PLANNER_NAME;
            //         boost::filesystem::path dir_save_path(strMLADisFolder);
            //         if (!boost::filesystem::exists(dir_save_path))
            //         {
            //             if (!boost::filesystem::create_directory(dir_save_path))
            //             {
            //                 std::cout << "dir failed to create: " << strMLADisFolder << std::endl;
            //             }
            //         }
            //         std::string triangulation_path = strMLADisFolder + ptrKey->StrRemoveLOD()+".png";
            //         cv::imwrite(triangulation_path, srcImage);
            //     }
            //
            //     cv::Mat_<float> mask_tri = cv::Mat::zeros(mi_height, mi_width, CV_32FC1);
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
            //             float4 n4 = adapt_MIPM.GetPriorPlaneParams(triangle, depths); // 修改过
            //             planeParams_tri.push_back(n4);
            //         }
            //     }
            //
            //     cv::Mat_<float> prior_depths = cv::Mat::zeros(mi_height, mi_width, CV_32FC1);
            //     for (int col = 0; col < mi_width; ++col)
            //     {
            //         for (int row = 0; row < mi_height; ++row)
            //         {
            //             if (mask_tri(row, col) > 0)
            //             {
            //                 float d = adapt_MIPM.GetDepthFromPlaneParam_LF_Tilekey(planeParams_tri[mask_tri(row, col) - 1], col, row);
            //                 if (d <= adapt_MIPM.GetMaxDepth() && d >= adapt_MIPM.GetMinDepth())
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
            //     adapt_MIPM.CudaPlanarPriorInitialization_LF_Tilekey(planeParams_tri, mask_tri);
            //     adapt_MIPM.RunPatchMatchCUDAForMI_plane();
            //
            //     // 存储结果
            //     for (int col = 0; col < mi_width; ++col)
            //     {
            //         for (int row = 0; row < mi_height; ++row)
            //         {
            //             int center = row * mi_width + col;
            //             float4 plane_hypothesis = adapt_MIPM.GetPlaneHypothesis(center);
            //             float cost = adapt_MIPM.GetCost(center);
            //             ptrDN->ph_cuda[center] = plane_hypothesis;
            //             ptrDN->d_cuda[center] = plane_hypothesis.w;
            //             ptrDN->c_cuda[center] = cost;
            //         }
            //     }
            //     ptrDN->m_StereoStage = eSS_PlannerPrior_Finished;
            // }
            adapt_MIPM.ReleaseMemory(); // TODO: LZD 释放内存or显存？
        }
        LOG_ERROR("MISM: StereoMatchingForMIA, End, garbge=", garbge);
    }

    void MIStereoMatch::StereoMatchingForMIA_SoftProxyRepair(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("MISM: StereoMatchingForMIA_SoftProxyRepair, Begin");
        // 微透镜阵列相关参数（硬件）
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();

        std::string& strRootPath = m_ptrDepthSolver->GetRootPath();
        int mi_width = params.mi_width_for_match;
        int mi_height = params.mi_height_for_match;

        // step2: 视觉匹配
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problems_map = itrFrame->second;
        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("DisNormalMap not find = ", strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        int garbge = 0;
        // 当前帧中，逐个微图像遍历进行视差匹配
        for(QuadTreeProblemMap::iterator itrP = problems_map.begin(); itrP != problems_map.end(); ++itrP)
        {
            MLA_Problem& problem = itrP->second;
            QuadTreeTileKeyPtr ptrKey = itrP->first;
            if (problem.m_bGarbage /*|| problem.m_bNeedMatch==false*/)
            {
                garbge++;
                continue;
            }

            // step1: 判断是否有可用于匹配的gpu
            const int top_gpu_device = m_ptrDepthSolver->GetTopGPUDevice();
            if (top_gpu_device == -1)
            {
                LOG_ERROR("PPLFT: Error! Find GPU device index is: ", top_gpu_device);
                continue;
            }
            cudaError_t err = cudaSetDevice(top_gpu_device);
            if (err != cudaSuccess)
            {
                LOG_ERROR("PPLFT: Error! cudaSetDevice: ", err);
                continue;
            }

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                LOG_ERROR("Current Image not found: ", ptrKey->StrRemoveLOD().c_str());
                continue;
            }

            std::vector<float4> planeVec;
            std::vector<float> costVec;
            AdaptMIPM adapt_MIPM(params);
            adapt_MIPM.SetTileKey(ptrKey->GetTileX(), ptrKey->GetTileY());
            adapt_MIPM.Initialize(MLA_info_map, problem, problems_map, planeVec, costVec);
            adapt_MIPM.RunPatchMatchCUDAForMI_SoftProxy_PatchRepair();
            // 深度估计结果：cuda--->host
            DisparityAndNormalPtr ptrDN = itr_DN->second;
            cv::Mat_<float> depths = cv::Mat::zeros(mi_height, mi_width, CV_32FC1);
#pragma omp parallel for schedule(dynamic)
            for (int row = 0; row < mi_height; row++)
            {
                for (int col = 0; col < mi_width; col++)
                {
                    int index = row*mi_width + col;
                    float4 plane_hypothesis = adapt_MIPM.GetPlaneHypothesis(index);
                    float cost = adapt_MIPM.GetCost(index);
                    unsigned int neig_viewBit = adapt_MIPM.GetSelected_viewIndexs(col, row);
                    float4 disp_baseline = adapt_MIPM.GetDisparityBaseline(index);

                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->d_cuda[index] = plane_hypothesis.w ; // disp_baseline.w
                    //std::cout<<"dis: "<<plane_hypothesis.w <<std::endl;
                    depths(row, col) = plane_hypothesis.w; // disp_baseline.w
                    // std::cout<<"d_real: "<<disp_baseline.y<<std::endl;
                    ptrDN->c_cuda[index] = cost;
                    ptrDN->selected_views[index] = neig_viewBit;
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;
            // 处理写出和测试
            if (g_Debug_Save >= 1)
            {
                TestWriteDisparityImage(strFrameName, ptrKey, adapt_MIPM);
            }
            adapt_MIPM.ReleaseMemory();
        }

        // 2) 全部完成 → 全局闭环验证（CPU，不用任何 AdaptMIPM）
        if (g_Debug_Static >= 1)
        {
            MICycleCheckStats stats;
            std::string str_chenck_save_root = m_ptrDepthSolver->GetRootPath() + LF_DEPTH_INTRA_NAME + strFrameName + LF_MVS_RESULT_DATA_NAME +"CycleCheck";
            MICycleCheckerCPU checker(params.mi_width_for_match, params.mi_height_for_match, params.baseline);

            MICycleClampConfig ccfg;
            ccfg.clamp_photo = true; ccfg.photo_u = 3.0;
            ccfg.clamp_geo = true; ccfg.geo_u_px = 2.0;
            ccfg.skip_on_geo = true; ccfg.skip_geo_u_px = 8.0;
            ccfg.skip_on_photo = true; ccfg.skip_photo_u = 8.0;
            ccfg.gate_geo_px = 0.5;
            ccfg.gate_photo_u = 0.5;
            ccfg.gate_min_good_neighbors = 2;
            // ① 全局检查 + 指标 + 可视化
            checker.CheckGlobal(strFrameName, MLA_info_map, problems_map, disNormals_map,
                                stats, 4,false,
                                str_chenck_save_root, true,
                                60, 2.0, 2.0,
                                3.0, ccfg);

            // ② 随机匹配可视化（跨图连线严格复用上一轮随机点）
            std::string str_chenck_save_root_match = m_ptrDepthSolver->GetRootPath() + LF_DEPTH_INTRA_NAME
            + strFrameName + LF_MVS_RESULT_DATA_NAME +"CycleCheck/Match";
            checker.VisualizeRandomMatchesGlobal(strFrameName, MLA_info_map, problems_map, disNormals_map,
                                8, 50, 4, str_chenck_save_root_match,
                                12345, true);

            // TODO：如果需要，可根据 stats 的阈值触发二次修复或降权
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ② 基于闭环一致性结果，立即剔除视差队列中的坏点/噪点，供后续虚拟深度图生成直接使用
        MIDisparityFilterConfig fcfg;
        fcfg.max_triplet = 4;
        fcfg.min_valid_disp = 0.0f;
        fcfg.use_selected_views_only = true;
        fcfg.clear_selected_views_when_invalid = true;
        fcfg.enable_cost_filter = true;
        fcfg.max_cost = 1.8f;
        fcfg.enable_cycle_geo_filter = true;
        fcfg.max_geo_err_px = 0.5;
        fcfg.enable_cycle_photo_filter = true;
        fcfg.max_photo_err_u = 0.5;
        fcfg.min_good_neighbors = 2;
        fcfg.enable_spike_filter = true;
        fcfg.spike_abs_diff = 1.0f;
        fcfg.spike_min_neighbors = 3;
        fcfg.dump_debug_mask = (g_Debug_Static >= 1);

        MIDisparityFilterStats filter_stats;
        std::string str_filter_save_root = m_ptrDepthSolver->GetRootPath() + LF_DEPTH_INTRA_NAME
                    + strFrameName + LF_MVS_RESULT_DATA_NAME + "CycleCheck/Filter";
        MIDisparityFilterCPU filter_cpu(params.mi_width_for_match, params.mi_height_for_match, params.baseline);
        filter_cpu.FilterGlobal(strFrameName, MLA_info_map, problems_map,
                            disNormals_map, fcfg, filter_stats,
                               str_filter_save_root);
        //////////////////////////////////////////////
        LOG_ERROR("MISM: StereoMatchingForMIA_SoftProxyRepair, End, garbge=", garbge);
    }

    void MIStereoMatch::StereoMatchingForMIA_FrameCrossViews(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("MISM: StereoMatchingForMIA_FrameCrossViews, Begin");

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();

        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problems_map = itrFrame->second;
        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("DisNormalMap not find = ", strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        const int top_gpu_device = m_ptrDepthSolver->GetTopGPUDevice();
        if (top_gpu_device == -1)
        {
            LOG_ERROR("PPLFT: Error! Find GPU device index is: ", top_gpu_device);
            return;
        }
        cudaError_t err = cudaSetDevice(top_gpu_device);
        if (err != cudaSuccess)
        {
            LOG_ERROR("PPLFT: Error! cudaSetDevice: ", err);
            return;
        }

        AdaptMIPMFrame adapt_frame(params);
        if (!adapt_frame.Initialize(MLA_info_map, problems_map))
        {
            LOG_ERROR("AdaptMIPMFrame.Initialize failed");
            return;
        }

        adapt_frame.RunPatchMatchCUDAForFrame();
        adapt_frame.WriteBackResults(disNormals_map);
        adapt_frame.ReleaseMemory();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ② 基于闭环一致性结果，立即剔除视差队列中的坏点/噪点，供后续虚拟深度图生成直接使用
        MIDisparityFilterConfig fcfg;
        fcfg.max_triplet = 4;
        fcfg.min_valid_disp = 0.0f;
        fcfg.use_selected_views_only = true;
        fcfg.clear_selected_views_when_invalid = true;
        fcfg.enable_cost_filter = true;
        fcfg.max_cost = 1.8f;
        fcfg.enable_cycle_geo_filter = true;
        fcfg.max_geo_err_px = 0.5;
        fcfg.enable_cycle_photo_filter = true;
        fcfg.max_photo_err_u = 0.5;
        fcfg.min_good_neighbors = 2;
        fcfg.enable_spike_filter = true;
        fcfg.spike_abs_diff = 1.0f;
        fcfg.spike_min_neighbors = 3;
        fcfg.dump_debug_mask = (g_Debug_Static >= 1);

        MIDisparityFilterStats filter_stats;
        std::string str_filter_save_root = m_ptrDepthSolver->GetRootPath() + LF_DEPTH_INTRA_NAME
                    + strFrameName + LF_MVS_RESULT_DATA_NAME + "CycleCheck/Filter";
        MIDisparityFilterCPU filter_cpu(params.mi_width_for_match, params.mi_height_for_match, params.baseline);
        filter_cpu.FilterGlobal(strFrameName, MLA_info_map, problems_map,
                            disNormals_map, fcfg, filter_stats,
                               str_filter_save_root);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        LOG_ERROR("MISM: StereoMatchingForMIA_FrameCrossViews, End");
    }

    void MIStereoMatch::StereoMatchingForMIA_SoftProxyPGRRepair(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("MISM: StereoMatchingForMIA_SoftProxyPGRRepair, Begin");

        // 微透镜阵列相关参数（硬件）
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        LightFieldParams& params = m_ptrDepthSolver->GetLightFieldParams();

        std::string& strRootPath = m_ptrDepthSolver->GetRootPath();
        int mi_width = params.mi_width_for_match;
        int mi_height = params.mi_height_for_match;

        // step1: 视觉匹配
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap& problems_map = itrFrame->second;
        QuadTreeDisNormalMapMap& disNormalMapMap = m_ptrDepthSolver->GetMLADisNormalMapMap();
        QuadTreeDisNormalMapMap::iterator itrDis = disNormalMapMap.find(strFrameName);
        if (itrDis == disNormalMapMap.end())
        {
            LOG_ERROR("DisNormalMap not find = ", strFrameName.c_str());
            return;
        }
        QuadTreeDisNormalMap& disNormals_map = itrDis->second;

        //
        // std::map<QuadTreeTileKeyPtr, QuadTreeTileKeyPtr> te_neigh_keys;
        // // 5-15 23-10 30-3
        // QuadTreeTileKeyPtr ptrKey_te = QuadTreeTileKey::CreateInstance(TileKey_None, 0, 23, 10);
        // MLA_Problem& problem_te = problems_map[ptrKey_te];
        // NeighScoreMap te_neig_map = problem_te.m_NeighScoreMapForMatch;
        // te_neig_map[ptrKey_te] = Neigh_Score();

        int garbge = 0;
        StereoResultInfoMap stereoResultMap;
        int propagation_Graph_size = 5;
        // 当前帧中，逐个微图像遍历进行视差匹配
// #pragma omp parallel for
        for(QuadTreeProblemMap::iterator itrP = problems_map.begin(); itrP != problems_map.end(); ++itrP)
        {
            MLA_Problem& problem = itrP->second;
            QuadTreeTileKeyPtr ptrKey = itrP->first;
            if (problem.m_bGarbage || problem.m_bNeedMatch==false)
            {
                garbge++;
                continue;
            }

            // Test
            // if (te_neig_map.find(ptrKey) == te_neig_map.end())
            // {
            //     continue;
            // }
            std::cout<<"1-current key: "<<ptrKey->StrRemoveLOD().c_str()<<std::endl;


            // step1: 判断是否有可用于匹配的gpu
            const int top_gpu_device = m_ptrDepthSolver->GetTopGPUDevice();
            if (top_gpu_device == -1)
            {
                std::cout<<"PPLFT: Error! Find GPU device index is: " << top_gpu_device << std::endl;
                continue;
            }
            cudaError_t err = cudaSetDevice(top_gpu_device);
            if (err != cudaSuccess)
            {
                std::cout<<"PPLFT: Error! cudaSetDevice: " << err << std::endl;
                continue;
            }

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                std::cout << "Current Image not found: "  << ptrKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }

            // TODO: 要重写LFACMP.
            std::vector<float4> planeVec;
            std::vector<float> costVec;
            AdaptMIPMPFPGR adapt_PFPGR(params);
            adapt_PFPGR.SetTileKey(ptrKey->GetTileX(), ptrKey->GetTileY());
            adapt_PFPGR.Initialize(MLA_info_map, problem, problems_map, planeVec, costVec);
            adapt_PFPGR.RunPatchMatchCUDAForMI_PFPGR_Collect();
            //adapt_PFPGR.TestWritePF_PGRInfo();
            //adapt_PFPGR.TestWriteNeighbour();
            adapt_PFPGR.TestWriteNeighbour_color(problem, problems_map);
            propagation_Graph_size = adapt_PFPGR.GetParamsGPU().propagation_Graph_size;

            // 深度估计结果：cuda--->host
            DisparityAndNormalPtr ptrDN = itr_DN->second;
// #pragma omp parallel for schedule(dynamic)
            for (int row = 0; row < mi_height; row++)
            {
                for (int col = 0; col < mi_width; col++)
                {
                    int index = row*mi_width + col;
                    float4 plane_hypothesis = adapt_PFPGR.GetPlaneHypothesis(index);
                    float cost = adapt_PFPGR.GetCost(index);
                    unsigned int neig_viewBit = adapt_PFPGR.GetSelected_viewIndexs(col, row);
                    float4 disp_baseline = adapt_PFPGR.GetDisparityBaseline(index);
                    int3 patchInfo = adapt_PFPGR.GetNeighborPatch(index);
                    int3 pgrInfo = adapt_PFPGR.GetNeighborPGR(index);

                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->d_cuda[index] = plane_hypothesis.w ; // disp_baseline.w
                    ptrDN->disp_v_cuda[index] = disp_baseline;
                    ptrDN->neighbor_Patch_info[index] = patchInfo;
                    ptrDN->neighbor_PGR_info[index] = pgrInfo;
                    ptrDN->c_cuda[index] = cost;
                    ptrDN->selected_views[index] = neig_viewBit;
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;
            // 处理写出和测试
            static bool bTestWriteDisImage = true;
            if (bTestWriteDisImage)
            {
                TestWriteDisparityImage(strFrameName, ptrKey, adapt_PFPGR);
            }

            StereoResultInfo* pStereoResult = new StereoResultInfo(mi_width, mi_height);
            memcpy(pStereoResult->plane_hypotheses_host , adapt_PFPGR.GetPlaneHypothesisVector(),
                (mi_width*mi_height)*sizeof(float4));
            memcpy(pStereoResult->costs_host , adapt_PFPGR.GetCostVector(),
(mi_width*mi_height)*sizeof(float));
            memcpy(pStereoResult->rand_states_host , adapt_PFPGR.GetRandStateVector(),
    (mi_width*mi_height)*sizeof(curandState));
            memcpy(pStereoResult->neighbor_patchFill_host , adapt_PFPGR.GetNeighborPatchFillVector(),
(mi_width*mi_height)*sizeof(int3));
            memcpy(pStereoResult->selected_views_host , adapt_PFPGR.GetSelected_view_vector(),
(mi_width*mi_height)*sizeof(unsigned int));
            stereoResultMap[ptrKey] = pStereoResult;
            adapt_PFPGR.ReleaseMemory(); // TODO: LZD 释放内存or显存？
        }
        // return;

        // step2: 匹配块和传播路径的补充（近似替代）
        std::map<QuadTreeTileKeyPtr, Proxy_DisPlane*, QuadTreeTileKeyMapCmpLess> mi_plandisp_map;
        // #pragma omp parallel for
        for(QuadTreeProblemMap::iterator itrP = problems_map.begin(); itrP != problems_map.end(); ++itrP)
        {
            MLA_Problem& problem = itrP->second;
            QuadTreeTileKeyPtr ptrKey = itrP->first;
            if (problem.m_bGarbage || problem.m_bNeedMatch==false)
            {
                garbge++;
                continue;
            }

            // Test
            // if (!(ptrKey->GetTileX() == 23 && ptrKey->GetTileY()==10))
            // {
            //     continue;
            // }

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                std::cout << "Current Image not found: "  << ptrKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }
            DisparityAndNormalPtr ptrDN = itr_DN->second;

            Proxy_DisPlane* proxy_dis_plane = new Proxy_DisPlane[mi_height*mi_width];
            // #pragma omp parallel for schedule(dynamic)
            for (int row = 0; row < mi_height; row++)
            {
                for (int col = 0; col < mi_width; col++)
                {
                    int2 center = make_int2(col, row);

                    // 处理传播路径
                    if (ptrDN->IsBroken(center, mi_height, mi_width, propagation_Graph_size))
                    {
                        ptrDN->CollectPropagationGraphLackedPixels(center, mi_width, mi_height, problem,
                            MLA_info_map, disNormals_map, proxy_dis_plane);
                    }
                }
            }
            mi_plandisp_map[ptrKey] = proxy_dis_plane;
        }

        // step3: 第二遍
        for(QuadTreeProblemMap::iterator itrP = problems_map.begin(); itrP != problems_map.end(); ++itrP)
        {
            MLA_Problem& problem = itrP->second;
            QuadTreeTileKeyPtr ptrKey = itrP->first;
            if (problem.m_bGarbage || problem.m_bNeedMatch==false)
            {
                garbge++;
                continue;
            }
            // step1: 判断是否有可用于匹配的gpu
            const int top_gpu_device = m_ptrDepthSolver->GetTopGPUDevice();
            if (top_gpu_device == -1)
            {
                std::cout<<"PPLFT: Error! Find GPU device index is: " << top_gpu_device << std::endl;
                continue;
            }
            cudaError_t err = cudaSetDevice(top_gpu_device);
            if (err != cudaSuccess)
            {
                std::cout<<"PPLFT: Error! cudaSetDevice: " << err << std::endl;
                continue;
            }

            // test
            // if (!(ptrKey->GetTileX() == 23 && ptrKey->GetTileY()==10))
            // {
            //     continue;
            // }
            // std::cout<<"current key: "<<ptrKey->StrRemoveLOD().c_str()<<std::endl;

            QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrKey);
            if (itr_DN == disNormals_map.end())
            {
                std::cout << "Current Image not found: "  << ptrKey->StrRemoveLOD().c_str() << std::endl;
                continue;
            }
            DisparityAndNormalPtr ptrDN = itr_DN->second;

            // TODO: 要重写LFACMP.
            Proxy_DisPlane* proxy_Plane = mi_plandisp_map[ptrKey];
            StereoResultInfo* pStereoResult = stereoResultMap[ptrKey];

            std::vector<float4> planeVec;
            std::vector<float> costVec;
            AdaptMIPMPFPGR adapt_PFPGR(params);
            adapt_PFPGR.SetTileKey(ptrKey->GetTileX(), ptrKey->GetTileY());
            adapt_PFPGR.Initialize_SecondStage(MLA_info_map, problem, problems_map, planeVec,
                                                costVec, proxy_Plane, pStereoResult);
            adapt_PFPGR.RunPatchMatchCUDAForMI_PFPGR_Repair();

            // 深度估计结果：cuda--->host
// #pragma omp parallel for schedule(dynamic)
            for (int row = 0; row < mi_height; row++)
            {
                for (int col = 0; col < mi_width; col++)
                {
                    int index = row*mi_width + col;
                    float4 plane_hypothesis = adapt_PFPGR.GetPlaneHypothesis(index);
                    float cost = adapt_PFPGR.GetCost(index);
                    unsigned int neig_viewBit = adapt_PFPGR.GetSelected_viewIndexs(col, row);
                    float4 disp_baseline = adapt_PFPGR.GetDisparityBaseline(index);

                    ptrDN->ph_cuda[index] = plane_hypothesis;
                    ptrDN->d_cuda[index] = plane_hypothesis.w ; // disp_baseline.w
                    ptrDN->disp_v_cuda[index] = disp_baseline;
                    ptrDN->c_cuda[index] = cost;
                    ptrDN->selected_views[index] = neig_viewBit;
                }
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;
            // 处理写出和测试
            static bool bTestWriteDisImage = true;
            if (bTestWriteDisImage)
            {
                TestWriteDisparityImage_PRG(strFrameName, ptrKey, adapt_PFPGR);
            }
            adapt_PFPGR.ReleaseMemory(); // TODO: LZD 释放内存or显存？
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ② 基于闭环一致性结果，立即剔除视差队列中的坏点/噪点，供后续虚拟深度图生成直接使用
        MIDisparityFilterConfig fcfg;
        fcfg.max_triplet = 4;
        fcfg.min_valid_disp = 0.0f;
        fcfg.use_selected_views_only = true;
        fcfg.clear_selected_views_when_invalid = true;
        fcfg.enable_cost_filter = true;
        fcfg.max_cost = 1.8f;
        fcfg.enable_cycle_geo_filter = true;
        fcfg.max_geo_err_px = 0.5;
        fcfg.enable_cycle_photo_filter = true;
        fcfg.max_photo_err_u = 0.5;
        fcfg.min_good_neighbors = 2;
        fcfg.enable_spike_filter = true;
        fcfg.spike_abs_diff = 1.0f;
        fcfg.spike_min_neighbors = 3;
        fcfg.dump_debug_mask = (g_Debug_Static >= 1);

        MIDisparityFilterStats filter_stats;
        std::string str_filter_save_root = m_ptrDepthSolver->GetRootPath() + LF_DEPTH_INTRA_NAME
                    + strFrameName + LF_MVS_RESULT_DATA_NAME + "CycleCheck/Filter";
        MIDisparityFilterCPU filter_cpu(params.mi_width_for_match, params.mi_height_for_match, params.baseline);
        filter_cpu.FilterGlobal(strFrameName, MLA_info_map, problems_map,
                            disNormals_map, fcfg, filter_stats,
                               str_filter_save_root);
        //////////////////////////////////////////////
        LOG_ERROR("MISM: StereoMatchingForMIA_SoftProxyPGRRepair, End, garbge=", garbge);
    }
}

