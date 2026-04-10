/********************************************************************
file base:      AdaptMIPMAlg.h
author:         LZD
created:        2025/06/07
purpose:        微图像的视差计算
*********************************************************************/

#ifndef ADAPTMIPMALG_H
#define ADAPTMIPMALG_H

// #include "Common/Common.h"
// #include "Common/CommonCUDA.h"

#include "MVStereo/LFDepthInfo.h"

namespace LFMVS
{
    struct StereoResultInfo
    {
        StereoResultInfo(const int mi_width, const int mi_height)
        {
            mi_image_width = mi_width;
            mi_image_height = mi_height;

            plane_hypotheses_host = new float4[mi_width*mi_height];
            costs_host = new float[mi_width*mi_height];
            rand_states_host = new curandState[mi_width*mi_height];
            neighbor_patchFill_host = new int3[mi_width*mi_height];
            disp_baseline_host = new float4[mi_width*mi_height];
            selected_views_host = new unsigned int[mi_width*mi_height];
        }

        int                 mi_image_width;
        int                 mi_image_height;
        float4*             plane_hypotheses_host;
        float*              costs_host;
        curandState*        rand_states_host;
        int3*               neighbor_patchFill_host; // (邻域微图像id, 对应点像素坐标 x,y )
        float4*             disp_baseline_host; // (标准视差d， 实际斜方视差d_real，baseline, 虚拟深度)
        unsigned int*       selected_views_host;
    };
    typedef std::map<QuadTreeTileKeyPtr, StereoResultInfo*, QuadTreeTileKeyMapCmpLess> StereoResultInfoMap;

    // 相当于LFACMP
    // Adaptive Micro-Image Propagation Matcher
    class AdaptMIPM
    {
    public:
        AdaptMIPM(LightFieldParams& params);
        ~AdaptMIPM();

    public: // 接口
        void RunPatchMatchCUDAForMI();
        void RunPatchMatchCUDAForMI_plane();

        // 边缘像素感知的微图像密集匹配算法：补全 patch+propagation
        void RunPatchMatchCUDAForMI_SoftProxy_PatchRepair();

        void SetTileKey(int tile_x, int tile_y);

        void Initialize(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                        QuadTreeProblemMap& problem_map,   std::vector<float4>& planeVec,
                        std::vector<float>& costVec);

        void Initialize_old(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                    QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                    std::vector<float>& costVec);

        void CudaPlanarPriorInitialization_LF_Tilekey(const std::vector<float4>& planeParams,
            const cv::Mat_<float>& masks);

        PatchMatchParamsLF GetParamsGPU()
        {
            return params;
        }

        float4 GetPlaneHypothesis(const int index);
        float4* GetPlaneHypothesisVector();

        float GetCost(const int index);
        float* GetCostVector();

        curandState* GetRandStateVector();
        int3* GetNeighborPatchFillVector();

        float4 GetDisparityBaseline(const int index);
        float4* GetDisparityBaselineVector();

        int3 GetNeighborPatch(const int index);

        unsigned int GetSelected_viewIndexs(const int pt_col, const int pt_row);
        unsigned int* GetSelected_view_vector();
        void SetPlanarPriorParams();

        void GetSupportPoints(std::vector<cv::Point>& support2DPoints);

        std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC,
                    const std::vector<cv::Point>& points);

        cv::Mat GetReferenceImage();

        float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
        float GetDepthFromPlaneParam_LF_Tilekey(const float4 plane_hypothesis, const int x, const int y);
        float GetMinDepth();
        float GetMaxDepth();

        void ReleaseMemory();

    protected:
        void InitializeGPU();

        void InitGPUParamsFromCPUParams();

        float3 Get3DPointonRefCamLF(const int x, const int y, const float depth);

        void CreateGrayImageObject(int image_index);
        void CreateBlurImageObject(int image_index);

        // test
        void TestWriteMIBeforeGPU();

    protected:
        LightFieldParamsCUDA                        m_ParamsCUDA;

        // tilekey
        int                                         m_tile_x;
        int                                         m_tile_y;

        //int                                         num_images;
        std::vector<cv::Mat>                        images_MI; // 自身和邻居
        std::vector<cv::Mat>                        blur_images_MI; // 自身和邻居
        float2*                                     centerPointS_MI; // center_points
        int2*                                       tileKeyS_MI; // (x,y) 列，行
        cudaTextureObjects                          texture_objects_host;

        float4*                                     plane_hypotheses_host; // w为标准视差d
        float*                                      costs_host;
        int3*                                       neighbor_patchFill_host; // (邻域微图像id, 对应点像素坐标 x,y )
        float4*                                     disp_baseline_host; // (标准视差d， 实际斜方视差d_real，baseline, 虚拟深度)
        float4*                                     prior_planes_host;
        unsigned int*                               plane_masks_host;
        curandState*                                rand_states_host;
        unsigned int*                               selected_views_host;

        // CUDA中使用的变量
        PatchMatchParamsLF                          params;
        Camera*                                     cameras_cuda;
        cudaArray*                                  cuArray[MAX_IMAGES];
        cudaArray*                                  cu_blur_Array[MAX_IMAGES];
        cudaArray*                                  cuDepthArray[MAX_IMAGES];
        cudaTextureObjects*                         texture_objects_cuda;
        cudaTextureObjects*                         texture_depths_cuda;
        float2*                                     centers_cuda;
        int2*                                       tilekeys_cuda;

        int   *d_nei_lin = nullptr;
        int   *d_kSteps  = nullptr;
        float *d_pTot    = nullptr;

        float4*                                     plane_hypotheses_cuda;
        curandState*                                rand_states_cuda;
        float*                                      costs_cuda;
        int3*                                       neighbor_patchFill_cuda; // (邻域微图像id, 对应点像素坐标 x,y )
        float4*                                     disp_baseline_cuda; // (标准视差d， 实际斜方视差d_real，baseline, 虚拟深度)
        unsigned int*                               selected_views_cuda;

        float*                                      depths_cuda;
        float4*                                     prior_planes_cuda;
        unsigned int*                               plane_masks_cuda;
        
        // 标志资源是否已被释放
        bool                                        m_bReleased;
    };
}

#endif //ADAPTMIPMALG_H
