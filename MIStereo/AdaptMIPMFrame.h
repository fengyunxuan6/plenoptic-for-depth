
/********************************************************************
file base:      AdaptMIPMFrame.h
author:         OpenAI + LZD
created:        2026/04/12
purpose:        整帧GPU版微图像视差匹配（跨视图proposal）
*********************************************************************/
#ifndef ADAPTMIPMFRAME_H
#define ADAPTMIPMFRAME_H

#include "MVStereo/LFDepthInfo.h"
#include "Util/Logger.h"

namespace LFMVS
{
    class AdaptMIPMFrame
    {
    public:
        AdaptMIPMFrame(LightFieldParams& params);
        ~AdaptMIPMFrame();

    public:
        bool Initialize(QuadTreeTileInfoMap& MLA_info_map,
                        QuadTreeProblemMap& problem_map);
        void RunPatchMatchCUDAForFrame();
        void WriteBackResults(QuadTreeDisNormalMap& disNormals_map);
        void ReleaseMemory();

        int GetValidViewCount() const { return m_num_views; }

    protected:
        void InitGPUParamsFromCPUParams();
        bool BuildViewIndexMap(QuadTreeTileInfoMap& MLA_info_map, QuadTreeProblemMap& problem_map);
        bool InitializeGPU();
        void CreateGrayImageObject(int image_index);
        void CreateBlurImageObject(int image_index);

    protected:
        LightFieldParamsCUDA                m_ParamsCUDA;
        PatchMatchParamsLF                  params;

        int                                 m_num_views;
        int                                 m_max_neighbors;
        int                                 m_pixels_per_view;

        bool                                m_bReleased;

        std::vector<QuadTreeTileKeyPtr>     m_view_keys;
        std::map<QuadTreeTileKeyPtr, int, QuadTreeTileKeyMapCmpLess> m_key_to_viewid;

        std::vector<cv::Mat>                images_MI;
        std::vector<cv::Mat>                blur_images_MI;

        float2*                             centerPointS_MI;
        int2*                               tileKeyS_MI;
        int*                                neighbor_ids_host;
        int*                                neighbor_counts_host;

        cudaTextureObjects                  texture_objects_host;
        cudaArray*                          cuArray[MAX_IMAGES];
        cudaArray*                          cu_blur_Array[MAX_IMAGES];

        cudaTextureObjects*                 texture_objects_cuda;
        float2*                             centers_cuda;
        int2*                               tilekeys_cuda;
        int*                                neighbor_ids_cuda;
        int*                                neighbor_counts_cuda;

        float4*                             plane_prev_cuda;
        float4*                             plane_next_cuda;
        float*                              cost_prev_cuda;
        float*                              cost_next_cuda;
        float4*                             disp_prev_cuda;
        float4*                             disp_next_cuda;
        unsigned int*                       selected_prev_cuda;
        unsigned int*                       selected_next_cuda;
        curandState*                        rand_states_cuda;

        float4*                             plane_final_host;
        float*                              cost_final_host;
        float4*                             disp_final_host;
        unsigned int*                       selected_final_host;
    };
}

#endif // ADAPTMIPMFRAME_H
