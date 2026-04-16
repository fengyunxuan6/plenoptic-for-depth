/********************************************************************
file base:      AdaptMIPMFrameACMM.h
author:         OpenAI + LZD workflow
created:        2026/04/14
purpose:        ACMM风格的整帧GPU版微图像视差匹配（多尺度 + 模糊分组/对齐）
*********************************************************************/
#ifndef ADAPTMIPMFRAME_ACMM_H
#define ADAPTMIPMFRAME_ACMM_H

#include "MVStereo/LFDepthInfo.h"
#include "Util/Logger.h"

namespace LFMVS
{
    const int MAX_ACMM_LEVELS = 4;

    struct ACMMFrameLevelHostData
    {
        int width = 0;
        int height = 0;
        float scale = 1.0f;                 // 相对原始微图像尺度
        std::vector<cv::Mat> images;        // CV_32FC1
        std::vector<cv::Mat> blur_images;   // CV_32FC1
        std::vector<float> blur_mean;       // 每个微图像的平均模糊值
    };

    class AdaptMIPMFrameACMM
    {
    public:
        explicit AdaptMIPMFrameACMM(LightFieldParams& params);
        ~AdaptMIPMFrameACMM();

    public:
        bool Initialize(QuadTreeTileInfoMap& MLA_info_map,
                        QuadTreeProblemMap& problem_map);
        void RunPatchMatchCUDAForFrameACMM();
        void WriteBackResults(QuadTreeDisNormalMap& disNormals_map);
        void ReleaseMemory();

        int GetValidViewCount() const { return m_num_views; }
        int GetNumLevels() const { return m_num_levels; }

    protected:
        void InitGPUParamsFromCPUParams(int level_id);
        bool BuildViewIndexMap(QuadTreeTileInfoMap& MLA_info_map, QuadTreeProblemMap& problem_map);
        bool BuildBlurAlignedPyramids();
        void BuildSingleLevelBlurAlignment(ACMMFrameLevelHostData& level_data) const;
        bool InitializeGPUForLevel(int level_id);
        void ReleaseGPUForCurrentLevel();
        void CreateGrayImageObject(int image_index, const cv::Mat& image_in);
        void CreateBlurImageObject(int image_index, const cv::Mat& blur_in);
        void DownloadCurrentLevelResults();
        void PrepareWarmStartFromPreviousLevel(int prev_level_id, int cur_level_id);
        void UploadWarmStartToGPU();

        float EstimateGaussianSigma(float blur_mean_src,
                                    float blur_mean_target,
                                    float level_scale) const;

    protected:
        LightFieldParamsCUDA                m_ParamsCUDA;
        PatchMatchParamsLF                  params;

        int                                 m_num_views;
        int                                 m_max_neighbors;
        int                                 m_pixels_per_view;
        int                                 m_num_levels;
        int                                 m_current_level;
        bool                                m_bReleased;

        float                               m_lambda_scale;
        float                               m_lambda_geo;
        float                               m_detail_th;
        float                               m_geom_clip;
        int                                 m_blur_group_count;

        std::vector<QuadTreeTileKeyPtr>     m_view_keys;
        std::map<QuadTreeTileKeyPtr, int, QuadTreeTileKeyMapCmpLess> m_key_to_viewid;

        std::vector<cv::Mat>                images_MI;
        std::vector<cv::Mat>                blur_images_MI;

        float2*                             centerPointS_MI;
        int2*                               tileKeyS_MI;
        int*                                neighbor_ids_host;
        int*                                neighbor_counts_host;

        std::vector<ACMMFrameLevelHostData> m_pyramid_levels;

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

        float4*                             plane_init_cuda;
        float*                              cost_init_cuda;
        float4*                             disp_init_cuda;
        unsigned int*                       selected_init_cuda;

        std::vector<float4>                 plane_level_host;
        std::vector<float>                  cost_level_host;
        std::vector<float4>                 disp_level_host;
        std::vector<unsigned int>           selected_level_host;

        std::vector<float4>                 plane_init_host;
        std::vector<float>                  cost_init_host;
        std::vector<float4>                 disp_init_host;
        std::vector<unsigned int>           selected_init_host;

        float4*                             plane_final_host;
        float*                              cost_final_host;
        float4*                             disp_final_host;
        unsigned int*                       selected_final_host;
    };
}

#endif // ADAPTMIPMFRAME_ACMM_H