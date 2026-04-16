/********************************************************************
file base:      AdaptMIPM_EdgewareUtil.cuh
author:         LZD
created:        2025/07/10
purpose:
*********************************************************************/
#ifndef ADAAPTMIPM_EDGEWARE_UTIL_CUH
#define ADAAPTMIPM_EDGEWARE_UTIL_CUH

#include "Common/Common.h"
#include "Common/CommonCUDA.h"
#include "MVStereo/LFDepthInfo.h"

namespace LFMVS
{
    // function

    __device__ bool IsBoundary(const int2 p, const PatchMatchParamsLF params);

    __device__ float SoftminScore(float rank_dist, float rank_cost, float lambda_d, float lambda_c);

    __device__ int SelectBestProxy_Softmin(const int2 p, const PatchMatchParamsLF params,
        float2* center_coord_vector, float2* correspond_coord_vector, float* cost_vector,
        float lambda_d, float lambda_c);

    __device__ void CollectNeighborsCoorespondInfo(const int2 p, float2* pcenters, float* cost_vector_input,
                float4* disp_baseline_vector, const PatchMatchParamsLF params,
                float2* center_coord_vector, float2* correspond_coord_vector, float* cost_vector);

    __device__ float ComputeBilateralNCC_MIPM_FillPatch(const cudaTextureObject_t ref_image,
            const cudaTextureObject_t ref_blur_image, float2 p0,
            const cudaTextureObject_t src_image, float2 p1,
            const cudaTextureObject_t src_blur_image, const int2 p,
            float* patch_pixels,
            float4 plane_hypothesis, const PatchMatchParamsLF params,
            float2& blur_value, float4& disparity_baseline);
    __device__ float ComputeBilateralNCC_MIPM_FillPatch_Hex(const cudaTextureObject_t ref_image,
        const cudaTextureObject_t ref_blur_image, float2 p0,
        const cudaTextureObject_t src_image, float2 p1,
        const cudaTextureObject_t src_blur_image, const int2 p,
        float* patch_pixels,
        float4 plane_hypothesis, const PatchMatchParamsLF params,
        float2& blur_value, float4& disparity_baseline,
        int2 tilekeys_ref, int2 tilekeys_neig);

    // 计算像素p的多邻域视图代价
    __device__ void ComputeMultiViewCostVector_MIPM_FillPatch(const cudaTextureObject_t* images,
                    const cudaTextureObject_t* blur_images, float2* pcenters,
                    const int2 p, int3* neighbor_patchFill_cuda,
                    float4 plane_hypothesis, float* cost_vector, float2* blure_array,
                    float4* disp_baseline_array, const PatchMatchParamsLF params);
    __device__ void ComputeMultiViewCostVector_MIPM_FillPatch_Hex(const cudaTextureObject_t* images,
                        const cudaTextureObject_t* blur_images, float2* pcenters,
                        const int2 p, int3* neighbor_patchFill_cuda,
                        float4 plane_hypothesis, float* cost_vector, float2* blure_array,
                        float4* disp_baseline_array, const PatchMatchParamsLF params, int2* tilekeys_cuda);

    __device__ void PlaneHypothesisRefinement_MIPM_FillPatch(const cudaTextureObject_t* images,
                const cudaTextureObject_t* blur_images,
                const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
                float2* pcenters, float* depth, float* cost, float4* disp_baseline, curandState* rand_state,
                const float* view_weights, const float weight_norm, float4* prior_planes,
                unsigned int* plane_masks, float* restricted_cost, const int2 p,
                int3* neighbor_patchFill_cuda,
                const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm);
    __device__ void PlaneHypothesisRefinement_MIPM_FillPatch_Hex(const cudaTextureObject_t* images,
            const cudaTextureObject_t* blur_images,
            const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
            float2* pcenters, float* depth, float* cost, float4* disp_baseline, curandState* rand_state,
            const float* view_weights, const float weight_norm, float4* prior_planes,
            unsigned int* plane_masks, float* restricted_cost, const int2 p,
            int3* neighbor_patchFill_cuda,
            const PatchMatchParamsLF params, const int WIDTH,
            const float* baseline_norm, int2* tilekeys_cuda);

    __device__ void PlaneHypothesisRefinement_MIPM_edgepixels(const cudaTextureObject_t* images,
                const cudaTextureObject_t* blur_images,
                const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
                float2* pcenters, float* depth, float* cost, float4* disp_baseline,
                float2* center_coord_vector, float2* correspond_coord_vector, float* cost_vector,
                curandState* rand_state,
                const float* view_weights, const float weight_norm, float4* prior_planes,
                unsigned int* plane_masks, float* restricted_cost, const int2 p,
                const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm);
    __device__ void PlaneHypothesisRefinement_MIPM_edgepixels(const cudaTextureObject_t* images,
            const cudaTextureObject_t* blur_images,
            const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
            float2* pcenters, float* depth, float* cost, float4* disp_baseline,
            float2* center_coord_vector, float2* correspond_coord_vector, float* cost_vector,
            curandState* rand_state,
            const float* view_weights, const float weight_norm, float4* prior_planes,
            unsigned int* plane_masks, float* restricted_cost, const int2 p,
            const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm);


    __device__ void CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware_FillPatch(
        const cudaTextureObjects* texture_objects,
        const cudaTextureObject_t* depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
        unsigned int* selected_views, float4* prior_planes,
        unsigned int* plane_masks, int3* neighbor_patchFill_cuda,
        const int2 p, const PatchMatchParamsLF params,
        const int iter, const int WIDTH, const int HEIGHT);

    __device__ void CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware_edgepixels(
            const cudaTextureObjects* texture_objects,
            const cudaTextureObject_t* depths, float4* plane_hypotheses,
            float2* pcenters, float* costs, float4* disp_baseline,
            float2* center_coord_vector, float2* correspond_coord_vector, float* cost_vector,
            curandState* rand_states, unsigned int* selected_views, float4* prior_planes,
            unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
            const int iter, const int WIDTH, const int HEIGHT, int2* tilekeys_cuda);

    __device__ void CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware(
        const cudaTextureObjects* texture_objects,
        const cudaTextureObject_t* depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
        unsigned int* selected_views, float4* prior_planes,
        unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
        const int iter, const int WIDTH, const int HEIGHT, int2* tilekeys_cuda);

    __global__ void BlackPixelUpdate_MIPM_Edgeaware(cudaTextureObjects* texture_objects,
            cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
            float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
            float4* prior_planes, unsigned int* plane_masks,
            int3* neighbor_patchFill_cuda,
            PatchMatchParamsLF params, const int iter,
            const int width, const int height, int2* tilekeys_cuda);

    __global__ void BlackPixelUpdate_MIPM_Edgeaware_FillPatch(cudaTextureObjects* texture_objects,
        cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
        float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
        float4* prior_planes, unsigned int* plane_masks,
        int3* neighbor_patchFill_cuda,
        const PatchMatchParamsLF params, const int iter,
        const int width, const int height, int2* tilekeys_cuda);

    __global__ void RedPixelUpdate_MIPM_Edgeaware(cudaTextureObjects *texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                int3* neighbor_patchFill_cuda,
                const PatchMatchParamsLF params, const int iter,
                const int width, const int height, int2* tilekeys_cuda);
    __global__ void RedPixelUpdate_MIPM_Edgeaware_FillPatch(cudaTextureObjects *texture_objects,
            cudaTextureObjects* texture_depths, float4* plane_hypotheses,
            float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
            unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
            int3* neighbor_patchFill_cuda,
            const PatchMatchParamsLF params, const int iter,
            const int width, const int height, int2* tilekeys_cuda);
}
#endif //ADAAPTMIPM_EDGEWARE_UTIL_CUH
