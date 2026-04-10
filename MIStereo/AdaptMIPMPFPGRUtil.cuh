/********************************************************************
file base:      AdaptMIPMPFPGRUtil.cuh
author:         LZD
created:        2025/07/12
purpose:
*********************************************************************/
#ifndef ADAPTMIPMPFPGRUTIL_CUH
#define ADAPTMIPMPFPGRUTIL_CUH
#include "Common/Common.h"
#include "Common/CommonCUDA.h"
#include "MVStereo/LFDepthInfo.h"

namespace LFMVS
{
    struct CollectNeighInfo
    {
        float2 center_coord; // 中心点
        float2 correspond_coord; // 对应点
        float cost; // 代价
    };

    //////////////////////////////////////////////////////////////////////
    __device__ bool IsBroken(const int2 p, const PatchMatchParamsLF params);

    __device__ int SelectBestProxy_Softmin_new(const int2 p, const PatchMatchParamsLF params,
                CollectNeighInfo* neig_info, float lambda_d, float lambda_c);

    __device__ void CollectNeighborInfo(const int2 p, float2* pcenters, float* cost_vector_input,
           float4* disp_baseline_vector, const PatchMatchParamsLF params,
           const bool bBound, const bool bBroken,
           CollectNeighInfo* boundary_info_vector, CollectNeighInfo* broken_info_vector);

    __device__ void ComputeMultiViewCostVector_PFPGR_Repair(const cudaTextureObject_t* images,
                    const cudaTextureObject_t* blur_images,
                    float2* pcenters, const int2 p,
                    int3* neighbor_patchFill_cuda,
                    float4 plane_hypothesis, float* cost_vector, float2* blure_array,
                    float4* disp_baseline_array, const PatchMatchParamsLF params);

    __device__ void PlaneHypothesisRefinement_PFPGR_Repair(const cudaTextureObject_t* images,
        const cudaTextureObject_t* blur_images,
        const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
        float2* pcenters, float* depth, float* cost, float4* disp_baseline, curandState* rand_state,
        const float* view_weights, const float weight_norm, float4* prior_planes,
        unsigned int* plane_masks, float* restricted_cost, const int2 p,
        int3* neighbor_patchFill_cuda,
        const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm);

    __device__ void PlaneHypothesisRefinement_MIPM_Repairpixels(const cudaTextureObject_t* images,
            const cudaTextureObject_t* blur_images,
            const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
            float2* pcenters, float* depth, float* cost, float4* disp_baseline,
            CollectNeighInfo* boundary_info_vector, CollectNeighInfo* broken_info_vector,
            curandState* rand_state,
            const float* view_weights, const float weight_norm, float4* prior_planes,
            unsigned int* plane_masks, float* restricted_cost, const int2 p,
            const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm,
            const bool bBound, const bool bBroken);

    __device__ void CheckerboardPropagation_MIPM_Repairpixels(
        const cudaTextureObjects* texture_objects,
        const cudaTextureObject_t* depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, float4* disp_baseline,
        CollectNeighInfo* boundary_info_vector, CollectNeighInfo* broken_info_vector,
        curandState* rand_states,
        unsigned int* selected_views, float4* prior_planes,
        unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
        const int iter, const int WIDTH, const int HEIGHT,
        const bool bBound, const bool bBroken);

    __device__ void CheckerboardPropagation_PFPGR_Repair(const cudaTextureObjects* texture_objects,
        const cudaTextureObject_t* depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
        unsigned int* selected_views, float4* prior_planes,
        unsigned int* plane_masks,
        int3* neighbor_patchFill_cuda, Proxy_DisPlane* proxy_plane_cuda,
        const int2 p, const PatchMatchParamsLF params,
        const int iter, const int WIDTH, const int HEIGHT);

    __global__ void BlackPixelUpdate_MIPM_PFPGR_Collect(cudaTextureObjects* texture_objects,
            cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
            float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
            float4* prior_planes, unsigned int* plane_masks,
            int3* neighbor_patchFill_cuda,
            int3* neighbor_PGR_cuda,
            const PatchMatchParamsLF params, const int iter,
            const int width, const int height, int2* tilekeys_cuda);
    __global__ void BlackPixelUpdate_PFPGR_Repair(cudaTextureObjects* texture_objects,
            cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
            float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
            float4* prior_planes, unsigned int* plane_masks,
            int3* neighbor_patchFill_cuda, Proxy_DisPlane* proxy_plane_cuda,
            const PatchMatchParamsLF params, const int iter,
            const int width, const int height);

    __global__ void RedPixelUpdate_MIPM_PFPGR_Collect(cudaTextureObjects *texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                int3* neighbor_patchFill_cuda,
                int3* neighbor_PGR_cuda,
                const PatchMatchParamsLF params, const int iter,
                const int width, const int height, int2* tilekeys_cuda);
    __global__ void RedPixelUpdate_PFPGR_Repair(cudaTextureObjects *texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                int3* neighbor_patchFill_cuda, Proxy_DisPlane* proxy_plane_cuda,
                const PatchMatchParamsLF params, const int iter,
                const int width, const int height);
}
#endif //ADAPTMIPMPFPGRUTIL_CUH
