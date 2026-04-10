/********************************************************************
file base:      CudaUtil.h
author:         LZD
created:        2025/06/13
purpose:
*********************************************************************/
#ifndef CUDAUTIL_H
#define CUDAUTIL_H

#include "Common/Common.h"
#include "Common/CommonCUDA.h"

#include "MVStereo/LFDepthInfo.h"

namespace LFMVS
{
    __device__  void setBit(unsigned int &input, const unsigned int n);
    __device__  int isSet(unsigned int input, const unsigned int n);
    __device__ void sort_small(float* d, const int n);

    __device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result);
    __device__ float Vec3DotVec3(const float4 vec1, const float4 vec2);
    __device__ void NormalizeVec3 (float4 *vec);

    __device__ int FindMinCostIndex(const float* costs, const int n);
    __device__ int FindMaxCostIndex(const float *costs, const int n);

    ////////////////////////////////////////////////////////////////
    __device__ float4 GetViewDirectionLF( const int2 p, const float depth);

    // 随机初始化
    __device__ float4 GeneratePerturbedNormalLF(const int2 p, const float4 normal,
                        curandState *rand_state, const float perturbation);
    __device__ float4 GenerateRandomPlaneHypothesisLF(const int2 p, curandState *rand_state,
        const float depth_min, const float depth_max);
    __device__ float4 GenerateRandomNormalLF( const int2 p, curandState* rand_state, const float depth);


    // 计算匹配代价值
    __device__ float ComputeMultiViewInitialCostandSelectedViewsLF(const cudaTextureObject_t* images,
                    float2* pcenters, const int2 p, const float4 plane_hypothesis,
                    unsigned int* selected_views, const PatchMatchParamsLF params);
    __device__ float ComputeBilateralNCCLF(const cudaTextureObject_t ref_image, float2 p0,
                    const cudaTextureObject_t src_image, float2 p1 , const int2 p,
                    const float4 plane_hypothesis, const PatchMatchParamsLF params);
    __device__ float3 DisparityPlane(const int2 p, const float4 plane_hypothesis);
    __device__ float to_disparity(const int2 p, float3 d_plane);
    __device__ float ComputeBilateralWeight(const float x_dist, const float y_dist,
        const float pix, const float center_pix, const float sigma_spatial, const float sigma_color);


    // 传播
    __global__ void BlackPixelUpdateLF(cudaTextureObjects* texture_objects,
            cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
            float* costs, curandState* rand_states, unsigned int* selected_views, float4* prior_planes,
            unsigned int* plane_masks, const PatchMatchParamsLF params,
            const int iter, const int width, const int height);
    __device__ void CheckerboardPropagationLF_Near(const cudaTextureObject_t* images,
                const cudaTextureObject_t* depths, float4* plane_hypotheses,
                float2* pcenters,float* costs, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes,
                unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
                const int iter, const int WIDTH, const int HEIGHT);
    __device__ void PlaneHypothesisRefinementLF(const cudaTextureObject_t* images,
                    const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
                    float2 * pcenters, float *depth, float *cost, curandState* rand_state,
                    const float* view_weights, const float weight_norm, float4* prior_planes,
                    unsigned int* plane_masks, float* restricted_cost, const int2 p,
                    const PatchMatchParamsLF params, const int WIDTH);
    __device__ void ComputeMultiViewCostVectorLF(const cudaTextureObject_t* images, float2* pcenters,
                        const int2 p, const float4 plane_hypothesis,
                        float* cost_vector, const PatchMatchParamsLF params);
    __device__ void TransformPDFToCDF(float* probs, const int num_probs);

    __global__ void RedPixelUpdateLF(cudaTextureObjects *texture_objects,
        cudaTextureObjects* texture_depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, curandState* rand_states,
        unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
        const PatchMatchParamsLF params, const int iter,
        const int width, const int height);

}

#endif //CUDAUTIL_H
