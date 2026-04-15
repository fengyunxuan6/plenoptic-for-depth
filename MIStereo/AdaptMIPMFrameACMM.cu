/********************************************************************
file base:      AdaptMIPMFrameACMM.cu
author:         OpenAI + LZD workflow
created:        2026/04/15
purpose:        ACMM风格的整帧GPU版微图像视差匹配（重写版，接近 ACMM 的“联合视图选择 + 棋盘传播”）
*********************************************************************/
#include "AdaptMIPMFrameACMM.h"

#include "CudaUtil.h"
#include "MIStereo/AdaptMIPMUtil.cuh"

namespace LFMVS
{
    static constexpr int ACMM_MAX_LOCAL_NEI = 32;
    static constexpr float ACMM_COST_MAX = 2.0f;

    static __device__ __forceinline__ int FrameIndexACMM(const int vid, const int x, const int y,
                                                         const int width, const int pixels_per_view)
    {
        return vid * pixels_per_view + y * width + x;
    }

    static __device__ __forceinline__ int FrameIndexACMM1D(const int vid, const int idx_local,
                                                           const int pixels_per_view)
    {
        return vid * pixels_per_view + idx_local;
    }

    static __device__ __forceinline__ float ClampCostACMM(float v)
    {
        return fminf(ACMM_COST_MAX, fmaxf(0.0f, v));
    }

    static __device__ __forceinline__ void setBitACMM(unsigned int &input, const unsigned int n)
    {
        input |= (unsigned int)(1u << n);
    }

    static __device__ __forceinline__ int isSetACMM(unsigned int input, const unsigned int n)
    {
        return (input >> n) & 1u;
    }

    static __device__ __forceinline__ float4 MakeFrontoPlaneACMM(float d)
    {
        return make_float4(0.0f, 0.0f, 1.0f, d);
    }

    static __device__ __forceinline__ float EvalDispACMM(const float3& d_plane, const int2& p)
    {
        return d_plane.x * p.x + d_plane.y * p.y + d_plane.z;
    }

    static __device__ float ComputeLFPairCostACMM(
        const cudaTextureObject_t ref_image,
        const cudaTextureObject_t ref_blur_image,
        const float2 c0,
        const int2 tk0,
        const cudaTextureObject_t src_image,
        const cudaTextureObject_t src_blur_image,
        const float2 c1,
        const int2 tk1,
        const int2 p,
        const float4 plane_hypothesis,
        const PatchMatchParamsLF params,
        float2& blur_value,
        float4& disparity_baseline)
    {
        const int radius = params.patch_size / 2;
        const float pixel_max = 255.0f;

        blur_value = make_float2(0.0f, 0.0f);

        float2 pt_anchor;
        DisparityGeometricMapOperate_Hex(c0, c1, p, plane_hypothesis, p, params,
                                         pt_anchor, disparity_baseline, tk0, tk1);

        if (pt_anchor.x < 0.0f || pt_anchor.x >= params.MLA_Mask_Width_Cuda ||
            pt_anchor.y < 0.0f || pt_anchor.y >= params.MLA_Mask_Height_Cuda)
            return ACMM_COST_MAX;

        const float3 d_plane = DisparityPlane(p, plane_hypothesis);
        const float inv_base = 1.0f / fmaxf(params.Base, 1e-6f);
        const float step_x = (c0.x - c1.x) * inv_base;
        const float step_y = (c0.y - c1.y) * inv_base;

        float sum_ref = 0.0f, sum_ref_ref = 0.0f;
        float sum_src = 0.0f, sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;

        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

        for (int j = -radius; j <= radius; j += params.radius_increment)
        {
            for (int i = -radius; i <= radius; i += params.radius_increment)
            {
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                if (ref_pt.x < 0 || ref_pt.x >= params.MLA_Mask_Width_Cuda ||
                    ref_pt.y < 0 || ref_pt.y >= params.MLA_Mask_Height_Cuda)
                    continue;

                const float d = EvalDispACMM(d_plane, ref_pt);
                const float2 src_pt = make_float2(ref_pt.x + step_x * d, ref_pt.y + step_y * d);
                if (src_pt.x < 0.0f || src_pt.x >= params.MLA_Mask_Width_Cuda ||
                    src_pt.y < 0.0f || src_pt.y >= params.MLA_Mask_Height_Cuda)
                    continue;

                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                const float ref_blur_v = tex2D<float>(ref_blur_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f) / pixel_max;
                const float src_blur_v = tex2D<float>(src_blur_image, src_pt.x + 0.5f, src_pt.y + 0.5f) / pixel_max;
                blur_value.x += ref_blur_v;
                blur_value.y += src_blur_v;

                const float weight = ComputeBilateralWeight((float)i, (float)j, ref_pix, ref_center_pix,
                                                            params.sigma_spatial, params.sigma_color);

                sum_ref += weight * ref_pix;
                sum_ref_ref += weight * ref_pix * ref_pix;
                sum_src += weight * src_pix;
                sum_src_src += weight * src_pix * src_pix;
                sum_ref_src += weight * ref_pix * src_pix;
                bilateral_weight_sum += weight;
            }
        }

        if (bilateral_weight_sum <= 1e-6f)
            return ACMM_COST_MAX;

        const float inv_w = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_w; sum_ref_ref *= inv_w;
        sum_src *= inv_w; sum_src_src *= inv_w;
        sum_ref_src *= inv_w;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;
        if (var_ref < 1e-5f || var_src < 1e-5f)
            return ACMM_COST_MAX;

        const float cov = sum_ref_src - sum_ref * sum_src;
        return ClampCostACMM(1.0f - cov / sqrtf(var_ref * var_src));
    }

    static __device__ void ComputeCostVectorACMM(
        const cudaTextureObjects* texture_objects,
        const float2* centers,
        const int2* tilekeys,
        const int* neighbor_ids,
        const int* neighbor_counts,
        const int max_neighbors,
        const PatchMatchParamsLF params,
        const int ref_vid,
        const int2 p,
        const float4 plane_hypothesis,
        float* cost_vector,
        float2* blur_vector,
        float4* disp_vector)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;
        const float2 c0 = centers[ref_vid];
        const int2 tk0 = tilekeys[ref_vid];
        const int neigh_count = neighbor_counts[ref_vid];

        for (int i = 0; i < ACMM_MAX_LOCAL_NEI; ++i)
        {
            cost_vector[i] = ACMM_COST_MAX;
            blur_vector[i] = make_float2(0.0f, 0.0f);
            disp_vector[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        for (int k = 0; k < neigh_count && k < max_neighbors && k < ACMM_MAX_LOCAL_NEI; ++k)
        {
            const int nvid = neighbor_ids[ref_vid * max_neighbors + k];
            if (nvid < 0)
                continue;

            float2 blur_v = make_float2(0.0f, 0.0f);
            float4 disp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float c = ComputeLFPairCostACMM(images[ref_vid], blur_images[ref_vid], c0, tk0,
                                            images[nvid], blur_images[nvid], centers[nvid], tilekeys[nvid],
                                            p, plane_hypothesis, params, blur_v, disp_b);
            const float blur_weight = expf(-0.007368f * (blur_v.x - blur_v.y) * (blur_v.x - blur_v.y));
            c = ClampCostACMM(c + 0.8f * (1.0f - blur_weight));
            cost_vector[k] = c;
            blur_vector[k] = blur_v;
            disp_vector[k] = disp_b;
        }
    }

    static __device__ void BuildSimpleTopKMaskACMM(
        const float* cost_vector,
        const int neigh_count,
        const int top_k,
        unsigned int& sel_mask,
        float& mean_cost,
        int& best_local_id)
    {
        sel_mask = 0;
        best_local_id = -1;
        mean_cost = ACMM_COST_MAX;

        int valid = 0;
        float tmp[ACMM_MAX_LOCAL_NEI];
        for (int i = 0; i < neigh_count && i < ACMM_MAX_LOCAL_NEI; ++i)
        {
            tmp[i] = cost_vector[i];
            if (cost_vector[i] < ACMM_COST_MAX)
            {
                ++valid;
                if (best_local_id < 0 || cost_vector[i] < cost_vector[best_local_id])
                    best_local_id = i;
            }
        }
        if (valid <= 0) return;

        const int k = min(valid, top_k);
        for (int i = 1; i < neigh_count; ++i)
        {
            float v = tmp[i];
            int j = i;
            while (j > 0 && tmp[j - 1] > v)
            {
                tmp[j] = tmp[j - 1];
                --j;
            }
            tmp[j] = v;
        }

        mean_cost = 0.0f;
        for (int i = 0; i < k; ++i) mean_cost += tmp[i];
        mean_cost /= float(k);
        const float threshold = tmp[k - 1];

        for (int i = 0; i < neigh_count && i < ACMM_MAX_LOCAL_NEI; ++i)
            if (cost_vector[i] <= threshold)
                setBitACMM(sel_mask, i);
    }

    static __device__ void ComputeJointViewWeightsACMM(
        const float neighbor_costs[8][ACMM_MAX_LOCAL_NEI],
        const bool valid_flags[8],
        const unsigned int* selected_prev,
        const int width,
        const int height,
        const int x,
        const int y,
        const int vid,
        const int pixels_per_view,
        const int neigh_count,
        const int iter,
        curandState* rand_state,
        float* view_weights)
    {
        for (int i = 0; i < ACMM_MAX_LOCAL_NEI; ++i) view_weights[i] = 0.0f;
        if (neigh_count <= 0) return;

        float priors[ACMM_MAX_LOCAL_NEI];
        for (int i = 0; i < ACMM_MAX_LOCAL_NEI; ++i) priors[i] = 0.0f;

        const int idx_up    = (y > 0)         ? FrameIndexACMM(vid, x, y - 1, width, pixels_per_view) : -1;
        const int idx_down  = (y + 1 < height)? FrameIndexACMM(vid, x, y + 1, width, pixels_per_view) : -1;
        const int idx_left  = (x > 0)         ? FrameIndexACMM(vid, x - 1, y, width, pixels_per_view) : -1;
        const int idx_right = (x + 1 < width) ? FrameIndexACMM(vid, x + 1, y, width, pixels_per_view) : -1;
        const int prior_ids[4] = {idx_up, idx_down, idx_left, idx_right};

        for (int pi = 0; pi < 4; ++pi)
        {
            if (prior_ids[pi] < 0) continue;
            const unsigned int mask = selected_prev[prior_ids[pi]];
            for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j)
                priors[j] += isSetACMM(mask, j) ? 0.9f : 0.1f;
        }

        float probs[ACMM_MAX_LOCAL_NEI];
        const float cost_threshold = 0.8f * expf(-(float)(iter * iter) / 90.0f);

        for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j)
        {
            float cnt_good = 0.0f;
            int cnt_bad = 0;
            float accum = 0.0f;
            for (int p = 0; p < 8; ++p)
            {
                if (!valid_flags[p]) continue;
                const float c = neighbor_costs[p][j];
                if (c < cost_threshold)
                {
                    accum += expf(-(c * c) / 0.18f);
                    cnt_good += 1.0f;
                }
                if (c > 1.2f) ++cnt_bad;
            }

            if (cnt_good > 2.0f && cnt_bad < 3)
                probs[j] = (accum / cnt_good) * fmaxf(priors[j], 0.1f);
            else if (cnt_bad < 3)
                probs[j] = expf(-(cost_threshold * cost_threshold) / 0.32f) * fmaxf(priors[j], 0.1f);
            else
                probs[j] = 0.0f;
        }

        float prob_sum = 0.0f;
        for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j) prob_sum += probs[j];
        if (prob_sum <= 1e-6f)
        {
            for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j) view_weights[j] = 1.0f;
            return;
        }

        for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j) probs[j] /= prob_sum;
        for (int j = 1; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j) probs[j] += probs[j - 1];

        for (int s = 0; s < 15; ++s)
        {
            const float r = curand_uniform(rand_state) - 1e-7f;
            for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j)
            {
                if (probs[j] > r)
                {
                    view_weights[j] += 1.0f;
                    break;
                }
            }
        }

        float wsum = 0.0f;
        for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j) wsum += view_weights[j];
        if (wsum <= 1e-6f)
        {
            for (int j = 0; j < neigh_count && j < ACMM_MAX_LOCAL_NEI; ++j) view_weights[j] = 1.0f;
        }
    }

    static __device__ float ComputeCycleGeoCostForLocalViewACMM(
        const float4* plane_prev,
        const float2* centers,
        const int2* tilekeys,
        const int* neighbor_ids,
        const int* neighbor_counts,
        const int max_neighbors,
        const int pixels_per_view,
        const PatchMatchParamsLF params,
        const int ref_vid,
        const int2 p,
        const float4 plane_hypothesis,
        const int local_view_id,
        const float geom_clip)
    {
        if (local_view_id < 0 || local_view_id >= neighbor_counts[ref_vid])
            return 0.0f;

        const int neigh_vid = neighbor_ids[ref_vid * max_neighbors + local_view_id];
        if (neigh_vid < 0)
            return 0.0f;

        float2 qf;
        float4 disp_fw;
        DisparityGeometricMapOperate_Hex(centers[ref_vid], centers[neigh_vid], p, plane_hypothesis, p,
                                         params, qf, disp_fw, tilekeys[ref_vid], tilekeys[neigh_vid]);

        const int qx = (int)floorf(qf.x + 0.5f);
        const int qy = (int)floorf(qf.y + 0.5f);
        if (qx < 0 || qx >= params.MLA_Mask_Width_Cuda || qy < 0 || qy >= params.MLA_Mask_Height_Cuda)
            return geom_clip;

        const int2 q = make_int2(qx, qy);
        const int q_idx = q.y * params.MLA_Mask_Width_Cuda + q.x;
        const float4 neigh_plane = plane_prev[FrameIndexACMM1D(neigh_vid, q_idx, pixels_per_view)];

        float2 pf_back;
        float4 disp_bw;
        DisparityGeometricMapOperate_Hex(centers[neigh_vid], centers[ref_vid], q, neigh_plane, q,
                                         params, pf_back, disp_bw, tilekeys[neigh_vid], tilekeys[ref_vid]);

        const float dx = pf_back.x - p.x;
        const float dy = pf_back.y - p.y;
        return fminf(geom_clip, sqrtf(dx * dx + dy * dy));
    }

    static __device__ float ComputePlaneWeightedCostACMM(
        const cudaTextureObjects* texture_objects,
        const float2* centers,
        const int2* tilekeys,
        const int* neighbor_ids,
        const int* neighbor_counts,
        const int max_neighbors,
        const float4* plane_prev,
        const float4* plane_init,
        const PatchMatchParamsLF params,
        const int pixels_per_view,
        const int ref_vid,
        const int2 p,
        const int idx,
        const float4 plane_hypothesis,
        const bool use_warm_start,
        const float lambda_scale,
        const float lambda_geo,
        const float geom_clip,
        const float* view_weights,
        float4& out_best_disp,
        unsigned int& out_sel_mask,
        int& out_best_local_id)
    {
        float cost_vector[ACMM_MAX_LOCAL_NEI];
        float2 blur_vector[ACMM_MAX_LOCAL_NEI];
        float4 disp_vector[ACMM_MAX_LOCAL_NEI];
        ComputeCostVectorACMM(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts,
                              max_neighbors, params, ref_vid, p, plane_hypothesis,
                              cost_vector, blur_vector, disp_vector);

        float photo_cost = 0.0f;
        float weight_norm = 0.0f;
        out_best_local_id = -1;
        out_best_disp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        out_sel_mask = 0;

        const int neigh_count = min(neighbor_counts[ref_vid], ACMM_MAX_LOCAL_NEI);
        for (int i = 0; i < neigh_count; ++i)
        {
            if (view_weights[i] > 0.0f && cost_vector[i] < ACMM_COST_MAX)
            {
                photo_cost += view_weights[i] * cost_vector[i];
                weight_norm += view_weights[i];
                setBitACMM(out_sel_mask, i);
                if (out_best_local_id < 0 || cost_vector[i] < cost_vector[out_best_local_id])
                    out_best_local_id = i;
            }
        }

        if (weight_norm <= 1e-6f)
        {
            float mean_cost;
            BuildSimpleTopKMaskACMM(cost_vector, neigh_count, params.top_k, out_sel_mask, mean_cost, out_best_local_id);
            photo_cost = mean_cost;
            weight_norm = 1.0f;
        }
        else
        {
            photo_cost /= weight_norm;
        }

        if (out_best_local_id >= 0)
            out_best_disp = disp_vector[out_best_local_id];

        float scale_cost = 0.0f;
        if (use_warm_start)
            scale_cost = fabsf(plane_hypothesis.w - plane_init[idx].w);

        float geo_cost = 0.0f;
        if (lambda_geo > 1e-6f && out_best_local_id >= 0)
            geo_cost = ComputeCycleGeoCostForLocalViewACMM(plane_prev, centers, tilekeys,
                                                           neighbor_ids, neighbor_counts, max_neighbors,
                                                           pixels_per_view, params, ref_vid, p,
                                                           plane_hypothesis, out_best_local_id, geom_clip);

        return ClampCostACMM(photo_cost + lambda_scale * scale_cost + lambda_geo * geo_cost);
    }

    static __device__ bool BuildCrossViewPlaneProposalACMM(
        const float4* plane_prev,
        const float2* centers,
        const int2* tilekeys,
        const int pixels_per_view,
        const PatchMatchParamsLF params,
        const int ref_vid,
        const int neigh_vid,
        const int2 p,
        const float4 current_plane,
        float4& out_plane)
    {
        float2 qf;
        float4 tmp_disp;
        DisparityGeometricMapOperate_Hex(centers[ref_vid], centers[neigh_vid], p, current_plane, p,
                                         params, qf, tmp_disp, tilekeys[ref_vid], tilekeys[neigh_vid]);
        const int qx = (int)floorf(qf.x + 0.5f);
        const int qy = (int)floorf(qf.y + 0.5f);
        if (qx < 0 || qx >= params.MLA_Mask_Width_Cuda || qy < 0 || qy >= params.MLA_Mask_Height_Cuda)
            return false;

        const int q_local = qy * params.MLA_Mask_Width_Cuda + qx;
        out_plane = plane_prev[FrameIndexACMM1D(neigh_vid, q_local, pixels_per_view)];
        return true;
    }

    __global__ void InitializeFrameACMM_Kernel(
        cudaTextureObjects* texture_objects,
        float2* centers,
        int2* tilekeys,
        int* neighbor_ids,
        int* neighbor_counts,
        int max_neighbors,
        float4* plane_prev,
        float* cost_prev,
        float4* disp_prev,
        unsigned int* selected_prev,
        float4* plane_init,
        float* cost_init,
        float4* disp_init,
        unsigned int* selected_init,
        curandState* rand_states,
        const PatchMatchParamsLF params,
        const int width,
        const int height,
        const int num_views,
        const bool use_warm_start)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int vid = blockIdx.z;
        if (vid >= num_views || x >= width || y >= height) return;

        const int pixels_per_view = width * height;
        const int idx = FrameIndexACMM(vid, x, y, width, pixels_per_view);
        const int2 p = make_int2(x, y);
        curand_init(1337ULL + (unsigned long long)idx, 0, 0, &rand_states[idx]);

        float4 plane = use_warm_start ? plane_init[idx]
                                      : GenerateRandomPlaneHypothesis_MIPM(p, &rand_states[idx], params.disparity_min, params.disparity_max);
        if (plane.w <= 0.0f)
            plane = MakeFrontoPlaneACMM(10.0f + 5.0f * curand_uniform(&rand_states[idx]));

        float cost_vector[ACMM_MAX_LOCAL_NEI];
        float2 blur_vector[ACMM_MAX_LOCAL_NEI];
        float4 disp_vector[ACMM_MAX_LOCAL_NEI];
        ComputeCostVectorACMM(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts,
                              max_neighbors, params, vid, p, plane,
                              cost_vector, blur_vector, disp_vector);

        unsigned int sel_mask = 0;
        float mean_cost = ACMM_COST_MAX;
        int best_id = -1;
        BuildSimpleTopKMaskACMM(cost_vector, min(neighbor_counts[vid], ACMM_MAX_LOCAL_NEI),
                                params.top_k, sel_mask, mean_cost, best_id);

        plane_prev[idx] = plane;
        cost_prev[idx] = mean_cost;
        disp_prev[idx] = (best_id >= 0) ? disp_vector[best_id] : make_float4(0,0,0,0);
        selected_prev[idx] = sel_mask;

        cost_init[idx] = mean_cost;
        disp_init[idx] = disp_prev[idx];
        selected_init[idx] = sel_mask;
    }

    __global__ void CheckerboardUpdateACMM_Kernel(
        cudaTextureObjects* texture_objects,
        float2* centers,
        int2* tilekeys,
        int* neighbor_ids,
        int* neighbor_counts,
        int max_neighbors,
        const float4* plane_prev,
        const float* cost_prev,
        const float4* disp_prev,
        const unsigned int* selected_prev,
        const float4* plane_init,
        const float* cost_init,
        float4* plane_next,
        float* cost_next,
        float4* disp_next,
        unsigned int* selected_next,
        curandState* rand_states,
        const PatchMatchParamsLF params,
        const int width,
        const int height,
        const int num_views,
        const int phase,
        const int iter,
        const bool use_warm_start,
        const float lambda_scale,
        const float lambda_geo,
        const float detail_th,
        const float geom_clip)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int vid = blockIdx.z;
        if (vid >= num_views || x >= width || y >= height) return;

        const int pixels_per_view = width * height;
        const int idx = FrameIndexACMM(vid, x, y, width, pixels_per_view);

        if (((x + y) & 1) != phase)
        {
            plane_next[idx] = plane_prev[idx];
            cost_next[idx] = cost_prev[idx];
            disp_next[idx] = disp_prev[idx];
            selected_next[idx] = selected_prev[idx];
            return;
        }

        const int2 p = make_int2(x, y);
        float4 best_plane = plane_prev[idx];
        float best_cost = cost_prev[idx];
        float4 best_disp = disp_prev[idx];
        unsigned int best_sel = selected_prev[idx];

        const int dxs[8] = { 0, 0, -1, 1, -3, 3, 0, 0 };
        const int dys[8] = { -1, 1, 0, 0, 0, 0, -3, 3 };
        bool valid_flags[8];
        float neighbor_costs[8][ACMM_MAX_LOCAL_NEI];
        for (int n = 0; n < 8; ++n)
        {
            valid_flags[n] = false;
            for (int j = 0; j < ACMM_MAX_LOCAL_NEI; ++j)
                neighbor_costs[n][j] = ACMM_COST_MAX;
        }

        for (int n = 0; n < 8; ++n)
        {
            const int nx = x + dxs[n];
            const int ny = y + dys[n];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
            valid_flags[n] = true;
            const int nidx = FrameIndexACMM(vid, nx, ny, width, pixels_per_view);

            float2 blur_vector[ACMM_MAX_LOCAL_NEI];
            float4 disp_vector[ACMM_MAX_LOCAL_NEI];
            ComputeCostVectorACMM(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts,
                                  max_neighbors, params, vid, p, plane_prev[nidx],
                                  neighbor_costs[n], blur_vector, disp_vector);
        }

        float view_weights[ACMM_MAX_LOCAL_NEI];
        ComputeJointViewWeightsACMM(neighbor_costs, valid_flags, selected_prev,
                                    width, height, x, y, vid, pixels_per_view,
                                    min(neighbor_counts[vid], ACMM_MAX_LOCAL_NEI), iter,
                                    &rand_states[idx], view_weights);

        for (int n = 0; n < 8; ++n)
        {
            const int nx = x + dxs[n];
            const int ny = y + dys[n];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            const int nidx = FrameIndexACMM(vid, nx, ny, width, pixels_per_view);
            float4 cand_disp;
            unsigned int cand_sel = 0;
            int best_local_id = -1;
            const float cand_cost = ComputePlaneWeightedCostACMM(texture_objects, centers, tilekeys,
                neighbor_ids, neighbor_counts, max_neighbors, plane_prev, plane_init, params,
                pixels_per_view, vid, p, idx, plane_prev[nidx],
                use_warm_start, lambda_scale, lambda_geo, geom_clip,
                view_weights, cand_disp, cand_sel, best_local_id);

            if (cand_cost < best_cost)
            {
                best_cost = cand_cost;
                best_plane = plane_prev[nidx];
                best_disp = cand_disp;
                best_sel = cand_sel;
            }
        }

        const int neigh_count = min(neighbor_counts[vid], max_neighbors);
        for (int k = 0; k < neigh_count && k < 4; ++k)
        {
            const int nvid = neighbor_ids[vid * max_neighbors + k];
            if (nvid < 0) continue;

            float4 proxy_plane;
            if (!BuildCrossViewPlaneProposalACMM(plane_prev, centers, tilekeys,
                                                 pixels_per_view, params, vid, nvid, p, best_plane, proxy_plane))
                continue;

            float4 cand_disp;
            unsigned int cand_sel = 0;
            int best_local_id = -1;
            const float cand_cost = ComputePlaneWeightedCostACMM(texture_objects, centers, tilekeys,
                neighbor_ids, neighbor_counts, max_neighbors, plane_prev, plane_init, params,
                pixels_per_view, vid, p, idx, proxy_plane,
                use_warm_start, lambda_scale, lambda_geo, geom_clip,
                view_weights, cand_disp, cand_sel, best_local_id);

            if (cand_cost < best_cost)
            {
                best_cost = cand_cost;
                best_plane = proxy_plane;
                best_disp = cand_disp;
                best_sel = cand_sel;
            }
        }

        for (int t = 0; t < 3; ++t)
        {
            float4 cand = (t == 0)
                ? GenerateRandomPlaneHypothesis_MIPM(p, &rand_states[idx], params.disparity_min, params.disparity_max)
                : best_plane;
            if (t > 0)
            {
                const float d = best_plane.w * (0.85f + 0.30f * curand_uniform(&rand_states[idx]));
                cand = MakeFrontoPlaneACMM(d);
            }

            float4 cand_disp;
            unsigned int cand_sel = 0;
            int best_local_id = -1;
            const float cand_cost = ComputePlaneWeightedCostACMM(texture_objects, centers, tilekeys,
                neighbor_ids, neighbor_counts, max_neighbors, plane_prev, plane_init, params,
                pixels_per_view, vid, p, idx, cand,
                use_warm_start, lambda_scale, lambda_geo, geom_clip,
                view_weights, cand_disp, cand_sel, best_local_id);

            if (cand_cost < best_cost)
            {
                best_cost = cand_cost;
                best_plane = cand;
                best_disp = cand_disp;
                best_sel = cand_sel;
            }
        }

        if (use_warm_start)
        {
            const float init_cost = cost_init[idx];
            const float prev_cost = cost_prev[idx];
            if (best_cost < prev_cost || (init_cost - best_cost) > detail_th)
            {
                plane_next[idx] = best_plane;
                cost_next[idx] = best_cost;
                disp_next[idx] = best_disp;
                selected_next[idx] = best_sel;
            }
            else
            {
                plane_next[idx] = plane_prev[idx];
                cost_next[idx] = cost_prev[idx];
                disp_next[idx] = disp_prev[idx];
                selected_next[idx] = selected_prev[idx];
            }
        }
        else
        {
            plane_next[idx] = best_plane;
            cost_next[idx] = best_cost;
            disp_next[idx] = best_disp;
            selected_next[idx] = best_sel;
        }
    }

    void RunPatchMatchCUDAForFrameACMM_Impl(cudaTextureObjects* texture_objects_cuda,
        float2* centers_cuda, int2* tilekeys_cuda, int* neighbor_ids_cuda, int* neighbor_counts_cuda,
        float4* plane_prev_cuda, float4* plane_next_cuda, float* cost_prev_cuda, float* cost_next_cuda,
        float4* disp_prev_cuda, float4* disp_next_cuda, unsigned int* selected_prev_cuda, unsigned int* selected_next_cuda,
        float4* plane_init_cuda, float* cost_init_cuda, float4* disp_init_cuda, unsigned int* selected_init_cuda,
        curandState* rand_states_cuda, PatchMatchParamsLF params, int num_views, int max_neighbors,
        bool use_warm_start, float lambda_scale, float lambda_geo, float detail_th, float geom_clip)
    {
        const int width = params.MLA_Mask_Width_Cuda;
        const int height = params.MLA_Mask_Height_Cuda;
        dim3 block(16, 16, 1);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y,
                  num_views);

        InitializeFrameACMM_Kernel<<<grid, block>>>(
            texture_objects_cuda, centers_cuda, tilekeys_cuda,
            neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
            plane_prev_cuda, cost_prev_cuda, disp_prev_cuda, selected_prev_cuda,
            plane_init_cuda, cost_init_cuda, disp_init_cuda, selected_init_cuda,
            rand_states_cuda, params, width, height, num_views, use_warm_start);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        float4* plane_in = plane_prev_cuda;
        float4* plane_out = plane_next_cuda;
        float* cost_in = cost_prev_cuda;
        float* cost_out = cost_next_cuda;
        float4* disp_in = disp_prev_cuda;
        float4* disp_out = disp_next_cuda;
        unsigned int* sel_in = selected_prev_cuda;
        unsigned int* sel_out = selected_next_cuda;

        for (int iter = 0; iter < params.max_iterations; ++iter)
        {
            CheckerboardUpdateACMM_Kernel<<<grid, block>>>(
                texture_objects_cuda, centers_cuda, tilekeys_cuda,
                neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
                plane_in, cost_in, disp_in, sel_in,
                plane_init_cuda, cost_init_cuda,
                plane_out, cost_out, disp_out, sel_out,
                rand_states_cuda, params, width, height, num_views, 0, iter,
                use_warm_start, lambda_scale, lambda_geo, detail_th, geom_clip);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_in, plane_out);
            std::swap(cost_in, cost_out);
            std::swap(disp_in, disp_out);
            std::swap(sel_in, sel_out);

            CheckerboardUpdateACMM_Kernel<<<grid, block>>>(
                texture_objects_cuda, centers_cuda, tilekeys_cuda,
                neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
                plane_in, cost_in, disp_in, sel_in,
                plane_init_cuda, cost_init_cuda,
                plane_out, cost_out, disp_out, sel_out,
                rand_states_cuda, params, width, height, num_views, 1, iter,
                use_warm_start, lambda_scale, lambda_geo, detail_th, geom_clip);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_in, plane_out);
            std::swap(cost_in, cost_out);
            std::swap(disp_in, disp_out);
            std::swap(sel_in, sel_out);
        }

        if (plane_in != plane_prev_cuda)
        {
            const size_t frame_pixels = (size_t)width * (size_t)height * (size_t)num_views;
            CUDA_SAFE_CALL(cudaMemcpy(plane_prev_cuda, plane_in, sizeof(float4) * frame_pixels, cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(cost_prev_cuda, cost_in, sizeof(float) * frame_pixels, cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(disp_prev_cuda, disp_in, sizeof(float4) * frame_pixels, cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(selected_prev_cuda, sel_in, sizeof(unsigned int) * frame_pixels, cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }
    }
}
