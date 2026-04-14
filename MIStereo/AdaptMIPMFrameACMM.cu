/********************************************************************
file base:      AdaptMIPMFrameACMM.cu
author:         OpenAI + LZD workflow
created:        2026/04/14
purpose:        ACMM风格的整帧GPU版微图像视差匹配（多尺度 + 几何一致性）
*********************************************************************/
#include "AdaptMIPMFrameACMM.h"

#include "CudaUtil.h"
#include "MIStereo/AdaptMIPMUtil.cuh"

namespace LFMVS
{
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

    static __device__ __forceinline__ float clamp_cost_acmm(float v)
    {
        return fminf(2.0f, fmaxf(0.0f, v));
    }

    static __device__ __forceinline__ float safe_denom_acmm(float v)
    {
        if (fabsf(v) < 1e-6f)
            return (v >= 0.0f) ? 1e-6f : -1e-6f;
        return v;
    }

    static __device__ __forceinline__ float4 EncodePlaneHypothesisFromDispPlaneACMM(
        const int2 anchor,
        const float alpha,
        const float beta,
        const float gamma)
    {
        float nx = -alpha;
        float ny = -beta;
        float nz =  1.0f;

        const float inv_norm = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
        nx *= inv_norm;
        ny *= inv_norm;
        nz *= inv_norm;

        const float d_anchor = alpha * anchor.x + beta * anchor.y + gamma;
        return make_float4(nx, ny, nz, d_anchor);
    }

    static __device__ __forceinline__ float EvalDisparityAtPixelACMM(const float3& d_plane, const int2& p)
    {
        return d_plane.x * p.x + d_plane.y * p.y + d_plane.z;
    }

    static __device__ __forceinline__ float2 MapRefToSrcFastHexACMM(const int2& ref_pt,
                                                                     const float disparity,
                                                                     const float step_x,
                                                                     const float step_y)
    {
        return make_float2(ref_pt.x + step_x * disparity,
                           ref_pt.y + step_y * disparity);
    }

    static __device__ float ComputeBilateralNCC_ACMM(
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
        const float cost_max = 2.0f;
        const int radius = params.patch_size / 2;
        const float pixel_max = 255.0f;

        blur_value = make_float2(0.0f, 0.0f);

        float2 pt_anchor;
        DisparityGeometricMapOperate_Hex(c0, c1, p, plane_hypothesis, p, params,
                                         pt_anchor, disparity_baseline, tk0, tk1);

        if (pt_anchor.x >= params.MLA_Mask_Width_Cuda || pt_anchor.x < 0.0f ||
            pt_anchor.y >= params.MLA_Mask_Height_Cuda || pt_anchor.y < 0.0f)
        {
            return cost_max;
        }

        const float3 d_plane = DisparityPlane(p, plane_hypothesis);
        const float inv_base = 1.0f / fmaxf(params.Base, 1e-6f);
        const float step_x = (c0.x - c1.x) * inv_base;
        const float step_y = (c0.y - c1.y) * inv_base;

        float sum_ref = 0.0f, sum_ref_ref = 0.0f;
        float sum_src = 0.0f, sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;

        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

        for (int i = -radius; i <= radius; i += params.radius_increment)
        {
            for (int j = -radius; j <= radius; j += params.radius_increment)
            {
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                if (ref_pt.x < 0 || ref_pt.x >= params.MLA_Mask_Width_Cuda ||
                    ref_pt.y < 0 || ref_pt.y >= params.MLA_Mask_Height_Cuda)
                    continue;

                const float d = EvalDisparityAtPixelACMM(d_plane, ref_pt);
                const float2 src_pt = MapRefToSrcFastHexACMM(ref_pt, d, step_x, step_y);
                if (src_pt.x < 0.0f || src_pt.x >= params.MLA_Mask_Width_Cuda ||
                    src_pt.y < 0.0f || src_pt.y >= params.MLA_Mask_Height_Cuda)
                    continue;

                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
                const float ref_blur_v = tex2D<float>(ref_blur_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f) / pixel_max;
                const float src_blur_v = tex2D<float>(src_blur_image, src_pt.x + 0.5f, src_pt.y + 0.5f) / pixel_max;
                blur_value.x += ref_blur_v;
                blur_value.y += src_blur_v;

                const float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix,
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
            return cost_max;

        const float inv = 1.0f / bilateral_weight_sum;
        sum_ref *= inv; sum_ref_ref *= inv;
        sum_src *= inv; sum_src_src *= inv;
        sum_ref_src *= inv;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;
        if (var_ref < 1e-5f || var_src < 1e-5f)
            return cost_max;

        const float covar = sum_ref_src - sum_ref * sum_src;
        const float denom = sqrtf(var_ref * var_src);
        return clamp_cost_acmm(1.0f - covar / denom);
    }

    static __device__ void ComputeMultiViewCostVector_ACMM(
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

        for (int i = 0; i < 32; ++i)
        {
            cost_vector[i] = 2.0f;
            blur_vector[i] = make_float2(0.0f, 0.0f);
            disp_vector[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            const int nvid = neighbor_ids[ref_vid * max_neighbors + k];
            if (nvid < 0)
                continue;

            float2 blur_v = make_float2(0.0f, 0.0f);
            float4 disp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float c = ComputeBilateralNCC_ACMM(images[ref_vid], blur_images[ref_vid], c0, tk0,
                                               images[nvid], blur_images[nvid], centers[nvid], tilekeys[nvid],
                                               p, plane_hypothesis, params, blur_v, disp_b);
            const float blur_weight = expf(-0.007368f * (blur_v.x - blur_v.y) * (blur_v.x - blur_v.y));
            c = clamp_cost_acmm(c + 2.0f * (1.0f - blur_weight));
            cost_vector[k] = c;
            blur_vector[k] = blur_v;
            disp_vector[k] = disp_b;
        }
    }

    static __device__ unsigned int ComputeSelectedViewsTopK_ACMM(const float* cost_vector,
                                                                  const int neigh_count,
                                                                  const int top_k,
                                                                  float& mean_cost,
                                                                  int& best_idx)
    {
        best_idx = -1;
        int valid = 0;
        for (int i = 0; i < neigh_count; ++i)
        {
            if (cost_vector[i] < 2.0f)
            {
                ++valid;
                if (best_idx < 0 || cost_vector[i] < cost_vector[best_idx])
                    best_idx = i;
            }
        }

        if (valid <= 0)
        {
            mean_cost = 2.0f;
            return 0;
        }

        const int k = min(valid, top_k);
        float tmp[32];
        for (int i = 0; i < 32; ++i) tmp[i] = cost_vector[i];
        for (int i = 0; i < k; ++i)
        {
            for (int j = i + 1; j < neigh_count; ++j)
            {
                if (tmp[j] < tmp[i])
                {
                    const float t = tmp[i];
                    tmp[i] = tmp[j];
                    tmp[j] = t;
                }
            }
        }

        mean_cost = 0.0f;
        for (int i = 0; i < k; ++i) mean_cost += tmp[i];
        mean_cost /= float(k);
        const float threshold = tmp[k - 1];

        unsigned int mask = 0;
        for (int i = 0; i < neigh_count && i < 32; ++i)
            if (cost_vector[i] <= threshold)
                setBit(mask, i);
        return mask;
    }

    static __device__ bool BuildCrossViewPlaneProposal_ACMM(
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

        const int2 q = make_int2(qx, qy);
        const int q_local = q.y * params.MLA_Mask_Width_Cuda + q.x;
        const float4 neigh_plane_h = plane_prev[FrameIndexACMM1D(neigh_vid, q_local, pixels_per_view)];
        const float3 neigh_disp_plane = DisparityPlane(q, neigh_plane_h);

        const float alpha = neigh_disp_plane.x;
        const float beta  = neigh_disp_plane.y;
        const float gamma = neigh_disp_plane.z;
        const float du = centers[ref_vid].x - centers[neigh_vid].x;
        const float dv = centers[ref_vid].y - centers[neigh_vid].y;
        const float denom = safe_denom_acmm(1.0f + alpha * du + beta * dv);

        out_plane = EncodePlaneHypothesisFromDispPlaneACMM(
            p, alpha / denom, beta / denom, gamma / denom);
        return true;
    }

    static __device__ float ComputeCycleGeoCost_ACMM(
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
        const int best_view_local_id,
        const float geom_clip)
    {
        if (best_view_local_id < 0 || best_view_local_id >= neighbor_counts[ref_vid])
            return 0.0f;
        const int neigh_vid = neighbor_ids[ref_vid * max_neighbors + best_view_local_id];
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

    static __device__ float ComputeTotalCost_ACMM(
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
        float4& out_best_disp,
        unsigned int& out_sel_mask)
    {
        float cost_vector[32];
        float2 blur_vector[32];
        float4 disp_vector[32];
        ComputeMultiViewCostVector_ACMM(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts,
                                        max_neighbors, params, ref_vid, p, plane_hypothesis,
                                        cost_vector, blur_vector, disp_vector);

        float photo_cost = 2.0f;
        int best_view_local_id = -1;
        out_sel_mask = ComputeSelectedViewsTopK_ACMM(cost_vector, neighbor_counts[ref_vid], params.top_k,
                                                     photo_cost, best_view_local_id);
        out_best_disp = (best_view_local_id >= 0) ? disp_vector[best_view_local_id] : make_float4(0,0,0,0);

        float scale_cost = 0.0f;
        if (use_warm_start)
            scale_cost = fabsf(plane_hypothesis.w - plane_init[idx].w);

        const float geo_cost = ComputeCycleGeoCost_ACMM(plane_prev, centers, tilekeys,
                                                        neighbor_ids, neighbor_counts, max_neighbors,
                                                        pixels_per_view, params, ref_vid, p,
                                                        plane_hypothesis, best_view_local_id, geom_clip);
        return clamp_cost_acmm(photo_cost + lambda_scale * scale_cost + lambda_geo * geo_cost);
    }

    static __device__ void PlaneHypothesisRefinement_ACMM(
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
        curandState* rand_state,
        const bool use_warm_start,
        const float lambda_scale,
        const float lambda_geo,
        const float geom_clip,
        float4& best_plane,
        float& best_cost,
        float4& best_disp,
        unsigned int& best_sel)
    {
        const float perturbation = 0.02f;
        const float depth_now = best_plane.w;
        float depth_rand;

        if (best_cost < 0.1f)
            depth_rand = (curand_uniform(rand_state) - 0.5f) * 2.0f + depth_now;
        else
            depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;

        float4 plane_rand_normal = GenerateRandomNormalLF(p, rand_state, depth_now);
        float4 plane_perturbed = GeneratePerturbedNormalLF(p, best_plane, rand_state, perturbation * M_PI);

        float depth_perturbed = depth_now;
        const float depth_min_perturbed = (1.0f - perturbation) * depth_perturbed;
        const float depth_max_perturbed = (1.0f + perturbation) * depth_perturbed;
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;

        float4 candidates[5];
        candidates[0] = GenerateRandomPlaneHypothesis_MIPM(p, rand_state, params.depth_min, params.depth_max);
        candidates[1] = plane_rand_normal;
        candidates[2] = plane_perturbed;
        candidates[3] = best_plane;
        candidates[4] = best_plane;
        candidates[4].w = depth_perturbed;

        for (int i = 1; i < 5; ++i)
        {
            float3 d_plane = DisparityPlane(p, candidates[i]);
            candidates[i].w = to_disparity(p, d_plane);
        }

        for (int i = 0; i < 5; ++i)
        {
            float4 cand_disp;
            unsigned int cand_sel = 0;
            const float c = ComputeTotalCost_ACMM(texture_objects, centers, tilekeys,
                                                  neighbor_ids, neighbor_counts, max_neighbors,
                                                  plane_prev, plane_init, params, pixels_per_view,
                                                  ref_vid, p, idx, candidates[i],
                                                  use_warm_start, lambda_scale, lambda_geo, geom_clip,
                                                  cand_disp, cand_sel);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = candidates[i];
                best_disp = cand_disp;
                best_sel = cand_sel;
            }
        }
    }

    __global__ void InitializeFrameACMM(
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
        const bool use_warm_start,
        const float lambda_scale,
        const float lambda_geo,
        const float geom_clip)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int vid = blockIdx.z;
        if (vid >= num_views || x >= width || y >= height)
            return;

        const int pixels_per_view = width * height;
        const int idx = FrameIndexACMM(vid, x, y, width, pixels_per_view);
        const int2 p = make_int2(x, y);
        curand_init(clock64(), (unsigned long long)vid * (unsigned long long)pixels_per_view + idx, 0, &rand_states[idx]);

        float4 plane = use_warm_start ? plane_init[idx]
                                      : GenerateRandomPlaneHypothesis_MIPM(p, &rand_states[idx], params.depth_min, params.depth_max);
        if (!use_warm_start && plane.w <= 0.0f)
            plane.w = curand_uniform(&rand_states[idx]) * (params.depth_max - params.depth_min) + params.depth_min;

        float4 best_disp;
        unsigned int sel_mask = 0;
        const float total_cost = ComputeTotalCost_ACMM(texture_objects, centers, tilekeys,
                                                       neighbor_ids, neighbor_counts, max_neighbors,
                                                       plane_prev, plane_init, params, pixels_per_view,
                                                       vid, p, idx, plane,
                                                       use_warm_start, lambda_scale, lambda_geo, geom_clip,
                                                       best_disp, sel_mask);

        plane_prev[idx] = plane;
        cost_prev[idx] = total_cost;
        disp_prev[idx] = best_disp;
        selected_prev[idx] = sel_mask;

        cost_init[idx] = total_cost;
        disp_init[idx] = best_disp;
        selected_init[idx] = sel_mask;
    }

    __global__ void CheckerboardUpdateFrameACMM(
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
        const float4* disp_init,
        const unsigned int* selected_init,
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
        const bool use_warm_start,
        const float lambda_scale,
        const float lambda_geo,
        const float detail_th,
        const float geom_clip)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int vid = blockIdx.z;
        if (vid >= num_views || x >= width || y >= height)
            return;

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
        for (int k = 0; k < 8; ++k)
        {
            const int nx = x + dxs[k];
            const int ny = y + dys[k];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
            const int nidx = FrameIndexACMM(vid, nx, ny, width, pixels_per_view);
            float4 cand_disp;
            unsigned int cand_sel = 0;
            const float c = ComputeTotalCost_ACMM(texture_objects, centers, tilekeys,
                                                  neighbor_ids, neighbor_counts, max_neighbors,
                                                  plane_prev, plane_init, params, pixels_per_view,
                                                  vid, p, idx, plane_prev[nidx],
                                                  use_warm_start, lambda_scale, lambda_geo, geom_clip,
                                                  cand_disp, cand_sel);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = plane_prev[nidx];
                best_disp = cand_disp;
                best_sel = cand_sel;
            }
        }

        const int neigh_count = neighbor_counts[vid];
        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            const int nvid = neighbor_ids[vid * max_neighbors + k];
            if (nvid < 0)
                continue;
            float4 proxy_plane;
            if (!BuildCrossViewPlaneProposal_ACMM(plane_prev, centers, tilekeys, pixels_per_view,
                                                  params, vid, nvid, p, best_plane, proxy_plane))
                continue;
            float4 proxy_disp;
            unsigned int proxy_sel = 0;
            const float c = ComputeTotalCost_ACMM(texture_objects, centers, tilekeys,
                                                  neighbor_ids, neighbor_counts, max_neighbors,
                                                  plane_prev, plane_init, params, pixels_per_view,
                                                  vid, p, idx, proxy_plane,
                                                  use_warm_start, lambda_scale, lambda_geo, geom_clip,
                                                  proxy_disp, proxy_sel);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = proxy_plane;
                best_disp = proxy_disp;
                best_sel = proxy_sel;
            }
        }

        PlaneHypothesisRefinement_ACMM(texture_objects, centers, tilekeys,
                                       neighbor_ids, neighbor_counts, max_neighbors,
                                       plane_prev, plane_init, params, pixels_per_view,
                                       vid, p, idx, &rand_states[idx],
                                       use_warm_start, lambda_scale, lambda_geo, geom_clip,
                                       best_plane, best_cost, best_disp, best_sel);

        if (use_warm_start)
        {
            const float init_cost = cost_init[idx];
            if ((init_cost - best_cost) > detail_th)
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
        dim3 block_size(16, 16, 1);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y,
                       num_views);

        InitializeFrameACMM<<<grid_size, block_size>>>(
            texture_objects_cuda, centers_cuda, tilekeys_cuda, neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
            plane_prev_cuda, cost_prev_cuda, disp_prev_cuda, selected_prev_cuda,
            plane_init_cuda, cost_init_cuda, disp_init_cuda, selected_init_cuda,
            rand_states_cuda, params, width, height, num_views,
            use_warm_start, lambda_scale, lambda_geo, geom_clip);
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
            CheckerboardUpdateFrameACMM<<<grid_size, block_size>>>(
                texture_objects_cuda, centers_cuda, tilekeys_cuda, neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
                plane_in, cost_in, disp_in, sel_in,
                plane_init_cuda, cost_init_cuda, disp_init_cuda, selected_init_cuda,
                plane_out, cost_out, disp_out, sel_out,
                rand_states_cuda, params, width, height, num_views, 0,
                use_warm_start, lambda_scale, lambda_geo, detail_th, geom_clip);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_in, plane_out); std::swap(cost_in, cost_out);
            std::swap(disp_in, disp_out); std::swap(sel_in, sel_out);

            CheckerboardUpdateFrameACMM<<<grid_size, block_size>>>(
                texture_objects_cuda, centers_cuda, tilekeys_cuda, neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
                plane_in, cost_in, disp_in, sel_in,
                plane_init_cuda, cost_init_cuda, disp_init_cuda, selected_init_cuda,
                plane_out, cost_out, disp_out, sel_out,
                rand_states_cuda, params, width, height, num_views, 1,
                use_warm_start, lambda_scale, lambda_geo, detail_th, geom_clip);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_in, plane_out); std::swap(cost_in, cost_out);
            std::swap(disp_in, disp_out); std::swap(sel_in, sel_out);
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
