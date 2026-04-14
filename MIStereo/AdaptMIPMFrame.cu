/********************************************************************
file base:      AdaptMIPMFrame.cu
author:         LZD
created:        2026/04/12
purpose:        整帧GPU版微图像视差匹配（跨视图proposal）
*********************************************************************/
#include "AdaptMIPMFrame.h"

#include "CudaUtil.h"
#include "MIStereo/AdaptMIPMUtil.cuh"

namespace LFMVS
{
    static __device__ __forceinline__ int FrameIndex(const int vid, const int x, const int y, const int width, const int pixels_per_view)
    {
        return vid * pixels_per_view + y * width + x;
    }

    static __device__ __forceinline__ int FrameIndex1D(const int vid, const int idx_local, const int pixels_per_view)
    {
        return vid * pixels_per_view + idx_local;
    }

    static __device__ __forceinline__ float clamp_cost(float v)
    {
        return fminf(2.0f, fmaxf(0.0f, v));
    }

    static __device__ __forceinline__ float SafeDenomDevice(float v)
    {
        if (fabsf(v) < 1e-6f)
            return (v >= 0.0f) ? 1e-6f : -1e-6f;
        return v;
    }

    static __device__ __forceinline__ float4 EncodePlaneHypothesisFromDispPlaneDevice(
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

    static __device__ __forceinline__ float EvalDisparityAtPixel_Frame(
        const float3& d_plane,
        const int2& p)
    {
        return d_plane.x * p.x + d_plane.y * p.y + d_plane.z;
    }

    static __device__ __forceinline__ float2 MapRefToSrcFast_Hex_Frame(
        const int2& ref_pt,
        const float disparity,
        const float step_x,
        const float step_y)
    {
        return make_float2(
            ref_pt.x + step_x * disparity,
            ref_pt.y + step_y * disparity);
    }

    static __device__ float ComputeBilateralNCC_Frame(
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
        DisparityGeometricMapOperate_Hex(
            c0, c1, p, plane_hypothesis, p, params,
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

        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
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
                {
                    continue;
                }

                const float d = EvalDisparityAtPixel_Frame(d_plane, ref_pt);
                const float2 src_pt = MapRefToSrcFast_Hex_Frame(ref_pt, d, step_x, step_y);

                if (src_pt.x < 0.0f || src_pt.x >= params.MLA_Mask_Width_Cuda ||
                    src_pt.y < 0.0f || src_pt.y >= params.MLA_Mask_Height_Cuda)
                {
                    continue;
                }

                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                const float ref_blur_v = tex2D<float>(ref_blur_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f) / pixel_max;
                const float src_blur_v = tex2D<float>(src_blur_image, src_pt.x + 0.5f, src_pt.y + 0.5f) / pixel_max;
                blur_value.x += ref_blur_v;
                blur_value.y += src_blur_v;

                const float weight = ComputeBilateralWeight(
                    i, j, ref_pix, ref_center_pix,
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
        sum_ref *= inv;
        sum_ref_ref *= inv;
        sum_src *= inv;
        sum_src_src *= inv;
        sum_ref_src *= inv;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;
        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar)
            return cost_max;

        const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
        const float var_ref_src = sqrtf(var_ref * var_src);
        return clamp_cost(1.0f - covar_src_ref / var_ref_src);
    }

    static __device__ void ComputeMultiViewCostVector_Frame(
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
            if (nvid < 0) continue;

            float2 blur_v = make_float2(0.0f, 0.0f);
            float4 disp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float c = ComputeBilateralNCC_Frame(images[ref_vid], blur_images[ref_vid], c0, tk0,
                                                images[nvid], blur_images[nvid], centers[nvid], tilekeys[nvid],
                                                p, plane_hypothesis, params, blur_v, disp_b);
            const float blur_weight = expf(-0.007368f * (blur_v.x - blur_v.y) * (blur_v.x - blur_v.y));
            c = clamp_cost(c + 2.0f * (1.0f - blur_weight));
            cost_vector[k] = c;
            blur_vector[k] = blur_v;
            disp_vector[k] = disp_b;
        }
    }

    static __device__ unsigned int ComputeSelectedViewsTopK_Frame(
        const float* cost_vector,
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
        for (int i = 0; i < neigh_count; ++i) tmp[i] = cost_vector[i];
        for (int i = neigh_count; i < 32; ++i) tmp[i] = 2.0f;

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
        {
            if (cost_vector[i] <= threshold)
                setBit(mask, i);
        }
        return mask;
    }

    static __device__ float ComputeWeightedCost_Frame(
        const float* cost_vector,
        const float2* blur_vector,
        const float* view_weights,
        const int neigh_count,
        float4* disp_vector,
        float4& out_best_disp)
    {
        float weight_norm = 0.0f;
        float total = 0.0f;
        float min_cost = 2.0f;
        int min_idx = -1;

        for (int i = 0; i < neigh_count; ++i)
        {
            if (view_weights[i] > 0.0f)
            {
                weight_norm += view_weights[i];
                total += view_weights[i] * cost_vector[i];
            }
            if (cost_vector[i] < min_cost)
            {
                min_cost = cost_vector[i];
                min_idx = i;
            }
        }

        if (min_idx >= 0) out_best_disp = disp_vector[min_idx];
        else out_best_disp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (weight_norm <= 1e-6f)
            return 2.0f;
        return total / weight_norm;
    }

    static __device__ void ComputeBaselineNorm_Frame(
        const float2* centers,
        const int* neighbor_ids,
        const int* neighbor_counts,
        const int max_neighbors,
        const int ref_vid,
        float* baseline_norm)
    {
        for (int i = 0; i < 32; ++i) baseline_norm[i] = 0.0f;
        const int neigh_count = neighbor_counts[ref_vid];
        float baseline_max = 1.0f;
        const float2 center = centers[ref_vid];

        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            const int nvid = neighbor_ids[ref_vid * max_neighbors + k];
            if (nvid < 0) continue;
            const float dx = center.x - centers[nvid].x;
            const float dy = center.y - centers[nvid].y;
            const float b = sqrtf(dx * dx + dy * dy);
            baseline_norm[k] = b;
            if (b > baseline_max) baseline_max = b;
        }

        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            const float tmp = baseline_norm[k] / baseline_max;
            baseline_norm[k] = expf(-tmp);
        }
    }

    static __device__ bool BuildCrossViewPlaneProposal(
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
        DisparityGeometricMapOperate_Hex(
            centers[ref_vid], centers[neigh_vid],
            p, current_plane, p, params,
            qf, tmp_disp,
            tilekeys[ref_vid], tilekeys[neigh_vid]);

        const int qx = (int)floorf(qf.x + 0.5f);
        const int qy = (int)floorf(qf.y + 0.5f);
        if (qx < 0 || qx >= params.MLA_Mask_Width_Cuda ||
            qy < 0 || qy >= params.MLA_Mask_Height_Cuda)
        {
            return false;
        }

        const int2 q = make_int2(qx, qy);
        const int q_local = q.y * params.MLA_Mask_Width_Cuda + q.x;
        const float4 plane_neigh_hyp = plane_prev[FrameIndex1D(neigh_vid, q_local, pixels_per_view)];
        const float3 disp_plane_neigh = DisparityPlane(q, plane_neigh_hyp);
        const float alpha = disp_plane_neigh.x;
        const float beta  = disp_plane_neigh.y;
        const float gamma = disp_plane_neigh.z;

        const float du = centers[ref_vid].x - centers[neigh_vid].x;
        const float dv = centers[ref_vid].y - centers[neigh_vid].y;

        float denom = 1.0f + alpha * du + beta * dv;
        denom = SafeDenomDevice(denom);

        const float aR = alpha / denom;
        const float bR = beta  / denom;
        const float cR = gamma / denom;

        out_plane = EncodePlaneHypothesisFromDispPlaneDevice(p, aR, bR, cR);
        return true;
    }

    static __device__ void PlaneHypothesisRefinement_Frame(
        const cudaTextureObjects* texture_objects,
        const float2* centers,
        const int2* tilekeys,
        const int* neighbor_ids,
        const int* neighbor_counts,
        const int max_neighbors,
        const PatchMatchParamsLF params,
        const int ref_vid,
        const int2 p,
        curandState* rand_state,
        const float* view_weights,
        float4& best_plane,
        float& best_cost,
        float4& best_disp)
    {
        const int neigh_count = neighbor_counts[ref_vid];
        if (neigh_count <= 0)
            return;

        const float perturbation = 0.02f;
        float depth_now = best_plane.w;
        float depth_rand;
        float4 plane_rand_normal;

        if (best_cost < 0.1f)
            depth_rand = (curand_uniform(rand_state) - 0.5f) * 2.0f + depth_now;
        else
            depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;

        plane_rand_normal = GenerateRandomNormalLF(p, rand_state, depth_now);
        float4 plane_perturbed = GeneratePerturbedNormalLF(p, best_plane, rand_state, perturbation * M_PI);

        float depth_perturbed = depth_now;
        const float depth_min_perturbed = (1.0f - perturbation) * depth_perturbed;
        const float depth_max_perturbed = (1.0f + perturbation) * depth_perturbed;
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;

        const int num_planes = 5;
        float4 candidates[num_planes];
        candidates[0] = GenerateRandomPlaneHypothesis_MIPM(p, rand_state, params.depth_min, params.depth_max);
        candidates[1] = plane_rand_normal;
        candidates[2] = plane_perturbed;
        candidates[3] = best_plane;
        candidates[4] = best_plane;
        candidates[4].w = depth_perturbed;

        for (int i = 1; i < num_planes; ++i)
        {
            float3 d_plane = DisparityPlane(p, candidates[i]);
            candidates[i].w = to_disparity(p, d_plane);
        }

        for (int i = 0; i < num_planes; ++i)
        {
            float cost_vector[32];
            float2 blur_vector[32];
            float4 disp_vector[32];
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys,
                                             neighbor_ids, neighbor_counts, max_neighbors,
                                             params, ref_vid, p, candidates[i],
                                             cost_vector, blur_vector, disp_vector);
            float4 tmp_disp;
            float c = ComputeWeightedCost_Frame(cost_vector, blur_vector, view_weights, neigh_count, disp_vector, tmp_disp);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = candidates[i];
                best_disp = tmp_disp;
            }
        }
    }


    __global__ void RandomInitializationForFrame(
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
        curandState* rand_states,
        const PatchMatchParamsLF params,
        const int width,
        const int height,
        const int num_views)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int vid = blockIdx.z;
        if (vid >= num_views || x >= width || y >= height)
            return;

        const int pixels_per_view = width * height;
        const int idx = FrameIndex(vid, x, y, width, pixels_per_view);
        const int2 p = make_int2(x, y);

        curand_init(clock64(), (unsigned long long)vid * (unsigned long long)pixels_per_view + idx, 0, &rand_states[idx]);

        float4 plane = GenerateRandomPlaneHypothesis_MIPM(p, &rand_states[idx], params.depth_min, params.depth_max);
        float cost_vector[32];
        float2 blur_vector[32];
        float4 disp_vector[32];
        ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys,
                                         neighbor_ids, neighbor_counts, max_neighbors,
                                         params, vid, p, plane,
                                         cost_vector, blur_vector, disp_vector);
        float mean_cost = 2.0f;
        int best_idx = -1;
        unsigned int sel = ComputeSelectedViewsTopK_Frame(cost_vector, neighbor_counts[vid], params.top_k, mean_cost, best_idx);
        plane_prev[idx] = plane;
        cost_prev[idx] = mean_cost;
        disp_prev[idx] = (best_idx >= 0) ? disp_vector[best_idx] : make_float4(0,0,0,0);
        selected_prev[idx] = sel;
    }

    __global__ void CheckerboardUpdateFrame(
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
        float4* plane_next,
        float* cost_next,
        float4* disp_next,
        unsigned int* selected_next,
        curandState* rand_states,
        const PatchMatchParamsLF params,
        const int width,
        const int height,
        const int num_views,
        const int iter,
        const int phase)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int vid = blockIdx.z;
        if (vid >= num_views || x >= width || y >= height)
            return;

        const int pixels_per_view = width * height;
        const int idx = FrameIndex(vid, x, y, width, pixels_per_view);

        if (((x + y) & 1) != phase)
        {
            plane_next[idx] = plane_prev[idx];
            cost_next[idx] = cost_prev[idx];
            disp_next[idx] = disp_prev[idx];
            selected_next[idx] = selected_prev[idx];
            return;
        }

        const int2 p = make_int2(x, y);
        const int neigh_count = neighbor_counts[vid];
        if (neigh_count <= 0)
        {
            plane_next[idx] = plane_prev[idx];
            cost_next[idx] = cost_prev[idx];
            disp_next[idx] = disp_prev[idx];
            selected_next[idx] = selected_prev[idx];
            return;
        }

        float cost_array[8][32];
        float2 blur_array[8][32];
        float4 disp_array[8][32];
        bool flag[8] = {false,false,false,false,false,false,false,false};
        int positions[8];

        for (int a = 0; a < 8; ++a)
        {
            positions[a] = idx;
            for (int b = 0; b < 32; ++b)
            {
                cost_array[a][b] = 2.0f;
                blur_array[a][b] = make_float2(0.0f, 0.0f);
                disp_array[a][b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        const int farDis = 2;
        const int nerDis = 2;
        const int center = y * width + x;
        int left_near = center - 1;
        int left_far = center - 3;
        int right_near = center + 1;
        int right_far = center + 3;
        int up_near = center - width;
        int up_far = center - 3 * width;
        int down_near = center + width;
        int down_far = center + 3 * width;

        float costMin;
        int costMinPoint;

        if (y > 2)
        {
            flag[1] = true;
            costMin = cost_prev[FrameIndex1D(vid, up_far, pixels_per_view)];
            costMinPoint = up_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (y > 2 + i)
                {
                    const int pointTemp = up_far - i * width;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin)
                    {
                        costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)];
                        costMinPoint = pointTemp;
                    }
                }
            }
            up_far = costMinPoint; positions[1] = up_far;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, up_far, pixels_per_view)], cost_array[1], blur_array[1], disp_array[1]);
        }
        if (y < height - 3)
        {
            flag[3] = true;
            costMin = cost_prev[FrameIndex1D(vid, down_far, pixels_per_view)];
            costMinPoint = down_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (y < height - 3 - i)
                {
                    const int pointTemp = down_far + i * width;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin)
                    {
                        costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)];
                        costMinPoint = pointTemp;
                    }
                }
            }
            down_far = costMinPoint; positions[3] = down_far;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, down_far, pixels_per_view)], cost_array[3], blur_array[3], disp_array[3]);
        }
        if (x > 2)
        {
            flag[5] = true;
            costMin = cost_prev[FrameIndex1D(vid, left_far, pixels_per_view)];
            costMinPoint = left_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (x > 2 + i)
                {
                    const int pointTemp = left_far - i;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin)
                    {
                        costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)];
                        costMinPoint = pointTemp;
                    }
                }
            }
            left_far = costMinPoint; positions[5] = left_far;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, left_far, pixels_per_view)], cost_array[5], blur_array[5], disp_array[5]);
        }
        if (x < width - 3)
        {
            flag[7] = true;
            costMin = cost_prev[FrameIndex1D(vid, right_far, pixels_per_view)];
            costMinPoint = right_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (x < width - 3 - i)
                {
                    const int pointTemp = right_far + i;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin)
                    {
                        costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)];
                        costMinPoint = pointTemp;
                    }
                }
            }
            right_far = costMinPoint; positions[7] = right_far;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, right_far, pixels_per_view)], cost_array[7], blur_array[7], disp_array[7]);
        }
        if (y > 0)
        {
            flag[0] = true;
            costMin = cost_prev[FrameIndex1D(vid, up_near, pixels_per_view)];
            costMinPoint = up_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (y > 1 + i && x > i)
                {
                    const int pointTemp = up_near - (1 + i) * width - i;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
                if (y > 1 + i && x < width - 1 - i)
                {
                    const int pointTemp = up_near - (1 + i) * width + i;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
            }
            up_near = costMinPoint; positions[0] = up_near;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, up_near, pixels_per_view)], cost_array[0], blur_array[0], disp_array[0]);
        }
        if (y <= 2 && y > 0)
        {
            flag[1] = true; positions[1] = up_near;
            for (int j = 0; j < 32; ++j) { cost_array[1][j] = cost_array[0][j]; blur_array[1][j] = blur_array[0][j]; disp_array[1][j] = disp_array[0][j]; }
        }
        if (y < height - 1)
        {
            flag[2] = true;
            costMin = cost_prev[FrameIndex1D(vid, down_near, pixels_per_view)];
            costMinPoint = down_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (y < height - 2 - i && x > i)
                {
                    const int pointTemp = down_near + (1 + i) * width - i;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
                if (y < height - 2 - i && x < width - 1 - i)
                {
                    const int pointTemp = down_near + (1 + i) * width + i;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
            }
            down_near = costMinPoint; positions[2] = down_near;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, down_near, pixels_per_view)], cost_array[2], blur_array[2], disp_array[2]);
        }
        if (y >= height - 3 && y < height - 1)
        {
            flag[3] = true; positions[3] = down_near;
            for (int j = 0; j < 32; ++j) { cost_array[3][j] = cost_array[2][j]; blur_array[3][j] = blur_array[2][j]; disp_array[3][j] = disp_array[2][j]; }
        }
        if (x > 0)
        {
            flag[4] = true;
            costMin = cost_prev[FrameIndex1D(vid, left_near, pixels_per_view)];
            costMinPoint = left_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (x > 1 + i && y > i)
                {
                    const int pointTemp = left_near - (1 + i) - i * width;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
                if (x > 1 + i && y < height - 1 - i)
                {
                    const int pointTemp = left_near - (1 + i) + i * width;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
            }
            left_near = costMinPoint; positions[4] = left_near;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, left_near, pixels_per_view)], cost_array[4], blur_array[4], disp_array[4]);
        }
        if (x <= 2 && x > 0)
        {
            flag[5] = true; positions[5] = left_near;
            for (int j = 0; j < 32; ++j) { cost_array[5][j] = cost_array[4][j]; blur_array[5][j] = blur_array[4][j]; disp_array[5][j] = disp_array[4][j]; }
        }
        if (x < width - 1)
        {
            flag[6] = true;
            costMin = cost_prev[FrameIndex1D(vid, right_near, pixels_per_view)];
            costMinPoint = right_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (x < width - 2 - i && y > i)
                {
                    const int pointTemp = right_near + (1 + i) - i * width;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
                if (x < width - 2 - i && y < height - 1 - i)
                {
                    const int pointTemp = right_near + (1 + i) + i * width;
                    if (cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)] < costMin) { costMin = cost_prev[FrameIndex1D(vid, pointTemp, pixels_per_view)]; costMinPoint = pointTemp; }
                }
            }
            right_near = costMinPoint; positions[6] = right_near;
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys, neighbor_ids, neighbor_counts, max_neighbors, params,
                                             vid, p, plane_prev[FrameIndex1D(vid, right_near, pixels_per_view)], cost_array[6], blur_array[6], disp_array[6]);
        }
        if (x >= width - 3 && x < width - 1)
        {
            flag[7] = true; positions[7] = right_near;
            for (int j = 0; j < 32; ++j) { cost_array[7][j] = cost_array[6][j]; blur_array[7][j] = blur_array[6][j]; disp_array[7][j] = disp_array[6][j]; }
        }

        float view_selection_priors[32] = {0.0f};
        const int direct_pos[4] = {
            (y > 0) ? (center - width) : -1,
            (y < height - 1) ? (center + width) : -1,
            (x > 0) ? (center - 1) : -1,
            (x < width - 1) ? (center + 1) : -1
        };
        const int same_flags[4] = {0, 2, 4, 6};
        for (int ii = 0; ii < 4; ++ii)
        {
            if (direct_pos[ii] < 0 || !flag[same_flags[ii]]) continue;
            for (int j = 0; j < neigh_count; ++j)
            {
                if (isSet(selected_prev[FrameIndex1D(vid, direct_pos[ii], pixels_per_view)], j) == 1)
                    view_selection_priors[j] += 0.9f;
                else
                    view_selection_priors[j] += 0.1f;
            }
        }

        float sampling_probs[32] = {0.0f};
        const float cost_threshold = 0.8f * expf((iter) * (iter) / (-90.0f));
        for (int i = 0; i < neigh_count; ++i)
        {
            float count = 0.0f;
            int count_false = 0;
            float tmpw = 0.0f;
            for (int j = 0; j < 8; ++j)
            {
                if (!flag[j]) continue;
                if (cost_array[j][i] < cost_threshold)
                {
                    tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                    count += 1.0f;
                }
                if (cost_array[j][i] > 1.2f) count_false++;
            }
            if (count > 2.0f && count_false < 3)
                sampling_probs[i] = tmpw / count;
            else if (count_false < 3)
                sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
            sampling_probs[i] *= view_selection_priors[i];
        }

        TransformPDFToCDF(sampling_probs, neigh_count);
        float view_weights[32] = {0.0f};
        for (int sample = 0; sample < 15; ++sample)
        {
            const float rand_prob = curand_uniform(&rand_states[idx]) - FLT_EPSILON;
            for (int image_id = 0; image_id < neigh_count; ++image_id)
            {
                const float prob = sampling_probs[image_id];
                if (prob > rand_prob)
                {
                    view_weights[image_id] += 1.0f;
                    break;
                }
            }
        }

        unsigned int temp_selected_views = 0;
        float weight_norm = 0.0f;
        for (int i = 0; i < neigh_count; ++i)
        {
            if (view_weights[i] > 0.0f)
            {
                setBit(temp_selected_views, i);
                weight_norm += view_weights[i];
            }
        }
        if (weight_norm <= 0.0f)
        {
            for (int i = 0; i < neigh_count; ++i)
            {
                view_weights[i] = 1.0f;
                setBit(temp_selected_views, i);
            }
            weight_norm = float(neigh_count);
        }

        float4 best_plane = plane_prev[idx];
        float best_cost = cost_prev[idx];
        float4 best_disp = disp_prev[idx];
        unsigned int best_sel = temp_selected_views;

        {
            float cost_vector_now[32];
            float2 blur_vector_now[32];
            float4 disp_vector_now[32];
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys,
                                             neighbor_ids, neighbor_counts, max_neighbors,
                                             params, vid, p, best_plane,
                                             cost_vector_now, blur_vector_now, disp_vector_now);
            best_cost = ComputeWeightedCost_Frame(cost_vector_now, blur_vector_now, view_weights, neigh_count, disp_vector_now, best_disp);
        }

        for (int i = 0; i < 8; ++i)
        {
            if (!flag[i]) continue;
            float4 cand_disp;
            const float c = ComputeWeightedCost_Frame(cost_array[i], blur_array[i], view_weights, neigh_count, disp_array[i], cand_disp);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = plane_prev[FrameIndex1D(vid, positions[i], pixels_per_view)];
                best_disp = cand_disp;
                best_sel = temp_selected_views;
            }
        }

        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            if (isSet(temp_selected_views, k) == 0)
                continue;

            const int nvid = neighbor_ids[vid * max_neighbors + k];
            if (nvid < 0)
                continue;

            float4 proxy_plane;
            if (!BuildCrossViewPlaneProposal(plane_prev, centers, tilekeys, pixels_per_view, params, vid, nvid, p, best_plane, proxy_plane))
                continue;

            float cost_vector_proxy[32];
            float2 blur_vector_proxy[32];
            float4 disp_vector_proxy[32];
            ComputeMultiViewCostVector_Frame(texture_objects, centers, tilekeys,
                                             neighbor_ids, neighbor_counts, max_neighbors,
                                             params, vid, p, proxy_plane,
                                             cost_vector_proxy, blur_vector_proxy, disp_vector_proxy);
            float4 proxy_disp;
            const float c = ComputeWeightedCost_Frame(cost_vector_proxy, blur_vector_proxy, view_weights, neigh_count, disp_vector_proxy, proxy_disp);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = proxy_plane;
                best_disp = proxy_disp;
                best_sel = temp_selected_views;
            }
        }

        PlaneHypothesisRefinement_Frame(texture_objects, centers, tilekeys,
                                        neighbor_ids, neighbor_counts, max_neighbors,
                                        params, vid, p, &rand_states[idx], view_weights,
                                        best_plane, best_cost, best_disp);

        plane_next[idx] = best_plane;
        cost_next[idx] = best_cost;
        disp_next[idx] = best_disp;
        selected_next[idx] = best_sel;
    }

    void RunPatchMatchCUDAForFrame_Impl(cudaTextureObjects* texture_objects_cuda,
        float2* centers_cuda, int2* tilekeys_cuda, int* neighbor_ids_cuda, int* neighbor_counts_cuda,
        float4* plane_prev_cuda, float4* plane_next_cuda, float* cost_prev_cuda, float* cost_next_cuda,
        float4* disp_prev_cuda, float4* disp_next_cuda, unsigned int* selected_prev_cuda, unsigned int* selected_next_cuda,
        curandState* rand_states_cuda, PatchMatchParamsLF params, int num_views, int max_neighbors)
    {
        const int width = params.MLA_Mask_Width_Cuda;
        const int height = params.MLA_Mask_Height_Cuda;
        dim3 block_size(16, 16, 1);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y,
                       num_views);

        RandomInitializationForFrame<<<grid_size, block_size>>>(
            texture_objects_cuda, centers_cuda, tilekeys_cuda,
            neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
            plane_prev_cuda, cost_prev_cuda, disp_prev_cuda, selected_prev_cuda,
            rand_states_cuda, params, width, height, num_views);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        for (int iter = 0; iter < params.max_iterations; ++iter)
        {
            CheckerboardUpdateFrame<<<grid_size, block_size>>>(
                texture_objects_cuda, centers_cuda, tilekeys_cuda,
                neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
                plane_prev_cuda, cost_prev_cuda, disp_prev_cuda, selected_prev_cuda,
                plane_next_cuda, cost_next_cuda, disp_next_cuda, selected_next_cuda,
                rand_states_cuda, params, width, height, num_views, iter, 0);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_prev_cuda, plane_next_cuda);
            std::swap(cost_prev_cuda, cost_next_cuda);
            std::swap(disp_prev_cuda, disp_next_cuda);
            std::swap(selected_prev_cuda, selected_next_cuda);

            CheckerboardUpdateFrame<<<grid_size, block_size>>>(
                texture_objects_cuda, centers_cuda, tilekeys_cuda,
                neighbor_ids_cuda, neighbor_counts_cuda, max_neighbors,
                plane_prev_cuda, cost_prev_cuda, disp_prev_cuda, selected_prev_cuda,
                plane_next_cuda, cost_next_cuda, disp_next_cuda, selected_next_cuda,
                rand_states_cuda, params, width, height, num_views, iter, 1);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_prev_cuda, plane_next_cuda);
            std::swap(cost_prev_cuda, cost_next_cuda);
            std::swap(disp_prev_cuda, disp_next_cuda);
            std::swap(selected_prev_cuda, selected_next_cuda);
        }
    }
}
