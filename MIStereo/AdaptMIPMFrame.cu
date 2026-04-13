
/********************************************************************
file base:      AdaptMIPMFrame.cu
author:         OpenAI + LZD
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

    static __device__ __forceinline__ void CopyIfOtherPhase(
        const int idx,
        const int x, const int y, const int phase,
        const float4* plane_prev, const float* cost_prev, const float4* disp_prev, const unsigned int* sel_prev,
        float4* plane_next, float* cost_next, float4* disp_next, unsigned int* sel_next)
    {
        if (((x + y) & 1) != phase)
        {
            plane_next[idx] = plane_prev[idx];
            cost_next[idx] = cost_prev[idx];
            disp_next[idx] = disp_prev[idx];
            sel_next[idx] = sel_prev[idx];
        }
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
        float2 pt;

        DisparityGeometricMapOperate_Hex(c0, c1, p, plane_hypothesis, p, params, pt, disparity_baseline, tk0, tk1);
        if (pt.x >= params.MLA_Mask_Width_Cuda || pt.x < 0.0f || pt.y >= params.MLA_Mask_Height_Cuda || pt.y < 0.0f)
            return cost_max;

        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;

        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
        const float pixel_max = 255.0f;

        for (int i = -radius; i <= radius; i += params.radius_increment)
        {
            for (int j = -radius; j <= radius; j += params.radius_increment)
            {
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                if (ref_pt.x < 0 || ref_pt.x >= params.MLA_Mask_Width_Cuda ||
                    ref_pt.y < 0 || ref_pt.y >= params.MLA_Mask_Height_Cuda)
                    continue;

                float2 src_pt;
                float4 tmp_db;
                DisparityGeometricMapOperate_Hex(c0, c1, p, plane_hypothesis, ref_pt, params, src_pt, tmp_db, tk0, tk1);
                if (src_pt.x < 0.0f || src_pt.x >= params.MLA_Mask_Width_Cuda ||
                    src_pt.y < 0.0f || src_pt.y >= params.MLA_Mask_Height_Cuda)
                    continue;

                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                const float ref_blur_v = tex2D<float>(ref_blur_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f) / pixel_max;
                const float src_blur_v = tex2D<float>(src_blur_image, src_pt.x + 0.5f, src_pt.y + 0.5f) / pixel_max;
                blur_value.x += ref_blur_v;
                blur_value.y += src_blur_v;

                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

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

    static __device__ float EvaluatePlaneForView_Frame(
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
        unsigned int* out_selected_views,
        float4* out_best_disp)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;

        float costs_local[32];
        float4 disp_local[32];
        float2 blur_local[32];
        const int neigh_count = neighbor_counts[ref_vid];
        for (int i = 0; i < 32; ++i) { costs_local[i] = 2.0f; disp_local[i] = make_float4(0,0,0,0); blur_local[i] = make_float2(0,0); }

        const float2 c0 = centers[ref_vid];
        const int2 tk0 = tilekeys[ref_vid];
        int valid = 0;
        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            const int nvid = neighbor_ids[ref_vid * max_neighbors + k];
            if (nvid < 0)
                continue;

            float2 blur_v = make_float2(0.0f, 0.0f);
            float4 disp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float c = ComputeBilateralNCC_Frame(images[ref_vid], blur_images[ref_vid], c0, tk0,
                                                images[nvid], blur_images[nvid], centers[nvid], tilekeys[nvid],
                                                p, plane_hypothesis, params, blur_v, disp_b);

            // 模糊差异惩罚
            float blur_weight = expf(-0.007368f * (blur_v.x - blur_v.y) * (blur_v.x - blur_v.y));
            c = clamp_cost(c + 2.0f * (1.0f - blur_weight));

            costs_local[k] = c;
            disp_local[k] = disp_b;
            blur_local[k] = blur_v;
            if (c < 2.0f) ++valid;
        }

        if (valid <= 0)
        {
            if (out_selected_views) *out_selected_views = 0;
            if (out_best_disp) *out_best_disp = make_float4(0,0,0,0);
            return 2.0f;
        }

        // simple top-k average
        const int top_k = min(valid, params.top_k);
        for (int i = 0; i < neigh_count && i < max_neighbors; ++i)
        {
            for (int j = i + 1; j < neigh_count && j < max_neighbors; ++j)
            {
                if (costs_local[j] < costs_local[i])
                {
                    float tmp = costs_local[i]; costs_local[i] = costs_local[j]; costs_local[j] = tmp;
                    float4 tmp4 = disp_local[i]; disp_local[i] = disp_local[j]; disp_local[j] = tmp4;
                    float2 tmp2 = blur_local[i]; blur_local[i] = blur_local[j]; blur_local[j] = tmp2;
                }
            }
        }

        float sum = 0.0f;
        float best_cost = costs_local[0];
        float4 best_disp = disp_local[0];
        for (int i = 0; i < top_k; ++i)
            sum += costs_local[i];

        unsigned int bitmask = 0;
        int picked = 0;
        for (int k = 0; k < neigh_count && k < max_neighbors && k < 32; ++k)
        {
            const int nvid = neighbor_ids[ref_vid * max_neighbors + k];
            if (nvid < 0)
                continue;

            float2 blur_v = make_float2(0.0f, 0.0f);
            float4 disp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float c = ComputeBilateralNCC_Frame(images[ref_vid], blur_images[ref_vid], c0, tk0,
                                                images[nvid], blur_images[nvid], centers[nvid], tilekeys[nvid],
                                                p, plane_hypothesis, params, blur_v, disp_b);
            float blur_weight = expf(-0.007368f * (blur_v.x - blur_v.y) * (blur_v.x - blur_v.y));
            c = clamp_cost(c + 2.0f * (1.0f - blur_weight));
            if (c <= costs_local[top_k - 1] && picked < 32)
            {
                setBit(bitmask, k);
                ++picked;
            }
        }

        if (out_selected_views) *out_selected_views = bitmask;
        if (out_best_disp) *out_best_disp = best_disp;
        return sum / top_k;
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
        DisparityGeometricMapOperate_Hex(centers[ref_vid], centers[neigh_vid], p, current_plane, p, params,
                                         qf, tmp_disp, tilekeys[ref_vid], tilekeys[neigh_vid]);

        const int qx = (int)floorf(qf.x + 0.5f);
        const int qy = (int)floorf(qf.y + 0.5f);
        if (qx < 0 || qx >= params.MLA_Mask_Width_Cuda || qy < 0 || qy >= params.MLA_Mask_Height_Cuda)
            return false;

        const int q_local = qy * params.MLA_Mask_Width_Cuda + qx;
        const float4 plane_neigh = plane_prev[FrameIndex1D(neigh_vid, q_local, pixels_per_view)];

        const float alpha = plane_neigh.x;
        const float beta  = plane_neigh.y;
        const float gamma = plane_neigh.z;
        const float du = centers[ref_vid].x - centers[neigh_vid].x;
        const float dv = centers[ref_vid].y - centers[neigh_vid].y;
        const float denom = 1.0f + alpha * du + beta * dv + 1e-6f;

        const float aR = alpha / denom;
        const float bR = beta  / denom;
        const float cR = gamma / denom;
        const float dR = aR * p.x + bR * p.y + cR;
        out_plane = make_float4(aR, bR, cR, dR);
        return true;
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
        unsigned int sel = 0;
        float4 best_disp = make_float4(0,0,0,0);
        float cost = EvaluatePlaneForView_Frame(texture_objects, centers, tilekeys,
                                                neighbor_ids, neighbor_counts, max_neighbors,
                                                params, vid, p, plane, &sel, &best_disp);

        plane_prev[idx] = plane;
        cost_prev[idx] = cost;
        disp_prev[idx] = best_disp;
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

        float4 best_plane = plane_prev[idx];
        float best_cost = cost_prev[idx];
        float4 best_disp = disp_prev[idx];
        unsigned int best_sel = selected_prev[idx];

        // re-evaluate current plane against current support snapshot
        {
            unsigned int sel = 0;
            float4 disp_b = make_float4(0,0,0,0);
            float c = EvaluatePlaneForView_Frame(texture_objects, centers, tilekeys,
                                                 neighbor_ids, neighbor_counts, max_neighbors,
                                                 params, vid, p, best_plane, &sel, &disp_b);
            best_cost = c;
            best_disp = disp_b;
            best_sel = sel;
        }

        // local same-view proposals: 8 directions
        const int local_x[8] = {x, x, x, x, x - 1, x - 3, x + 1, x + 3};
        const int local_y[8] = {y - 1, y - 3, y + 1, y + 3, y, y, y, y};
        for (int k = 0; k < 8; ++k)
        {
            const int nx = local_x[k];
            const int ny = local_y[k];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            const int nidx = FrameIndex(vid, nx, ny, width, pixels_per_view);
            const float4 cand = plane_prev[nidx];

            unsigned int sel = 0;
            float4 disp_b = make_float4(0,0,0,0);
            float c = EvaluatePlaneForView_Frame(texture_objects, centers, tilekeys,
                                                 neighbor_ids, neighbor_counts, max_neighbors,
                                                 params, vid, p, cand, &sel, &disp_b);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = cand;
                best_disp = disp_b;
                best_sel = sel;
            }
        }

        // cross-view proposals from each support view
        const int neigh_count = neighbor_counts[vid];
        for (int k = 0; k < neigh_count && k < max_neighbors; ++k)
        {
            const int nvid = neighbor_ids[vid * max_neighbors + k];
            if (nvid < 0)
                continue;

            float4 proxy_plane;
            if (!BuildCrossViewPlaneProposal(plane_prev, centers, tilekeys, pixels_per_view, params, vid, nvid, p, best_plane, proxy_plane))
                continue;

            unsigned int sel = 0;
            float4 disp_b = make_float4(0,0,0,0);
            float c = EvaluatePlaneForView_Frame(texture_objects, centers, tilekeys,
                                                 neighbor_ids, neighbor_counts, max_neighbors,
                                                 params, vid, p, proxy_plane, &sel, &disp_b);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = proxy_plane;
                best_disp = disp_b;
                best_sel = sel;
            }
        }

        // one random refinement proposal
        {
            float4 rand_plane = GenerateRandomPlaneHypothesis_MIPM(p, &rand_states[idx], params.depth_min, params.depth_max);
            unsigned int sel = 0;
            float4 disp_b = make_float4(0,0,0,0);
            float c = EvaluatePlaneForView_Frame(texture_objects, centers, tilekeys,
                                                 neighbor_ids, neighbor_counts, max_neighbors,
                                                 params, vid, p, rand_plane, &sel, &disp_b);
            if (c < best_cost)
            {
                best_cost = c;
                best_plane = rand_plane;
                best_disp = disp_b;
                best_sel = sel;
            }
        }

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
                rand_states_cuda, params, width, height, num_views, 0);
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
                rand_states_cuda, params, width, height, num_views, 1);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::swap(plane_prev_cuda, plane_next_cuda);
            std::swap(cost_prev_cuda, cost_next_cuda);
            std::swap(disp_prev_cuda, disp_next_cuda);
            std::swap(selected_prev_cuda, selected_next_cuda);
        }
    }
}
