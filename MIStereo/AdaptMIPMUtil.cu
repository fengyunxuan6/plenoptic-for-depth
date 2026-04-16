/********************************************************************
file base:      AdaptMIPMUtil.cu
author:         LZD
created:        2025/06/26
purpose:
*********************************************************************/
#include "AdaptMIPMUtil.cuh"
// Note: Hex ring utilities available via LFMVS::HexGrid in AdaptMIPMUtil_with_hexring.cuh

#include "CudaUtil.h"

namespace LFMVS
{

__device__ int*   g_nei_lin   = nullptr;
__device__ float* g_pTot_list = nullptr;
__device__ int*   g_k_list    = nullptr;
__device__ int    g_M         = 0;
__device__ int    g_ref_lin   = -1;

void SetHexPerCallPTot(int* d_nei_lin, float* d_pTot, int* d_kSteps, int M, int ref_lin){
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_nei_lin,   &d_nei_lin,   sizeof(int*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_pTot_list, &d_pTot,      sizeof(float*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_k_list,    &d_kSteps,    sizeof(int*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_M,         &M,           sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ref_lin,   &ref_lin,     sizeof(int)));
}
    // 计算像素p的代价值（局部块窗口：与邻域图像对应的像素）
    __device__ float ComputeMultiViewInitialCostandSelectedViews_MIPM(const cudaTextureObjects* texture_objects,
        float2* pcenters, const int2 p, const float4 plane_hypothesis,
        unsigned int* selected_views, const PatchMatchParamsLF params)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;

        float cost_max = 2.0f;
        float cost_vector[32] = {2.0f};
        float cost_vector_copy[32] = {2.0f};
        float4 disp_baseline_array[32];
        float2 blur_array[32] = {0.0f};
        int cost_count = 0;
        int num_valid_views = 0;

        // 遍历所有的邻居
        for (int i = 1; i < params.num_images; ++i)
        {
            //printf("%d\n", params.num_images);
            float cost_value = ComputeBilateralNCC_MIPM(images[0], blur_images[0],
                pcenters[0], images[i],
                pcenters[i], blur_images[i], p, plane_hypothesis, params,
                blur_array[i],disp_baseline_array[i]);
            cost_vector[i-1] = cost_value;
            cost_vector_copy[i-1] = cost_value;
            cost_count++;
            if (cost_value < cost_max)
            {
                num_valid_views++;
            }
        }
        //printf("%d\n", cost_count);
        sort_small(cost_vector, cost_count);
        *selected_views = 0;

        int top_k = min(num_valid_views, params.top_k);
        //printf("num_valid_views: %d\n", num_valid_views);

        if (top_k > 0)
        {
            float cost = 0.0f;
            for (int i = 0; i < top_k; ++i)
            {
                cost += cost_vector[i];
            }
            float cost_threshold = cost_vector[top_k - 1];
            for (int i = 0; i < params.num_images - 1; ++i)
            {
                if (cost_vector_copy[i] <= cost_threshold)
                {
                    // 利用位运算存储前top_k个邻居图像的编号，作为当前图像当前像素的最佳用于匹配的邻居
                    setBit(*selected_views, i);
                }
            }
            return cost / top_k;
        }
        else
        {
            return cost_max;
        }
    }

    __device__ float4 FindMinCostWithDispBaseline(float* min_cost_array, float4* disp_baseline_array,
        const PatchMatchParamsLF params)
    {
        float4 disp_baseline_min;
        float min_cost = FLT_MAX; // todo
        int min_view_index = -1;
        for (int j = 0; j < params.num_images - 1; ++j)
        {
            float tmp = min_cost_array[j];
            if (min_cost > tmp)
            {
                min_cost = tmp;
                min_view_index = j;
            }
        }
        disp_baseline_min = disp_baseline_array[min_view_index];

        // if (min_cost < 2.0)
        // {
        //     printf("mincost: %f, view_index:%d, d_real:%f, d:%f\n", min_cost,
        //         min_view_index, disp_baseline_min.y, disp_baseline_min.x);
        // }
        return disp_baseline_min;
    }

    __device__ float2 ComputeAverageVirtualDepth(float2* disp_baseline_array, const PatchMatchParamsLF params)
    {
        float2 v;
        v.x = 0.0; // 视差
        v.y = 0.0; // 虚拟深度
        for (int j = 0; j < params.num_images - 1; ++j)
        {
            v.x = v.x+disp_baseline_array[j].x;
            v.y = v.y+(disp_baseline_array[j].y/disp_baseline_array[j].x);
        }
        v.x /= (params.num_images - 1);
        v.y /= (params.num_images - 1);
    }

        __device__ void PlaneHypothesisRefinement_MIPM(const cudaTextureObject_t* images,
        const cudaTextureObject_t* blur_images,
        const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
        float2* pcenters, float* depth, float* cost, float4* disp_baseline, curandState* rand_state,
        const float* view_weights, const float weight_norm, float4* prior_planes,
        unsigned int* plane_masks, float* restricted_cost, const int2 p,
        const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm)
    {
        float perturbation = 0.02f;
        const int center = p.y * WIDTH + p.x;

        float gamma = 0.5f;
        float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
        float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
        float angle_sigma = M_PI * (5.0f / 180.0f);
        float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
        float beta = 0.18f;
        float depth_prior = 0.0f;

        float depth_rand;
        float4 plane_hypothesis_rand;
        if (params.planar_prior && plane_masks[center] > 0)
        {
            depth_prior = prior_planes[center].w;
            depth_rand = curand_uniform(rand_state) * 6 * depth_sigma + (depth_prior - 3 * depth_sigma);
            plane_hypothesis_rand = GeneratePerturbedNormalLF(p, prior_planes[center], rand_state, angle_sigma);
        }
        else
        {
            if(cost[center]<0.1)
            {
                depth_rand = (curand_uniform(rand_state)-0.5) * 2.0 + plane_hypothesis[center].w;
            }
            else
            {
                depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
            }
            plane_hypothesis_rand = GenerateRandomNormalLF(p, rand_state, *depth);
        }
        float depth_perturbed = *depth;
        const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
        const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
        do
        {
            depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed-depth_min_perturbed)
                                + depth_min_perturbed;
        } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);

        float4 plane_hypothesis_perturbed = GeneratePerturbedNormalLF( p, *plane_hypothesis, rand_state, perturbation * M_PI); // GeneratePertubedPlaneHypothesis(cameras[0], p, rand_state, perturbation, *plane_hypothesis, *depth, params.depth_min, params.depth_max);

        // lzd 构建了多个新的假设视差平面，继续算，得到最小代价，作为本次迭代的最终视差平面+代价
        const int num_planes = 5;
        float depths[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
        float4 normals[num_planes] = {*plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis};

        for (int i = 0; i < num_planes; ++i)
        {
            float cost_vector[32] = {2.0f};
            float2 blur_vector[32] = {0.0f};
            float4 disp_baseline_vector[32];

            float4 temp_plane_hypothesis = normals[i];

            // todo: 满足视差平面计算视差值的公式
            float3 d_plane = DisparityPlane(p, temp_plane_hypothesis);
            temp_plane_hypothesis.w = to_disparity(p, d_plane);

            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, temp_plane_hypothesis,
                                            cost_vector, blur_vector, disp_baseline_vector, params);

            float temp_cost = 0.0f;
            for (int j = 0; j < params.num_images-1; ++j)
            {
                if (view_weights[j] > 0)
                {
                    if (params.geom_consistency)
                    {
                        //temp_cost += view_weights[j] * (cost_vector[j] + 0.1f * ComputeGeomConsistencyCost(depth_images[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                    }
                    else
                    {
                        //temp_cost += view_weights[j] * cost_vector[j];

                        // todo: blurWeight  融合模糊差异的代价累积
                        float blur_weight = exp(-0.007368*pow((blur_vector[i].x - blur_vector[i].y), 2));
                        temp_cost += view_weights[j] * (cost_vector[j] + 2.0*(1.0-blur_weight));
        //                 if (blur_vector[i].x > 0.5 && blur_vector[i].y>0.5)
        //                 {
        //                     printf("cost: %f, blur_weight: %f,%f,  %f\n", cost_vector[i], blur_vector[i].x,
        // blur_vector[i].y, 1-blur_weight);
        //                 }
                    }
                }
            }
            temp_cost /= weight_norm;
            float4 disp_baseline_min = FindMinCostWithDispBaseline(cost_vector, disp_baseline_vector, params);
            //printf("refine: %f\n", disp_baseline_min.y);

            // TODO：可以与ACMP类似，存储虚拟深度
            float disp_before = temp_plane_hypothesis.w ; // 标准视差 d
            if (disp_before >= params.depth_min && disp_before <= params.depth_max)
            {
                if (params.planar_prior && plane_masks[center] > 0)
                {
                    float depth_diff = depths[i] - depth_prior;
                    float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
                    float angle_diff = acos(angle_cos);
                    float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                    float restricted_temp_cost = exp(-temp_cost * temp_cost / beta) * prior;
                    if (restricted_temp_cost > *restricted_cost)
                    {
                        *depth = disp_before;
                        *plane_hypothesis = temp_plane_hypothesis;
                        *cost = temp_cost;
                        *restricted_cost = restricted_temp_cost;

                        // todo: 计算虚拟深度值  长基线+代价小的算
                        float virtual_depth_before = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_vector,
    params, baseline_norm, cost_vector);
                        (*disp_baseline).w = virtual_depth_before;
                    }
                }
                else
                {
                    if (temp_cost < *cost)
                    {
                        *depth = disp_before;
                        *plane_hypothesis = temp_plane_hypothesis;
                        *cost = temp_cost;

                        // todo: 计算虚拟深度值  长基线+代价小的算
                        float virtual_depth_before = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_vector,
    params, baseline_norm, cost_vector);
                        (*disp_baseline).w = virtual_depth_before;
                    }
                }
            }
        }
    }

    __device__ void PlaneHypothesisRefinement_MIPM_Hex(const cudaTextureObject_t* images,
        const cudaTextureObject_t* blur_images,
        const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
        float2* pcenters, float* depth, float* cost, float4* disp_baseline, curandState* rand_state,
        const float* view_weights, const float weight_norm, float4* prior_planes,
        unsigned int* plane_masks, float* restricted_cost, const int2 p,
        const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm, int2* tilekeys_cuda)
    {
        float perturbation = 0.02f;
        const int center = p.y * WIDTH + p.x;

        float gamma = 0.5f;
        float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
        float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
        float angle_sigma = M_PI * (5.0f / 180.0f);
        float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
        float beta = 0.18f;
        float depth_prior = 0.0f;

        float depth_rand;
        float4 plane_hypothesis_rand;
        if (params.planar_prior && plane_masks[center] > 0)
        {
            depth_prior = prior_planes[center].w;
            depth_rand = curand_uniform(rand_state) * 6 * depth_sigma + (depth_prior - 3 * depth_sigma);
            plane_hypothesis_rand = GeneratePerturbedNormalLF(p, prior_planes[center], rand_state, angle_sigma);
        }
        else
        {
            if(cost[center]<0.1)
            {
                depth_rand = (curand_uniform(rand_state)-0.5) * 2.0 + plane_hypothesis[center].w;
            }
            else
            {
                depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
            }
            plane_hypothesis_rand = GenerateRandomNormalLF(p, rand_state, *depth);
        }
        float depth_perturbed = *depth;
        const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
        const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
        do
        {
            depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed-depth_min_perturbed)
                                + depth_min_perturbed;
        } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);

        float4 plane_hypothesis_perturbed = GeneratePerturbedNormalLF( p, *plane_hypothesis, rand_state, perturbation * M_PI); // GeneratePertubedPlaneHypothesis(cameras[0], p, rand_state, perturbation, *plane_hypothesis, *depth, params.depth_min, params.depth_max);

        // lzd 构建了多个新的假设视差平面，继续算，得到最小代价，作为本次迭代的最终视差平面+代价
        const int num_planes = 5;
        float depths[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
        float4 normals[num_planes] = {*plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis};

        for (int i = 0; i < num_planes; ++i)
        {
            float cost_vector[32] = {2.0f};
            float2 blur_vector[32] = {0.0f};
            float4 disp_baseline_vector[32];

            float4 temp_plane_hypothesis = normals[i];

            // todo: 满足视差平面计算视差值的公式
            float3 d_plane = DisparityPlane(p, temp_plane_hypothesis);
            temp_plane_hypothesis.w = to_disparity(p, d_plane);

            ComputeMultiViewCostVector_MIPM_Hex(images, blur_images, pcenters, p, temp_plane_hypothesis,
                                            cost_vector, blur_vector,
                                            disp_baseline_vector, params, tilekeys_cuda);

            float temp_cost = 0.0f;
            for (int j = 0; j < params.num_images-1; ++j)
            {
                if (view_weights[j] > 0)
                {
                    if (params.geom_consistency)
                    {
                        //temp_cost += view_weights[j] * (cost_vector[j] + 0.1f * ComputeGeomConsistencyCost(depth_images[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                    }
                    else
                    {
                        //temp_cost += view_weights[j] * cost_vector[j];

                        // todo: blurWeight  融合模糊差异的代价累积
                        float blur_weight = exp(-0.007368*pow((blur_vector[i].x - blur_vector[i].y), 2));
                        temp_cost += view_weights[j] * (cost_vector[j] + 2.0*(1.0-blur_weight));
        //                 if (blur_vector[i].x > 0.5 && blur_vector[i].y>0.5)
        //                 {
        //                     printf("cost: %f, blur_weight: %f,%f,  %f\n", cost_vector[i], blur_vector[i].x,
        // blur_vector[i].y, 1-blur_weight);
        //                 }
                    }
                }
            }
            temp_cost /= weight_norm;
            float4 disp_baseline_min = FindMinCostWithDispBaseline(cost_vector, disp_baseline_vector, params);
            //printf("refine: %f\n", disp_baseline_min.y);

            // TODO：可以与ACMP类似，存储虚拟深度
            float disp_before = temp_plane_hypothesis.w ; // 标准视差 d
            if (disp_before >= params.depth_min && disp_before <= params.depth_max)
            {
                if (params.planar_prior && plane_masks[center] > 0)
                {
                    float depth_diff = depths[i] - depth_prior;
                    float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
                    float angle_diff = acos(angle_cos);
                    float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                    float restricted_temp_cost = exp(-temp_cost * temp_cost / beta) * prior;
                    if (restricted_temp_cost > *restricted_cost)
                    {
                        *depth = disp_before;
                        *plane_hypothesis = temp_plane_hypothesis;
                        *cost = temp_cost;
                        *restricted_cost = restricted_temp_cost;

                        // todo: 计算虚拟深度值  长基线+代价小的算
                        float virtual_depth_before = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_vector,
    params, baseline_norm, cost_vector);
                        (*disp_baseline).w = virtual_depth_before;
                    }
                }
                else
                {
                    if (temp_cost < *cost)
                    {
                        *depth = disp_before;
                        *plane_hypothesis = temp_plane_hypothesis;
                        *cost = temp_cost;

                        // todo: 计算虚拟深度值  长基线+代价小的算
                        float virtual_depth_before = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_vector,
    params, baseline_norm, cost_vector);
                        (*disp_baseline).w = virtual_depth_before;
                    }
                }
            }
        }
    }

    __device__ float ComputeVirtualDepthConsiderCostAndBaseline(const float4* disp_baseline_vector,
                                                            const PatchMatchParamsLF& params,
                                                            const float* baseline_norm,
                                                            const float* cost_vector)
    {
        int baseline_max_idx = -1;
        float best_temp = 0.0;
        for (int i = 1; i < params.num_images; ++i)
        {
            const float temp_cost = cost_vector[i];
            const float4& disp_baseline_pair = disp_baseline_vector[i-1];
            const float temp_disp= disp_baseline_pair.y;//实际视差
            float temp_select = (1-temp_cost)*std::exp(-1.0*std::pow(temp_disp-params.base_height_ratio,2)/(2.0*params.base_height_sigma*params.base_height_sigma));
            if(temp_select>best_temp)
            {
                best_temp = temp_select;
                baseline_max_idx = i;
            }
        }
        const float4& disp_baseline_pair = disp_baseline_vector[baseline_max_idx-1];
        float virtual_depth = (disp_baseline_pair.z / (disp_baseline_pair.y+1e-7));
        return virtual_depth;
    }

    __device__ void ComputeBaselineNorm(float2* pcenters, const PatchMatchParamsLF& params, float* baseline_norm)
    {
        float baseline_max = 1.0;
        float2& center_coord = pcenters[0];
        for (int i = 1; i < params.num_images; ++i)
        {
            float2& neig_center_coord = pcenters[i];
            float d_x = center_coord.x - neig_center_coord.x;
            float d_y = center_coord.y - neig_center_coord.y;

            float baseline = sqrt((d_x*d_x) + (d_y*d_y));
            if (baseline_max < baseline)
            {
                baseline_max = baseline;
            }
            baseline_norm[i-1] = baseline;
        }

        // 归一化
        for (int i = 1; i < params.num_images; ++i)
        {
             float tmp = baseline_norm[i-1] / baseline_max;
            // tmp在(0, 1)之间
             baseline_norm[i-1] = std::exp(-tmp);
            //printf("%d: y=%f, x=%f\n", i, baseline_norm[i-1], tmp);
        }
    }

    __device__ float4 GenerateRandomNormal_MIPM( const int2 p, curandState *rand_state, const float depth,const float depth_min, const float depth_max)
    {
        float4 normal;
        float q1 = 1.0f;
        float q2 = 1.0f;
        float s = 2.0f;
        while (s >= 1.0f)
        {
            q1 = 2.0f * curand_uniform(rand_state) -1.0f;
            q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
            s = q1 * q1 + q2 * q2;
        }
        const float sq = sqrt(1.0f - s);
        normal.x = 2.0f * q1 * sq;
        normal.y = 2.0f * q2 * sq;
        normal.z = 1.0f - 2.0f * s;
        normal.w = 0;
        // float nx = 2.0f * q1 * sq;
        // float ny = 2.0f * q2 * sq;
        // float nz = 1.0f - 2.0f * s;
        // float ndisparity = curand_uniform(rand_state)*depth_max;
        // normal.x = -nx/(nz+0.0000001);
        // normal.y = -ny/(nz+0.0000001);
        // normal.z = ndisparity-(normal.x *p.x+normal.y *p.y);
        // normal.w = ndisparity;

        NormalizeVec3(&normal);

        return normal;
    }

    __device__ float4 GenerateRandomPlaneHypothesis_MIPM(const int2 p, curandState *rand_state, const float depth_min, const float depth_max)
    {
        // depth 视差
        float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
        float4 plane_hypothesis = GenerateRandomNormal_MIPM(p, rand_state, depth,depth_min,depth_max);
        plane_hypothesis.w = depth;
        return plane_hypothesis;
    }

    __device__ void DisparityGeometricMapOperate_Hex_TY(float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p0, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline,
    int2 tilekey_ref, int2 tilekey_neig)
    {
        // 1.从视差平面计算当前视角对的视差d
        float3 d_plane = DisparityPlane(p_for_plane, plane_hypothesis);
        float d = to_disparity(p0, d_plane);

        int k = HexGrid::get_ring_number(c0.x,c0.y,params.Base,c1.x,c1.y);
        //plane_hypothesis.w = d;
        //printf("%f\n", d);
        // 2. 计算基线（参考微图像中心点到邻域微图像中心点的距离）单位：像素
        float baseline = sqrt(pow((c1.y - c0.y), 2) + pow((c1.x -c0.x), 2));

        //float d_x = (c0.x - c1.x) / baseline;//这句代码啥意思，作了个距离比值
        //float d_y = (c0.y - c1.y) / baseline;
        float d_x = (c0.x - c1.x);//这句代码啥意思，作了个距离比值
        float d_y = (c0.y - c1.y);
        //float pA = baseline/k;
        //printf("scale:%f, baseline:%f, Base:%f\n", scale, baseline, params.Base);
        // 视差平面几何映射：计算邻域图像中对应点
        p1.x = p0.x + d_x*d*k/baseline;
        p1.y = p0.y + d_y*d*k/baseline;

        float d_real = sqrt((c0.x - c1.x)* (p0.x-p1.x)+(c0.y - c1.y)*(p0.y -p1.y));

        disparity_basline.x = d;
        disparity_basline.y = d_real;
        disparity_basline.z = baseline;
        disparity_basline.w = baseline/(d_real+1e-7); // 防止除以零

        //printf("%f\n", d_real);


        // if (p1.x < params.MLA_Mask_Width_Cuda && p1.x > 0.0f
        //     && p1.y < params.MLA_Mask_Height_Cuda && p1.y > 0.0f)
        // {
        //     printf("d=%f, d_real=%f, baseline=%f, v=%f\n ",
        //         d, d_real, baseline, disparity_basline.w);
        //     // printf("c0=(%f,%f), c1=(%f,%f), p0=(%d,%d), p1=(%f,%f), d=%f, d_real=%f, baseline=%f, v=%f\n ",
        //     // c0.x, c0.y, c1.x, c1.y, p.x, p.y, p1.x, p1.y, d,
        //     // d_real, baseline, disparity_basline.w);
        // }
    }

    __device__ void DisparityGeometricMapOperate_Hex_error(float2 c0, float2 c1, const int2 p_for_plane,
        float4 plane_hypothesis, const int2 p0, const PatchMatchParamsLF params,
        float2& p1, float4& disparity_basline,
        int2 tilekey_ref, int2 tilekey_neig)
    {
        // 1.从视差平面计算当前视角对的视差d
        float3 d_plane = DisparityPlane(p_for_plane, plane_hypothesis);
        float d = to_disparity(p0, d_plane);

        // 视差平面几何映射：计算邻域图像中对应点
        //int k = HexGrid::get_ring_number(c0.x,c0.y,params.Base,c1.x,c1.y);
        int k = lfhex::hex_ring_from_points_with_radius_kind<double>(c1.x,c1.y,c0.x,c0.y,params.Base*0.5,1);
        float baseline = sqrt(pow((c1.y - c0.y), 2) + pow((c1.x -c0.x), 2));
        float d_x = (c0.x - c1.x);//这句代码啥意思，作了个距离比值
        float d_y = (c0.y - c1.y);
        p1.x = p0.x + d_x*d*k/baseline;
        p1.y = p0.y + d_y*d*k/baseline;

        ///
        // 视差平面几何映射：计算邻域图像中对应点
        // float baseline = sqrt(pow((c1.y - c0.y), 2) + pow((c1.x -c0.x), 2));
        // float d_x = (c0.x - c1.x) / baseline;//这句代码啥意思，作了个距离比值
        // float d_y = (c0.y - c1.y) / baseline;
        // float scale = baseline/params.Base;
        // p1.x = p0.x + d_x*d*scale;
        // p1.y = p0.y + d_y*d*scale;
        ///

        float d_real = sqrt((c0.x - c1.x)* (p0.x-p1.x)+(c0.y - c1.y)*(p0.y -p1.y));

        disparity_basline.x = d;
        disparity_basline.y = d_real;
        disparity_basline.z = baseline;
        disparity_basline.w = baseline/(d_real+1e-7); // 防止除以零
    }

    __device__ __forceinline__ int lin16(const int2 k)
{
    return ((k.x & 0xFFF) << 16) | (k.y & 0xFFF);
}

    // —— 工具：tilekey -> 线性下标（务必与 Host 一致：y*W + x）——
    __device__ __forceinline__ int linFromTile(const int2 tk, const PatchMatchParamsLF& P){
        return tk.y * P.MLA_Mask_Width_Cuda + tk.x;
    }

    __device__ __forceinline__ bool HexPTotFetchByKey(const int2 tilekey_neig,
                                                  const PatchMatchParamsLF& params,
                                                  float& pTot_out, int& kStep_out)
{
    pTot_out = 0.f; kStep_out = 1;
    if (LFMVS::g_nei_lin == nullptr || LFMVS::g_M <= 0) return false;

    const int neig_lin = linFromTile(tilekey_neig, params);
    int hit = -1;
    #pragma unroll
    for (int t = 0; t < LFMVS::g_M; ++t){
        if (LFMVS::g_nei_lin[t] == neig_lin){ hit = t; break; }
    }
    if (hit < 0) return false;

    kStep_out  = max(1, LFMVS::g_k_list[hit]);
    pTot_out   = fmaxf(1e-6f, LFMVS::g_pTot_list[hit]);
    return true;
}

    /**
     * @brief Hex路径：从视差平面映射到邻域图像中的同名点（使用 p_Δ^{tot}）
     * 签名保持不变；内部改为：
     *   p1 = p0 + (d * k / pTot) * b
     * 其中：
     *   d    : to_disparity() 给出的视差（单位：像素/步）
     *   k    : 六边形步数（来自 Host 在 Initialize() 为本参考下发的小表）
     *   pTot : p_Δ^{tot}（单位：像素；来自小表）
     *   b    : 参考→邻域的基线向量（像素） = c0 - c1
     */
    __device__ void DisparityGeometricMapOperate_Hex(
        float2 c0, float2 c1, const int2 p_for_plane,
        float4 plane_hypothesis, const int2 p0, const PatchMatchParamsLF params,
        float2& p1, float4& disparity_basline,
        int2 tilekey_ref, int2 tilekey_neig)
    {
        // 1.从视差平面计算当前视角对的视差d
        float3 d_plane = DisparityPlane(p_for_plane, plane_hypothesis);
        float d = to_disparity(p0, d_plane);
        // 2. 计算基线（参考微图像中心点到邻域微图像中心点的距离）单位：像素
        float baseline = sqrt(pow((c1.y - c0.y), 2) + pow((c1.x -c0.x), 2));

        // 视差平面几何映射：计算邻域图像中对应点
        float d_x = (c0.x - c1.x) / baseline;//这句代码啥意思，作了个距离比值
        float d_y = (c0.y - c1.y) / baseline;
        float scale = baseline/params.Base;
        p1.x = p0.x + d_x*d*scale;
        p1.y = p0.y + d_y*d*scale;

        float d_real = sqrt((c0.x - c1.x)* (p0.x-p1.x)+(c0.y - c1.y)*(p0.y -p1.y));
        disparity_basline.x = d;
        disparity_basline.y = d_real;
        disparity_basline.z = baseline;
        disparity_basline.w = baseline/(d_real+1e-7); // 防止除以零
    }

    // TODO：效率很低，待改进
    __device__ void DisparityGeometricMapOperate_Hex_TimeToken(
    float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p0, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline,
    int2 tilekey_ref, int2 tilekey_neig)
    {
        // 1) 视差（单位：像素/步）
        const float3 d_plane = DisparityPlane(p_for_plane, plane_hypothesis);
        const float  d       = to_disparity(p0, d_plane);

        // 2) 基线向量（参考 → 邻域），注意方向必须是 c0 - c1
        const float2 b  = make_float2(c0.x - c1.x, c0.y - c1.y);
        const float  bn = sqrtf(fmaxf(0.f, b.x*b.x + b.y*b.y)); // |b|

        // 3) 直接通过 HexPTotFetchByKey() 查询 (k, pTot)
        int   k    = 1;
        float pTot = 0.f;
        bool  ok   = HexPTotFetchByKey(tilekey_neig, params, pTot, k);

        // 4) 兜底（少数未命中小表时）：避免崩溃，回退到粗略近似
        if (!ok || pTot <= 0.f){
    #if defined(USE_LFHEX_DEVICE) || defined(HAS_LFHEX_DEVICE)
            k = hex_ring_index_device(c0.x, c0.y, c1.x, c1.y, params.Base*0.5, 1);
            k = max(1, k);
    #else
            const float bn = sqrtf(fmaxf(0.f, (c0.x-c1.x)*(c0.x-c1.x) + (c0.y-c1.y)*(c0.y-c1.y)));
            k = max(1, (int)floorf(bn / fmaxf(1e-6f, params.Base) + 0.5f));
    #endif
            pTot = fmaxf(1e-6f, params.Base * (float)k);
        }


        // 5) 新的几何映射：p1 = p0 + (d * k / pTot) * b
        const float alpha = d * ((float)k / pTot);
        p1.x = p0.x + alpha * b.x;
        p1.y = p0.y + alpha * b.y;

        // 6) 诊断量（便于调试/日志核对）
        const float d_real = (bn > 1e-6f)
            ? fabsf( b.x * (p0.x - p1.x) + b.y * (p0.y - p1.y) ) / bn
            : 0.f;

        // disparity_basline：
        //   x = d（px/step）; y = d_real（回投沿 b 的标量视差，px）
        //   z = |b| baseline（px）; w = pTot（px）
        disparity_basline = make_float4(d, d_real, bn, pTot);
    }

    __device__ void DisparityGeometricMapOperate(float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p0, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline)
    {
        // 1.从视差平面计算当前视角对的视差d
        float3 d_plane = DisparityPlane(p_for_plane, plane_hypothesis);
        float d = to_disparity(p0, d_plane);
        // 2. 计算基线（参考微图像中心点到邻域微图像中心点的距离）单位：像素
        float baseline = sqrt(pow((c1.y - c0.y), 2) + pow((c1.x -c0.x), 2));

        // 视差平面几何映射：计算邻域图像中对应点
        float d_x = (c0.x - c1.x) / baseline;//这句代码啥意思，作了个距离比值
        float d_y = (c0.y - c1.y) / baseline;
        float scale = baseline/params.Base;
        p1.x = p0.x + d_x*d*scale;
        p1.y = p0.y + d_y*d*scale;

        float d_real = sqrt((c0.x - c1.x)* (p0.x-p1.x)+(c0.y - c1.y)*(p0.y -p1.y));
        disparity_basline.x = d;
        disparity_basline.y = d_real;
        disparity_basline.z = baseline;
        disparity_basline.w = baseline/(d_real+1e-7); // 防止除以零
    }

    __device__ float ComputeBilateralNCC_MIPM(const cudaTextureObject_t ref_image,
                const cudaTextureObject_t ref_blur_image, float2 p0,
                const cudaTextureObject_t src_image, float2 p1,
                const cudaTextureObject_t src_blur_image, const int2 p,
                float4 plane_hypothesis, const PatchMatchParamsLF params,
                float2& blur_value, float4& disparity_baseline)
    {
        // p为参考图像中的当前像素
        const float cost_max = 2.0f;
        int radius = params.patch_size / 2; // 5
        float2 pt;
        //计算参考图像像素p对应的领域图像同名点
        DisparityGeometricMapOperate(p0, p1, p, plane_hypothesis, p, params,
            pt, disparity_baseline);

        if (pt.x >= params.MLA_Mask_Width_Cuda || pt.x < 0.0f
            || pt.y >= params.MLA_Mask_Height_Cuda || pt.y < 0.0f)
        {
            return cost_max;
        }
        //printf("MIPM290");

        float cost = 0.0f;
        {
            float sum_ref = 0.0f;
            float sum_ref_ref = 0.0f;
            float sum_src = 0.0f;
            float sum_src_src = 0.0f;
            float sum_ref_src = 0.0f;
            float bilateral_weight_sum = 0.0f;
            const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
            //printf("%f\n", ref_center_pix);

            float ref_blur_pixel_count = 0.0;
            float src_blur_pixel_count = 0.0;
            float pixel_max = 255.0;
            // LZD：块匹配中的窗口：以当前像素为中心，计算窗口大小内的像素组建的代价
            for (int i = -radius; i < radius + 1; i += params.radius_increment)
            {
                float sum_ref_row = 0.0f;
                float sum_src_row = 0.0f;
                float sum_ref_ref_row = 0.0f;
                float sum_src_src_row = 0.0f;
                float sum_ref_src_row = 0.0f;
                float bilateral_weight_sum_row = 0.0f;

                for (int j = -radius; j < radius + 1; j += params.radius_increment)
                {
                    const int2 ref_pt = make_int2(p.x + i, p.y + j);

                    // if (ref_pt.x >= params.MLA_Mask_Width_Cuda || ref_pt.x < 0.0f
                    //     || ref_pt.y >= params.MLA_Mask_Height_Cuda || ref_pt.y < 0.0f)
                    // {
                    //     //printf("buquan\n");
                    //     continue;
                    // }

                    float2 src_pt; // 邻居图像中的像素坐标：通过视差平面和基线向量计算得来
                    float4 disparity_baseline_src;
                    DisparityGeometricMapOperate(p0, p1, p, plane_hypothesis, ref_pt,
                        params, src_pt, disparity_baseline_src);
                    // if (src_pt.x >= params.MLA_Mask_Width_Cuda || src_pt.x < 0.0f
                    //     || src_pt.y >= params.MLA_Mask_Height_Cuda || src_pt.y < 0.0f)
                    // {
                    //     //printf("buquan\n");
                    //     continue;
                    // }

                    const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                    const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                    // 模糊差异程度
                    float ref_blur_v = tex2D<float>(ref_blur_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                    ref_blur_v /= pixel_max;
                    blur_value.x += ref_blur_v;
                    if (ref_blur_v > 0.5) // 128
                    {
                        ref_blur_pixel_count++;
                    }
                    float src_blur_v = tex2D<float>(src_blur_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
                    src_blur_v /= pixel_max;
                    blur_value.y += src_blur_v;
                    if (src_blur_v > 0.5) // 128
                    {
                        src_blur_pixel_count++;
                    }

                    float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix,
                        params.sigma_spatial, params.sigma_color);

                    sum_ref_row += weight * ref_pix;
                    sum_ref_ref_row += weight * ref_pix * ref_pix;
                    sum_src_row += weight * src_pix;
                    sum_src_src_row += weight * src_pix * src_pix;
                    sum_ref_src_row += weight * ref_pix * src_pix;
                    bilateral_weight_sum_row += weight;
                }

                sum_ref += sum_ref_row;
                sum_ref_ref += sum_ref_ref_row;
                sum_src += sum_src_row;
                sum_src_src += sum_src_src_row;
                sum_ref_src += sum_ref_src_row;
                bilateral_weight_sum += bilateral_weight_sum_row;
            }

            const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
            sum_ref *= inv_bilateral_weight_sum;
            sum_ref_ref *= inv_bilateral_weight_sum;
            sum_src *= inv_bilateral_weight_sum;
            sum_src_src *= inv_bilateral_weight_sum;
            sum_ref_src *= inv_bilateral_weight_sum;

            const float var_ref = sum_ref_ref - sum_ref * sum_ref;
            const float var_src = sum_src_src - sum_src * sum_src;

            const float kMinVar = 1e-5f;
            if (var_ref < kMinVar || var_src < kMinVar)
            {
                return cost_max;
            }
            else
            {
                const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
                const float var_ref_src = sqrt(var_ref * var_src);
                return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
            }
        }
    }

        __device__ float ComputeBilateralNCC_MIPM_Hex(const cudaTextureObject_t ref_image,
                const cudaTextureObject_t ref_blur_image, float2 p0,
                const cudaTextureObject_t src_image, float2 p1,
                const cudaTextureObject_t src_blur_image, const int2 p,
                float4 plane_hypothesis, const PatchMatchParamsLF params,
                float2& blur_value, float4& disparity_baseline,
                int2 tilekey_ref, int2 tilekey_neig)
    {
        // p为参考图像中的当前像素
        const float cost_max = 2.0f;
        int radius = params.patch_size / 2; // 5
        float2 pt;
        //计算参考图像像素p对应的领域图像同名点
        DisparityGeometricMapOperate_Hex(p0, p1, p, plane_hypothesis, p, params,
            pt, disparity_baseline, tilekey_ref, tilekey_neig);

        if (pt.x >= params.MLA_Mask_Width_Cuda || pt.x < 0.0f
            || pt.y >= params.MLA_Mask_Height_Cuda || pt.y < 0.0f)
        {
            return cost_max;
        }
        //printf("MIPM290");

        float cost = 0.0f;
        {
            float sum_ref = 0.0f;
            float sum_ref_ref = 0.0f;
            float sum_src = 0.0f;
            float sum_src_src = 0.0f;
            float sum_ref_src = 0.0f;
            float bilateral_weight_sum = 0.0f;
            const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
            //printf("%f\n", ref_center_pix);

            float ref_blur_pixel_count = 0.0;
            float src_blur_pixel_count = 0.0;
            float pixel_max = 255.0;
            // LZD：块匹配中的窗口：以当前像素为中心，计算窗口大小内的像素组建的代价
            for (int i = -radius; i < radius + 1; i += params.radius_increment)
            {
                float sum_ref_row = 0.0f;
                float sum_src_row = 0.0f;
                float sum_ref_ref_row = 0.0f;
                float sum_src_src_row = 0.0f;
                float sum_ref_src_row = 0.0f;
                float bilateral_weight_sum_row = 0.0f;

                for (int j = -radius; j < radius + 1; j += params.radius_increment)
                {
                    const int2 ref_pt = make_int2(p.x + i, p.y + j);

                    // if (ref_pt.x >= params.MLA_Mask_Width_Cuda || ref_pt.x < 0.0f
                    //     || ref_pt.y >= params.MLA_Mask_Height_Cuda || ref_pt.y < 0.0f)
                    // {
                    //     //printf("buquan\n");
                    //     continue;
                    // }

                    float2 src_pt; // 邻居图像中的像素坐标：通过视差平面和基线向量计算得来
                    float4 disparity_baseline_src;
                    DisparityGeometricMapOperate(p0, p1, p, plane_hypothesis, ref_pt,
                        params, src_pt, disparity_baseline_src);
                    // if (src_pt.x >= params.MLA_Mask_Width_Cuda || src_pt.x < 0.0f
                    //     || src_pt.y >= params.MLA_Mask_Height_Cuda || src_pt.y < 0.0f)
                    // {
                    //     //printf("buquan\n");
                    //     continue;
                    // }

                    const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                    const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                    // 模糊差异程度
                    float ref_blur_v = tex2D<float>(ref_blur_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                    ref_blur_v /= pixel_max;
                    blur_value.x += ref_blur_v;
                    if (ref_blur_v > 0.5) // 128
                    {
                        ref_blur_pixel_count++;
                    }
                    float src_blur_v = tex2D<float>(src_blur_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
                    src_blur_v /= pixel_max;
                    blur_value.y += src_blur_v;
                    if (src_blur_v > 0.5) // 128
                    {
                        src_blur_pixel_count++;
                    }

                    float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix,
                        params.sigma_spatial, params.sigma_color);

                    sum_ref_row += weight * ref_pix;
                    sum_ref_ref_row += weight * ref_pix * ref_pix;
                    sum_src_row += weight * src_pix;
                    sum_src_src_row += weight * src_pix * src_pix;
                    sum_ref_src_row += weight * ref_pix * src_pix;
                    bilateral_weight_sum_row += weight;
                }

                sum_ref += sum_ref_row;
                sum_ref_ref += sum_ref_ref_row;
                sum_src += sum_src_row;
                sum_src_src += sum_src_src_row;
                sum_ref_src += sum_ref_src_row;
                bilateral_weight_sum += bilateral_weight_sum_row;
            }

            const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
            sum_ref *= inv_bilateral_weight_sum;
            sum_ref_ref *= inv_bilateral_weight_sum;
            sum_src *= inv_bilateral_weight_sum;
            sum_src_src *= inv_bilateral_weight_sum;
            sum_ref_src *= inv_bilateral_weight_sum;

            const float var_ref = sum_ref_ref - sum_ref * sum_ref;
            const float var_src = sum_src_src - sum_src * sum_src;

            const float kMinVar = 1e-5f;
            if (var_ref < kMinVar || var_src < kMinVar)
            {
                return cost_max;
            }
            else
            {
                const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
                const float var_ref_src = sqrt(var_ref * var_src);
                return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
            }
        }
    }

    __device__ void ComputeMultiViewCostVector_MIPM(const cudaTextureObject_t* images,
                            const cudaTextureObject_t* blur_images, float2* pcenters,
                            const int2 p, float4 plane_hypothesis,
                            float* cost_vector, float2* blure_array,
                            float4* disp_baseline_array, const PatchMatchParamsLF params)
    {
        for (int i = 1; i < params.num_images; ++i)
        {
            cost_vector[i-1] = ComputeBilateralNCC_MIPM(images[0], blur_images[0],
                pcenters[0], images[i], pcenters[i],
                blur_images[i], p, plane_hypothesis,
                params, blure_array[i-1],
                disp_baseline_array[i-1]);
        }
    }

    __device__ void ComputeMultiViewCostVector_MIPM_Hex(const cudaTextureObject_t* images,
                        const cudaTextureObject_t* blur_images, float2* pcenters,
                        const int2 p, float4 plane_hypothesis,
                        float* cost_vector, float2* blure_array,
                        float4* disp_baseline_array, const PatchMatchParamsLF params, int2* tilekeys_cuda)
    {
        for (int i = 1; i < params.num_images; ++i)
        {
            cost_vector[i-1] = ComputeBilateralNCC_MIPM_Hex(images[0], blur_images[0],
                pcenters[0], images[i], pcenters[i],
                blur_images[i], p, plane_hypothesis,
                params, blure_array[i-1],
                disp_baseline_array[i-1],
                tilekeys_cuda[0], tilekeys_cuda[i]);
        }
    }

    __device__ void CheckerboardPropagation_MIPM(const cudaTextureObjects* texture_objects,
            const cudaTextureObject_t* depths, float4* plane_hypotheses,
            float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
            unsigned int* selected_views, float4* prior_planes,
            unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
            const int iter, const int WIDTH, const int HEIGHT)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;

        int width = WIDTH;
        int height = HEIGHT;
        // 边界检查
        if (p.x >= width || p.y >= height)
        {
            return;
        }

        int farDis = 2; // 11 2
        int nerDis = 2; // 3 2

        const int center = p.y * width + p.x;
        int left_near = center - 1;
        int left_far = center - 3;
        int right_near = center + 1;
        int right_far = center + 3;
        int up_near = center - width;
        int up_far = center - 3 * width;
        int down_near = center + width;
        int down_far = center + 3 * width; // 邻近八个像素

        // Adaptive Checkerboard Sampling
        float cost_array[8][32] = {2.0f};
        float2 blur_array[8][32] = {0.0};
        float4 disp_baseline_array[8][32];
        // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
        bool flag[8] = {false};
        int num_valid_pixels = 0;

        float costMin;
        int costMinPoint;

        // 计算归一化的基线
        float baseline_norm[32] = {0.0};
        ComputeBaselineNorm(pcenters, params, baseline_norm);

        // up_far
        if (p.y > 2)
        {
            flag[1] = true;
            num_valid_pixels++;
            costMin = costs[up_far];
            costMinPoint = up_far;
            // 正上方寻找代价最小的像素作为真正的up_far
            for (int i = 1; i < farDis; ++i) // 原 i < 11
            {
                if (p.y > 2 + i)
                {
                    int pointTemp = up_far - i*width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            up_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[up_far],
                cost_array[1], blur_array[1], disp_baseline_array[1], params);
        }

        // down_far
        if (p.y < height - 3)
        {
            flag[3] = true;
            num_valid_pixels++;
            costMin = costs[down_far];
            costMinPoint = down_far;
            for (int i = 1; i < farDis; ++i) // 11
            {
                if (p.y < height - 3 - i)
                {
                    int pointTemp = down_far + i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            down_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[down_far],
                cost_array[3], blur_array[3], disp_baseline_array[3], params);
        }

        // left_far
        if (p.x > 2)
        {
            flag[5] = true;
            num_valid_pixels++;
            costMin = costs[left_far];
            costMinPoint = left_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (p.x > 2 + i)
                {
                    int pointTemp = left_far - i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            left_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[left_far],
                cost_array[5], blur_array[5], disp_baseline_array[5], params);
        }

        // right_far
        if (p.x < width - 3)
        {
            flag[7] = true;
            num_valid_pixels++;
            costMin = costs[right_far];
            costMinPoint = right_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (p.x < width - 3 - i)
                {
                    int pointTemp = right_far + i;
                    if (costMin < costs[pointTemp])
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            right_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[right_far],
                cost_array[7], blur_array[7], disp_baseline_array[7], params);
        }

        // up_near
        if (p.y > 0)
        {
            flag[0] = true;
            num_valid_pixels++;
            costMin = costs[up_near];
            costMinPoint = up_near;
            for (int i = 0; i < nerDis; ++i) // 3
            {
                if (p.y > 1 + i && p.x > i)
                {
                    int pointTemp = up_near - (1 + i) * width - i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.y > 1 + i && p.x < width - 1 - i)
                {
                    int pointTemp = up_near - (1 + i) * width + i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            up_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[up_near],
                cost_array[0], blur_array[0], disp_baseline_array[0], params);
        }

        // down_near
        if (p.y < height - 1)
        {
            flag[2] = true;
            num_valid_pixels++;
            costMin = costs[down_near];
            costMinPoint = down_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (p.y < height - 2 - i && p.x > i)
                {
                    int pointTemp = down_near + (1 + i) * width - i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.y < height - 2 - i && p.x < width - 1 - i)
                {
                    int pointTemp = down_near + (1 + i) * width + i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            down_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[down_near],
                cost_array[2], blur_array[2], disp_baseline_array[2], params);
        }

        // left_near
        if (p.x > 0)
        {
            flag[4] = true;
            num_valid_pixels++;
            costMin = costs[left_near];
            costMinPoint = left_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (p.x > 1 + i && p.y > i)
                {
                    int pointTemp = left_near - (1 + i) - i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.x > 1 + i && p.y < height - 1 - i)
                {
                    int pointTemp = left_near - (1 + i) + i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            left_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[left_near],
                cost_array[4], blur_array[4], disp_baseline_array[4], params);
        }

        // right_near
        if (p.x < width - 1)
        {
            flag[6] = true;
            num_valid_pixels++;
            costMin = costs[right_near];
            costMinPoint = right_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (p.x < width - 2 - i && p.y > i)
                {
                    int pointTemp = right_near + (1 + i) - i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.x < width - 2 - i && p.y < height - 1- i)
                {
                    int pointTemp = right_near + (1 + i) + i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            right_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[right_near],
                cost_array[6], blur_array[6], disp_baseline_array[6], params);
        }
        const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

        //printf("%f\n ",cost_array[7][0] );

        // Multi-hypothesis Joint View Selection
        float view_weights[32] = {0.0f};
        float view_selection_priors[32] = {0.0f};
        int neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};
        for (int i = 0; i < 4; ++i)
        {
            if (flag[2*i])
            {
                for (int j = 0; j < params.num_images - 1; ++j)
                {
                    if (isSet(selected_views[neighbor_positions[i]], j) == 1)
                    {
                        //权重1：4[i]*18[j]=72
                        view_selection_priors[j] += 0.9f;
                    }
                    else
                    {
                        view_selection_priors[j] += 0.1f;
                    }
                }
            }
        }

        float sampling_probs[32] = {0.0f};
        float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f)); // 0.8 0.79116 0.765223
        for (int i = 0; i < params.num_images - 1; i++)
        {
            float count = 0;
            int count_false = 0;
            float tmpw = 0;
            for (int j = 0; j < 8; j++)
            {
                if (cost_array[j][i] < cost_threshold)
                {
                    tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                    count++;
                }
                if (cost_array[j][i] > 1.2f)
                {
                    count_false++;
                }
            }
            if (count > 2 && count_false < 3)
            {
                //权重2：8[j]*18[i]=144
                sampling_probs[i] = tmpw / count;
            }
            else if (count_false < 3)
            {
                sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
            }
            sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
        }

        TransformPDFToCDF(sampling_probs, params.num_images - 1);
        for (int sample = 0; sample < 15; ++sample)
        {
            const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

            for (int image_id = 0; image_id < params.num_images - 1; ++image_id)
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
        int num_selected_view = 0;
        float weight_norm = 0;
        for (int i = 0; i < params.num_images - 1; ++i)
        {
            //printf("view_weight: %d,%f\n", i, view_weights[i]);
            if (view_weights[i] > 0)
            {
                setBit(temp_selected_views, i);
                weight_norm += view_weights[i];
                num_selected_view++;
            }
        }
        //printf("weight_norm=%f\n", weight_norm);

        float final_costs[8] = {0.0f};
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < params.num_images - 1; ++j)
            {
                if (view_weights[j] > 0)
                {
                    if (params.geom_consistency)
                    {
                        if (flag[i])
                        {
                            //final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * ComputeGeomConsistencyCost(depths[j+1], cameras[0], cameras[j+1], plane_hypotheses[positions[i]], p));
                        }
                        else
                        {
                            final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * 5.0f);
                        }
                    }
                    else
                    {
                        //final_costs[i] += view_weights[j] * cost_array[i][j];

                        // todo：blurWeight
                        float blur_weight = exp(-0.007368*pow((blur_array[i][j].x - blur_array[i][j].y), 2));
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 2.0*(1-blur_weight));
                    }
                }
            }
            final_costs[i] /= weight_norm;
        }
        const int min_cost_idx = FindMinCostIndex(final_costs, 8);
        float4 disp_baseline_min = FindMinCostWithDispBaseline(cost_array[min_cost_idx],
                                    disp_baseline_array[min_cost_idx], params);
        float virtual_depth_before = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_array[min_cost_idx],
                                                                        params,
                                                                        baseline_norm,
                                                                        cost_array[min_cost_idx]);

        // 利用自己的平面假设计算代价
        float cost_vector_now[32] = {2.0f};
        float2 blur_vector_now[32] = {0.0f};
        float4 disp_baseline_vector_now[32] = {0.0};
        ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[center],
                                        cost_vector_now, blur_vector_now,
                                        disp_baseline_vector_now, params);

        float cost_now = 0.0f;
        for (int i = 0; i < params.num_images-1; ++i)
        {
            if (params.geom_consistency)
            {
                //cost_now += view_weights[i] * (cost_vector_now[i] + 0.1f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p));
            }
            else
            {
                //cost_now += view_weights[i] * cost_vector_now[i] ;

                // todo：blurWeight
                float blur_weight = exp(-0.007368*pow((blur_vector_now[i].x - blur_vector_now[i].y), 2));
                cost_now += view_weights[i] * (cost_vector_now[i] + 2.0*(1-blur_weight));
            }
        }
        cost_now /= weight_norm; // 平均代价
        costs[center] = cost_now;
        float4 disp_baseine_min_now = FindMinCostWithDispBaseline(cost_vector_now,
                                    disp_baseline_vector_now, params);
        float virtual_depth_now = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_vector_now,
                                    params, baseline_norm, cost_vector_now);

        float depth_now = plane_hypotheses[center].w;
        //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);

        float restricted_cost = 0.0f;
        if (params.planar_prior)
        {
            float restricted_final_costs[8] = {0.0f};
            float gamma = 0.5f;
            float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
            float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
            float angle_sigma = M_PI * (5.0f / 180.0f);
            float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
            //float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
            float depth_prior =  prior_planes[center].w;
            float beta = 0.18f;

            if (plane_masks[center] > 0)
            {
                for (int i = 0; i < 8; i++)
                {
                    if (flag[i])
                    {
                        //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[i]], p);
                        float depth_now = plane_hypotheses[center].w;
                        float depth_diff = depth_now - depth_prior;
                        float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[positions[i]]);
                        float angle_diff = acos(angle_cos);
                        float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                        restricted_final_costs[i] = exp(-final_costs[i] * final_costs[i] / beta) * prior;
                    }
                }
                const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

                float restricted_cost_now = 0.0f;
                //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
                float depth_now = plane_hypotheses[center].w;
                float depth_diff = depth_now - depth_prior;
                float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[center]);
                float angle_diff = acos(angle_cos);
                float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                restricted_cost_now = exp(-cost_now * cost_now / beta) * prior;

                if (flag[max_cost_idx])
                {
                    float disp_before = disp_baseline_array[min_cost_idx][0].x;
                    // 存储虚拟深度值： 长基线+代价小，加权平均
                    disp_baseline[center] = disp_baseline_min;
                    disp_baseline[center].w = virtual_depth_now;
                    if (disp_before >= params.depth_min && disp_before <= params.depth_max
                        && restricted_final_costs[max_cost_idx] > restricted_cost_now)
                    {
                        depth_now  = disp_before;
                        plane_hypotheses[center] = plane_hypotheses[positions[max_cost_idx]]; // lzd 赋值新的平面、视差
                        costs[center] = final_costs[max_cost_idx];
                        restricted_cost = restricted_final_costs[max_cost_idx];

                        // 存储虚拟深度值： 长基线+代价小，加权平均
                        disp_baseline[center] = disp_baseline_min;
                        disp_baseline[center].w = virtual_depth_before;

                        selected_views[center] = temp_selected_views;
                    }
                }
            }
            else if (flag[min_cost_idx])
            {
                //float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
                float disp_before = disp_baseline_array[min_cost_idx][0].x;
                // 存储虚拟深度值： 长基线+代价小，加权平均
                disp_baseline[center] = disp_baseline_min;
                disp_baseline[center].w = virtual_depth_now;
                if (disp_before >= params.depth_min && disp_before <= params.depth_max
                    && final_costs[min_cost_idx] < cost_now)
                {
                    depth_now = disp_before; // 标准视差 d

                    plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]]; // lzd 赋值新的平面、视差
                    costs[center] = final_costs[min_cost_idx]; // 平均代价

                    // 存储虚拟深度值： 长基线+代价小，加权平均
                    disp_baseline[center] = disp_baseline_min;
                    disp_baseline[center].w = virtual_depth_before;

                    selected_views[center] = temp_selected_views;
                }
            }
        }

        if (!params.planar_prior && flag[min_cost_idx])
        {
            float disp_before = disp_baseline_array[min_cost_idx][0].x;
            // 存储虚拟深度值： 长基线+代价小，加权平均
            disp_baseline[center] = disp_baseline_min;
            disp_baseline[center].w = virtual_depth_now;
            if (disp_before >= params.depth_min && disp_before <= params.depth_max
                && final_costs[min_cost_idx] < cost_now)
            {
                depth_now = disp_before; // 标准视差 d
                plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]]; // lzd 赋值新的平面、视差
                costs[center] = final_costs[min_cost_idx]; // 平均代价

                // 存储虚拟深度值： 长基线+代价小，加权平均
                disp_baseline[center] = disp_baseline_min;
                disp_baseline[center].w = virtual_depth_before;

                selected_views[center] = temp_selected_views;
            }
        }
        PlaneHypothesisRefinement_MIPM(images, blur_images, depths,
            &plane_hypotheses[center], pcenters,
            &depth_now, &costs[center], &disp_baseline[center],
            &rand_states[center],
            view_weights, weight_norm, prior_planes, plane_masks,
            &restricted_cost, p, params,WIDTH, baseline_norm);
    }

    __device__ void CheckerboardPropagation_MIPM_ConsiderBorderPixels(const cudaTextureObjects* texture_objects,
            const cudaTextureObject_t* depths, float4* plane_hypotheses,
            float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
            unsigned int* selected_views, float4* prior_planes,
            unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
            const int iter, const int WIDTH, const int HEIGHT)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;

        int width = WIDTH;
        int height = HEIGHT;
        // 边界检查
        if (p.x >= width || p.y >= height)
        {
            return;
        }

        int farDis = 2; // 11 2
        int nerDis = 2; // 3 2

        const int center = p.y * width + p.x;
        int left_near = center - 1;
        int left_far = center - 3;
        int right_near = center + 1;
        int right_far = center + 3;
        int up_near = center - width;
        int up_far = center - 3 * width;
        int down_near = center + width;
        int down_far = center + 3 * width; // 邻近八个像素

        // Adaptive Checkerboard Sampling
        float cost_array[8][32] = {2.0f};
        float2 blur_array[8][32] = {0.0};
        float4 disp_baseline_array[8][32];
        // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
        bool flag[8] = {false};
        int num_valid_pixels = 0;

        float costMin;
        int costMinPoint;

        // 计算归一化的基线
        float baseline_norm[32] = {0.0};
        ComputeBaselineNorm(pcenters, params, baseline_norm);

        // up_far
        if (p.y > 2 )
        {
            flag[1] = true;
            num_valid_pixels++;
            costMin = costs[up_far];
            costMinPoint = up_far;
            // 正上方寻找代价最小的像素作为真正的up_far
            for (int i = 1; i < farDis; ++i) // 原 i < 11
            {
                if (p.y > 2 + i)
                {
                    int pointTemp = up_far - i*width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            up_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[up_far],
                cost_array[1], blur_array[1], disp_baseline_array[1], params);
        }

        // down_far
        if (p.y < height - 3)
        {
            flag[3] = true;
            num_valid_pixels++;
            costMin = costs[down_far];
            costMinPoint = down_far;
            for (int i = 1; i < farDis; ++i) // 11
            {
                if (p.y < height - 3 - i)
                {
                    int pointTemp = down_far + i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            down_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[down_far],
                cost_array[3], blur_array[3], disp_baseline_array[3], params);
        }

        // left_far
        if (p.x > 2)
        {
            flag[5] = true;
            num_valid_pixels++;
            costMin = costs[left_far];
            costMinPoint = left_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (p.x > 2 + i)
                {
                    int pointTemp = left_far - i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            left_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[left_far],
                cost_array[5], blur_array[5], disp_baseline_array[5], params);
        }

        // right_far
        if (p.x < width - 3)
        {
            flag[7] = true;
            num_valid_pixels++;
            costMin = costs[right_far];
            costMinPoint = right_far;
            for (int i = 1; i < farDis; ++i)
            {
                if (p.x < width - 3 - i)
                {
                    int pointTemp = right_far + i;
                    if (costMin < costs[pointTemp])
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            right_far = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[right_far],
                cost_array[7], blur_array[7], disp_baseline_array[7], params);
        }

        // up_near
        if (p.y > 0)
        {
            flag[0] = true;
            num_valid_pixels++;
            costMin = costs[up_near];
            costMinPoint = up_near;
            for (int i = 0; i < nerDis; ++i) // 3
            {
                if (p.y > 1 + i && p.x > i)
                {
                    int pointTemp = up_near - (1 + i) * width - i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.y > 1 + i && p.x < width - 1 - i)
                {
                    int pointTemp = up_near - (1 + i) * width + i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            up_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[up_near],
                cost_array[0], blur_array[0], disp_baseline_array[0], params);
        }

        if (p.y <= 2 && p.y > 0)
        {
            flag[1] = true; num_valid_pixels++;
            up_far = up_near;
            // 复制上近邻的代价和视差信息到上远邻槽位
            for (int j = 0; j < params.num_images - 1; ++j) {
                cost_array[1][j] = cost_array[0][j];
                blur_array[1][j] = blur_array[0][j];
                disp_baseline_array[1][j] = disp_baseline_array[0][j];
            }
        }

        // down_near
        if (p.y < height - 1)
        {
            flag[2] = true;
            num_valid_pixels++;
            costMin = costs[down_near];
            costMinPoint = down_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (p.y < height - 2 - i && p.x > i)
                {
                    int pointTemp = down_near + (1 + i) * width - i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.y < height - 2 - i && p.x < width - 1 - i)
                {
                    int pointTemp = down_near + (1 + i) * width + i;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            down_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[down_near],
                cost_array[2], blur_array[2], disp_baseline_array[2], params);
        }

        // **边界：填充下远邻**
    if (p.y >= height - 3 && p.y < height - 1) {
        flag[3] = true; num_valid_pixels++;
        down_far = down_near;
        for (int j = 0; j < params.num_images - 1; ++j) {
            cost_array[3][j] = cost_array[2][j];
            blur_array[3][j] = blur_array[2][j];
            disp_baseline_array[3][j] = disp_baseline_array[2][j];
        }
    }

        // left_near
        if (p.x > 0)
        {
            flag[4] = true;
            num_valid_pixels++;
            costMin = costs[left_near];
            costMinPoint = left_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (p.x > 1 + i && p.y > i)
                {
                    int pointTemp = left_near - (1 + i) - i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.x > 1 + i && p.y < height - 1 - i)
                {
                    int pointTemp = left_near - (1 + i) + i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            left_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[left_near],
                cost_array[4], blur_array[4], disp_baseline_array[4], params);
        }

        // **边界：填充左远邻**
        if (p.x <= 2 && p.x > 0) {
            flag[5] = true; num_valid_pixels++;
            left_far = left_near;
            for (int j = 0; j < params.num_images - 1; ++j) {
                cost_array[5][j] = cost_array[4][j];
                blur_array[5][j] = blur_array[4][j];
                disp_baseline_array[5][j] = disp_baseline_array[4][j];
            }
        }

        // right_near
        if (p.x < width - 1)
        {
            flag[6] = true;
            num_valid_pixels++;
            costMin = costs[right_near];
            costMinPoint = right_near;
            for (int i = 0; i < nerDis; ++i)
            {
                if (p.x < width - 2 - i && p.y > i)
                {
                    int pointTemp = right_near + (1 + i) - i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
                if (p.x < width - 2 - i && p.y < height - 1- i)
                {
                    int pointTemp = right_near + (1 + i) + i * width;
                    if (costs[pointTemp] < costMin)
                    {
                        costMin = costs[pointTemp];
                        costMinPoint = pointTemp;
                    }
                }
            }
            right_near = costMinPoint;
            ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[right_near],
                cost_array[6], blur_array[6], disp_baseline_array[6], params);
        }

        // **边界：填充右远邻**
        if (p.x >= width - 3 && p.x < width - 1) {
            flag[7] = true; num_valid_pixels++;
            right_far = right_near;
            for (int j = 0; j < params.num_images - 1; ++j) {
                cost_array[7][j] = cost_array[6][j];
                blur_array[7][j] = blur_array[6][j];
                disp_baseline_array[7][j] = disp_baseline_array[6][j];
            }
        }

        const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

        //printf("%f\n ",cost_array[7][0] );

        // Multi-hypothesis Joint View Selection
        float view_weights[32] = {0.0f};
        float view_selection_priors[32] = {0.0f};
        int neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};
        for (int i = 0; i < 4; ++i)
        {
            if (flag[2*i])
            {
                for (int j = 0; j < params.num_images - 1; ++j)
                {
                    if (isSet(selected_views[neighbor_positions[i]], j) == 1)
                    {
                        //权重1：4[i]*18[j]=72
                        view_selection_priors[j] += 0.9f;
                    }
                    else
                    {
                        view_selection_priors[j] += 0.1f;
                    }
                }
            }
        }

        float sampling_probs[32] = {0.0f};
        float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f)); // 0.8 0.79116 0.765223
        for (int i = 0; i < params.num_images - 1; i++)
        {
            float count = 0;
            int count_false = 0;
            float tmpw = 0;
            for (int j = 0; j < 8; j++)
            {
                if (cost_array[j][i] < cost_threshold)
                {
                    tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                    count++;
                }
                if (cost_array[j][i] > 1.2f)
                {
                    count_false++;
                }
            }
            if (count > 2 && count_false < 3)
            {
                //权重2：8[j]*18[i]=144
                sampling_probs[i] = tmpw / count;
            }
            else if (count_false < 3)
            {
                sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
            }
            sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
        }

        TransformPDFToCDF(sampling_probs, params.num_images - 1);
        for (int sample = 0; sample < 15; ++sample)
        {
            const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

            for (int image_id = 0; image_id < params.num_images - 1; ++image_id)
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
        int num_selected_view = 0;
        float weight_norm = 0;
        for (int i = 0; i < params.num_images - 1; ++i)
        {
            //printf("view_weight: %d,%f\n", i, view_weights[i]);
            if (view_weights[i] > 0)
            {
                setBit(temp_selected_views, i);
                weight_norm += view_weights[i];
                num_selected_view++;
            }
        }
        //printf("weight_norm=%f\n", weight_norm);
        // **边界：如果无视图被选中，则分配最小正权重**
        if (weight_norm <= 0.0f) {
            for (int i = 0; i < params.num_images - 1; ++i) {
                view_weights[i] = 1.0f;
                setBit(temp_selected_views, i);
            }
            weight_norm = float(params.num_images - 1);
            num_selected_view = params.num_images - 1;
        }

        float final_costs[8] = {0.0f};
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < params.num_images - 1; ++j)
            {
                if (view_weights[j] > 0)
                {
                    if (params.geom_consistency)
                    {
                        if (flag[i])
                        {
                            //final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * ComputeGeomConsistencyCost(depths[j+1], cameras[0], cameras[j+1], plane_hypotheses[positions[i]], p));
                        }
                        else
                        {
                            final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * 5.0f);
                        }
                    }
                    else
                    {
                        //final_costs[i] += view_weights[j] * cost_array[i][j];

                        // todo：blurWeight
                        float blur_weight = exp(-0.007368*pow((blur_array[i][j].x - blur_array[i][j].y), 2));
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 2.0*(1-blur_weight));
                    }
                }
            }
            final_costs[i] /= weight_norm;
        }
        const int min_cost_idx = FindMinCostIndex(final_costs, 8);
        float4 disp_baseline_min = FindMinCostWithDispBaseline(cost_array[min_cost_idx],
                                    disp_baseline_array[min_cost_idx], params);
        float virtual_depth_before = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_array[min_cost_idx],
                                                                        params,
                                                                        baseline_norm,
                                                                        cost_array[min_cost_idx]);

        // 利用自己的平面假设计算代价
        float cost_vector_now[32] = {2.0f};
        float2 blur_vector_now[32] = {0.0f};
        float4 disp_baseline_vector_now[32] = {0.0};
        ComputeMultiViewCostVector_MIPM(images, blur_images, pcenters, p, plane_hypotheses[center],
                                        cost_vector_now, blur_vector_now,
                                        disp_baseline_vector_now, params);

        float cost_now = 0.0f;
        for (int i = 0; i < params.num_images-1; ++i)
        {
            if (params.geom_consistency)
            {
                //cost_now += view_weights[i] * (cost_vector_now[i] + 0.1f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p));
            }
            else
            {
                //cost_now += view_weights[i] * cost_vector_now[i] ;

                // todo：blurWeight
                float blur_weight = exp(-0.007368*pow((blur_vector_now[i].x - blur_vector_now[i].y), 2));
                cost_now += view_weights[i] * (cost_vector_now[i] + 2.0*(1-blur_weight));
            }
        }
        cost_now /= weight_norm; // 平均代价
        costs[center] = cost_now;
        float4 disp_baseine_min_now = FindMinCostWithDispBaseline(cost_vector_now,
                                    disp_baseline_vector_now, params);
        float virtual_depth_now = ComputeVirtualDepthConsiderCostAndBaseline(disp_baseline_vector_now,
                                    params, baseline_norm, cost_vector_now);

        float depth_now = plane_hypotheses[center].w;
        //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);

        float restricted_cost = 0.0f;
        if (params.planar_prior)
        {
            float restricted_final_costs[8] = {0.0f};
            float gamma = 0.5f;
            float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
            float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
            float angle_sigma = M_PI * (5.0f / 180.0f);
            float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
            //float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
            float depth_prior =  prior_planes[center].w;
            float beta = 0.18f;

            if (plane_masks[center] > 0)
            {
                for (int i = 0; i < 8; i++)
                {
                    if (flag[i])
                    {
                        //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[i]], p);
                        float depth_now = plane_hypotheses[center].w;
                        float depth_diff = depth_now - depth_prior;
                        float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[positions[i]]);
                        float angle_diff = acos(angle_cos);
                        float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                        restricted_final_costs[i] = exp(-final_costs[i] * final_costs[i] / beta) * prior;
                    }
                }
                const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

                float restricted_cost_now = 0.0f;
                //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
                float depth_now = plane_hypotheses[center].w;
                float depth_diff = depth_now - depth_prior;
                float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[center]);
                float angle_diff = acos(angle_cos);
                float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                restricted_cost_now = exp(-cost_now * cost_now / beta) * prior;

                if (flag[max_cost_idx])
                {
                    float disp_before = disp_baseline_array[min_cost_idx][0].x;
                    // 存储虚拟深度值： 长基线+代价小，加权平均
                    disp_baseline[center] = disp_baseline_min;
                    disp_baseline[center].w = virtual_depth_now;
                    if (disp_before >= params.depth_min && disp_before <= params.depth_max
                        && restricted_final_costs[max_cost_idx] > restricted_cost_now)
                    {
                        depth_now  = disp_before;
                        plane_hypotheses[center] = plane_hypotheses[positions[max_cost_idx]]; // lzd 赋值新的平面、视差
                        costs[center] = final_costs[max_cost_idx];
                        restricted_cost = restricted_final_costs[max_cost_idx];

                        // 存储虚拟深度值： 长基线+代价小，加权平均
                        disp_baseline[center] = disp_baseline_min;
                        disp_baseline[center].w = virtual_depth_before;

                        selected_views[center] = temp_selected_views;
                    }
                }
            }
            else if (flag[min_cost_idx])
            {
                //float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
                float disp_before = disp_baseline_array[min_cost_idx][0].x;
                // 存储虚拟深度值： 长基线+代价小，加权平均
                disp_baseline[center] = disp_baseline_min;
                disp_baseline[center].w = virtual_depth_now;
                if (disp_before >= params.depth_min && disp_before <= params.depth_max
                    && final_costs[min_cost_idx] < cost_now)
                {
                    depth_now = disp_before; // 标准视差 d

                    plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]]; // lzd 赋值新的平面、视差
                    costs[center] = final_costs[min_cost_idx]; // 平均代价

                    // 存储虚拟深度值： 长基线+代价小，加权平均
                    disp_baseline[center] = disp_baseline_min;
                    disp_baseline[center].w = virtual_depth_before;

                    selected_views[center] = temp_selected_views;
                }
            }
        }

        if (!params.planar_prior && flag[min_cost_idx])
        {
            float disp_before = disp_baseline_array[min_cost_idx][0].x;
            // 存储虚拟深度值： 长基线+代价小，加权平均
            disp_baseline[center] = disp_baseline_min;
            disp_baseline[center].w = virtual_depth_now;
            if (disp_before >= params.depth_min && disp_before <= params.depth_max
                && final_costs[min_cost_idx] < cost_now)
            {
                depth_now = disp_before; // 标准视差 d
                plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]]; // lzd 赋值新的平面、视差
                costs[center] = final_costs[min_cost_idx]; // 平均代价

                // 存储虚拟深度值： 长基线+代价小，加权平均
                disp_baseline[center] = disp_baseline_min;
                disp_baseline[center].w = virtual_depth_before;

                selected_views[center] = temp_selected_views;
            }
        }
        PlaneHypothesisRefinement_MIPM(images, blur_images, depths,
            &plane_hypotheses[center], pcenters,
            &depth_now, &costs[center], &disp_baseline[center],
            &rand_states[center],
            view_weights, weight_norm, prior_planes, plane_masks,
            &restricted_cost, p, params,WIDTH, baseline_norm);
    }

    __global__ void BlackPixelUpdate_MIPM(cudaTextureObjects* texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
                float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
                float4* prior_planes, unsigned int* plane_masks,
                const PatchMatchParamsLF params, const int iter,
                const int width, const int height)
    {
        const int base_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int base_y = blockIdx.y * blockDim.y + threadIdx.y;
        int2 p = make_int2(base_x, base_y);
        if (threadIdx.x % 2 == 0)
        {
            p.y = p.y * 2;
        }
        else
        {
            p.y = p.y * 2 + 1;
        }

        CheckerboardPropagation_MIPM_ConsiderBorderPixels(texture_objects, texture_depths[0].images,
            plane_hypotheses, pcenters, costs, disp_baseline, rand_states, selected_views, prior_planes,
            plane_masks, p, params, iter, width, height);
    }

    __global__ void RedPixelUpdate_MIPM(cudaTextureObjects *texture_objects,
                    cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                    float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                    unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                    const PatchMatchParamsLF params, const int iter,
                    const int width, const int height)
    {
        int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
        if (threadIdx.x % 2 == 0)
        {
            p.y = p.y * 2 + 1;
        }
        else
        {
            p.y = p.y * 2;
        }
        CheckerboardPropagation_MIPM_ConsiderBorderPixels(texture_objects, texture_depths[0].images,
            plane_hypotheses, pcenters, costs, disp_baseline, rand_states, selected_views, prior_planes, plane_masks,
            p, params, iter, width, height);
    }

    __global__ void RandomInitializationForMI(cudaTextureObjects* texture_objects,
        float4* plane_hypotheses, float2* pcenters, float* costs,
        curandState* rand_states, unsigned int* selected_views,
        float4* prior_planes, unsigned int* plane_masks, const PatchMatchParamsLF params,
        const int width, const int height)
    {
        const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
        if (p.x >= width || p.y >= height)
        {
            return;
        }
        const int center = p.y * width + p.x; // 将当前像素的二维数组表现形式转换为一维数组

        curand_init(clock64(), p.y, p.x, &rand_states[center]);

        if (params.geom_consistency)
        {
            //printf("%s\n", "000000");

            //float4 plane_hypothesis = plane_hypotheses[center];
            //plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
            //float depth = plane_hypothesis.w;
            //plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
            //plane_hypotheses[center] = plane_hypothesis;
            //costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images_MI, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
        else if (params.planar_prior)
        {
            if (plane_masks[center] > 0 && costs[center] >= 0.15f)
            {
                //printf("%s\n", "平面先验1");
                float perturbation = 0.02f;
                float4 plane_hypothesis = prior_planes[center];
                float depth_perturbed = plane_hypothesis.w;
                //printf("%f\n", depth_perturbed);
                const float depth_min_perturbed = (1 - 1 * perturbation) * depth_perturbed;
                const float depth_max_perturbed = (1 + 1 * perturbation) * depth_perturbed;

                // 生成随机深度值
                depth_perturbed = curand_uniform(&rand_states[center]) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;

                //生成新的随机平面
                float4 plane_hypothesis_perturbed = GeneratePerturbedNormalLF( p, plane_hypothesis, &rand_states[center], 1 * perturbation * M_PI);
                plane_hypothesis_perturbed.w = depth_perturbed;
                plane_hypotheses[center] = plane_hypothesis_perturbed;

                // 下面函数执行后，得到的结果包括：当前像素的匹配代价值（前top_k个的cost的平均值），当前像素的top_k个邻居图像的编号
                costs[center] = ComputeMultiViewInitialCostandSelectedViews_MIPM(texture_objects, pcenters, p, plane_hypotheses[center], &selected_views[center], params);
                //printf("%f\n", costs[center]);
            }
            else
            {
                //printf("%s\n", "平面先验2");
                //float4 plane_hypothesis = plane_hypotheses[center];
                //float depth = plane_hypothesis.w;
                //plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
                //plane_hypotheses[center] = plane_hypothesis;
                //printf("%f\n", plane_hypotheses[center].w);
                //
                costs[center] = ComputeMultiViewInitialCostandSelectedViews_MIPM(texture_objects,
                    pcenters, p, plane_hypotheses[center], &selected_views[center], params);
            }
        }
        else
        {
            if (plane_hypotheses[center].w == 0)
            {
                plane_hypotheses[center] = GenerateRandomPlaneHypothesis_MIPM(p, &rand_states[center],
                    params.depth_min, params.depth_max);
            }
            //plane_hypotheses[center] = GenerateRandomPlaneHypothesisLF( p, &rand_states[center], params.depth_min, params.depth_max);
            costs[center] = ComputeMultiViewInitialCostandSelectedViews_MIPM(texture_objects,
                pcenters, p, plane_hypotheses[center], &selected_views[center], params);
        }
    }

}
