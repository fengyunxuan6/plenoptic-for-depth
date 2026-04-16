/********************************************************************
file base:      AdaptMIPMPFPGRUtil.cu
author:         LZD
created:        2025/07/12
purpose:
*********************************************************************/
#include "AdaptMIPMPFPGRUtil.cuh"

#include "CudaUtil.h"
#include "AdaptMIPMUtil.cuh"
#include "AdaaptMIPM_EdgewareUtil.cuh"

namespace LFMVS
{
    __device__ int SelectBestProxy_Softmin_new(const int2 p, const PatchMatchParamsLF params,
                CollectNeighInfo* neig_info, float lambda_d, float lambda_c)
    {
        float dist_array[32];
        // step1: compute all distances and costs
        for (int i=1; i < params.num_images; i++)
        {
            CollectNeighInfo& info = neig_info[i-1];
            float2 neig_center = info.center_coord;
            float2 neig_correspond = info.correspond_coord;
            float dx = neig_correspond.x-neig_center.x;
            float dy = neig_correspond.y-neig_center.y;
            dist_array[i-1] = sqrtf(dx*dx+dy*dy);
        }

        // step2: compute ranks
        int rank_dist[32];
        int rank_cost[32];
        for (int i=1; i<params.num_images; i++)
        {
            rank_dist[i-1] = 0;
            rank_cost[i-1] = 0;
            for (int j=1; j<params.num_images; j++)
            {
                if (dist_array[j-1] < dist_array[i-1])
                    rank_dist[i-1]++;
                if (neig_info[j-1].cost < neig_info[i-1].cost)
                    rank_cost[i-1]++;
            }
        }

        // step3: softmin scoring
        float best_score = -1.0f;
        int best_idx = 0;
        for (int i=1; i<params.num_images;i++)
        {
            float r_d = float(rank_dist[i-1])/float(params.num_images);
            float r_c = float(rank_cost[i-1])/float(params.num_images);
            float s = SoftminScore(r_d, r_c, lambda_d, lambda_c);
            if (s > best_score)
            {
                best_score = s;
                best_idx = i;
            }
        }
        return best_idx;
    }

    __device__ bool IsBroken(const int2 p, const PatchMatchParamsLF params)
    {
        int propagation_graph_radius = params.propagation_Graph_size; // 5
        const int width = params.MLA_Mask_Width_Cuda;
        const int height = params.MLA_Mask_Height_Cuda;

        if (p.x<propagation_graph_radius || p.x>=width-propagation_graph_radius ||
            p.y<propagation_graph_radius || p.y>=height-propagation_graph_radius)
        {
            return true;
        }
        return false;
    }

    __device__ void CollectNeighborInfo(const int2 p, float2* pcenters, float* cost_vector_input,
           float4* disp_baseline_vector, const PatchMatchParamsLF params,
           const bool bBound, const bool bBroken,
           CollectNeighInfo* boundary_info_vector, CollectNeighInfo* broken_info_vector)
    {
        for (int i=1; i < params.num_images; i++)
        {
            if (bBound)
            {
                boundary_info_vector[i-1].center_coord = pcenters[i];
                boundary_info_vector[i-1].cost = cost_vector_input[i-1]; // 代价直接赋值

                // 视差平面几何映射：计算邻域图像中对应点
                float disp_norm = disp_baseline_vector[i-1].x; // 标准视差
                float2 ref_center = pcenters[0];
                float2 neig_center = pcenters[i];
                float baseline = sqrt(pow((neig_center.y - ref_center.y), 2) + pow((neig_center.x -ref_center.x), 2));
                float d_x = (ref_center.x - neig_center.x) / baseline;//这句代码啥意思，作了个距离比值
                float d_y = (ref_center.y - neig_center.y) / baseline;
                float scale = baseline/params.Base;
                boundary_info_vector[i-1].correspond_coord.x = p.x + d_x*disp_norm*scale;
                boundary_info_vector[i-1].correspond_coord.y = p.y + d_y*disp_norm*scale;
            }
            if (bBroken)
            {
                broken_info_vector[i-1].center_coord = pcenters[i];
                broken_info_vector[i-1].cost = cost_vector_input[i-1]; // 代价直接赋值

                // 视差平面几何映射：计算邻域图像中对应点
                float disp_norm = disp_baseline_vector[i-1].x; // 标准视差
                float2 ref_center = pcenters[0];
                float2 neig_center = pcenters[i];
                float baseline = sqrt(pow((neig_center.y - ref_center.y), 2) + pow((neig_center.x -ref_center.x), 2));
                float d_x = (ref_center.x - neig_center.x) / baseline;//这句代码啥意思，作了个距离比值
                float d_y = (ref_center.y - neig_center.y) / baseline;
                float scale = baseline/params.Base;
                broken_info_vector[i-1].correspond_coord.x = p.x + d_x*disp_norm*scale;
                broken_info_vector[i-1].correspond_coord.y = p.y + d_y*disp_norm*scale;
            }
        }
    }

    __device__ void PlaneHypothesisRefinement_MIPM_Repairpixels(const cudaTextureObject_t* images,
            const cudaTextureObject_t* blur_images,
            const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
            float2* pcenters, float* depth, float* cost, float4* disp_baseline,
            CollectNeighInfo* boundary_info_vector, CollectNeighInfo* broken_info_vector,
            curandState* rand_state,
            const float* view_weights, const float weight_norm, float4* prior_planes,
            unsigned int* plane_masks, float* restricted_cost, const int2 p,
            const PatchMatchParamsLF params, const int WIDTH, const float* baseline_norm,
            const bool bBound, const bool bBroken)
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
                        //temp_cost += view_weights[j] * (cost[j] + 0.1f * ComputeGeomConsistencyCost(depth_images[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                    }
                    else
                    {
                        //temp_cost += view_weights[j] * cost[j];

                        // todo: blurWeight  融合模糊差异的代价累积
                        float blur_weight = exp(-0.007368*pow((blur_vector[i].x - blur_vector[i].y), 2));
                        temp_cost += view_weights[j] * (cost_vector[j] + 2.0*(1.0-blur_weight));
        //                 if (blur_vector[i].x > 0.5 && blur_vector[i].y>0.5)
        //                 {
        //                     printf("cost: %f, blur_weight: %f,%f,  %f\n", cost[i], blur_vector[i].x,
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

                        CollectNeighborInfo(p, pcenters, cost_vector, disp_baseline_vector, params,
                            bBound, bBroken, boundary_info_vector, broken_info_vector);
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

                        CollectNeighborInfo(p, pcenters, cost_vector, disp_baseline_vector, params,
                            bBound, bBroken, boundary_info_vector, broken_info_vector);
                    }
                }
            }
        }
    }

    __device__ void CheckerboardPropagation_MIPM_Repairpixels(const cudaTextureObjects* texture_objects,
        const cudaTextureObject_t* depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, float4* disp_baseline,
        CollectNeighInfo* boundary_info_vector, CollectNeighInfo* broken_info_vector,
        curandState* rand_states,
        unsigned int* selected_views, float4* prior_planes,
        unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
        const int iter, const int WIDTH, const int HEIGHT, const bool bBound, const bool bBroken)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;

        int width = WIDTH;
        int height = HEIGHT;
        // p为参考图像中的当前像素
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
        if (p.y >= height - 3 && p.y < height - 1)
        {
            flag[3] = true; num_valid_pixels++;
            down_far = down_near;
            for (int j = 0; j < params.num_images - 1; ++j)
            {
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
        if (p.x <= 2 && p.x > 0)
        {
            flag[5] = true; num_valid_pixels++;
            left_far = left_near;
            for (int j = 0; j < params.num_images - 1; ++j)
            {
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
        if (p.x >= width - 3 && p.x < width - 1)
        {
            flag[7] = true; num_valid_pixels++;
            right_far = right_near;
            for (int j = 0; j < params.num_images - 1; ++j)
            {
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
        if (weight_norm <= 0.0f)
        {
            for (int i = 0; i < params.num_images - 1; ++i)
            {
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
        CollectNeighborInfo(p, pcenters, cost_vector_now, disp_baseline_vector_now, params,
                            bBound, bBroken, boundary_info_vector, broken_info_vector);
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

                CollectNeighborInfo(p, pcenters, cost_array[min_cost_idx],
                                     disp_baseline_array[min_cost_idx], params,
                                     bBound, bBroken, boundary_info_vector, broken_info_vector);

                selected_views[center] = temp_selected_views;
            }
        }
        PlaneHypothesisRefinement_MIPM_Repairpixels(images, blur_images, depths,
            &plane_hypotheses[center], pcenters,
            &depth_now, &costs[center], &disp_baseline[center],
            boundary_info_vector, broken_info_vector,
            &rand_states[center],
            view_weights, weight_norm, prior_planes, plane_masks,
            &restricted_cost, p, params,WIDTH, baseline_norm,bBound, bBroken);
    }

    __global__ void BlackPixelUpdate_MIPM_PFPGR_Collect(cudaTextureObjects* texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
                float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
                float4* prior_planes, unsigned int* plane_masks,
                int3* neighbor_patchFill_cuda,
                int3* neighbor_PGR_cuda,
                const PatchMatchParamsLF params, const int iter,
                const int width, const int height, int2* tilekeys_cuda)
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
        if (p.x >= width || p.y >= height)
        {
            return;
        }

        if (iter < params.max_iterations-1)
        {
            CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware(texture_objects,
                texture_depths[0].images, plane_hypotheses, pcenters, costs,
                disp_baseline, rand_states, selected_views, prior_planes,
                plane_masks, p, params, iter, width, height, tilekeys_cuda);
        }
        else if (iter == params.max_iterations-1)
        {
            int center = p.y * width + p.x;
            neighbor_patchFill_cuda[center] = make_int3(-1,-1,-1); // 初始化
            neighbor_PGR_cuda[center] = make_int3(-1,-1,-1); // 初始化

            // 为边界附近的像素寻找patch的候补像素（来源于邻域微图像）
            // 判断是否为边界
            bool bBound = IsBoundary(p, params);
            bool bBroken = IsBroken(p, params);
            if (bBound || bBroken)
            {
                // 下面的32表示最大的邻域微图像数量
                CollectNeighInfo boundary_info_vector[32];
                CollectNeighInfo broken_info_vector[32];
                CheckerboardPropagation_MIPM_Repairpixels(texture_objects,
                    texture_depths[0].images, plane_hypotheses, pcenters, costs, disp_baseline,
                    boundary_info_vector, broken_info_vector,
                    rand_states, selected_views, prior_planes,
                    plane_masks, p, params, iter, width, height, bBound, bBroken);

                float lambda_d = 0.5;
                float lambda_c = 0.5;
                int center = p.y * width + p.x;
                if (bBound)
                {
                    int selected_neig_idx = SelectBestProxy_Softmin_new(p, params, boundary_info_vector,
                                                                        lambda_d, lambda_c);
                    int3 selected_proxy = make_int3(selected_neig_idx,
                      boundary_info_vector[selected_neig_idx-1].correspond_coord.x,
                      boundary_info_vector[selected_neig_idx-1].correspond_coord.y); // (neig_mi_idx, neig_coorespond_p)
                    neighbor_patchFill_cuda[center] = selected_proxy; // 压入
    //                 printf("index=%d, pF:(%i, %i,%i)\n", center,
    // selected_proxy.x, selected_proxy.y, selected_proxy.z);
                }
                if (bBroken)
                {
                    int selected_neig_idx = SelectBestProxy_Softmin_new(p, params, broken_info_vector,
                                                                    lambda_d, lambda_c);
                    int3 selected_proxy = make_int3(selected_neig_idx,
                      (int)broken_info_vector[selected_neig_idx-1].correspond_coord.x,
                      (int)broken_info_vector[selected_neig_idx-1].correspond_coord.y); // (neig_mi_idx, neig_coorespond_p)
                    neighbor_PGR_cuda[center] = selected_proxy; // 压入
//                     printf("index=%d, pgr:(%i, %i,%i)\n", center,
// selected_proxy.x, selected_proxy.y, selected_proxy.z);
                }
            }
            else
            {
                CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware(texture_objects,
                        texture_depths[0].images, plane_hypotheses, pcenters, costs,
                        disp_baseline, rand_states, selected_views, prior_planes,
                        plane_masks, p, params, iter, width, height, tilekeys_cuda);
            }
        }
    }

    __global__ void RedPixelUpdate_MIPM_PFPGR_Collect(cudaTextureObjects *texture_objects,
                    cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                    float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                    unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                    int3* neighbor_patchFill_cuda,
                    int3* neighbor_PGR_cuda,
                    const PatchMatchParamsLF params, const int iter,
                    const int width, const int height, int2* tilekeys_cuda)
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
        if (p.x >= width || p.y >= height)
        {
            return;
        }

         if (iter < params.max_iterations-1)
         {
             CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware(texture_objects,
                 texture_depths[0].images, plane_hypotheses, pcenters, costs,
                 disp_baseline, rand_states, selected_views, prior_planes,
                 plane_masks, p, params, iter, width, height, tilekeys_cuda);
         }
         else if (iter == params.max_iterations-1)
         {
             int center = p.y * width + p.x;
             neighbor_patchFill_cuda[center] = make_int3(-1,-1,-1); // 初始化
             neighbor_PGR_cuda[center] = make_int3(-1,-1,-1); // 初始化

            // 为边界附近的像素寻找patch的候补像素（来源于邻域微图像）
            // 判断是否为边界
            bool bBound = IsBoundary(p, params);
            bool bBroken = IsBroken(p, params);
            if (bBound || bBroken)
            {
                // 下面的32表示最大的邻域微图像数量
                CollectNeighInfo boundary_info_vector[32];
                CollectNeighInfo broken_info_vector[32];
                CheckerboardPropagation_MIPM_Repairpixels(texture_objects,
                    texture_depths[0].images, plane_hypotheses, pcenters, costs, disp_baseline,
                    boundary_info_vector, broken_info_vector,
                    rand_states, selected_views, prior_planes,
                    plane_masks, p, params, iter, width, height, bBound, bBroken);

                float lambda_d = 0.5;
                float lambda_c = 0.5;
                if (bBound)
                {
                    int selected_neig_idx = SelectBestProxy_Softmin_new(p, params, boundary_info_vector,
                                                                        lambda_d, lambda_c);
                    int3 selected_proxy = make_int3(selected_neig_idx,
                      boundary_info_vector[selected_neig_idx-1].correspond_coord.x,
                      boundary_info_vector[selected_neig_idx-1].correspond_coord.y); // (neig_mi_idx, neig_coorespond_p)
                    neighbor_patchFill_cuda[center] = selected_proxy; // 压入
//                     printf("index=(%d,%d), pF:(%d, %d, %d)\n", p.x, p.y,
// selected_proxy.x, selected_proxy.y, selected_proxy.z);
                }
                if (bBroken)
                {
                    int selected_neig_idx = SelectBestProxy_Softmin_new(p, params, broken_info_vector,
                                                                    lambda_d, lambda_c);
                    int3 selected_proxy = make_int3(selected_neig_idx,
                      broken_info_vector[selected_neig_idx-1].correspond_coord.x,
                      broken_info_vector[selected_neig_idx-1].correspond_coord.y); // (neig_mi_idx, neig_coorespond_p)
                    neighbor_PGR_cuda[center] = selected_proxy; // 压入
//                     printf("index=(%d,%d), pgr:(%d, %d,%d)\n", p.x, p.y,
// selected_proxy.x, selected_proxy.y, selected_proxy.z);
                }
            }
            else
            {
                CheckerboardPropagation_MIPM_ConsiderBorderPixels_Edgeaware(texture_objects,
                        texture_depths[0].images, plane_hypotheses, pcenters, costs,
                        disp_baseline, rand_states, selected_views, prior_planes,
                        plane_masks, p, params, iter, width, height, tilekeys_cuda);
            }
         }
    }

    __device__ void ComputeMultiViewCostVector_PFPGR_Repair(const cudaTextureObject_t* images,
                        const cudaTextureObject_t* blur_images,
                        float2* pcenters, const int2 p,
                        int3* neighbor_patchFill_cuda,
                        float4 plane_hypothesis, float* cost_vector, float2* blure_array,
                        float4* disp_baseline_array, const PatchMatchParamsLF params)
    {
        bool bSuccessFill = false;

        float patch_pixels[MAX_PATCH_SIZE*MAX_PATCH_SIZE] = {0.0};

        bool bBound = IsBoundary(p, params);
        if (bBound)
        {
            int center = p.y * params.MLA_Mask_Width_Cuda + p.x;
            if (neighbor_patchFill_cuda[center].x >= 0) // 有候补
            {
                //printf("neighbor_patchFill_cuda[center].x >= 0, %d\n", neighbor_patchFill_cuda[center].x);
                bSuccessFill = true;
                // 邻域微图像近似像素候补自身patch

                // neighbor image fetch
                int radius = params.patch_size / 2; // 5
                int3 proxy_src = neighbor_patchFill_cuda[center];
                cudaTextureObject_t ref_image = images[0];
                cudaTextureObject_t proxy_src_image = images[proxy_src.x];
                const int2 proxy_src_p = make_int2(proxy_src.y, proxy_src.z);
                for (int i = -radius; i < radius + 1; i += params.radius_increment) // x
                {
                    for (int j = -radius; j < radius + 1; j += params.radius_increment) // y
                    {
                        int2 pt = make_int2(p.x+i, p.y+j);
                        if (pt.x >= params.MLA_Mask_Width_Cuda || pt.x < 0.0f
                            || pt.y >= params.MLA_Mask_Height_Cuda || pt.y < 0.0f)
                        {
                            int dx = pt.x-p.x;
                            int dy = pt.y-p.y;
                            int2 proxy_src_pt = make_int2(proxy_src_p.x+dx, proxy_src_p.y+dy);
                            float src_center_pix = tex2D<float>(proxy_src_image,
                                proxy_src_pt.x + 0.5f, proxy_src_pt.y + 0.5f);
                            patch_pixels[j*MAX_PATCH_SIZE + i] = src_center_pix;
                        }
                        else
                        {
                            float src_center_pix = tex2D<float>(ref_image,pt.x + 0.5f, pt.y + 0.5f);
                            patch_pixels[j*MAX_PATCH_SIZE + i] = src_center_pix;
                        }
                    }
                }
            }
        }

        // imp
        if (bSuccessFill)
        {
            for (int i = 1; i < params.num_images; ++i)
            {
                cost_vector[i-1] = ComputeBilateralNCC_MIPM_FillPatch(images[0], blur_images[0],
                    pcenters[0], images[i], pcenters[i],
                    blur_images[i], p,
                    patch_pixels,
                    plane_hypothesis,
                    params, blure_array[i-1],
                    disp_baseline_array[i-1]);
            }
        }
        else
        {
            for (int i = 1; i < params.num_images; ++i)
            {
                cost_vector[i-1] = ComputeBilateralNCC_MIPM(images[0], blur_images[0],
                    pcenters[0], images[i], pcenters[i],
                    blur_images[i], p,
                    plane_hypothesis,
                    params, blure_array[i-1],
                    disp_baseline_array[i-1]);
            }
        }
    }

    __device__ void PlaneHypothesisRefinement_PFPGR_Repair(const cudaTextureObject_t* images,
            const cudaTextureObject_t* blur_images,
            const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
            float2* pcenters, float* depth, float* cost, float4* disp_baseline, curandState* rand_state,
            const float* view_weights, const float weight_norm, float4* prior_planes,
            unsigned int* plane_masks, float* restricted_cost, const int2 p,
            int3* neighbor_patchFill_cuda,
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

            ComputeMultiViewCostVector_MIPM_FillPatch(images, blur_images, pcenters, p,
                                            neighbor_patchFill_cuda, temp_plane_hypothesis,
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

    __device__ void CheckerboardPropagation_PFPGR_Repair(const cudaTextureObjects* texture_objects,
            const cudaTextureObject_t* depths, float4* plane_hypotheses,
            float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
            unsigned int* selected_views, float4* prior_planes,
            unsigned int* plane_masks,
            int3* neighbor_patchFill_cuda, Proxy_DisPlane* proxy_plane_cuda,
            const int2 p, const PatchMatchParamsLF params,
            const int iter, const int WIDTH, const int HEIGHT)
    {
        const cudaTextureObject_t* images = texture_objects[0].images;
        const cudaTextureObject_t* blur_images = texture_objects[0].blur_images;

        int width = WIDTH;
        int height = HEIGHT;
        // p为参考图像中的当前像素
        // 边界检查
        if (p.x >= width || p.y >= height)
        {
            return;
        }

        int farDis = 2; // 11 2
        int nerDis = 2; // 3 2

        const int center = p.y * width + p.x;
        const Proxy_DisPlane& proxy_plane_hypothes = proxy_plane_cuda[center];

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

            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[up_far],
                cost_array[1], blur_array[1], disp_baseline_array[1], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[1],
                        cost_array[1], blur_array[1],
                        disp_baseline_array[1], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[down_far],
                cost_array[3], blur_array[3], disp_baseline_array[3], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[3],
                        cost_array[3], blur_array[3],
                        disp_baseline_array[3], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[left_far],
                cost_array[5], blur_array[5], disp_baseline_array[5], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[5],
                        cost_array[5], blur_array[5],
                        disp_baseline_array[5], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[right_far],
                cost_array[7], blur_array[7], disp_baseline_array[7], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[7],
                        cost_array[7], blur_array[7],
                        disp_baseline_array[7], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[up_near],
                cost_array[0], blur_array[0], disp_baseline_array[0], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[0],
                        cost_array[0], blur_array[0],
                        disp_baseline_array[0], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[down_near],
                cost_array[2], blur_array[2], disp_baseline_array[2], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[2],
                        cost_array[2], blur_array[2],
                        disp_baseline_array[2], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                neighbor_patchFill_cuda, plane_hypotheses[left_near],
                cost_array[4], blur_array[4], disp_baseline_array[4], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[4],
                        cost_array[4], blur_array[4],
                        disp_baseline_array[4], params);
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
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
            neighbor_patchFill_cuda, plane_hypotheses[right_near],
                cost_array[6], blur_array[6], disp_baseline_array[6], params);
        }
        else
        {
            ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                        neighbor_patchFill_cuda,
                        proxy_plane_hypothes.plane[6],
                        cost_array[6], blur_array[6],
                        disp_baseline_array[6], params);
        }
        const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

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
        if (weight_norm <= 0.0f)
        {
            for (int i = 0; i < params.num_images - 1; ++i)
            {
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
        ComputeMultiViewCostVector_PFPGR_Repair(images, blur_images, pcenters, p,
                                        neighbor_patchFill_cuda,
                                        plane_hypotheses[center],
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

        // 此处，不需要进行传播路径的补充，因为在refine中没有传播，只有扰动
        PlaneHypothesisRefinement_PFPGR_Repair(images, blur_images, depths,
            &plane_hypotheses[center], pcenters,
            &depth_now, &costs[center], &disp_baseline[center],
            &rand_states[center],
            view_weights, weight_norm, prior_planes, plane_masks,
            &restricted_cost, p,
            neighbor_patchFill_cuda,
            params,WIDTH, baseline_norm);
    }

    __global__ void BlackPixelUpdate_PFPGR_Repair(cudaTextureObjects* texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
                float* costs, float4* disp_baseline, curandState* rand_states, unsigned int* selected_views,
                float4* prior_planes, unsigned int* plane_masks,
                int3* neighbor_patchFill_cuda, Proxy_DisPlane* proxy_plane_cuda,
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
        CheckerboardPropagation_PFPGR_Repair(texture_objects,
                    texture_depths[0].images, plane_hypotheses,
                    pcenters, costs, disp_baseline, rand_states,
                    selected_views, prior_planes, plane_masks,
                    neighbor_patchFill_cuda, proxy_plane_cuda,
                    p, params, iter, width, height);
    }

    __global__ void RedPixelUpdate_PFPGR_Repair(cudaTextureObjects *texture_objects,
                    cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                    float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                    unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                    int3* neighbor_patchFill_cuda, Proxy_DisPlane* proxy_plane_cuda,
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

        CheckerboardPropagation_PFPGR_Repair(texture_objects,
            texture_depths[0].images, plane_hypotheses,
            pcenters, costs, disp_baseline, rand_states,
            selected_views, prior_planes, plane_masks,
            neighbor_patchFill_cuda, proxy_plane_cuda,
            p, params, iter, width, height);
    }

}
