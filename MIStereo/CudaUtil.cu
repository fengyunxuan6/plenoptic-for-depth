/********************************************************************
file base:      CudaUtil.cu
author:         LZD
created:        2025/06/13
purpose:
*********************************************************************/
#include "CudaUtil.h"

namespace LFMVS
{
    /////////////////////////////////////////////////////////////////////////////////////////
    // 共用函数
    __device__  void setBit(unsigned int &input, const unsigned int n)
    {
        input |= (unsigned int)(1 << n);
    }

    __device__  int isSet(unsigned int input, const unsigned int n)
    {
        return (input >> n) & 1;
    }

    __device__ void sort_small(float* d, const int n)
    {
        int j;
        for (int i = 1; i < n; i++)
        {
            float tmp = d[i];
            for (j = i; j >= 1 && tmp < d[j-1]; j--)
            {
                d[j] = d[j-1];
            }
            d[j] = tmp;
        }
    }

    __device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
    {
        result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
        result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
        result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
    }

    __device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
    {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    }

    __device__ void NormalizeVec3 (float4 *vec)
    {
        const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
        const float inverse_sqrt = rsqrtf (normSquared);
        vec->x *= inverse_sqrt;
        vec->y *= inverse_sqrt;
        vec->z *= inverse_sqrt;
    }

    __device__ float4 GetViewDirectionLF( const int2 p, const float depth)
    {
        float X[3];
        X[0] = p.x;
        X[1] = p.y;
        X[2] = depth;
        float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

        float4 view_direction;
        view_direction.x = X[0] / norm;
        view_direction.y = X[1] / norm;
        view_direction.z =  X[2] / norm;
        view_direction.w = 0;
        return view_direction;
    }

    __device__ int FindMinCostIndex(const float* costs, const int n)
    {
        float min_cost = costs[0];
        int min_cost_idx = 0;
        for (int idx = 1; idx < n; ++idx)
        {
            if (costs[idx] <= min_cost)
            {
                min_cost = costs[idx];
                min_cost_idx = idx;
            }
        }
        return min_cost_idx;
    }

    __device__ int FindMaxCostIndex(const float *costs, const int n)
    {
        float max_cost = costs[0];
        int max_cost_idx = 0;
        for (int idx = 1; idx < n; ++idx)
        {
            if (costs[idx] >= max_cost)
            {
                max_cost = costs[idx];
                max_cost_idx = idx;
            }
        }
        return max_cost_idx;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    __device__ float4 GeneratePerturbedNormalLF(const int2 p, const float4 normal,
    curandState *rand_state, const float perturbation)
    {
        float4 view_direction = GetViewDirectionLF(p, 1.0f);

        const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
        const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
        const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

        const float sin_a1 = sin(a1);
        const float sin_a2 = sin(a2);
        const float sin_a3 = sin(a3);
        const float cos_a1 = cos(a1);
        const float cos_a2 = cos(a2);
        const float cos_a3 = cos(a3);

        float R[9];
        R[0] = cos_a2 * cos_a3;
        R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
        R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
        R[3] = cos_a2 * sin_a3;
        R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
        R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
        R[6] = -sin_a2;
        R[7] = cos_a2 * sin_a1;
        R[8] = cos_a1 * cos_a2;

        float4 normal_perturbed;
        Mat33DotVec3(R, normal, &normal_perturbed);

        if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f)
        {
            normal_perturbed = normal;
        }

        NormalizeVec3(&normal_perturbed);
        return normal_perturbed;
    }


    __device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
    {
        const float spatial_dist = sqrt(x_dist*x_dist + y_dist*y_dist);
        const float color_dist = fabs(pix - center_pix);
        return exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
    }

    __device__ float to_disparity(const int2 p, float3 d_plane)
    {
        // 视差d=A*x+B*y+C
        return d_plane.x*p.x +d_plane.y*p.y + d_plane.z*1.0;
    }

    __device__ float3 DisparityPlane(const int2 p, const float4 plane_hypothesis)
    {
        // 目的：将3D平面参数转换为像素坐标系下的视差平面方程
        // 这个平面在3D空间中表示为：A*X + B*Y + C*Z = D
        // 其中，（A，B，C，D）是平面参数，即plane_hypothesis; (X,Y,Z)是3D点坐标
        // 返回的是视差平面方程的参数。disparity = plane.x*x + plane.y*y + plane.z,(x,y)为像素坐标
        float3 plane;
        plane.x = -plane_hypothesis.x / plane_hypothesis.z; // A
        plane.y = -plane_hypothesis.y / plane_hypothesis.z; // B
        plane.z = (plane_hypothesis.x*p.x +plane_hypothesis.y*p.y
                    +plane_hypothesis.z *plane_hypothesis.w)/plane_hypothesis.z; // C
        return plane;
    }

    __device__ float ComputeBilateralNCCLF(const cudaTextureObject_t ref_image, float2 p0 ,
    const cudaTextureObject_t src_image, float2 p1 , const int2 p,
    const float4 plane_hypothesis, const PatchMatchParamsLF params)
    {
        // p为参考图像中的当前像素
        
        const float cost_max = 2.0f;
        int radius = params.patch_size / 2; // 5
        float3 d_plane = DisparityPlane(p, plane_hypothesis); // 视差平面
        //printf("%f\n", d_plane.y);

        // 参考图像的中心点到邻域图像中心点的距离
        float Base_line = sqrt(pow((p1.y - p0.y), 2) + pow((p1.x -p0.x), 2));

        float d_x = (p0.x - p1.x) / Base_line;//这句代码啥意思，作了个距离比值
        float d_y = (p0.y - p1.y) / Base_line;
        float B = Base_line / params.Base;

        float2 pt;
        //float d = plane_hypothesis.w;
        float d = to_disparity(p, d_plane); // 视差，相邻微图像之间的视差
        //printf("%f\n", d);

        //pt.x = p.x + d_x*d;
        pt.x = p.x + d_x*d*B;
        pt.y = p.y + d_y*d*B;
        //printf("%f\n", pt.y);
        if (pt.x >= params.MLA_Mask_Width_Cuda || pt.x < 0.0f || pt.y >= params.MLA_Mask_Height_Cuda || pt.y < 0.0f)
        {
            return cost_max;
        }

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
                    const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);

                    float2 src_pt; // 邻居图像中的像素坐标：通过视差平面和基线向量计算得来
                    float d1 = to_disparity(ref_pt, d_plane);
                    //float d1 = plane_hypothesis.w;
                    //printf("%f\n", d1);
                    if (d1<params.depth_min || d1>params.depth_max)
                    {
                        //printf("%s\n", "视差不符合阈值");
                        return cost_max;
                    }
                    src_pt.x = ref_pt.x + d_x*d1*B;
                    src_pt.y = ref_pt.y + d_y*d1*B;
                    const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                    float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

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
                return cost = cost_max;
            }
            else
            {
                const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
                const float var_ref_src = sqrt(var_ref * var_src);
                return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
            }
        }
    }

    // 计算像素p的代价值（局部块窗口：与邻域图像对应的像素）
    __device__ float ComputeMultiViewInitialCostandSelectedViewsLF(const cudaTextureObject_t* images,
        float2* pcenters, const int2 p, const float4 plane_hypothesis,
        unsigned int* selected_views, const PatchMatchParamsLF params)
    {
        float cost_max = 2.0f;
        float cost_vector[32] = {2.0f};
        float cost_vector_copy[32] = {2.0f};
        int cost_count = 0;
        int num_valid_views = 0;

        // 遍历所有的邻居
        for (int i = 1; i < params.num_images; ++i)
        {
            //printf("%d\n", params.num_images);
            float cost_value = ComputeBilateralNCCLF(images[0], pcenters[0], images[i],
                pcenters[i], p, plane_hypothesis, params);
            cost_vector[i - 1] = cost_value;
            cost_vector_copy[i - 1] = cost_value;
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

    __device__ float4 GenerateRandomNormalLF( const int2 p, curandState *rand_state, const float depth)
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

        NormalizeVec3(&normal);
        return normal;
    }

    __device__ float4 GenerateRandomPlaneHypothesisLF(const int2 p, curandState *rand_state, const float depth_min, const float depth_max)
    {
        //depth 视差
        float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
        float4 plane_hypothesis = GenerateRandomNormalLF(p, rand_state, depth);
        plane_hypothesis.w = depth;
        return plane_hypothesis;
    }

    __device__ void ComputeMultiViewCostVectorLF(const cudaTextureObject_t* images, float2* pcenters,
                            const int2 p, const float4 plane_hypothesis,
                            float* cost_vector, const PatchMatchParamsLF params)
    {
        for (int i = 1; i < params.num_images; ++i)
        {
            cost_vector[i-1] = ComputeBilateralNCCLF(images[0], pcenters[0], images[i],
                pcenters[i], p, plane_hypothesis, params);
        }
    }

    __device__ void PlaneHypothesisRefinementLF(const cudaTextureObject_t* images,
        const cudaTextureObject_t* depth_images,  float4* plane_hypothesis,
        float2 * pcenters, float *depth, float *cost, curandState* rand_state,
        const float* view_weights, const float weight_norm, float4* prior_planes,
        unsigned int* plane_masks, float* restricted_cost, const int2 p,
        const PatchMatchParamsLF params, const int WIDTH)
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
            depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
        } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);

        float4 plane_hypothesis_perturbed = GeneratePerturbedNormalLF( p, *plane_hypothesis, rand_state, perturbation * M_PI); // GeneratePertubedPlaneHypothesis(cameras[0], p, rand_state, perturbation, *plane_hypothesis, *depth, params.depth_min, params.depth_max);

        // lzd 构建了多个新的假设视差平面，继续算，得到最小代价，作为本次迭代的最终视差平面+代价
        const int num_planes = 5;
        float depths[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
        float4 normals[num_planes] = {*plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis};

        for (int i = 0; i < num_planes; ++i)
        {
            float cost_vector[32] = {2.0f};
            float4 temp_plane_hypothesis = normals[i];
            temp_plane_hypothesis.w =  temp_plane_hypothesis.w; // dists[i];
            ComputeMultiViewCostVectorLF(images, pcenters, p, temp_plane_hypothesis, cost_vector, params);

            float temp_cost = 0.0f;
            for (int j = 0; j < params.num_images - 1; ++j)
            {
                if (view_weights[j] > 0)
                {
                    if (params.geom_consistency)
                    {
                        //temp_cost += view_weights[j] * (cost_vector[j] + 0.1f * ComputeGeomConsistencyCost(depth_images[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                    }
                    else
                    {
                        temp_cost += view_weights[j] * cost_vector[j];
                    }
                }
            }
            temp_cost /= weight_norm;

            // TODO：可以与ACMP类似，存储虚拟深度
            float depth_before =  temp_plane_hypothesis.w; // lzd 代码有问题

            if (params.planar_prior && plane_masks[center] > 0)
            {
                float depth_diff = depths[i] - depth_prior;
                float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
                float angle_diff = acos(angle_cos);
                float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                float restricted_temp_cost = exp(-temp_cost * temp_cost / beta) * prior;
                if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_temp_cost > *restricted_cost) {
                    *depth = depth_before;
                    *plane_hypothesis = temp_plane_hypothesis;
                    *cost = temp_cost;
                    *restricted_cost = restricted_temp_cost;
                }
            }
            else
            {
                if (depth_before >= params.depth_min &&
                    depth_before <= params.depth_max
                    && temp_cost < *cost)
                {
                    *depth = depth_before;
                    *plane_hypothesis = temp_plane_hypothesis;
                    *cost = temp_cost;
                }
            }
        }
    }

    __device__ void TransformPDFToCDF(float* probs, const int num_probs)
    {
        float prob_sum = 0.0f;
        for (int i = 0; i < num_probs; ++i)
        {
            prob_sum += probs[i];
        }
        const float inv_prob_sum = 1.0f / prob_sum;

        float cum_prob = 0.0f;
        for (int i = 0; i < num_probs; ++i)
        {
            const float prob = probs[i] * inv_prob_sum;
            cum_prob += prob;
            probs[i] = cum_prob;
        }
    }

    __device__ void CheckerboardPropagationLF_Near(const cudaTextureObject_t* images,
                    const cudaTextureObject_t* depths, float4* plane_hypotheses,
                    float2* pcenters,float* costs, curandState* rand_states,
                    unsigned int* selected_views, float4* prior_planes,
                    unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
                    const int iter, const int WIDTH, const int HEIGHT)
    {
        int width = WIDTH;
        int height = HEIGHT;

        // 边界检查
        if (p.x >= width || p.y >= height)
        {
            return;
        }

        int farDis = 2; // 11
        int nerDis = 2; // 3

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
        // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
        bool flag[8] = {false};
        int num_valid_pixels = 0;

        float costMin;
        int costMinPoint;

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
            // lzd 利用up_far的视差平面（plane_hypotheses)来计算当前像素p的视差+代价
            // 每个邻域图像都与当前图像计算一个视差+代价值。但是，实际仅仅存了代价值
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[up_far], cost_array[1], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[down_far], cost_array[3], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[left_far], cost_array[5], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[right_far], cost_array[7], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[up_near], cost_array[0], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[down_near], cost_array[2], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[left_near], cost_array[4], params);
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
            ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[right_near], cost_array[6], params);
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
            if (view_weights[i] > 0)
            {
                setBit(temp_selected_views, i);
                weight_norm += view_weights[i];
                num_selected_view++;
            }
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
                        final_costs[i] += view_weights[j] * cost_array[i][j];
                    }
                }
            }
            final_costs[i] /= weight_norm;
        }

        // 找出8个邻居像素的视差平面参数计算当前像素p的代价，params.num_images数量的代价累加后，最小的那组代价对应的
        // 邻居像素的索引=min_cost_idx
        const int min_cost_idx = FindMinCostIndex(final_costs, 8);

        // 计算全是自己参数（视差平面）的代价值=cost_vector_now
        float cost_vector_now[32] = {2.0f};
        ComputeMultiViewCostVectorLF(images, pcenters, p, plane_hypotheses[center], cost_vector_now, params);
        float cost_now = 0.0f;
        for (int i = 0; i < params.num_images - 1; ++i)
        {
            if (params.geom_consistency)
            {
                //cost_now += view_weights[i] * (cost_vector_now[i] + 0.1f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p));
            }
            else
            {
                cost_now += view_weights[i] * cost_vector_now[i];
            }
        }
        cost_now /= weight_norm;
        costs[center] = cost_now;
        //float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
        float depth_now = plane_hypotheses[center].w;

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
                    //float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[max_cost_idx]], p);
                    float depth_before = plane_hypotheses[positions[max_cost_idx]].w;
                    if (depth_before >= params.depth_min && depth_before <= params.depth_max
                        && restricted_final_costs[max_cost_idx] > restricted_cost_now)
                    {
                        depth_now   = depth_before;
                        plane_hypotheses[center] = plane_hypotheses[positions[max_cost_idx]]; // lzd 赋值新的平面、视差
                        costs[center] = final_costs[max_cost_idx];
                        restricted_cost = restricted_final_costs[max_cost_idx];
                        selected_views[center] = temp_selected_views;
                    }
                }
            }
            else if (flag[min_cost_idx])
            {
                //float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
                float depth_before = plane_hypotheses[positions[min_cost_idx]].w;
                if (depth_before >= params.depth_min && depth_before <= params.depth_max
                    && final_costs[min_cost_idx] < cost_now)
                {
                    depth_now = depth_before;
                    plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]]; // lzd 赋值新的平面、视差
                    costs[center] = final_costs[min_cost_idx];
                }
            }
        }

        if (!params.planar_prior && flag[min_cost_idx])
        {
            //float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
            float disp_before = plane_hypotheses[positions[min_cost_idx]].w;

            // if (final_costs[min_cost_idx] < cost_now)
            // {
            //     printf("%f\n", disp_before);
            // }

            if (disp_before >= params.depth_min && disp_before <= params.depth_max
                && final_costs[min_cost_idx] < cost_now)
            {
                depth_now = disp_before;
                plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]]; // lzd 赋值新的平面、视差
                costs[center] = final_costs[min_cost_idx];
                selected_views[center] = temp_selected_views;
            }
        }
        PlaneHypothesisRefinementLF(images, depths,
            &plane_hypotheses[center], pcenters,&depth_now,
            &costs[center], &rand_states[center], view_weights,
            weight_norm, prior_planes, plane_masks, &restricted_cost,
            p, params,WIDTH);
    }

    __global__ void BlackPixelUpdateLF(cudaTextureObjects* texture_objects,
                    cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
                    float* costs, curandState* rand_states, unsigned int* selected_views,
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

        CheckerboardPropagationLF_Near(texture_objects[0].images, texture_depths[0].images,
            plane_hypotheses, pcenters, costs, rand_states, selected_views, prior_planes,
            plane_masks, p, params, iter, width, height);
    }

    __global__ void RedPixelUpdateLF(cudaTextureObjects *texture_objects,
        cudaTextureObjects* texture_depths, float4* plane_hypotheses,
        float2* pcenters, float* costs, curandState* rand_states,
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
        CheckerboardPropagationLF_Near(texture_objects[0].images, texture_depths[0].images,
            plane_hypotheses, pcenters,costs, rand_states, selected_views, prior_planes, plane_masks,
            p, params, iter, width, height);
    }
}
