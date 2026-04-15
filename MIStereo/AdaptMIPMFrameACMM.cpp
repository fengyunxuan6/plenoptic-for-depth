/********************************************************************
file base:      AdaptMIPMFrameACMM.cpp
author:         OpenAI + LZD workflow
created:        2026/04/15
purpose:        ACMM风格整帧微图像匹配（v5：保留原LF有效内核，仅增加多尺度/暖启动）
*********************************************************************/
#include "AdaptMIPMFrameACMM.h"

#include "CudaUtil.h"
#include "MIStereo/AdaptMIPMUtil.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace LFMVS
{
    AdaptMIPMFrameACMM::AdaptMIPMFrameACMM(LightFieldParams& paramsIn)
        : m_ParamsCUDA(paramsIn)
        , m_num_views(0)
        , m_max_neighbors(16)
        , m_pixels_per_view(0)
        , m_num_levels(3)
        , m_current_level(-1)
        , m_bReleased(false)
        , m_lambda_scale(0.0f)
        , m_lambda_geo(0.0f)
        , m_detail_th(0.0f)
        , m_geom_clip(2.0f)
        , m_blur_group_count(3)
        , centerPointS_MI(nullptr)
        , tileKeyS_MI(nullptr)
        , neighbor_ids_host(nullptr)
        , neighbor_counts_host(nullptr)
        , texture_objects_cuda(nullptr)
        , centers_cuda(nullptr)
        , tilekeys_cuda(nullptr)
        , neighbor_ids_cuda(nullptr)
        , neighbor_counts_cuda(nullptr)
        , plane_prev_cuda(nullptr)
        , plane_next_cuda(nullptr)
        , cost_prev_cuda(nullptr)
        , cost_next_cuda(nullptr)
        , disp_prev_cuda(nullptr)
        , disp_next_cuda(nullptr)
        , selected_prev_cuda(nullptr)
        , selected_next_cuda(nullptr)
        , rand_states_cuda(nullptr)
        , plane_init_cuda(nullptr)
        , cost_init_cuda(nullptr)
        , disp_init_cuda(nullptr)
        , selected_init_cuda(nullptr)
        , plane_final_host(nullptr)
        , cost_final_host(nullptr)
        , disp_final_host(nullptr)
        , selected_final_host(nullptr)
    {
        memset(&texture_objects_host, 0, sizeof(texture_objects_host));
        memset(cuArray, 0, sizeof(cuArray));
        memset(cu_blur_Array, 0, sizeof(cu_blur_Array));
    }

    AdaptMIPMFrameACMM::~AdaptMIPMFrameACMM()
    {
        ReleaseMemory();
    }

    void AdaptMIPMFrameACMM::ReleaseGPUForCurrentLevel()
    {
        for (int i = 0; i < m_num_views; ++i)
        {
            if (texture_objects_host.images[i]) {
                cudaDestroyTextureObject(texture_objects_host.images[i]);
                texture_objects_host.images[i] = 0;
            }
            if (cuArray[i]) {
                cudaFreeArray(cuArray[i]);
                cuArray[i] = nullptr;
            }
            if (texture_objects_host.blur_images[i]) {
                cudaDestroyTextureObject(texture_objects_host.blur_images[i]);
                texture_objects_host.blur_images[i] = 0;
            }
            if (cu_blur_Array[i]) {
                cudaFreeArray(cu_blur_Array[i]);
                cu_blur_Array[i] = nullptr;
            }
        }

        if (texture_objects_cuda) cudaFree(texture_objects_cuda);
        if (centers_cuda) cudaFree(centers_cuda);
        if (tilekeys_cuda) cudaFree(tilekeys_cuda);
        if (neighbor_ids_cuda) cudaFree(neighbor_ids_cuda);
        if (neighbor_counts_cuda) cudaFree(neighbor_counts_cuda);
        if (plane_prev_cuda) cudaFree(plane_prev_cuda);
        if (plane_next_cuda) cudaFree(plane_next_cuda);
        if (cost_prev_cuda) cudaFree(cost_prev_cuda);
        if (cost_next_cuda) cudaFree(cost_next_cuda);
        if (disp_prev_cuda) cudaFree(disp_prev_cuda);
        if (disp_next_cuda) cudaFree(disp_next_cuda);
        if (selected_prev_cuda) cudaFree(selected_prev_cuda);
        if (selected_next_cuda) cudaFree(selected_next_cuda);
        if (rand_states_cuda) cudaFree(rand_states_cuda);
        if (plane_init_cuda) cudaFree(plane_init_cuda);
        if (cost_init_cuda) cudaFree(cost_init_cuda);
        if (disp_init_cuda) cudaFree(disp_init_cuda);
        if (selected_init_cuda) cudaFree(selected_init_cuda);

        texture_objects_cuda = nullptr;
        centers_cuda = nullptr;
        tilekeys_cuda = nullptr;
        neighbor_ids_cuda = nullptr;
        neighbor_counts_cuda = nullptr;
        plane_prev_cuda = nullptr;
        plane_next_cuda = nullptr;
        cost_prev_cuda = nullptr;
        cost_next_cuda = nullptr;
        disp_prev_cuda = nullptr;
        disp_next_cuda = nullptr;
        selected_prev_cuda = nullptr;
        selected_next_cuda = nullptr;
        rand_states_cuda = nullptr;
        plane_init_cuda = nullptr;
        cost_init_cuda = nullptr;
        disp_init_cuda = nullptr;
        selected_init_cuda = nullptr;
    }

    void AdaptMIPMFrameACMM::ReleaseMemory()
    {
        if (m_bReleased)
            return;
        m_bReleased = true;

        ReleaseGPUForCurrentLevel();

        images_MI.clear();
        blur_images_MI.clear();
        m_view_keys.clear();
        m_key_to_viewid.clear();
        m_pyramid_levels.clear();

        delete[] centerPointS_MI; centerPointS_MI = nullptr;
        delete[] tileKeyS_MI; tileKeyS_MI = nullptr;
        delete[] neighbor_ids_host; neighbor_ids_host = nullptr;
        delete[] neighbor_counts_host; neighbor_counts_host = nullptr;

        delete[] plane_final_host; plane_final_host = nullptr;
        delete[] cost_final_host; cost_final_host = nullptr;
        delete[] disp_final_host; disp_final_host = nullptr;
        delete[] selected_final_host; selected_final_host = nullptr;
    }

    void AdaptMIPMFrameACMM::InitGPUParamsFromCPUParams(int level_id)
    {
        const ACMMFrameLevelHostData& level = m_pyramid_levels[level_id];
        params.max_iterations = 3;
        params.patch_size = 5;
        params.patch_Bound_size = 5;
        params.propagation_Graph_size = 5;
        params.num_images = m_max_neighbors + 1;
        params.top_k = 5;

        // 与原有有效LF内核保持同量纲；每层只按当前Base同步缩放
        params.Base = m_ParamsCUDA.baseline * level.scale;
        params.depth_min = 0.0f;
        params.depth_max = params.Base * 0.5f;
        params.disparity_min = 0.0f;
        params.disparity_max = params.Base * 0.5f;

        params.MLA_Mask_Width_Cuda = level.width;
        params.MLA_Mask_Height_Cuda = level.height;
        params.base_height_ratio = 0.2f;
        params.base_height_sigma = 0.05f;
        params.geom_consistency = false;

        LOG_INFO("ACMM-v5 params: level=", level_id,
                 ", width=", level.width,
                 ", height=", level.height,
                 ", Base=", params.Base,
                 ", disp_range=[", params.disparity_min, ", ", params.disparity_max, "]");
    }

    bool AdaptMIPMFrameACMM::BuildViewIndexMap(QuadTreeTileInfoMap& MLA_info_map, QuadTreeProblemMap& problem_map)
    {
        m_view_keys.clear();
        m_key_to_viewid.clear();
        images_MI.clear();
        blur_images_MI.clear();

        for (QuadTreeProblemMap::iterator itrP = problem_map.begin(); itrP != problem_map.end(); ++itrP)
        {
            MLA_Problem& problem = itrP->second;
            QuadTreeTileKeyPtr ptrKey = itrP->first;
            if (problem.m_bGarbage || !problem.m_bNeedMatch)
                continue;

            QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(ptrKey);
            if (itrInfo == MLA_info_map.end())
                continue;

            const int vid = (int)m_view_keys.size();
            if (vid >= MAX_IMAGES)
            {
                LOG_ERROR("AdaptMIPMFrameACMM: valid views exceed MAX_IMAGES=", MAX_IMAGES);
                return false;
            }

            m_view_keys.push_back(ptrKey);
            m_key_to_viewid[ptrKey] = vid;

            cv::Mat image_float;
            problem.m_Image_gray.convertTo(image_float, CV_32FC1);
            images_MI.push_back(image_float);

            cv::Mat blur_float;
            problem.m_Image_Blureness.convertTo(blur_float, CV_32FC1);
            blur_images_MI.push_back(blur_float);
        }

        m_num_views = (int)m_view_keys.size();
        if (m_num_views <= 0)
        {
            LOG_ERROR("AdaptMIPMFrameACMM: no valid micro-images for current frame.");
            return false;
        }

        centerPointS_MI = new float2[m_num_views];
        tileKeyS_MI = new int2[m_num_views];
        neighbor_counts_host = new int[m_num_views];
        neighbor_ids_host = new int[m_num_views * m_max_neighbors];
        for (int i = 0; i < m_num_views * m_max_neighbors; ++i)
            neighbor_ids_host[i] = -1;

        for (int vid = 0; vid < m_num_views; ++vid)
        {
            QuadTreeTileKeyPtr key = m_view_keys[vid];
            MLA_InfoPtr ptrInfo = MLA_info_map[key];
            centerPointS_MI[vid] = make_float2(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
            tileKeyS_MI[vid] = make_int2(key->GetTileX(), key->GetTileY());

            MLA_Problem& problem = problem_map[key];
            int count = 0;
            for (size_t i = 0; i < problem.m_NeighsSortVecForMatch.size() && count < (size_t)m_max_neighbors; ++i)
            {
                QuadTreeTileKeyPtr neighKey = problem.m_NeighsSortVecForMatch[i];
                std::map<QuadTreeTileKeyPtr, int, QuadTreeTileKeyMapCmpLess>::iterator itr = m_key_to_viewid.find(neighKey);
                if (itr == m_key_to_viewid.end())
                    continue;
                neighbor_ids_host[vid * m_max_neighbors + count] = itr->second;
                ++count;
            }
            neighbor_counts_host[vid] = count;
        }

        LOG_INFO("AdaptMIPMFrameACMM: valid micro-image views=", m_num_views,
                 ", max_neighbors=", m_max_neighbors);
        return true;
    }

    float AdaptMIPMFrameACMM::EstimateGaussianSigma(float blur_mean_src,
                                                    float blur_mean_target,
                                                    float level_scale) const
    {
        if (blur_mean_target <= blur_mean_src)
            return 0.0f;
        const float diff = (blur_mean_target - blur_mean_src) / 255.0f;
        const float sigma = std::max(0.0f, 2.0f * diff / std::max(level_scale, 0.25f));
        return std::min(2.5f, sigma);
    }

    void AdaptMIPMFrameACMM::BuildSingleLevelBlurAlignment(ACMMFrameLevelHostData& level_data) const
    {
        // v5.1：
        // 先关闭额外的图像模糊对齐。当前问题主要不是“粗层不稳定”，而是“细层精度不够、纹理泄漏明显”。
        // 你原LF代价本身已经包含 blur-aware 约束，再对灰度图做高斯对齐会进一步削弱可判别纹理。
        (void)level_data;
        return;
    }

    bool AdaptMIPMFrameACMM::BuildBlurAlignedPyramids()
    {
        if (images_MI.empty())
            return false;

        const int full_w = images_MI[0].cols;
        const int full_h = images_MI[0].rows;

        int max_levels = 1;
        int cur_w = full_w, cur_h = full_h;
        while (max_levels < MAX_ACMM_LEVELS && cur_w >= 24 && cur_h >= 24)
        {
            cur_w /= 2;
            cur_h /= 2;
            if (cur_w < 12 || cur_h < 12) break;
            ++max_levels;
        }
        m_num_levels = std::max(2, std::min(3, max_levels));

        m_pyramid_levels.clear();
        m_pyramid_levels.resize(m_num_levels);

        for (int lid = 0; lid < m_num_levels; ++lid)
        {
            const int coarse_id = m_num_levels - 1 - lid;
            const float scale = 1.0f / std::pow(2.0f, (float)coarse_id);
            const int width = std::max(8, (int)std::round(full_w * scale));
            const int height = std::max(8, (int)std::round(full_h * scale));

            ACMMFrameLevelHostData& level = m_pyramid_levels[lid];
            level.width = width;
            level.height = height;
            level.scale = scale;
            level.images.resize(m_num_views);
            level.blur_images.resize(m_num_views);

            for (int vid = 0; vid < m_num_views; ++vid)
            {
                if (width == full_w && height == full_h)
                {
                    level.images[vid] = images_MI[vid].clone();
                    level.blur_images[vid] = blur_images_MI[vid].clone();
                }
                else
                {
                    cv::resize(images_MI[vid], level.images[vid], cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
                    cv::resize(blur_images_MI[vid], level.blur_images[vid], cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
                }
            }
            BuildSingleLevelBlurAlignment(level);
        }
        return true;
    }

    void AdaptMIPMFrameACMM::CreateGrayImageObject(int image_index, const cv::Mat& image_in)
    {
        const int rows = image_in.rows;
        const int cols = image_in.cols;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[image_index], &channelDesc, cols, rows);
        cudaMemcpy2DToArray(cuArray[image_index], 0, 0, image_in.ptr<float>(),
            image_in.step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[image_index];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[image_index]), &resDesc, &texDesc, NULL);
    }

    void AdaptMIPMFrameACMM::CreateBlurImageObject(int image_index, const cv::Mat& blur_in)
    {
        const int rows = blur_in.rows;
        const int cols = blur_in.cols;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cu_blur_Array[image_index], &channelDesc, cols, rows);
        cudaMemcpy2DToArray(cu_blur_Array[image_index], 0, 0, blur_in.ptr<float>(),
            blur_in.step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cu_blur_Array[image_index];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.blur_images[image_index]), &resDesc, &texDesc, NULL);
    }

    bool AdaptMIPMFrameACMM::InitializeGPUForLevel(int level_id)
    {
        const ACMMFrameLevelHostData& level = m_pyramid_levels[level_id];
        m_pixels_per_view = level.width * level.height;
        const size_t frame_pixels = (size_t)m_pixels_per_view * (size_t)m_num_views;
        if (m_pixels_per_view <= 0 || m_num_views <= 0 || frame_pixels == 0)
        {
            LOG_ERROR("AdaptMIPMFrameACMM::InitializeGPUForLevel invalid shape, level=", level_id,
                      ", pixels_per_view=", m_pixels_per_view, ", num_views=", m_num_views);
            return false;
        }

        ReleaseGPUForCurrentLevel();
        memset(&texture_objects_host, 0, sizeof(texture_objects_host));

        for (int i = 0; i < m_num_views; ++i)
        {
            CreateGrayImageObject(i, level.images[i]);
            CreateBlurImageObject(i, level.blur_images[i]);
        }

        CUDA_SAFE_CALL(cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects)));
        CUDA_SAFE_CALL(cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice));

        std::vector<float2> level_centers(m_num_views);
        for (int vid = 0; vid < m_num_views; ++vid)
            level_centers[vid] = make_float2(centerPointS_MI[vid].x * level.scale, centerPointS_MI[vid].y * level.scale);

        CUDA_SAFE_CALL(cudaMalloc((void**)&centers_cuda, sizeof(float2) * m_num_views));
        CUDA_SAFE_CALL(cudaMemcpy(centers_cuda, level_centers.data(), sizeof(float2) * m_num_views, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc((void**)&tilekeys_cuda, sizeof(int2) * m_num_views));
        CUDA_SAFE_CALL(cudaMemcpy(tilekeys_cuda, tileKeyS_MI, sizeof(int2) * m_num_views, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc((void**)&neighbor_counts_cuda, sizeof(int) * m_num_views));
        CUDA_SAFE_CALL(cudaMemcpy(neighbor_counts_cuda, neighbor_counts_host, sizeof(int) * m_num_views, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc((void**)&neighbor_ids_cuda, sizeof(int) * m_num_views * m_max_neighbors));
        CUDA_SAFE_CALL(cudaMemcpy(neighbor_ids_cuda, neighbor_ids_host, sizeof(int) * m_num_views * m_max_neighbors, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc((void**)&plane_prev_cuda, sizeof(float4) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&plane_next_cuda, sizeof(float4) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&cost_prev_cuda, sizeof(float) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&cost_next_cuda, sizeof(float) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&disp_prev_cuda, sizeof(float4) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&disp_next_cuda, sizeof(float4) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&selected_prev_cuda, sizeof(unsigned int) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&selected_next_cuda, sizeof(unsigned int) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * frame_pixels));

        CUDA_SAFE_CALL(cudaMalloc((void**)&plane_init_cuda, sizeof(float4) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&cost_init_cuda, sizeof(float) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&disp_init_cuda, sizeof(float4) * frame_pixels));
        CUDA_SAFE_CALL(cudaMalloc((void**)&selected_init_cuda, sizeof(unsigned int) * frame_pixels));

        plane_level_host.resize(frame_pixels);
        cost_level_host.resize(frame_pixels);
        disp_level_host.resize(frame_pixels);
        selected_level_host.resize(frame_pixels);

        plane_init_host.resize(frame_pixels, make_float4(0, 0, 1, 0));
        cost_init_host.resize(frame_pixels, 2.0f);
        disp_init_host.resize(frame_pixels, make_float4(0, 0, 0, 0));
        selected_init_host.resize(frame_pixels, 0);

        LOG_INFO("ACMM-v5 level host buffers resized: level=", level_id,
                 ", frame_pixels=", (double)frame_pixels,
                 ", cost_level_host.size=", (double)cost_level_host.size());
        return true;
    }

    bool AdaptMIPMFrameACMM::Initialize(QuadTreeTileInfoMap& MLA_info_map, QuadTreeProblemMap& problem_map)
    {
        if (!BuildViewIndexMap(MLA_info_map, problem_map)) return false;
        if (!BuildBlurAlignedPyramids()) return false;

        const size_t frame_pixels_final = (size_t)images_MI[0].cols * (size_t)images_MI[0].rows * (size_t)m_num_views;
        plane_final_host = new float4[frame_pixels_final];
        cost_final_host = new float[frame_pixels_final];
        disp_final_host = new float4[frame_pixels_final];
        selected_final_host = new unsigned int[frame_pixels_final];

        LOG_INFO("AdaptMIPMFrameACMM::Initialize success, final_frame_pixels=", (double)frame_pixels_final,
                 ", levels=", m_num_levels, ", views=", m_num_views);
        return true;
    }

    void AdaptMIPMFrameACMM::PrepareWarmStartFromPreviousLevel(int prev_level_id, int cur_level_id)
    {
        const ACMMFrameLevelHostData& prev_level = m_pyramid_levels[prev_level_id];
        const ACMMFrameLevelHostData& cur_level = m_pyramid_levels[cur_level_id];
        const int prev_pixels = prev_level.width * prev_level.height;
        const int cur_pixels = cur_level.width * cur_level.height;

        if ((int)plane_level_host.size() != prev_pixels * m_num_views)
        {
            std::fill(plane_init_host.begin(), plane_init_host.end(), make_float4(0, 0, 1, 0));
            std::fill(cost_init_host.begin(), cost_init_host.end(), 2.0f);
            std::fill(disp_init_host.begin(), disp_init_host.end(), make_float4(0, 0, 0, 0));
            std::fill(selected_init_host.begin(), selected_init_host.end(), 0);
            return;
        }

        plane_init_host.assign(cur_pixels * m_num_views, make_float4(0, 0, 1, 0));
        cost_init_host.assign(cur_pixels * m_num_views, 2.0f);
        disp_init_host.assign(cur_pixels * m_num_views, make_float4(0, 0, 0, 0));
        selected_init_host.assign(cur_pixels * m_num_views, 0);

        const float sx = (float)prev_level.width / std::max(1, cur_level.width);
        const float sy = (float)prev_level.height / std::max(1, cur_level.height);
        const float disp_scale = cur_level.scale / std::max(prev_level.scale, 1e-6f);

        for (int vid = 0; vid < m_num_views; ++vid)
        {
            for (int y = 0; y < cur_level.height; ++y)
            {
                const int py = std::min(prev_level.height - 1, (int)std::floor(y * sy));
                for (int x = 0; x < cur_level.width; ++x)
                {
                    const int px = std::min(prev_level.width - 1, (int)std::floor(x * sx));
                    const int prev_idx = vid * prev_pixels + py * prev_level.width + px;
                    const int cur_idx = vid * cur_pixels + y * cur_level.width + x;

                    float4 prev_plane = plane_level_host[prev_idx];
                    prev_plane.w *= disp_scale; // 保留原LF内核估计到的plane方向，只缩放锚点视差
                    plane_init_host[cur_idx] = prev_plane;
                    cost_init_host[cur_idx] = cost_level_host[prev_idx];
                    disp_init_host[cur_idx] = disp_level_host[prev_idx];
                    selected_init_host[cur_idx] = selected_level_host[prev_idx];
                }
            }
        }
    }

    void AdaptMIPMFrameACMM::UploadWarmStartToGPU()
    {
        const size_t frame_pixels = (size_t)m_pixels_per_view * (size_t)m_num_views;
        CUDA_SAFE_CALL(cudaMemcpy(plane_init_cuda, plane_init_host.data(), sizeof(float4) * frame_pixels, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(cost_init_cuda, cost_init_host.data(), sizeof(float) * frame_pixels, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(disp_init_cuda, disp_init_host.data(), sizeof(float4) * frame_pixels, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(selected_init_cuda, selected_init_host.data(), sizeof(unsigned int) * frame_pixels, cudaMemcpyHostToDevice));
    }

    void AdaptMIPMFrameACMM::DownloadCurrentLevelResults()
    {
        const size_t frame_pixels = (size_t)m_pixels_per_view * (size_t)m_num_views;
        CUDA_SAFE_CALL(cudaMemcpy(plane_level_host.data(), plane_prev_cuda, sizeof(float4) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(cost_level_host.data(), cost_prev_cuda, sizeof(float) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(disp_level_host.data(), disp_prev_cuda, sizeof(float4) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(selected_level_host.data(), selected_prev_cuda, sizeof(unsigned int) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void RunPatchMatchCUDAForFrameACMM_Impl(cudaTextureObjects* texture_objects_cuda,
        float2* centers_cuda, int2* tilekeys_cuda, int* neighbor_ids_cuda, int* neighbor_counts_cuda,
        float4* plane_prev_cuda, float4* plane_next_cuda, float* cost_prev_cuda, float* cost_next_cuda,
        float4* disp_prev_cuda, float4* disp_next_cuda, unsigned int* selected_prev_cuda, unsigned int* selected_next_cuda,
        float4* plane_init_cuda, float* cost_init_cuda, float4* disp_init_cuda, unsigned int* selected_init_cuda,
        curandState* rand_states_cuda, PatchMatchParamsLF params, int num_views, int max_neighbors,
        bool use_warm_start, float lambda_scale, float lambda_geo, float detail_th, float geom_clip);

    void AdaptMIPMFrameACMM::RunPatchMatchCUDAForFrameACMM()
    {
        for (int lid = 0; lid < m_num_levels; ++lid)
        {
            m_current_level = lid;
            InitGPUParamsFromCPUParams(lid);
            if (!InitializeGPUForLevel(lid))
            {
                LOG_ERROR("AdaptMIPMFrameACMM::InitializeGPUForLevel failed, level=", lid);
                return;
            }

            if (lid == 0)
            {
                std::fill(plane_init_host.begin(), plane_init_host.end(), make_float4(0, 0, 1, 0));
                std::fill(cost_init_host.begin(), cost_init_host.end(), 2.0f);
                std::fill(disp_init_host.begin(), disp_init_host.end(), make_float4(0, 0, 0, 0));
                std::fill(selected_init_host.begin(), selected_init_host.end(), 0);
            }
            else
            {
                PrepareWarmStartFromPreviousLevel(lid - 1, lid);
            }
            UploadWarmStartToGPU();

            RunPatchMatchCUDAForFrameACMM_Impl(texture_objects_cuda,
                centers_cuda, tilekeys_cuda, neighbor_ids_cuda, neighbor_counts_cuda,
                plane_prev_cuda, plane_next_cuda, cost_prev_cuda, cost_next_cuda,
                disp_prev_cuda, disp_next_cuda, selected_prev_cuda, selected_next_cuda,
                plane_init_cuda, cost_init_cuda, disp_init_cuda, selected_init_cuda,
                rand_states_cuda, params, m_num_views, m_max_neighbors,
                (lid > 0), 0.0f, 0.0f, 0.0f, m_geom_clip);

            DownloadCurrentLevelResults();

            float cost_min = std::numeric_limits<float>::max();
            float cost_max = -std::numeric_limits<float>::max();
            float disp_min = std::numeric_limits<float>::max();
            float disp_max = -std::numeric_limits<float>::max();
            double cost_mean = 0.0, disp_mean = 0.0;
            size_t valid_selected = 0, good_cost = 0;
            for (size_t i = 0; i < cost_level_host.size(); ++i)
            {
                const float c = cost_level_host[i];
                const float d = plane_level_host[i].w;
                cost_min = std::min(cost_min, c);
                cost_max = std::max(cost_max, c);
                disp_min = std::min(disp_min, d);
                disp_max = std::max(disp_max, d);
                cost_mean += c; disp_mean += d;
                if (selected_level_host[i] != 0) ++valid_selected;
                if (c < 1.8f) ++good_cost;
            }
            const double denom = std::max<size_t>(1, cost_level_host.size());
            LOG_INFO("ACMM-v5 level=", lid,
                     ", scale=", m_pyramid_levels[lid].scale,
                     ", valid_selected_ratio=", (double)valid_selected / denom,
                     ", good_cost_ratio=", (double)good_cost / denom,
                     ", cost[min/mean/max]=", cost_min, "/", cost_mean / denom, "/", cost_max,
                     ", disp[min/mean/max]=", disp_min, "/", disp_mean / denom, "/", disp_max);
        }

        const ACMMFrameLevelHostData& last_level = m_pyramid_levels.back();
        const size_t final_pixels = (size_t)last_level.width * (size_t)last_level.height * (size_t)m_num_views;
        std::copy(plane_level_host.begin(), plane_level_host.begin() + final_pixels, plane_final_host);
        std::copy(cost_level_host.begin(), cost_level_host.begin() + final_pixels, cost_final_host);
        std::copy(disp_level_host.begin(), disp_level_host.begin() + final_pixels, disp_final_host);
        std::copy(selected_level_host.begin(), selected_level_host.begin() + final_pixels, selected_final_host);
    }

    void AdaptMIPMFrameACMM::WriteBackResults(QuadTreeDisNormalMap& disNormals_map)
    {
        const ACMMFrameLevelHostData& last_level = m_pyramid_levels.back();
        const int mi_width = last_level.width;
        const int mi_height = last_level.height;
        const int pixels = mi_width * mi_height;

        for (int vid = 0; vid < m_num_views; ++vid)
        {
            QuadTreeTileKeyPtr key = m_view_keys[vid];
            QuadTreeDisNormalMap::iterator itr = disNormals_map.find(key);
            if (itr == disNormals_map.end()) continue;

            DisparityAndNormalPtr ptrDN = itr->second;
            const int offset = vid * pixels;
            for (int idx = 0; idx < pixels; ++idx)
            {
                ptrDN->ph_cuda[idx] = plane_final_host[offset + idx];
                ptrDN->d_cuda[idx] = plane_final_host[offset + idx].w;
                ptrDN->c_cuda[idx] = cost_final_host[offset + idx];
                ptrDN->disp_v_cuda[idx] = disp_final_host[offset + idx];
                ptrDN->selected_views[idx] = selected_final_host[offset + idx];
            }
            ptrDN->m_StereoStage = eSS_ACMH_Finished;
        }
    }
}
