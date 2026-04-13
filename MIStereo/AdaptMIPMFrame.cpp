/********************************************************************
file base:      AdaptMIPMFrame.cpp
author:         LZD
created:        2026/04/12
purpose:        整帧GPU版微图像视差匹配（跨视图proposal）
*********************************************************************/
#include "AdaptMIPMFrame.h"

#include "CudaUtil.h"
#include "MIStereo/AdaptMIPMUtil.cuh"

namespace LFMVS
{
    AdaptMIPMFrame::AdaptMIPMFrame(LightFieldParams& paramsIn)
        : m_ParamsCUDA(paramsIn)
        , m_num_views(0)
        , m_max_neighbors(16)
        , m_pixels_per_view(0)
        , m_bReleased(false)
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
        , plane_final_host(nullptr)
        , cost_final_host(nullptr)
        , disp_final_host(nullptr)
        , selected_final_host(nullptr)
    {
        memset(&texture_objects_host, 0, sizeof(texture_objects_host));
        memset(cuArray, 0, sizeof(cuArray));
        memset(cu_blur_Array, 0, sizeof(cu_blur_Array));
    }

    AdaptMIPMFrame::~AdaptMIPMFrame()
    {
        ReleaseMemory();
    }

    void AdaptMIPMFrame::ReleaseMemory()
    {
        if (m_bReleased)
            return;
        m_bReleased = true;

        images_MI.clear();
        blur_images_MI.clear();
        m_view_keys.clear();
        m_key_to_viewid.clear();

        delete[] centerPointS_MI; centerPointS_MI = nullptr;
        delete[] tileKeyS_MI; tileKeyS_MI = nullptr;
        delete[] neighbor_ids_host; neighbor_ids_host = nullptr;
        delete[] neighbor_counts_host; neighbor_counts_host = nullptr;

        delete[] plane_final_host; plane_final_host = nullptr;
        delete[] cost_final_host; cost_final_host = nullptr;
        delete[] disp_final_host; disp_final_host = nullptr;
        delete[] selected_final_host; selected_final_host = nullptr;

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
    }

    void AdaptMIPMFrame::InitGPUParamsFromCPUParams()
    {
        params.max_iterations = 3;
        params.patch_size = 5;
        params.patch_Bound_size = 5;
        params.propagation_Graph_size = 5;
        params.num_images = m_max_neighbors + 1;
        params.top_k = 5;
        params.depth_min = 0.0f;
        params.depth_max = m_ParamsCUDA.baseline * 0.5f;
        params.disparity_min = 0.0f;
        params.disparity_max = m_ParamsCUDA.baseline * 0.5f;
        params.Base = m_ParamsCUDA.baseline;
        params.MLA_Mask_Width_Cuda = m_ParamsCUDA.mi_width_for_match;
        params.MLA_Mask_Height_Cuda = m_ParamsCUDA.mia_height_for_match;
        params.base_height_ratio = 0.2f;
        params.base_height_sigma = 0.05f;
    }

    bool AdaptMIPMFrame::BuildViewIndexMap(QuadTreeTileInfoMap& MLA_info_map, QuadTreeProblemMap& problem_map)
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
                LOG_ERROR("AdaptMIPMFrame: valid views exceed MAX_IMAGES=", MAX_IMAGES);
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
            LOG_ERROR("AdaptMIPMFrame: no valid micro-images for current frame.");
            return false;
        }
        LOG_INFO("AdaptMIPMFrame: valid micro-images loaded = ", m_num_views,
                 ", MAX_IMAGES = ", MAX_IMAGES,
                 ", m_max_neighbors = ", m_max_neighbors);

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
            for (size_t i = 0; i < problem.m_NeighsSortVecForMatch.size() && count < m_max_neighbors; ++i)
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

        return true;
    }

    void AdaptMIPMFrame::CreateGrayImageObject(int image_index)
    {
        int image_rows = images_MI[image_index].rows;
        int image_cols = images_MI[image_index].cols;
        cv::Mat& gray_image = images_MI[image_index];

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[image_index], &channelDesc, image_cols, image_rows);
        cudaMemcpy2DToArray(cuArray[image_index], 0, 0, gray_image.ptr<float>(),
            gray_image.step[0], image_cols * sizeof(float), image_rows, cudaMemcpyHostToDevice);

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

    void AdaptMIPMFrame::CreateBlurImageObject(int image_index)
    {
        int image_rows = blur_images_MI[image_index].rows;
        int image_cols = blur_images_MI[image_index].cols;
        cv::Mat& blur_image = blur_images_MI[image_index];

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cu_blur_Array[image_index], &channelDesc, image_cols, image_rows);
        cudaMemcpy2DToArray(cu_blur_Array[image_index], 0, 0, blur_image.ptr<float>(),
            blur_image.step[0], image_cols * sizeof(float), image_rows, cudaMemcpyHostToDevice);

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

    bool AdaptMIPMFrame::InitializeGPU()
    {
        if (m_num_views <= 0)
            return false;
        if ((int)images_MI.size() != m_num_views || (int)blur_images_MI.size() != m_num_views)
            return false;

        const int mi_width = m_ParamsCUDA.mi_width_for_match;
        const int mi_height = m_ParamsCUDA.mia_height_for_match;
        m_pixels_per_view = mi_width * mi_height;
        const size_t frame_pixels = (size_t)m_pixels_per_view * (size_t)m_num_views;

        for (int i = 0; i < m_num_views; ++i)
        {
            CreateGrayImageObject(i);
            CreateBlurImageObject(i);
        }

        CUDA_SAFE_CALL(cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects)));
        CUDA_SAFE_CALL(cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc((void**)&centers_cuda, sizeof(float2) * m_num_views));
        CUDA_SAFE_CALL(cudaMemcpy(centers_cuda, centerPointS_MI, sizeof(float2) * m_num_views, cudaMemcpyHostToDevice));

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

        plane_final_host = new float4[frame_pixels];
        cost_final_host = new float[frame_pixels];
        disp_final_host = new float4[frame_pixels];
        selected_final_host = new unsigned int[frame_pixels];

        return true;
    }

    bool AdaptMIPMFrame::Initialize(QuadTreeTileInfoMap& MLA_info_map, QuadTreeProblemMap& problem_map)
    {
        if (!BuildViewIndexMap(MLA_info_map, problem_map))
            return false;
        InitGPUParamsFromCPUParams();
        return InitializeGPU();
    }

    void RunPatchMatchCUDAForFrame_Impl(cudaTextureObjects* texture_objects_cuda,
        float2* centers_cuda, int2* tilekeys_cuda, int* neighbor_ids_cuda, int* neighbor_counts_cuda,
        float4* plane_prev_cuda, float4* plane_next_cuda, float* cost_prev_cuda, float* cost_next_cuda,
        float4* disp_prev_cuda, float4* disp_next_cuda, unsigned int* selected_prev_cuda, unsigned int* selected_next_cuda,
        curandState* rand_states_cuda, PatchMatchParamsLF params, int num_views, int max_neighbors);

    void AdaptMIPMFrame::RunPatchMatchCUDAForFrame()
    {
        RunPatchMatchCUDAForFrame_Impl(texture_objects_cuda,
            centers_cuda, tilekeys_cuda, neighbor_ids_cuda, neighbor_counts_cuda,
            plane_prev_cuda, plane_next_cuda, cost_prev_cuda, cost_next_cuda,
            disp_prev_cuda, disp_next_cuda, selected_prev_cuda, selected_next_cuda,
            rand_states_cuda, params, m_num_views, m_max_neighbors);

        const size_t frame_pixels = (size_t)m_pixels_per_view * (size_t)m_num_views;
        CUDA_SAFE_CALL(cudaMemcpy(plane_final_host, plane_prev_cuda, sizeof(float4) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(cost_final_host, cost_prev_cuda, sizeof(float) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(disp_final_host, disp_prev_cuda, sizeof(float4) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(selected_final_host, selected_prev_cuda, sizeof(unsigned int) * frame_pixels, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void AdaptMIPMFrame::WriteBackResults(QuadTreeDisNormalMap& disNormals_map)
    {
        const int mi_width = m_ParamsCUDA.mi_width_for_match;
        const int mi_height = m_ParamsCUDA.mia_height_for_match;
        const int pixels = mi_width * mi_height;

        for (int vid = 0; vid < m_num_views; ++vid)
        {
            QuadTreeTileKeyPtr key = m_view_keys[vid];
            QuadTreeDisNormalMap::iterator itr = disNormals_map.find(key);
            if (itr == disNormals_map.end())
                continue;

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
