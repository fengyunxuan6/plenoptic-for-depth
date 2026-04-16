/********************************************************************
file base:      AdaptMIPM.cu
author:         LZD
created:        2025/06/12
purpose:
*********************************************************************/
#include "AdaptMIPM.h"

#include <future>

#include "CudaUtil.h"
#include "AdaptMIPMUtil.cuh"
#include "AdaaptMIPM_EdgewareUtil.cuh"

namespace LFMVS
{
    // CPU--->GPU的接口函数
    void AdaptMIPM::RunPatchMatchCUDAForMI()
    {
        int width = images_MI[0].cols;
             const int height = images_MI[0].rows;
        //std::cout << width << " " << height << std::endl;

        int BLOCK_W = 32;
        int BLOCK_H = (BLOCK_W / 2);

        dim3 grid_size_randinit;
        grid_size_randinit.x = (width + 16 - 1) / 16;
        grid_size_randinit.y=(height + 16 - 1) / 16;
        grid_size_randinit.z = 1;
        dim3 block_size_randinit;
        block_size_randinit.x = 16;
        block_size_randinit.y = 16;
        block_size_randinit.z = 1;

        dim3 grid_size_checkerboard;
        grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
        grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
        grid_size_checkerboard.z = 1;
        dim3 block_size_checkerboard;
        block_size_checkerboard.x = BLOCK_W;
        block_size_checkerboard.y = BLOCK_H;
        block_size_checkerboard.z = 1;
        int max_iterations = params.max_iterations;

        if (texture_objects_cuda==nullptr)
            printf("Error-1: texture_objects_cuda\n");
        if (plane_hypotheses_cuda==nullptr)
            printf("Error-2: plane_hypotheses_cuda\n");

        // 随机初始化：随机视差、根据随机视差计算了代价
        RandomInitializationForMI<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda,
            plane_hypotheses_cuda, centers_cuda, costs_cuda, rand_states_cuda, selected_views_cuda,
            prior_planes_cuda, plane_masks_cuda, params, width, height);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // 传播：代价更新
        for (int i = 0; i < max_iterations; ++i)
        {
            BlackPixelUpdate_MIPM<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda,
                params, i, width, height);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                std::cerr<<"Sync error: "<<cudaGetErrorString(syncErr) << std::endl;
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            RedPixelUpdate_MIPM<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda,
                params, i, width, height);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //printf("iteration: %d\n", ia);
        }

        // 回传：GPU--->CPU
        cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(disp_baseline_host, disp_baseline_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(selected_views_host, selected_views_cuda, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void AdaptMIPM::RunPatchMatchCUDAForMI_plane()
    {
        const int width = images_MI[0].cols;
        const int height = images_MI[0].rows;
        //std::cout << width << " " << height << std::endl;

        int BLOCK_W = 32;
        int BLOCK_H = (BLOCK_W / 2);

        dim3 grid_size_randinit;
        grid_size_randinit.x = (width + 16 - 1) / 16;
        grid_size_randinit.y=(height + 16 - 1) / 16;
        grid_size_randinit.z = 1;
        dim3 block_size_randinit;
        block_size_randinit.x = 16;
        block_size_randinit.y = 16;
        block_size_randinit.z = 1;

        dim3 grid_size_checkerboard;
        grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
        grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
        grid_size_checkerboard.z = 1;
        dim3 block_size_checkerboard;
        block_size_checkerboard.x = BLOCK_W;
        block_size_checkerboard.y = BLOCK_H;
        block_size_checkerboard.z = 1;

        int max_iterations = params.max_iterations;

        // 随机初始化
        RandomInitializationForMI<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda,
            plane_hypotheses_cuda, centers_cuda,costs_cuda, rand_states_cuda, selected_views_cuda,
            prior_planes_cuda, plane_masks_cuda, params, width, height);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // 图像内的传播与更新
        for (int i = 0; i < max_iterations; ++i)
        {
            BlackPixelUpdateLF<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, rand_states_cuda,
                selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, i, width, height);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            RedPixelUpdateLF<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, rand_states_cuda,
                selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, i, width, height);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //printf("iteration: %d\n", i);
        }
        cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(selected_views_host, selected_views_cuda, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    ///////////////////////////////////////////////////////////////////////////
    // 边缘像素感知的微图像密集匹配算法：补全 patch+propagation
    void AdaptMIPM::RunPatchMatchCUDAForMI_SoftProxy_PatchRepair()
    {
        const int width = images_MI[0].cols;
        const int height = images_MI[0].rows;
        //std::cout << width << " " << height << std::endl;

        int BLOCK_W = 32;
        int BLOCK_H = (BLOCK_W / 2);

        dim3 grid_size_randinit;
        grid_size_randinit.x = (width + 16 - 1) / 16;
        grid_size_randinit.y=(height + 16 - 1) / 16;
        grid_size_randinit.z = 1;
        dim3 block_size_randinit;
        block_size_randinit.x = 16;
        block_size_randinit.y = 16;
        block_size_randinit.z = 1;

        dim3 grid_size_checkerboard;
        grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
        grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
        grid_size_checkerboard.z = 1;
        dim3 block_size_checkerboard;
        block_size_checkerboard.x = BLOCK_W;
        block_size_checkerboard.y = BLOCK_H;
        block_size_checkerboard.z = 1;
        int max_iterations = params.max_iterations;

        if (texture_objects_cuda==nullptr)
            printf("Error-1: texture_objects_cuda\n");
        if (plane_hypotheses_cuda==nullptr)
            printf("Error-2: plane_hypotheses_cuda\n");

        // 随机初始化：随机视差、根据随机视差计算了代价
        RandomInitializationForMI<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda,
            plane_hypotheses_cuda, centers_cuda, costs_cuda, rand_states_cuda, selected_views_cuda,
            prior_planes_cuda, plane_masks_cuda, params, width, height);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // 传播：代价更新
        for (int i = 0; i < max_iterations*0.5; ++i)
        {
            BlackPixelUpdate_MIPM_Edgeaware<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda,
                neighbor_patchFill_cuda, params, i, width, height, tilekeys_cuda);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                std::cerr<<"Sync error: "<<cudaGetErrorString(syncErr) << std::endl;
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            RedPixelUpdate_MIPM_Edgeaware<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda,
                neighbor_patchFill_cuda, params, i, width, height, tilekeys_cuda);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //printf("iteration: %d\n", ia);
        }

        // 传播：代价更新 (候补像素)
        for (int i = 0; i < max_iterations*0.5; ++i)
        {
            //printf("candidate iteration: %d\n", i);
            BlackPixelUpdate_MIPM_Edgeaware_FillPatch<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda,
                neighbor_patchFill_cuda, params, i, width, height, tilekeys_cuda);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                std::cerr<<"Sync error: "<<cudaGetErrorString(syncErr) << std::endl;
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            RedPixelUpdate_MIPM_Edgeaware_FillPatch<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda, costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda,
                neighbor_patchFill_cuda, params, i, width, height, tilekeys_cuda);
             CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //printf("iteration: %d\n", ia);
        }

        // 回传：GPU--->CPU
        cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(neighbor_patchFill_host, neighbor_patchFill_cuda, sizeof(int3) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(disp_baseline_host, disp_baseline_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(selected_views_host, selected_views_cuda, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
}
