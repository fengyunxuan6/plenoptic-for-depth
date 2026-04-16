/********************************************************************
file base:      AdaptMIPMPFPGR.cu
author:         LZD
created:        2025/07/12
purpose:
*********************************************************************/
#include "AdaptMIPMPFPGR.h"
#include "AdaptMIPM.h"

#include <future>

#include "CudaUtil.h"
#include "AdaptMIPMUtil.cuh"
#include "AdaaptMIPM_EdgewareUtil.cuh"
#include "AdaptMIPMPFPGRUtil.cuh"

namespace LFMVS
{
    void AdaptMIPMPFPGR::RunPatchMatchCUDAForMI_PFPGR_Collect()
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
        for (int i = 0; i < max_iterations; ++i)
        {
            BlackPixelUpdate_MIPM_PFPGR_Collect<<<grid_size_checkerboard, block_size_checkerboard>>>(
                texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda,
                costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda,
                prior_planes_cuda, plane_masks_cuda,
                neighbor_patchFill_cuda,
                neighbor_PGR_cuda,
                params, i, width, height, tilekeys_cuda);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                std::cerr<<"Sync error: "<<cudaGetErrorString(syncErr) << std::endl;
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            RedPixelUpdate_MIPM_PFPGR_Collect<<<grid_size_checkerboard, block_size_checkerboard>>>(
                texture_objects_cuda,
                texture_depths_cuda, plane_hypotheses_cuda, centers_cuda,
                costs_cuda, disp_baseline_cuda,
                rand_states_cuda, selected_views_cuda,
                prior_planes_cuda, plane_masks_cuda,
                neighbor_patchFill_cuda,
                neighbor_PGR_cuda,
                params, i, width, height, tilekeys_cuda);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //printf("iteration: %d\n", ia);
        }

        // 回传：GPU--->CPU
        cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(rand_states_host, rand_states_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(neighbor_patchFill_host, neighbor_patchFill_cuda, sizeof(int3) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(neighbor_PGR_host, neighbor_PGR_cuda, sizeof(int3) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(disp_baseline_host, disp_baseline_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(selected_views_host, selected_views_cuda, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void AdaptMIPMPFPGR::RunPatchMatchCUDAForMI_PFPGR_Repair()
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

        // 传播：代价更新
        for (int i = 0; i < max_iterations; ++i)
        {
            BlackPixelUpdate_PFPGR_Repair<<<grid_size_checkerboard, block_size_checkerboard>>>(
                texture_objects_cuda,
                texture_depths_cuda,
                plane_hypotheses_cuda,
                centers_cuda,
                costs_cuda,
                disp_baseline_cuda,
                rand_states_cuda,
                selected_views_cuda,
                prior_planes_cuda,
                plane_masks_cuda,
                neighbor_patchFill_cuda,
                proxy_plane_cuda,
                params, i, width, height);
            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                std::cerr<<"Sync error: "<<cudaGetErrorString(syncErr) << std::endl;
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            RedPixelUpdate_PFPGR_Repair<<<grid_size_checkerboard, block_size_checkerboard>>>(
                texture_objects_cuda,
                texture_depths_cuda,
                plane_hypotheses_cuda,
                centers_cuda,
                costs_cuda,
                disp_baseline_cuda,
                rand_states_cuda,
                selected_views_cuda,
                prior_planes_cuda,
                plane_masks_cuda,
                neighbor_patchFill_cuda,
                proxy_plane_cuda,
                params, i, width, height);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //printf("iteration: %d\n", ia);
        }

        // 回传：GPU--->CPU
        cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(neighbor_patchFill_host, neighbor_patchFill_cuda, sizeof(int3) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(neighbor_PGR_host, neighbor_PGR_cuda, sizeof(int3) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(disp_baseline_host, disp_baseline_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
        cudaMemcpy(selected_views_host, selected_views_cuda, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
}
