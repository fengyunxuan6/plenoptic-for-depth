/********************************************************************
file base:      CommonCUDA.h
author:         LZD
created:        2024/06/19
purpose:
*********************************************************************/
#ifndef _COMMONCUDA_H_
#define _COMMONCUDA_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

#include "Common/Common.h"

namespace LFMVS
{
    // 读写
    int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
    int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
    int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
    int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

    Camera ReadCamera(const std::string &cam_path);
    void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> & src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);

    float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
    float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera);

    void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);

    float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
    void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);

    #define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
    #define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)
    void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
    void CudaCheckError(const char* file, const int line);

    struct cudaTextureObjects
    {
        cudaTextureObject_t images[MAX_IMAGES];
        cudaTextureObject_t blur_images[MAX_IMAGES];
    };

    struct PatchMatchParams
    {
        int         max_iterations = 3;
        int         patch_size = 11;
        int         num_images = 5;
        int         max_image_size=3200;
        int         radius_increment = 2;
        float       sigma_spatial = 5.0f;
        float       sigma_color = 3.0f;
        int         top_k = 4;
        float       baseline = 0.54f;
        float       depth_min = 0.0f;
        float       depth_max = 1.0f;
        float       disparity_min = 0.0f;
        float       disparity_max = 1.0f;
        bool        geom_consistency = false;
        bool        multi_geometry = false;
        bool        planar_prior = false;
    };
}
#endif // _COMMONCUDA_H_