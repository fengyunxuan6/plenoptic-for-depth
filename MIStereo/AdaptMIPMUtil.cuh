/********************************************************************
file base:      AdaptMIPMUtil.cuh
author:         LZD
created:        2025/06/26
purpose:
*********************************************************************/
#ifndef ADAPTMIPMUTIL_CUH
#define ADAPTMIPMUTIL_CUH

#include "Common/Common.h"
#include "Common/CommonCUDA.h"
#include "MVStereo/LFDepthInfo.h"

namespace LFMVS
{
    // 为“本参考微图像的一次调用”设置邻域小表（Hex 路径在 device 端查表，不改 kernel 签名）
void SetHexPerCallPTot(int*  d_nei_lin,   // [M]  本参考所有邻域的线性下标（y*W + x）
                       float* d_pTot,     // [M]  对应 p_Δ^{tot}（像素）
                       int*   d_kSteps,   // [M]  对应 k_Δ（六边形步数）
                       int    M,          // 邻域个数
                       int    ref_lin);   // 本参考微图像的线性下标（y*W + x）
    __device__ float ComputeVirtualDepthConsiderCostAndBaseline(const float4* disp_baseline_vector,
                                                            const PatchMatchParamsLF& params,
                                                            const float* baseline_norm,
                                                            const float* cost_vector);
    __device__ void ComputeBaselineNorm(float2* pcenters, const PatchMatchParamsLF& params, float* baseline_norm);

    __device__ float4 GenerateRandomNormal_MIPM( const int2 p, curandState *rand_state, const float depth,const float depth_min, const float depth_max);
    __device__ float4 GenerateRandomPlaneHypothesis_MIPM(const int2 p, curandState *rand_state, const float depth_min, const float depth_max);
    __device__ float ComputeMultiViewInitialCostandSelectedViews_MIPM(const cudaTextureObjects* texture_objects,
    float2* pcenters, const int2 p, const float4 plane_hypothesis,
    unsigned int* selected_views, const PatchMatchParamsLF params);

    __device__ float4 FindMinCostWithDispBaseline(float* min_cost_array, float4* disp_baseline_array, const PatchMatchParamsLF params);
    __device__ float2 ComputeAverageVirtualDepth(float2* disp_baseline_array, const PatchMatchParamsLF params);

    __device__ void PlaneHypothesisRefinement_MIPM(const cudaTextureObject_t* images,
                        const cudaTextureObject_t* blur_images,
                        const cudaTextureObject_t* depth_images, float4* plane_hypothesis,
                        float2 * pcenters, float *depth, float* cost, float4* disp_baseline,
                        curandState* rand_state, const float* view_weights, const float weight_norm,
                        float4* prior_planes, unsigned int* plane_masks,
                        float* restricted_cost, const int2 p,
                        const PatchMatchParamsLF params, const int WIDTH,
                        const float* baseline_norm);
    __device__ void PlaneHypothesisRefinement_MIPM_Hex(const cudaTextureObject_t* images,
                    const cudaTextureObject_t* blur_images,
                    const cudaTextureObject_t* depth_images, float4* plane_hypothesis,
                    float2 * pcenters, float *depth, float* cost, float4* disp_baseline,
                    curandState* rand_state, const float* view_weights, const float weight_norm,
                    float4* prior_planes, unsigned int* plane_masks,
                    float* restricted_cost, const int2 p,
                    const PatchMatchParamsLF params, const int WIDTH,
                    const float* baseline_norm, int2* tilekeys_cuda);

    __device__ void DisparityGeometricMapOperate_Hex_TY(float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline, int2 tilekey_ref, int2 tilekey_neig);
    __device__ void DisparityGeometricMapOperate_Hex_error(float2 c0, float2 c1, const int2 p_for_plane,
        float4 plane_hypothesis, const int2 p, const PatchMatchParamsLF params,
        float2& p1, float4& disparity_basline, int2 tilekey_ref, int2 tilekey_neig);
    __device__ void DisparityGeometricMapOperate_Hex(float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline, int2 tilekey_ref, int2 tilekey_neig);
    __device__ void DisparityGeometricMapOperate_Hex_TimeToken(float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline, int2 tilekey_ref, int2 tilekey_neig);

    __device__ void DisparityGeometricMapOperate(float2 c0, float2 c1, const int2 p_for_plane,
    float4 plane_hypothesis, const int2 p, const PatchMatchParamsLF params,
    float2& p1, float4& disparity_basline);

    __device__ float ComputeBilateralNCC_MIPM(const cudaTextureObject_t ref_image,
                const cudaTextureObject_t ref_blur_image, float2 p0,
                const cudaTextureObject_t src_image, float2 p1,
                const cudaTextureObject_t src_blur_image, const int2 p,
                float4 plane_hypothesis, const PatchMatchParamsLF params,
                float2& blur_value, float4& disparity_baseline);
    __device__ float ComputeBilateralNCC_MIPM_Hex(const cudaTextureObject_t ref_image,
        const cudaTextureObject_t ref_blur_image, float2 p0,
        const cudaTextureObject_t src_image, float2 p1,
        const cudaTextureObject_t src_blur_image, const int2 p,
        float4 plane_hypothesis, const PatchMatchParamsLF params,
        float2& blur_value, float4& disparity_baseline,
        int2 tilekey_ref, int2 tilekey_neig);

    // 计算像素p的多邻域视图代价
    __device__ void ComputeMultiViewCostVector_MIPM(const cudaTextureObject_t* images,
                        const cudaTextureObject_t* blur_images, float2* pcenters,
                        const int2 p, float4 plane_hypothesis,
                        float* cost_vector, float2* blure_array,
                        float4* disp_baseline_array, const PatchMatchParamsLF params);

    __device__ void ComputeMultiViewCostVector_MIPM_Hex(const cudaTextureObject_t* images,
                    const cudaTextureObject_t* blur_images, float2* pcenters,
                    const int2 p, float4 plane_hypothesis,
                    float* cost_vector, float2* blure_array,
                    float4* disp_baseline_array, const PatchMatchParamsLF params, int2* tilekeys_cuda);

    // 传播
    __device__ void CheckerboardPropagation_MIPM(const cudaTextureObjects* texture_objects,
                const cudaTextureObject_t* depths, float4* plane_hypotheses,
                float2* pcenters,float* costs, float4* disp_baseline, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes,
                unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
                const int iter, const int WIDTH, const int HEIGHT);
    __device__ void CheckerboardPropagation_MIPM_ConsiderBorderPixels(const cudaTextureObjects* texture_objects,
                const cudaTextureObject_t* depths, float4* plane_hypotheses,
                float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes,
                unsigned int* plane_masks, const int2 p, const PatchMatchParamsLF params,
                const int iter, const int WIDTH, const int HEIGHT);

    // 传播
    __global__ void BlackPixelUpdate_MIPM(cudaTextureObjects* texture_objects,
            cudaTextureObjects* texture_depths, float4* plane_hypotheses, float2* pcenters,
            float* costs, float4* disp_baseline, curandState* rand_states,
            unsigned int* selected_views, float4* prior_planes,
            unsigned int* plane_masks, const PatchMatchParamsLF params,
            const int iter, const int width, const int height);

    __global__ void RedPixelUpdate_MIPM(cudaTextureObjects *texture_objects,
                cudaTextureObjects* texture_depths, float4* plane_hypotheses,
                float2* pcenters, float* costs, float4* disp_baseline, curandState* rand_states,
                unsigned int* selected_views, float4* prior_planes, unsigned int* plane_masks,
                const PatchMatchParamsLF params, const int iter,
                const int width, const int height);

    __global__ void RandomInitializationForMI(cudaTextureObjects* texture_objects,
            float4* plane_hypotheses, float2* pcenters, float* costs,
            curandState* rand_states, unsigned int* selected_views,
            float4* prior_planes, unsigned int* plane_masks, const PatchMatchParamsLF params,
            const int width, const int height);

}

// ==================== Hex Grid Ring Utilities (host/device) ====================
// Embedded from user-provided C++ implementation, adapted for CUDA (__host__ __device__).
// Compute the ring index k in a pointy-topped hex grid given pixel coords and base spacing.
// All functions are inline and usable both on host and device.
#include <math.h>

namespace LFMVS {
    namespace HexGrid {

    struct Cube {
        double q, r, s;
        __host__ __device__ Cube(double q_=0.0, double r_=0.0, double s_=0.0) : q(q_), r(r_), s(s_) {}
    };

    struct IntCube {
        int q, r, s;
        __host__ __device__ IntCube(int q_=0, int r_=0, int s_=0) : q(q_), r(r_), s(s_) {}
    };

    // Round fractional cube to nearest hex lattice point
    __host__ __device__ inline IntCube cube_round(const Cube& frac) {
        int qi = (int) ::llround(frac.q);
        int ri = (int) ::llround(frac.r);
        int si = (int) ::llround(frac.s);

        double diff_q = fabs((double)qi - frac.q);
        double diff_r = fabs((double)ri - frac.r);
        double diff_s = fabs((double)si - frac.s);

        if (diff_q > diff_r && diff_q > diff_s) {
            qi = -ri - si;
        } else if (diff_r > diff_s) {
            ri = -qi - si;
        } else {
            si = -qi - ri;
        }
        return IntCube(qi, ri, si);
    }

    // Convert pixel (x,y) to cube coords for pointy-topped hex layout
    __host__ __device__ inline Cube pixel_to_cube(double x, double y, double base) {
        // base: center-to-center spacing between adjacent micro-images in pixels
        double q = (2.0 / 3.0 * x) / base;
        double r = ((-1.0 / 3.0 * x) + (1.7320508075688772 / 3.0 * y)) / base; // sqrt(3) ~ 1.732...
        double s = -q - r;
        return Cube(q, r, s);
    }

    // Hex distance (Manhattan distance in cube space / 2)
    __host__ __device__ inline int hex_distance(const IntCube& a, const IntCube& b) {
        return (abs(a.q - b.q) + abs(a.r - b.r) + abs(a.s - b.s)) / 2;
    }

    __host__ __device__ inline int hex_steps_k(int dq,int dr){
        int x = dq;
        int z = dr;
        int y = -(x + z);
        int a = abs(x), b = abs(y), c = abs(z);
        int max_ab = a > b ? a : b;
        return max_ab > c ? max_ab : c;
    }

    // Public API: ring index k relative to a reference center
    __host__ __device__ inline int get_ring_number(double x, double y, double base, double ref_x, double ref_y) {
        Cube c_ref = pixel_to_cube(ref_x, ref_y, base);
        IntCube hex_ref = cube_round(c_ref);

        Cube c_tgt = pixel_to_cube(x, y, base);
        IntCube hex_tgt = cube_round(c_tgt);

        return hex_distance(hex_ref, hex_tgt);
    }

    // Convenience overloads for float inputs
    __host__ __device__ inline int get_ring_number(float x, float y, float base, float ref_x, float ref_y) {
        return get_ring_number((double)x, (double)y, (double)base, (double)ref_x, (double)ref_y);
    }

    // Compute ring number directly from pcenters buffer indices (device/host)
    __host__ __device__ inline int ring_from_centers(const float2* pcenters, int idx, int ref_idx, float base) {
        const float2 p  = pcenters[idx];
        const float2 pr = pcenters[ref_idx];
        return get_ring_number(p.x, p.y, base, pr.x, pr.y);
    }

}} // namespace LFMVS::HexGrid
// ==============================================================================
#include <math.h>   // floor, ceil, fabs
#ifdef __CUDACC__
#  define LF_HD __host__ __device__
#  define LF_FORCE_INLINE __forceinline__
#else
#  define LF_HD
#  define LF_FORCE_INLINE inline
#endif

namespace LFMVS
{
    namespace lfhex
    {
        // ---- small helpers ----
        LF_HD LF_FORCE_INLINE int iabs_int(int v) { return (v < 0) ? -v : v; }

        template <typename T>
        LF_HD LF_FORCE_INLINE int round_to_int(T x)
        {
            // half-away-from-zero rounding
            return (int)((x >= (T)0) ? ::floor(x + (T)0.5) : ::ceil(x - (T)0.5));
        }

        template <typename T>
        LF_HD LF_FORCE_INLINE T to_circumradius(T radius_px, int radius_is_apothem)
        {
            // size = apothem * 2 / sqrt(3)
            const T SQRT3 = (T)1.7320508075688772935;
            return radius_is_apothem ? (radius_px * (T)2 / SQRT3) : radius_px;
        }

        // ---- core: pixel delta -> ring index ----
        template <typename T>
        LF_HD LF_FORCE_INLINE int hex_ring_from_delta(T dx, T dy, T size_circumradius)
        {
            if (size_circumradius <= (T)0) return -1;

            const T inv = (T)1 / size_circumradius;
            const T SQRT3 = (T)1.7320508075688772935;

            // pixel -> axial (pointy-top)
            T q = ((SQRT3 / (T)3) * dx - ((T)1 / (T)3) * dy) * inv;
            T r = (((T)2 / (T)3) * dy) * inv;

            // axial -> cube
            T cx = q;
            T cz = r;
            T cy = -cx - cz;

            // cube round
            int rx = round_to_int(cx);
            int ry = round_to_int(cy);
            int rz = round_to_int(cz);

            T dxr = ::fabs((T)rx - cx);
            T dyr = ::fabs((T)ry - cy);
            T dzr = ::fabs((T)rz - cz);

            if (dxr > dyr && dxr > dzr) rx = -ry - rz;
            else if (dyr > dzr) ry = -rx - rz;
            else rz = -rx - ry;

            // distance (ring index)
            int dist = (iabs_int(rx) + iabs_int(ry) + iabs_int(rz)) / 2;
            return dist;
        }

        // ---- convenience: two points + size ----
        template <typename T>
        LF_HD LF_FORCE_INLINE int hex_ring_from_points(T px, T py, T refx, T refy, T size_circumradius)
        {
            return hex_ring_from_delta<T>(px - refx, py - refy, size_circumradius);
        }

        // ---- convenience: two points + radius kind (circumradius/apothem) ----
        template <typename T>
        LF_HD LF_FORCE_INLINE int hex_ring_from_points_with_radius_kind(T px, T py,
                                                                        T refx, T refy,
                                                                        T radius_px,
                                                                        int radius_is_apothem)
        {
            T size = to_circumradius<T>(radius_px, radius_is_apothem);
            return hex_ring_from_points<T>(px, py, refx, refy, size);
        }
    } // namespace lfhex
}
#endif //ADAPTMIPMUTIL_CUH
