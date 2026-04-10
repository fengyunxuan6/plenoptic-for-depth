/********************************************************************
file base:      MICycleCheck.h
author:         LZD (merged by ChatGPT)
created:        2025/08/13
purpose:        闭环一致性检查 + 随机匹配可视化 + RMSE + 直方图 +
               固定色条热图 + 纹理掩膜(Masked) + 梯度分桶曲线
*********************************************************************/
#ifndef MICYCLECHECK_H
#define MICYCLECHECK_H

// MICycleCheck.h (complete, pair-panel reuse)
#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>
#include <cstdint>  // for int64_t
#include <random>
#include "Common/CommonUtil.h"
#include "MVStereo/LFDepthInfo.h"

namespace LFMVS {

struct MICycleCheckStats {
    double mean_geo_err_px   = 0.0;
    double median_geo_err_px = 0.0;  // approx
    double p90_geo_err_px    = 0.0;  // approx
    double mean_photo_err    = 0.0;
    int    num_samples       = 0;
    double rmse_geo_err_px   = 0.0;  // sqrt(sum e_g^2 / N)
    double rmse_photo_err    = 0.0;  // sqrt(sum e_p^2 / N)

    // ==== Completeness (mask & K-pass coverage) ====
    double  completeness               = 0.0; // good_pixels / masked_total
    double  completeness_geo           = 0.0; // good_pixels / masked_total
    double  completeness_photo         = 0.0; // good_pixels / masked_total
    int64_t completeness_masked_pixels = 0;   // denominator: pixels passed patch_mean_std()
    int64_t completeness_good_pixels   = 0;   // numerator: pixels with >=K neighbors passing thresholds
    int64_t completeness_good_pixels_geo   = 0;   // numerator: pixels with >=K neighbors passing thresholds
    int64_t completeness_good_pixels_photo   = 0;   // numerator: pixels with >=K neighbors passing thresholds

};

struct MICycleMaskParams {
    double sobel_thresh = 8.0;
    double var_thresh   = 60.0;
    bool   use_and      = true;
    int    grad_hist_bins = 50;
    double grad_hist_max  = 50.0;
};


// ===== Runtime config for clamping, skipping and per-pixel K-neighbor gating =====
struct MICycleClampConfig
{
    // Photometric clamping (phe)
    bool   clamp_photo = true;
    double photo_u     = 3.0;   // z-score or abs fallback 光度误差的阈值上限参数

    // Geometric clamping (geo_e)
    bool   clamp_geo   = false;
    double geo_u_px    = 2.0;   // 几何误差的阈值上限参数

    // Neighbor-level skipping (raw thresholds) -- optional legacy
    bool   skip_on_geo   = false;
    double skip_geo_u_px = 8.0; // pixels
    bool   skip_on_photo = false;
    double skip_photo_u  = 10.0;

    // Pixel-level K-neighbor gating (count pixel only if >=K neighbors pass thresholds)
    bool   pixel_gate_enabled       = false;
    int    gate_min_good_neighbors  = 3;     // K
    double gate_geo_px              = 2.0;   // pixels
    double gate_photo_u             = 3.0;   // z-score or abs fallback
    bool   gate_use_clamped         = false; // false: use raw (geo_e/phe_raw) for gating; true: use clamped values
};

class MICycleCheckerCPU
{
public:
    MICycleCheckerCPU(int mi_w, int mi_h, float baseline_unit);

    // Layer A：闭环一致性 + 指标 + 可视化（热图/直方图/纹理分桶）
    void CheckGlobal(
        const std::string& frameName,
        const QuadTreeTileInfoMap& MLA_info_map,
        QuadTreeProblemMap&  problems_map,
        QuadTreeDisNormalMap& dis_normals,
        MICycleCheckStats& out_stats,
        int max_triplet = 4,
        bool dump_heatmap = true,
        const std::string& save_root = "CycleCheck",
        bool dump_hist = true,
        int  hist_bins = 60,
        double hist_geo_max_x = 2.0,
        double vis_geo_max_px = 2.0,
        double vis_photo_max  = 3.0,  // photometric residual (z-score) clamp
        const MICycleClampConfig& ccfg = MICycleClampConfig()
        ) const;

    // 随机匹配可视化（含跨图连线；严格复用上一轮随机样本）
    void VisualizeRandomMatchesGlobal(
        const std::string& frameName,
        const QuadTreeTileInfoMap& MLA_info_map,
        const QuadTreeProblemMap&  problems_map,
        const QuadTreeDisNormalMap& dis_normals,
        int tiles_to_pick = 8,
        int points_per_tile = 60,
        int max_triplet = 4,
        const std::string& save_root = "CycleCheck/Matches",
        unsigned int rng_seed = 12345,
        bool draw_lines = true
    ) const;

private:
    const int   W_, H_;
    const float m_baseline_unit; // = params.Base（像素）

    struct TileInputs
    {
        cv::Mat ref_gray_f32;
        std::vector<cv::Mat>       neigh_gray_f32;
        std::vector<cv::Point2f>   baselines_px; // c_nei - c_ref (像素)
        std::vector<QuadTreeTileKeyPtr> neigh_keys;
    };

    bool BuildTileInputsFromMaps(
        const QuadTreeTileKeyPtr& key,
        const QuadTreeTileInfoMap& MLA_info_map,
        const QuadTreeProblemMap&  problems_map,
        TileInputs& out) const;

    void CheckTile(
        const std::string& frameName,
        QuadTreeTileKeyPtr& key,
        QuadTreeProblemMap&  problems_map,
        QuadTreeDisNormalMap& dis_normals,
        const TileInputs& in,
        const DisparityAndNormalPtr& ptrDN,
        MICycleCheckStats& io_stats,
        int max_triplet,
        bool dump_heatmap,
        const std::string& save_root,
        bool dump_hist,
        int  hist_bins,
        double hist_geo_max_x,
        double vis_geo_max_px,
        double vis_photo_max,
        const MICycleClampConfig& ccfg = MICycleClampConfig()) const;

    // 工具
    static void   EnumerateSetBits(unsigned int mask, std::vector<int>& out_ids, int max_needed);
    static bool   BilinearSampleF32(const cv::Mat& img, float x, float y, float& v);
    static double Mean(const std::vector<float>& v);
    static double Percentile(std::vector<float>& v, double p);

    static void   DumpHeatMap(const std::string& path, const cv::Mat& f32map);
    static void   DumpHeatMapFixed(const std::string& path, const cv::Mat& f32map,
                                   double vmax, const std::string& unit_text);
    static cv::Mat RenderHistogramImage(const std::vector<float>& data, int bins, double max_x,
                                        int img_w=800, int img_h=300);
    static cv::Mat RenderBucketCurve(const std::vector<double>& edges,
                                     const std::vector<double>& vals,
                                     const std::string& title,
                                     int img_w=800, int img_h=300);
    static void BuildTextureMaskAndViz(const cv::Mat& gray_f32, const MICycleMaskParams& mcfg,
                                       cv::Mat& grad_mag, cv::Mat& var7x7, cv::Mat& mask_good,
                                       std::string save_dir);
    static void DumpHeatMapFixedMasked(const std::string& path, const cv::Mat& f32map,
                                       const cv::Mat& mask_u8, double vmax, const std::string& unit_text);
    static void SaveFloatAsPNG(const std::string& path, const cv::Mat& f32);

private:
    void VisualizeRandomMatchesTile(
        const std::string& frameName,
        const QuadTreeTileKeyPtr& key,
        const TileInputs& in,
        const QuadTreeProblemMap& problems_map,
        const QuadTreeDisNormalMap& dis_normals,
        const DisparityAndNormalPtr& refDN,
        int points_per_tile,
        int max_triplet,
        const std::string& save_root,
        std::mt19937& rng,
        bool draw_lines
    ) const;
};
} // namespace LFMVS
#endif // MICYCLECHECK_H
