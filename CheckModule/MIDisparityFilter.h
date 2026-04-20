/********************************************************************
file base:      MIDisparityFilter.h
author:         LZD
created:        2026/04/10
purpose:        基于闭环一致性与局部连续性的视差噪点剔除，
                并在 FilterGlobal() 后输出整张光场影像级别的重投影误差图
*********************************************************************/
#ifndef MIDISPARITYFILTER_H
#define MIDISPARITYFILTER_H

#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

#include "Common/CommonUtil.h"
#include "MVStereo/LFDepthInfo.h"

namespace LFMVS
{
    struct MIDisparityFilterConfig
    {
        int   max_triplet = 4;              // 每个像素最多检查多少个已选邻域视图
        float min_valid_disp = 0.0f;        // <= 该值视为无效

        bool  use_selected_views_only = true;
        bool  clear_selected_views_when_invalid = true;

        // 1) 匹配代价约束
        bool  enable_cost_filter = true;
        float max_cost = 1.8f;

        // 2) 基于闭环检查的跨视图一致性
        bool  enable_cycle_geo_filter = true;
        float max_geo_err_px = 0.5f;

        bool  enable_cycle_photo_filter = true;
        float max_photo_err_u = 0.5f;

        int   min_good_neighbors = 2;       // 至少有 K 个邻域同时通过 geo/photo 约束

        // 3) 局部椒盐/毛刺去除（同一微图像内部）
        bool  enable_spike_filter = true;
        float spike_abs_diff = 1.0f;        // 与 8 邻域中值相差过大则剔除
        int   spike_min_neighbors = 3;      // 至少收集到这么多有效邻域，才做毛刺判断

        bool  dump_debug_mask = true;

        // 4) 整张光场影像级别的一致性误差图（重投影误差）
        // 说明：这里不是 GT 真误差，而是 FilterGlobal() 中闭环几何重投影误差的整图输出。
        bool  dump_disp_error_map = true;
        float disp_error_thresh_px  = 0.5f; // 色条阈值标线 / pass mask 阈值
        float disp_error_vis_max_px = 2.0f; // 整图只保留 <= 该值的误差；同时也是热图色条上限
    };

    struct MIDisparityFilterStats
    {
        int64_t total_pixels = 0;
        int64_t initially_valid_pixels = 0;
        int64_t kept_pixels = 0;
        int64_t removed_pixels = 0;

        int64_t removed_by_cost = 0;
        int64_t removed_by_no_support = 0;
        int64_t removed_by_cycle = 0;
        int64_t removed_by_spike = 0;

        // 整图重投影误差统计（仅统计最终保留且 <= disp_error_vis_max_px 的像素）
        int64_t num_disp_error_samples = 0;
        int64_t num_disp_error_le_thresh = 0;
        double  mean_disp_error_px = 0.0;
        double  rmse_disp_error_px = 0.0;
        double  p90_disp_error_px  = 0.0;
    };

    class MIDisparityFilterCPU
    {
    public:
        MIDisparityFilterCPU(int mi_w, int mi_h, float baseline_unit);

        void FilterGlobal(
            const std::string& frameName,
            const QuadTreeTileInfoMap& MLA_info_map,
            const QuadTreeProblemMap& problems_map,
            QuadTreeDisNormalMap& dis_normals,
            const MIDisparityFilterConfig& cfg,
            MIDisparityFilterStats& out_stats,
            const std::string& save_root = "") const;

    private:
        struct TileInputs
        {
            cv::Mat ref_gray_f32;
            std::vector<cv::Mat> neigh_gray_f32;
            std::vector<cv::Point2f> baselines_px;
            std::vector<QuadTreeTileKeyPtr> neigh_keys;
        };

        enum PixelRejectReason : unsigned char
        {
            PR_Keep       = 0,
            PR_Cost       = 80,
            PR_NoSupport  = 140,
            PR_Cycle      = 200,
            PR_Spike      = 255
        };

        bool BuildTileInputsFromMaps(
            const QuadTreeTileKeyPtr& key,
            const QuadTreeTileInfoMap& MLA_info_map,
            const QuadTreeProblemMap& problems_map,
            TileInputs& out) const;

        void FilterTile(
            const std::string& frameName,
            const QuadTreeTileKeyPtr& key,
            const TileInputs& in,
            QuadTreeDisNormalMap& dis_normals,
            const DisparityAndNormalPtr& ptrDN,
            const MIDisparityFilterConfig& cfg,
            MIDisparityFilterStats& io_stats,
            const std::string& save_root) const;

        void ComputeTileReprojectionErrorMap(
            const TileInputs& in,
            const QuadTreeDisNormalMap& dis_normals,
            const DisparityAndNormalPtr& ptrDN,
            const MIDisparityFilterConfig& cfg,
            cv::Mat& out_err_map) const;

        void DumpWholeFrameReprojectionError(
            const std::string& frameName,
            const QuadTreeTileInfoMap& MLA_info_map,
            const QuadTreeProblemMap& problems_map,
            const QuadTreeDisNormalMap& dis_normals,
            const MIDisparityFilterConfig& cfg,
            MIDisparityFilterStats& io_stats,
            const std::string& save_root) const;

        static void EnumerateSetBits(unsigned int mask, std::vector<int>& out_ids, int max_needed);
        static bool BilinearSampleF32(const cv::Mat& img, float x, float y, float& v);
        static float QuantileFromSorted(const std::vector<float>& sorted_vals, float q);

        float BilinearDisparity(const DisparityAndNormalPtr& dn, float x, float y, float min_valid_disp) const;
        bool PatchMeanStd3x3(const cv::Mat& img, float fx, float fy, float& mean, float& stddev) const;
        float ComputePhotoResidual(const cv::Mat& ref_img, const cv::Mat& nei_img,
                                   float px, float py, float qx, float qy) const;
        void InvalidatePixel(const DisparityAndNormalPtr& ptrDN, int idx, bool clear_selected_views) const;
        void DumpMask(const std::string& path, const cv::Mat& reason_mask) const;

        static void DumpDispErrorGrayPNG(const std::string& path,
                                         const cv::Mat& err_f32,
                                         float vis_max_px);
        static void DumpDispErrorPassMask(const std::string& path,
                                          const cv::Mat& err_f32,
                                          float thresh_px);
        static void DumpDispErrorOverlay(const std::string& path,
                                         const cv::Mat& err_f32,
                                         float thresh_px);
        static void DumpDispErrorHeatMapFixedWithThreshold(const std::string& path,
                                                           const cv::Mat& err_f32,
                                                           float vmax_px,
                                                           float thresh_px,
                                                           const std::string& unit_text = "px");

    private:
        const int   W_;
        const int   H_;
        const float baseline_unit_;
    };
}

#endif // MIDISPARITYFILTER_H
