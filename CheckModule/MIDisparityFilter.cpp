/********************************************************************
file base:      MIDisparityFilter.cpp
author:         LZD
created:        2026/04/10
purpose:        基于闭环一致性与局部连续性的视差噪点剔除，
                并在 FilterGlobal() 后输出整张光场影像级别的重投影误差图
*********************************************************************/
#include "MIDisparityFilter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <boost/filesystem.hpp>

#include "Util/Logger.h"

namespace LFMVS
{
    MIDisparityFilterCPU::MIDisparityFilterCPU(int mi_w, int mi_h, float baseline_unit)
        : W_(mi_w), H_(mi_h), baseline_unit_(baseline_unit)
    {
    }

    void MIDisparityFilterCPU::EnumerateSetBits(unsigned int mask, std::vector<int>& out_ids, int max_needed)
    {
        out_ids.clear();
        for (int b = 0; mask && (int)out_ids.size() < max_needed; ++b)
        {
            if (mask & 1u)
                out_ids.push_back(b);
            mask >>= 1u;
        }
    }

    float MIDisparityFilterCPU::QuantileFromSorted(const std::vector<float>& sorted_vals, float q)
    {
        if (sorted_vals.empty())
            return 0.0f;
        if (sorted_vals.size() == 1)
            return sorted_vals[0];

        const float qq = std::max(0.0f, std::min(1.0f, q));
        const float pos = qq * static_cast<float>(sorted_vals.size() - 1);
        const int lo = static_cast<int>(std::floor(pos));
        const int hi = static_cast<int>(std::ceil(pos));
        const float t = pos - static_cast<float>(lo);
        if (lo == hi)
            return sorted_vals[lo];
        return sorted_vals[lo] * (1.0f - t) + sorted_vals[hi] * t;
    }

    bool MIDisparityFilterCPU::BilinearSampleF32(const cv::Mat& img, float x, float y, float& v)
    {
        if (img.empty() || img.type() != CV_32FC1)
            return false;
        const int W = img.cols;
        const int H = img.rows;
        if (x < 0 || y < 0 || x > W - 1 || y > H - 1)
            return false;

        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(x0 + 1, W - 1);
        const int y1 = std::min(y0 + 1, H - 1);
        const float ax = x - x0;
        const float ay = y - y0;

        const float v00 = img.at<float>(y0, x0);
        const float v01 = img.at<float>(y1, x0);
        const float v10 = img.at<float>(y0, x1);
        const float v11 = img.at<float>(y1, x1);
        v = (1.0f - ax) * (1.0f - ay) * v00 +
            (1.0f - ax) * ay * v01 +
            ax * (1.0f - ay) * v10 +
            ax * ay * v11;
        return true;
    }

    float MIDisparityFilterCPU::BilinearDisparity(const DisparityAndNormalPtr& dn,
                                                  float x, float y,
                                                  float min_valid_disp) const
    {
        if (!dn)
            return std::numeric_limits<float>::quiet_NaN();
        if (x < 0 || y < 0 || x > W_ - 1 || y > H_ - 1)
            return std::numeric_limits<float>::quiet_NaN();

        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(x0 + 1, W_ - 1);
        const int y1 = std::min(y0 + 1, H_ - 1);
        const float ax = x - x0;
        const float ay = y - y0;

        auto d_at = [&](int xi, int yi) -> float
        {
            const int idx = yi * W_ + xi;
            float d = dn->ph_cuda[idx].w;
            if (!std::isfinite(d))
                d = dn->d_cuda[idx];
            if (!std::isfinite(d) || d <= min_valid_disp)
                return std::numeric_limits<float>::quiet_NaN();
            return d;
        };

        const float d00 = d_at(x0, y0);
        const float d01 = d_at(x0, y1);
        const float d10 = d_at(x1, y0);
        const float d11 = d_at(x1, y1);
        if (!std::isfinite(d00) || !std::isfinite(d01) || !std::isfinite(d10) || !std::isfinite(d11))
            return std::numeric_limits<float>::quiet_NaN();

        return (1.0f - ax) * (1.0f - ay) * d00 +
               (1.0f - ax) * ay * d01 +
               ax * (1.0f - ay) * d10 +
               ax * ay * d11;
    }

    bool MIDisparityFilterCPU::PatchMeanStd3x3(const cv::Mat& img, float fx, float fy,
                                               float& mean, float& stddev) const
    {
        const int cx = static_cast<int>(std::round(fx));
        const int cy = static_cast<int>(std::round(fy));
        if (cx < 1 || cy < 1 || cx >= img.cols - 1 || cy >= img.rows - 1)
            return false;
        if (img.empty() || img.type() != CV_32FC1)
            return false;

        const cv::Rect roi(cx - 1, cy - 1, 3, 3);
        cv::Scalar m, s;
        cv::meanStdDev(img(roi), m, s);
        mean = static_cast<float>(m[0]);
        stddev = static_cast<float>(s[0]);
        return std::isfinite(mean) && std::isfinite(stddev) && stddev > 1.0e-3f;
    }

    float MIDisparityFilterCPU::ComputePhotoResidual(const cv::Mat& ref_img, const cv::Mat& nei_img,
                                                     float px, float py, float qx, float qy) const
    {
        float Ir = 0.0f, In = 0.0f;
        if (!BilinearSampleF32(ref_img, px, py, Ir))
            return std::numeric_limits<float>::quiet_NaN();
        if (!BilinearSampleF32(nei_img, qx, qy, In))
            return std::numeric_limits<float>::quiet_NaN();

        float mr, sr, mn, sn;
        if (PatchMeanStd3x3(ref_img, px, py, mr, sr) && PatchMeanStd3x3(nei_img, qx, qy, mn, sn))
        {
            const float Irn = (Ir - mr) / sr;
            const float Inn = (In - mn) / sn;
            return std::fabs(Irn - Inn);
        }

        return std::fabs(In - Ir);
    }

    void MIDisparityFilterCPU::InvalidatePixel(const DisparityAndNormalPtr& ptrDN,
                                               int idx,
                                               bool clear_selected_views) const
    {
        ptrDN->d_cuda[idx] = -1.0f;
        ptrDN->ph_cuda[idx].x = 0.0f;
        ptrDN->ph_cuda[idx].y = 0.0f;
        ptrDN->ph_cuda[idx].z = 0.0f;
        ptrDN->ph_cuda[idx].w = -1.0f;
        ptrDN->c_cuda[idx] = 2.0f;
        ptrDN->disp_v_cuda[idx].x = -1.0f;
        ptrDN->disp_v_cuda[idx].y = -1.0f;
        ptrDN->disp_v_cuda[idx].z = 0.0f;
        ptrDN->disp_v_cuda[idx].w = 0.0f;
        if (clear_selected_views)
            ptrDN->selected_views[idx] = 0u;
    }

    void MIDisparityFilterCPU::DumpMask(const std::string& path, const cv::Mat& reason_mask) const
    {
        if (path.empty() || reason_mask.empty())
            return;
        boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());

        cv::Mat color(reason_mask.rows, reason_mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < reason_mask.rows; ++y)
        {
            for (int x = 0; x < reason_mask.cols; ++x)
            {
                const unsigned char v = reason_mask.at<unsigned char>(y, x);
                cv::Vec3b c(0, 0, 0);
                if (v == PR_Keep)      c = cv::Vec3b(0, 180, 0);
                if (v == PR_Cost)      c = cv::Vec3b(0, 255, 255);
                if (v == PR_NoSupport) c = cv::Vec3b(255, 128, 0);
                if (v == PR_Cycle)     c = cv::Vec3b(0, 0, 255);
                if (v == PR_Spike)     c = cv::Vec3b(255, 0, 255);
                color.at<cv::Vec3b>(y, x) = c;
            }
        }
        cv::imwrite(path, color);
    }

    void MIDisparityFilterCPU::DumpDispErrorGrayPNG(const std::string& path,
                                                    const cv::Mat& err_f32,
                                                    float vis_max_px)
    {
        if (path.empty() || err_f32.empty() || err_f32.type() != CV_32FC1)
            return;
        if (vis_max_px <= 1.0e-6f)
            vis_max_px = 1.0f;

        boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());

        cv::Mat gray(err_f32.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < err_f32.rows; ++y)
        {
            const float* ep = err_f32.ptr<float>(y);
            unsigned char* gp = gray.ptr<unsigned char>(y);
            for (int x = 0; x < err_f32.cols; ++x)
            {
                if (!std::isfinite(ep[x]) || ep[x] < 0.0f)
                {
                    gp[x] = 0;
                    continue;
                }
                const float e = std::max(0.0f, std::min(ep[x], vis_max_px));
                const float u = e / vis_max_px;
                // 0 保留给 invalid，所以有效像素映射到 [1,255]
                gp[x] = static_cast<unsigned char>(std::max(1.0f, std::round(1.0f + u * 254.0f)));
            }
        }
        cv::imwrite(path, gray);
    }

    void MIDisparityFilterCPU::DumpDispErrorPassMask(const std::string& path,
                                                     const cv::Mat& err_f32,
                                                     float thresh_px)
    {
        if (path.empty() || err_f32.empty() || err_f32.type() != CV_32FC1)
            return;
        boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());

        cv::Mat mask(err_f32.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < err_f32.rows; ++y)
        {
            const float* ep = err_f32.ptr<float>(y);
            unsigned char* mp = mask.ptr<unsigned char>(y);
            for (int x = 0; x < err_f32.cols; ++x)
            {
                if (std::isfinite(ep[x]) && ep[x] >= 0.0f && ep[x] <= thresh_px)
                    mp[x] = 255;
            }
        }
        cv::imwrite(path, mask);
    }

    void MIDisparityFilterCPU::DumpDispErrorOverlay(const std::string& path,
                                                    const cv::Mat& err_f32,
                                                    float thresh_px)
    {
        if (path.empty() || err_f32.empty() || err_f32.type() != CV_32FC1)
            return;
        boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());

        cv::Mat vis(err_f32.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < err_f32.rows; ++y)
        {
            const float* ep = err_f32.ptr<float>(y);
            cv::Vec3b* vp = vis.ptr<cv::Vec3b>(y);
            for (int x = 0; x < err_f32.cols; ++x)
            {
                if (!std::isfinite(ep[x]) || ep[x] < 0.0f)
                    continue;
                if (ep[x] <= thresh_px)
                    vp[x] = cv::Vec3b(0, 255, 0);
                else
                    vp[x] = cv::Vec3b(0, 0, 255);
            }
        }
        cv::imwrite(path, vis);
    }

    void MIDisparityFilterCPU::DumpDispErrorHeatMapFixedWithThreshold(const std::string& path,
                                                                      const cv::Mat& err_f32,
                                                                      float vmax_px,
                                                                      float thresh_px,
                                                                      const std::string& unit_text)
    {
        if (path.empty() || err_f32.empty() || err_f32.type() != CV_32FC1)
            return;
        if (vmax_px <= 1.0e-6f)
            vmax_px = 1.0f;

        boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());

        cv::Mat img8(err_f32.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat valid_mask(err_f32.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < err_f32.rows; ++y)
        {
            const float* ep = err_f32.ptr<float>(y);
            unsigned char* ip = img8.ptr<unsigned char>(y);
            unsigned char* mp = valid_mask.ptr<unsigned char>(y);
            for (int x = 0; x < err_f32.cols; ++x)
            {
                if (!std::isfinite(ep[x]) || ep[x] < 0.0f)
                    continue;
                const float e = std::max(0.0f, std::min(ep[x], vmax_px));
                ip[x] = static_cast<unsigned char>(std::round(e / vmax_px * 255.0f));
                mp[x] = 255;
            }
        }

        cv::Mat color;
        cv::applyColorMap(img8, color, cv::COLORMAP_JET);
        for (int y = 0; y < color.rows; ++y)
        {
            const unsigned char* mp = valid_mask.ptr<unsigned char>(y);
            cv::Vec3b* cp = color.ptr<cv::Vec3b>(y);
            for (int x = 0; x < color.cols; ++x)
            {
                if (mp[x] == 0)
                    cp[x] = cv::Vec3b(0, 0, 0);
            }
        }

        const int bar_w = 60;
        cv::Mat canvas(color.rows, color.cols + bar_w, CV_8UC3, cv::Scalar(0, 0, 0));
        color.copyTo(canvas(cv::Rect(0, 0, color.cols, color.rows)));

        cv::Mat bar(color.rows, bar_w, CV_8UC3);
        for (int y = 0; y < color.rows; ++y)
        {
            const unsigned char v = static_cast<unsigned char>(std::round(255.0 * (1.0 - static_cast<double>(y) / std::max(1, color.rows - 1))));
            cv::Mat tmp(1, 1, CV_8U, cv::Scalar(v));
            cv::applyColorMap(tmp, tmp, cv::COLORMAP_JET);
            const cv::Vec3b c = tmp.at<cv::Vec3b>(0, 0);
            for (int x = 0; x < bar_w; ++x)
                bar.at<cv::Vec3b>(y, x) = c;
        }
        bar.copyTo(canvas(cv::Rect(color.cols, 0, bar_w, color.rows)));

        auto putTextSafe = [&](const std::string& s, int y)
        {
            cv::putText(canvas, s,
                        cv::Point(color.cols + 5, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        };

        putTextSafe("0", color.rows - 5);
        putTextSafe(cv::format("%.2f", vmax_px * 0.5f), color.rows / 2);
        putTextSafe(cv::format("%.2f %s", vmax_px, unit_text.c_str()), 12);

        const float t = std::max(0.0f, std::min(thresh_px, vmax_px));
        const int y_thresh = static_cast<int>(std::round((1.0f - t / vmax_px) * (color.rows - 1)));
        cv::line(canvas,
                 cv::Point(color.cols, y_thresh),
                 cv::Point(color.cols + bar_w - 1, y_thresh),
                 cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        putTextSafe(cv::format("T=%.2f", thresh_px), std::max(14, y_thresh - 4));

        cv::imwrite(path, canvas);
    }

    bool MIDisparityFilterCPU::BuildTileInputsFromMaps(
        const QuadTreeTileKeyPtr& key,
        const QuadTreeTileInfoMap& MLA_info_map,
        const QuadTreeProblemMap& problems_map,
        TileInputs& out) const
    {
        auto itProb = problems_map.find(key);
        if (itProb == problems_map.end())
            return false;
        const MLA_Problem& problem = itProb->second;
        if (problem.m_Image_gray.empty())
            return false;

        if (problem.m_Image_gray.type() != CV_32FC1)
            problem.m_Image_gray.convertTo(out.ref_gray_f32, CV_32FC1);
        else
            out.ref_gray_f32 = problem.m_Image_gray;

        auto itMLA = MLA_info_map.find(key);
        if (itMLA == MLA_info_map.end())
            return false;
        const cv::Point2f c_ref = itMLA->second->GetCenter();

        out.neigh_keys.clear();
        if (!problem.m_NeighsSortVecForMatch.empty())
            out.neigh_keys = problem.m_NeighsSortVecForMatch;
        else
        {
            for (size_t i = 0; i < problem.m_Res_Image_KeyVec.size(); ++i)
                out.neigh_keys.push_back(problem.m_Res_Image_KeyVec[i].m_ptrKey);
        }

        out.neigh_gray_f32.clear();
        out.baselines_px.clear();
        out.neigh_gray_f32.reserve(out.neigh_keys.size());
        out.baselines_px.reserve(out.neigh_keys.size());

        for (size_t i = 0; i < out.neigh_keys.size(); ++i)
        {
            const QuadTreeTileKeyPtr& nk = out.neigh_keys[i];
            cv::Mat g;
            auto itN = problems_map.find(nk);
            if (itN != problems_map.end())
            {
                if (itN->second.m_Image_gray.type() != CV_32FC1)
                    itN->second.m_Image_gray.convertTo(g, CV_32FC1);
                else
                    g = itN->second.m_Image_gray;
            }
            out.neigh_gray_f32.push_back(g);

            cv::Point2f B(0.0f, 0.0f);
            auto itMN = MLA_info_map.find(nk);
            if (itMN != MLA_info_map.end())
                B = c_ref - itMN->second->GetCenter();
            out.baselines_px.push_back(B);
        }

        return !out.ref_gray_f32.empty();
    }

    void MIDisparityFilterCPU::FilterTile(
        const std::string& frameName,
        const QuadTreeTileKeyPtr& key,
        const TileInputs& in,
        QuadTreeDisNormalMap& dis_normals,
        const DisparityAndNormalPtr& ptrDN,
        const MIDisparityFilterConfig& cfg,
        MIDisparityFilterStats& io_stats,
        const std::string& save_root) const
    {
        if (!ptrDN)
            return;

        const int N = W_ * H_;
        std::vector<unsigned char> keep_stage1(N, 0);
        std::vector<unsigned char> keep_final(N, 0);
        cv::Mat reason_mask(H_, W_, CV_8UC1, cv::Scalar(PR_Keep));

        auto read_disp = [&](int idx) -> float
        {
            float d = ptrDN->ph_cuda[idx].w;
            if (!std::isfinite(d))
                d = ptrDN->d_cuda[idx];
            return d;
        };

        auto read_cost = [&](const DisparityAndNormalPtr& dn, int idx) -> float
        {
            if (!dn)
                return std::numeric_limits<float>::quiet_NaN();
            if (idx < 0 || idx >= W_ * H_)
                return std::numeric_limits<float>::quiet_NaN();
            return dn->c_cuda[idx];
        };

        auto bilinear_cost = [&](const DisparityAndNormalPtr& dn, float x, float y) -> float
        {
            if (!dn)
                return std::numeric_limits<float>::quiet_NaN();
            if (x < 0 || y < 0 || x > W_ - 1 || y > H_ - 1)
                return std::numeric_limits<float>::quiet_NaN();

            const int x0 = static_cast<int>(std::floor(x));
            const int y0 = static_cast<int>(std::floor(y));
            const int x1 = std::min(x0 + 1, W_ - 1);
            const int y1 = std::min(y0 + 1, H_ - 1);
            const float ax = x - x0;
            const float ay = y - y0;

            const float c00 = read_cost(dn, y0 * W_ + x0);
            const float c01 = read_cost(dn, y1 * W_ + x0);
            const float c10 = read_cost(dn, y0 * W_ + x1);
            const float c11 = read_cost(dn, y1 * W_ + x1);
            if (!std::isfinite(c00) || !std::isfinite(c01) || !std::isfinite(c10) || !std::isfinite(c11))
                return std::numeric_limits<float>::quiet_NaN();

            return (1.0f - ax) * (1.0f - ay) * c00 +
                   (1.0f - ax) * ay * c01 +
                   ax * (1.0f - ay) * c10 +
                   ax * ay * c11;
        };

        for (int y = 0; y < H_; ++y)
        {
            for (int x = 0; x < W_; ++x)
            {
                const int idx = y * W_ + x;
                ++io_stats.total_pixels;

                const float d = read_disp(idx);
                if (!std::isfinite(d) || d <= cfg.min_valid_disp)
                    continue;
                ++io_stats.initially_valid_pixels;

                const float cost = read_cost(ptrDN, idx);
                if (cfg.enable_cost_filter)
                {
                    const bool invalid_cost = (!std::isfinite(cost) || cost < 0.0f);
                    const bool high_cost = (std::isfinite(cost) && cost > cfg.max_cost);
                    if (invalid_cost || high_cost)
                    {
                        reason_mask.at<unsigned char>(y, x) = PR_Cost;
                        ++io_stats.removed_by_cost;
                        continue;
                    }
                }

                std::vector<int> vids;
                if (cfg.use_selected_views_only)
                {
                    EnumerateSetBits(ptrDN->selected_views[idx], vids, cfg.max_triplet);
                }
                else
                {
                    const int lim = std::min(static_cast<int>(in.neigh_keys.size()), cfg.max_triplet);
                    vids.resize(lim);
                    for (int i = 0; i < lim; ++i)
                        vids[i] = i;
                }

                if (vids.empty())
                {
                    reason_mask.at<unsigned char>(y, x) = PR_NoSupport;
                    ++io_stats.removed_by_no_support;
                    continue;
                }

                const float px = static_cast<float>(x);
                const float py = static_cast<float>(y);
                int support_neighbors = 0;
                int good_neighbors = 0;

                for (size_t vi = 0; vi < vids.size(); ++vi)
                {
                    const int b = vids[vi];
                    if (b < 0 || b >= static_cast<int>(in.neigh_keys.size()) || b >= static_cast<int>(in.baselines_px.size()))
                        continue;
                    if (b >= static_cast<int>(in.neigh_gray_f32.size()) || in.neigh_gray_f32[b].empty())
                        continue;

                    const cv::Point2f& Bv = in.baselines_px[b];
                    const float baseline = std::hypot(Bv.x, Bv.y);
                    if (baseline < 1.0e-6f)
                        continue;

                    const float d_x = Bv.x / baseline;
                    const float d_y = Bv.y / baseline;
                    const float scale = baseline / baseline_unit_;

                    const float qx = px + d_x * d * scale;
                    const float qy = py + d_y * d * scale;
                    if (qx < 0 || qy < 0 || qx > W_ - 1 || qy > H_ - 1)
                        continue;

                    auto itN = dis_normals.find(in.neigh_keys[b]);
                    if (itN == dis_normals.end() || !itN->second)
                        continue;
                    const DisparityAndNormalPtr neiDN = itN->second;

                    const float d_nei = BilinearDisparity(neiDN, qx, qy, cfg.min_valid_disp);
                    if (!std::isfinite(d_nei))
                        continue;

                    if (cfg.enable_cost_filter)
                    {
                        const float nei_cost = bilinear_cost(neiDN, qx, qy);
                        const bool invalid_nei_cost = (!std::isfinite(nei_cost) || nei_cost < 0.0f);
                        const bool high_nei_cost = (std::isfinite(nei_cost) && nei_cost > cfg.max_cost);
                        if (invalid_nei_cost || high_nei_cost)
                            continue;
                    }

                    ++support_neighbors;

                    bool pass_geo = true;
                    bool pass_photo = true;

                    if (cfg.enable_cycle_geo_filter)
                    {
                        const float phx = qx - d_x * d_nei * scale;
                        const float phy = qy - d_y * d_nei * scale;
                        const float geo_e = std::hypot(phx - px, phy - py);
                        pass_geo = std::isfinite(geo_e) && geo_e <= cfg.max_geo_err_px;
                    }

                    if (cfg.enable_cycle_photo_filter)
                    {
                        const float photo_e = ComputePhotoResidual(in.ref_gray_f32, in.neigh_gray_f32[b], px, py, qx, qy);
                        pass_photo = std::isfinite(photo_e) && photo_e <= cfg.max_photo_err_u;
                    }

                    if (pass_geo && pass_photo)
                        ++good_neighbors;
                }

                if (support_neighbors < cfg.min_good_neighbors)
                {
                    reason_mask.at<unsigned char>(y, x) = PR_NoSupport;
                    ++io_stats.removed_by_no_support;
                    continue;
                }
                if (good_neighbors < cfg.min_good_neighbors)
                {
                    reason_mask.at<unsigned char>(y, x) = PR_Cycle;
                    ++io_stats.removed_by_cycle;
                    continue;
                }

                keep_stage1[idx] = 1;
                keep_final[idx] = 1;
                reason_mask.at<unsigned char>(y, x) = PR_Keep;
            }
        }

        if (cfg.enable_spike_filter)
        {
            for (int y = 0; y < H_; ++y)
            {
                for (int x = 0; x < W_; ++x)
                {
                    const int idx = y * W_ + x;
                    if (!keep_stage1[idx])
                        continue;

                    std::vector<float> neigh_vals;
                    neigh_vals.reserve(8);
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0)
                                continue;
                            const int xn = x + dx;
                            const int yn = y + dy;
                            if (xn < 0 || yn < 0 || xn >= W_ || yn >= H_)
                                continue;
                            const int nidx = yn * W_ + xn;
                            if (!keep_stage1[nidx])
                                continue;

                            const float dv = read_disp(nidx);
                            if (!std::isfinite(dv) || dv <= cfg.min_valid_disp)
                                continue;
                            neigh_vals.push_back(dv);
                        }
                    }

                    if (static_cast<int>(neigh_vals.size()) < cfg.spike_min_neighbors)
                        continue;

                    std::nth_element(neigh_vals.begin(),
                                     neigh_vals.begin() + neigh_vals.size() / 2,
                                     neigh_vals.end());
                    const float med = neigh_vals[neigh_vals.size() / 2];
                    const float cur = read_disp(idx);
                    if (!std::isfinite(cur))
                        continue;

                    if (std::fabs(cur - med) > cfg.spike_abs_diff)
                    {
                        keep_final[idx] = 0;
                        reason_mask.at<unsigned char>(y, x) = PR_Spike;
                        ++io_stats.removed_by_spike;
                    }
                }
            }
        }

        for (int idx = 0; idx < N; ++idx)
        {
            const float d = read_disp(idx);
            if (!std::isfinite(d) || d <= cfg.min_valid_disp)
                continue;

            if (keep_final[idx])
            {
                ++io_stats.kept_pixels;
            }
            else
            {
                ++io_stats.removed_pixels;
                InvalidatePixel(ptrDN, idx, cfg.clear_selected_views_when_invalid);
            }
        }

        if (cfg.dump_debug_mask && !save_root.empty())
        {
            const std::string mask_path = save_root + "/" + frameName + "/" + key->StrRemoveLOD() + "/reject_mask.png";
            DumpMask(mask_path, reason_mask);
        }
    }

    void MIDisparityFilterCPU::ComputeTileReprojectionErrorMap(
        const TileInputs& in,
        const QuadTreeDisNormalMap& dis_normals,
        const DisparityAndNormalPtr& ptrDN,
        const MIDisparityFilterConfig& cfg,
        cv::Mat& out_err_map) const
    {
        out_err_map = cv::Mat(H_, W_, CV_32FC1, cv::Scalar(-1.0f));
        if (!ptrDN)
            return;

        auto read_disp = [&](int idx) -> float
        {
            float d = ptrDN->ph_cuda[idx].w;
            if (!std::isfinite(d))
                d = ptrDN->d_cuda[idx];
            return d;
        };

        auto read_cost = [&](const DisparityAndNormalPtr& dn, int idx) -> float
        {
            if (!dn)
                return std::numeric_limits<float>::quiet_NaN();
            if (idx < 0 || idx >= W_ * H_)
                return std::numeric_limits<float>::quiet_NaN();
            return dn->c_cuda[idx];
        };

        auto bilinear_cost = [&](const DisparityAndNormalPtr& dn, float x, float y) -> float
        {
            if (!dn)
                return std::numeric_limits<float>::quiet_NaN();
            if (x < 0 || y < 0 || x > W_ - 1 || y > H_ - 1)
                return std::numeric_limits<float>::quiet_NaN();

            const int x0 = static_cast<int>(std::floor(x));
            const int y0 = static_cast<int>(std::floor(y));
            const int x1 = std::min(x0 + 1, W_ - 1);
            const int y1 = std::min(y0 + 1, H_ - 1);
            const float ax = x - x0;
            const float ay = y - y0;

            const float c00 = read_cost(dn, y0 * W_ + x0);
            const float c01 = read_cost(dn, y1 * W_ + x0);
            const float c10 = read_cost(dn, y0 * W_ + x1);
            const float c11 = read_cost(dn, y1 * W_ + x1);
            if (!std::isfinite(c00) || !std::isfinite(c01) || !std::isfinite(c10) || !std::isfinite(c11))
                return std::numeric_limits<float>::quiet_NaN();

            return (1.0f - ax) * (1.0f - ay) * c00 +
                   (1.0f - ax) * ay * c01 +
                   ax * (1.0f - ay) * c10 +
                   ax * ay * c11;
        };

        for (int y = 0; y < H_; ++y)
        {
            for (int x = 0; x < W_; ++x)
            {
                const int idx = y * W_ + x;
                const float d = read_disp(idx);
                if (!std::isfinite(d) || d <= cfg.min_valid_disp)
                    continue;

                if (cfg.enable_cost_filter)
                {
                    const float c = read_cost(ptrDN, idx);
                    const bool invalid_cost = (!std::isfinite(c) || c < 0.0f);
                    const bool high_cost = (std::isfinite(c) && c > cfg.max_cost);
                    if (invalid_cost || high_cost)
                        continue;
                }

                std::vector<int> vids;
                if (cfg.use_selected_views_only)
                {
                    EnumerateSetBits(ptrDN->selected_views[idx], vids, cfg.max_triplet);
                }
                else
                {
                    const int lim = std::min(static_cast<int>(in.neigh_keys.size()), cfg.max_triplet);
                    vids.resize(lim);
                    for (int i = 0; i < lim; ++i)
                        vids[i] = i;
                }
                if (vids.empty())
                    continue;

                const float px = static_cast<float>(x);
                const float py = static_cast<float>(y);
                std::vector<float> geo_errs;
                geo_errs.reserve(vids.size());

                for (size_t vi = 0; vi < vids.size(); ++vi)
                {
                    const int b = vids[vi];
                    if (b < 0 || b >= static_cast<int>(in.neigh_keys.size()) || b >= static_cast<int>(in.baselines_px.size()))
                        continue;
                    if (b >= static_cast<int>(in.neigh_gray_f32.size()) || in.neigh_gray_f32[b].empty())
                        continue;

                    const cv::Point2f& Bv = in.baselines_px[b];
                    const float baseline = std::hypot(Bv.x, Bv.y);
                    if (baseline < 1.0e-6f)
                        continue;

                    const float d_x = Bv.x / baseline;
                    const float d_y = Bv.y / baseline;
                    const float scale = baseline / baseline_unit_;

                    const float qx = px + d_x * d * scale;
                    const float qy = py + d_y * d * scale;
                    if (qx < 0 || qy < 0 || qx > W_ - 1 || qy > H_ - 1)
                        continue;

                    auto itN = dis_normals.find(in.neigh_keys[b]);
                    if (itN == dis_normals.end() || !itN->second)
                        continue;
                    const DisparityAndNormalPtr neiDN = itN->second;

                    const float d_nei = BilinearDisparity(neiDN, qx, qy, cfg.min_valid_disp);
                    if (!std::isfinite(d_nei))
                        continue;

                    if (cfg.enable_cost_filter)
                    {
                        const float nei_cost = bilinear_cost(neiDN, qx, qy);
                        const bool invalid_nei_cost = (!std::isfinite(nei_cost) || nei_cost < 0.0f);
                        const bool high_nei_cost = (std::isfinite(nei_cost) && nei_cost > cfg.max_cost);
                        if (invalid_nei_cost || high_nei_cost)
                            continue;
                    }

                    const float phx = qx - d_x * d_nei * scale;
                    const float phy = qy - d_y * d_nei * scale;
                    const float geo_e = std::hypot(phx - px, phy - py);
                    if (std::isfinite(geo_e))
                        geo_errs.push_back(geo_e);
                }

                if (static_cast<int>(geo_errs.size()) >= cfg.min_good_neighbors)
                {
                    std::sort(geo_errs.begin(), geo_errs.end());
                    out_err_map.at<float>(y, x) = QuantileFromSorted(geo_errs, 0.5f);
                }
            }
        }
    }

    void MIDisparityFilterCPU::DumpWholeFrameReprojectionError(
        const std::string& frameName,
        const QuadTreeTileInfoMap& MLA_info_map,
        const QuadTreeProblemMap& problems_map,
        const QuadTreeDisNormalMap& dis_normals,
        const MIDisparityFilterConfig& cfg,
        MIDisparityFilterStats& io_stats,
        const std::string& save_root) const
    {
        if (save_root.empty())
            return;

        int min_col = std::numeric_limits<int>::max();
        int max_col = std::numeric_limits<int>::min();
        int min_row = std::numeric_limits<int>::max();
        int max_row = std::numeric_limits<int>::min();

        for (QuadTreeProblemMap::const_iterator it = problems_map.begin(); it != problems_map.end(); ++it)
        {
            const QuadTreeTileKeyPtr& key = it->first;
            if (!key)
                continue;
            min_col = std::min(min_col, key->GetTileX());
            max_col = std::max(max_col, key->GetTileX());
            min_row = std::min(min_row, key->GetTileY());
            max_row = std::max(max_row, key->GetTileY());
        }

        if (min_col > max_col || min_row > max_row)
            return;

        const int image_width  = (max_col - min_col + 1) * W_;
        const int image_height = (max_row - min_row + 1) * H_;
        cv::Mat frame_err(image_height, image_width, CV_32FC1, cv::Scalar(-1.0f));

        std::vector<float> err_vals;
        err_vals.reserve(static_cast<size_t>(image_width) * static_cast<size_t>(image_height) / 4);

        for (QuadTreeProblemMap::const_iterator it = problems_map.begin(); it != problems_map.end(); ++it)
        {
            const QuadTreeTileKeyPtr& key = it->first;
            if (!key)
                continue;

            auto itDN = dis_normals.find(key);
            if (itDN == dis_normals.end() || !itDN->second)
                continue;

            TileInputs ti;
            if (!BuildTileInputsFromMaps(key, MLA_info_map, problems_map, ti))
                continue;

            cv::Mat tile_err;
            ComputeTileReprojectionErrorMap(ti, dis_normals, itDN->second, cfg, tile_err);
            if (tile_err.empty())
                continue;

            const int base_x = (key->GetTileX() - min_col) * W_;
            const int base_y = (key->GetTileY() - min_row) * H_;

            for (int row = 0; row < H_; ++row)
            {
                const float* tep = tile_err.ptr<float>(row);
                float* fep = frame_err.ptr<float>(base_y + row);
                for (int col = 0; col < W_; ++col)
                {
                    const float e = tep[col];
                    if (!std::isfinite(e) || e < 0.0f)
                        continue;
                    if (e > cfg.disp_error_vis_max_px)
                        continue; // 整图中只保留 <= disp_error_vis_max_px 的误差值

                    fep[base_x + col] = e;
                    err_vals.push_back(e);
                }
            }
        }

        int num_le_thresh = 0;
        double sum = 0.0;
        double sqsum = 0.0;
        if (!err_vals.empty())
        {
            for (size_t i = 0; i < err_vals.size(); ++i)
            {
                const float e = err_vals[i];
                sum += e;
                sqsum += static_cast<double>(e) * e;
                if (e <= cfg.disp_error_thresh_px)
                    ++num_le_thresh;
            }
            std::sort(err_vals.begin(), err_vals.end());
            io_stats.num_disp_error_samples = static_cast<int64_t>(err_vals.size());
            io_stats.num_disp_error_le_thresh = num_le_thresh;
            io_stats.mean_disp_error_px = sum / static_cast<double>(err_vals.size());
            io_stats.rmse_disp_error_px = std::sqrt(sqsum / static_cast<double>(err_vals.size()));
            io_stats.p90_disp_error_px = QuantileFromSorted(err_vals, 0.9f);
        }
        else
        {
            io_stats.num_disp_error_samples = 0;
            io_stats.num_disp_error_le_thresh = 0;
            io_stats.mean_disp_error_px = 0.0;
            io_stats.rmse_disp_error_px = 0.0;
            io_stats.p90_disp_error_px = 0.0;
        }

        boost::filesystem::create_directories(save_root);
        const std::string prefix = save_root + "/" + frameName + "_reproj_error";
        DumpDispErrorGrayPNG(prefix + "_gray.png", frame_err, cfg.disp_error_vis_max_px);
        DumpDispErrorHeatMapFixedWithThreshold(prefix + "_heatmap.png", frame_err,
                                               cfg.disp_error_vis_max_px,
                                               cfg.disp_error_thresh_px,
                                               "px");
        DumpDispErrorPassMask(prefix + "_passmask.png", frame_err, cfg.disp_error_thresh_px);
        DumpDispErrorOverlay(prefix + "_overlay.png", frame_err, cfg.disp_error_thresh_px);
    }

    void MIDisparityFilterCPU::FilterGlobal(
        const std::string& frameName,
        const QuadTreeTileInfoMap& MLA_info_map,
        const QuadTreeProblemMap& problems_map,
        QuadTreeDisNormalMap& dis_normals,
        const MIDisparityFilterConfig& cfg,
        MIDisparityFilterStats& out_stats,
        const std::string& save_root) const
    {
        out_stats = MIDisparityFilterStats{};

        for (QuadTreeProblemMap::const_iterator it = problems_map.begin(); it != problems_map.end(); ++it)
        {
            const QuadTreeTileKeyPtr& key = it->first;
            auto itDN = dis_normals.find(key);
            if (itDN == dis_normals.end() || !itDN->second)
                continue;

            TileInputs ti;
            if (!BuildTileInputsFromMaps(key, MLA_info_map, problems_map, ti))
                continue;

            FilterTile(frameName, key, ti, dis_normals, itDN->second, cfg, out_stats, save_root);
        }

        if (cfg.dump_disp_error_map)
        {
            DumpWholeFrameReprojectionError(frameName, MLA_info_map, problems_map,
                                            dis_normals, cfg, out_stats, save_root);
        }

        LOG_ERROR("[MIDisparityFilter] frame=", frameName.c_str(),
                  " total=", out_stats.total_pixels,
                  " valid_in=", out_stats.initially_valid_pixels,
                  " kept=", out_stats.kept_pixels,
                  " removed=", out_stats.removed_pixels,
                  " cost=", out_stats.removed_by_cost,
                  " no_support=", out_stats.removed_by_no_support,
                  " cycle=", out_stats.removed_by_cycle,
                  " spike=", out_stats.removed_by_spike,
                  " reprojErrN=", out_stats.num_disp_error_samples,
                  " reprojErrMean=", out_stats.mean_disp_error_px,
                  " reprojErrRMSE=", out_stats.rmse_disp_error_px,
                  " reprojErrP90=", out_stats.p90_disp_error_px,
                  " reprojErr<=T=", out_stats.num_disp_error_le_thresh,
                  " visMax=", cfg.disp_error_vis_max_px,
                  " thresh=", cfg.disp_error_thresh_px);
    }
}
