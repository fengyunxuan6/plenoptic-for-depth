/********************************************************************
file base:      MICycleCheck.cpp
author:         LZD (merged by ChatGPT)
created:        2025/08/13
purpose:        闭环一致性 + 随机匹配可视化 + RMSE + 直方图 +
                固定色条热图 + 纹理掩膜(Masked) + 梯度分桶曲线
*********************************************************************/
#include "MICycleCheck.h"
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>

#include "Util/Logger.h"

namespace LFMVS {

#ifndef MCCHECK_DEBUG_PHOTO
#define MCCHECK_DEBUG_PHOTO 1
#endif

#if MCHECK_DEBUG_PHOTO
    static int64_t g_dbg_zscore_cnt = 0;
    static int64_t g_dbg_fallback_cnt = 0;
    static float g_dbg_phe_max = 0.0f;
    static float g_dbg_phe_masked_max = 0.0f;
#endif


// —— 文件作用域（Masked 累加器）——
static std::vector<int64_t> g_mask_hist_bins;     // 全局：mask_good 下 geo 误差的直方图（0.5px 分箱）
static const double G_MASK_BIN_W = 0.5;           // 分箱宽度（像素）
static double g_mask_geo_min = std::numeric_limits<double>::infinity();
static double g_mask_geo_max = 0.0;

static double  g_mask_geo_sq_sum = 0.0, g_mask_photo_sq_sum = 0.0;
static double  g_mask_geo_sum    = 0.0, g_mask_photo_sum    = 0.0;
static int64_t g_mask_count      = 0;
static int64_t g_all_count_seen  = 0;


// ==== Completeness Global Counters ====
static int64_t g_comp_mask_total  = 0; // denominator: mask-ok pixels
static int64_t g_comp_good_pixels = 0; // numerator: pixels with >=K good neighbors
static int64_t g_comp_good_pixels_geo = 0; // numerator: pixels with >=K good neighbors
static int64_t g_comp_good_pixels_photo = 0; // numerator: pixels with >=K good neighbors
MICycleCheckerCPU::MICycleCheckerCPU(int mi_w, int mi_h, float baseline_unit)
: W_(mi_w), H_(mi_h), m_baseline_unit(baseline_unit) {}

void MICycleCheckerCPU::EnumerateSetBits(unsigned int mask, std::vector<int>& out_ids, int max_needed){
    out_ids.clear();
    for (int b=0; mask && (int)out_ids.size()<max_needed; ++b){
        if (mask & 1u) out_ids.push_back(b);
        mask >>= 1u;
    }
}

bool MICycleCheckerCPU::BilinearSampleF32(const cv::Mat& img, float x, float y, float& v){
    if (img.empty() || img.type()!=CV_32FC1) return false;
    const int W=img.cols,H=img.rows;
    if (x<0||y<0||x>W-1||y>H-1) return false;
    int x0=(int)std::floor(x), y0=(int)std::floor(y);
    int x1=std::min(x0+1,W-1), y1=std::min(y0+1,H-1);
    float ax=x-x0, ay=y-y0;
    float v00=img.at<float>(y0,x0), v01=img.at<float>(y1,x0);
    float v10=img.at<float>(y0,x1), v11=img.at<float>(y1,x1);
    v=(1-ax)*(1-ay)*v00 + (1-ax)*ay*v01 + ax*(1-ay)*v10 + ax*ay*v11;
    return true;
}

double MICycleCheckerCPU::Mean(const std::vector<float>& v){
    if (v.empty()) return 0.0; double s=0; for(float x:v) s+=x; return s/v.size();
}
double MICycleCheckerCPU::Percentile(std::vector<float>& v,double p){
    if (v.empty()) return 0.0; std::sort(v.begin(),v.end());
    size_t k=(size_t)std::max(0.0,std::min((double)v.size()-1.0,p*(v.size()-1.0)));
    return (double)v[k];
}

void MICycleCheckerCPU::DumpHeatMap(const std::string& path, const cv::Mat& f32map){
    cv::Mat img8, color; double mn,mx; cv::minMaxLoc(f32map,&mn,&mx); if(mx<=1e-6) mx=1.0;
    cv::convertScaleAbs(f32map,img8,255.0/mx); cv::applyColorMap(img8,color,cv::COLORMAP_JET);
    boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());
    cv::imwrite(path, color);
}
void MICycleCheckerCPU::DumpHeatMapFixed(const std::string& path, const cv::Mat& f32map,
                                         double vmax, const std::string& unit_text){
    CV_Assert(f32map.type()==CV_32FC1);
    cv::Mat clipped; f32map.copyTo(clipped);
    if (vmax<=1e-9) vmax=1.0;
    cv::threshold(clipped, clipped, (float)vmax, (float)vmax, cv::THRESH_TRUNC);
    cv::Mat img8, color; clipped.convertTo(img8, CV_8U, 255.0/vmax, 0.0);
    cv::applyColorMap(img8, color, cv::COLORMAP_JET);

    const int bar_w=50;
    cv::Mat canvas(color.rows, color.cols+bar_w, CV_8UC3, cv::Scalar(0,0,0));
    color.copyTo(canvas(cv::Rect(0,0,color.cols,color.rows)));
    cv::Mat bar(color.rows, bar_w, CV_8UC3);
    for (int y=0; y<color.rows; ++y){
        uchar v=(uchar)std::round(255.0*(1.0 - (double)y/(color.rows-1)));
        cv::Mat tmp(1,1,CV_8U, cv::Scalar(v));
        cv::applyColorMap(tmp, tmp, cv::COLORMAP_JET);
        cv::Vec3b c = tmp.at<cv::Vec3b>(0,0);
        for (int x=0;x<bar_w;++x) bar.at<cv::Vec3b>(y,x)=c;
    }
    bar.copyTo(canvas(cv::Rect(color.cols,0,bar_w,color.rows)));
    auto put=[&](const std::string& s, int y){
        cv::putText(canvas, s, cv::Point(color.cols+5,y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255),1,cv::LINE_AA);
    };
    put("0", color.rows-5);
    put(cv::format("%.2f", vmax*0.5), color.rows/2);
    put(cv::format("%.2f %s", vmax, unit_text.c_str()), 12);
    boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());
    cv::imwrite(path, canvas);
}
cv::Mat MICycleCheckerCPU::RenderHistogramImage(const std::vector<float>& data,
                                                int bins, double max_x,
                                                int img_w, int img_h){
    if (bins<=1) bins=10;
    double mx=max_x; if (mx<=0){ for(float v:data) if (std::isfinite(v)) mx=std::max(mx,(double)v); if(mx<=1e-6) mx=1.0; }
    std::vector<int> cnt(bins,0); const double bw=mx/bins;
    for(float v:data){ if(!std::isfinite(v)||v<0) continue; int b=(int)std::floor(v/bw); b=std::max(0,std::min(b,bins-1)); cnt[b]++; }
    int W=img_w,H=img_h; cv::Mat canvas(H,W,CV_8UC3,cv::Scalar(255,255,255));
    int ml=50,mr=10,mt=10,mb=30; int pw=W-ml-mr, ph=H-mt-mb;
    cv::rectangle(canvas, cv::Rect(ml,mt,pw,ph), cv::Scalar(220,220,220),1);
    int maxc=1; for(int c:cnt) maxc=std::max(maxc,c);
    double cw=(double)pw/bins;
    for(int i=0;i<bins;++i){
        int h=(int)std::round((double)cnt[i]/maxc*(ph-2));
        int x0=ml+(int)std::round(i*cw);
        int x1=ml+(int)std::round((i+1)*cw)-1;
        int y1=mt+ph-1, y0=y1-h;
        cv::rectangle(canvas, cv::Point(x0,y0), cv::Point(x1,y1), cv::Scalar(100,160,240), -1);
    }
    cv::putText(canvas, "Geo cycle error histogram", cv::Point(ml+5, mt-2), 0, 0.45, cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::putText(canvas, "0", cv::Point(ml-8, mt+ph+18), 0, 0.4, cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::putText(canvas, cv::format("%.2f px", mx), cv::Point(ml+pw-45, mt+ph+18), 0, 0.4, cv::Scalar(0,0,0),1,cv::LINE_AA);
    return canvas;
}
cv::Mat MICycleCheckerCPU::RenderBucketCurve(const std::vector<double>& edges,
                                             const std::vector<double>& vals,
                                             const std::string& title,
                                             int img_w, int img_h){
    int W=img_w,H=img_h; cv::Mat canvas(H,W,CV_8UC3,cv::Scalar(255,255,255));
    int ml=50,mr=10,mt=20,mb=30; int pw=W-ml-mr, ph=H-mt-mb;
    cv::rectangle(canvas, cv::Rect(ml,mt,pw,ph), cv::Scalar(230,230,230),1);
    if (edges.empty() || vals.empty()) return canvas;
    double xmax=edges.back(), ymax=0.0; for(double v:vals) ymax=std::max(ymax,v); if (ymax<=1e-9) ymax=1.0;
    auto X=[&](double x){ return ml + (int)std::round(x/xmax*pw); };
    auto Y=[&](double y){ return mt + ph - (int)std::round(y/ymax*ph); };
    for (size_t i=1;i<edges.size();++i){
        cv::line(canvas, cv::Point(X(edges[i-1]), Y(vals[i-1])),
                         cv::Point(X(edges[i]),   Y(vals[i])), cv::Scalar(40,120,200), 2, cv::LINE_AA);
    }
    cv::putText(canvas, title, cv::Point(ml+5, mt-4), 0, 0.5, cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::putText(canvas, "grad", cv::Point(ml+pw-35, mt+ph+20), 0, 0.45, cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::putText(canvas, "geo(px)", cv::Point(5, mt+15), 0, 0.45, cv::Scalar(0,0,0),1,cv::LINE_AA);
    return canvas;
}
void MICycleCheckerCPU::SaveFloatAsPNG(const std::string& path, const cv::Mat& f32){
    if (f32.empty() || f32.type()!=CV_32FC1) return;
    double mn,mx; cv::minMaxLoc(f32, &mn, &mx); if (mx<=mn) mx=mn+1.0;
    cv::Mat u8; f32.convertTo(u8, CV_8U, 255.0/(mx-mn), -mn*255.0/(mx-mn));
    boost::filesystem::create_directories(boost::filesystem::path(path).parent_path());
    cv::imwrite(path, u8);
}
void MICycleCheckerCPU::DumpHeatMapFixedMasked(const std::string& path,
                                               const cv::Mat& f32map,
                                               const cv::Mat& mask_u8,
                                               double vmax, const std::string& unit_text){
    CV_Assert(f32map.type()==CV_32FC1); CV_Assert(mask_u8.empty() || mask_u8.type()==CV_8UC1);
    cv::Mat masked = f32map.clone();
    if (!mask_u8.empty()){
        for (int y=0;y<masked.rows;++y){
            const uchar* m=mask_u8.ptr<uchar>(y); float* f=masked.ptr<float>(y);
            for (int x=0;x<masked.cols;++x){ if (!m[x]) f[x]=0.f; }
        }
    }
    DumpHeatMapFixed(path, masked, vmax, unit_text);
}
void MICycleCheckerCPU::BuildTextureMaskAndViz(
    const cv::Mat& gray_f32, const MICycleMaskParams& mcfg,
    cv::Mat& grad_mag, cv::Mat& var7x7, cv::Mat& mask_good,
    std::string save_dir)
    {
        CV_Assert(gray_f32.type()==CV_32FC1);
        cv::Mat gx, gy; cv::Sobel(gray_f32, gx, CV_32F, 1,0,3); cv::Sobel(gray_f32, gy, CV_32F, 0,1,3);
        cv::magnitude(gx, gy, grad_mag);
        cv::Mat mean, mean2; cv::boxFilter(gray_f32, mean, CV_32F, cv::Size(7,7));
        cv::boxFilter(gray_f32.mul(gray_f32), mean2, CV_32F, cv::Size(7,7));
        var7x7 = mean2 - mean.mul(mean);
        cv::Mat m1=(grad_mag > (float)mcfg.sobel_thresh);
        cv::Mat m2=(var7x7  > (float)mcfg.var_thresh);
        mask_good = (mcfg.use_and ? (m1 & m2) : (m1 | m2));
        mask_good.convertTo(mask_good, CV_8U, 255);

        if (0) // todo:lzd
        {
            if(!save_dir.empty())
            {
                boost::filesystem::create_directories(save_dir);
                SaveFloatAsPNG(save_dir + "/texture_grad.png", grad_mag);
                SaveFloatAsPNG(save_dir + "/texture_var.png",  var7x7);
                cv::imwrite(save_dir + "/texture_mask.png", mask_good);
            }
        }
    }

// === 构建 tile 输入 ===
bool MICycleCheckerCPU::BuildTileInputsFromMaps(
    const QuadTreeTileKeyPtr& key,
    const QuadTreeTileInfoMap& MLA_info_map,
    const QuadTreeProblemMap&  problems_map,
    TileInputs& out) const
{
    auto itProb = problems_map.find(key);
    if (itProb == problems_map.end())
        return false;
    const MLA_Problem& problem = itProb->second;
    if (problem.m_Image_gray.empty())
        return false;

    // 自身像素：灰度图
    if (problem.m_Image_gray.type()!=CV_32FC1)
        problem.m_Image_gray.convertTo(out.ref_gray_f32, CV_32FC1);
    else
        out.ref_gray_f32 = problem.m_Image_gray;

    auto itMLA = MLA_info_map.find(key);
    if (itMLA == MLA_info_map.end())
        return false;

    const cv::Point2f c_ref = itMLA->second->GetCenter();

    // 邻域key
    out.neigh_keys.clear();
    if (!problem.m_NeighsSortVecForMatch.empty())
    {
        out.neigh_keys = problem.m_NeighsSortVecForMatch;
    }
    else
    {
        for (auto const& item : problem.m_Res_Image_KeyVec)
        {
            out.neigh_keys.push_back(item.m_ptrKey);
        }
    }

    out.neigh_gray_f32.clear();
    out.baselines_px.clear();
    out.neigh_gray_f32.reserve(out.neigh_keys.size());
    out.baselines_px.reserve(out.neigh_keys.size());

    for (auto& nk : out.neigh_keys)
    {
        auto itN = problems_map.find(nk);
        cv::Mat g;
        if (itN != problems_map.end())
        {
            if (itN->second.m_Image_gray.type()!=CV_32FC1)
                itN->second.m_Image_gray.convertTo(g, CV_32FC1);
            else
                g = itN->second.m_Image_gray;
        }
        out.neigh_gray_f32.push_back(g); // 邻域灰度图

        cv::Point2f B(0,0);
        auto itMN = MLA_info_map.find(nk);
        if (itMN != MLA_info_map.end())
            B = c_ref-itMN->second->GetCenter(); // c0-c1
        out.baselines_px.push_back(B); // 基线
    }
    return !out.ref_gray_f32.empty() && !out.neigh_gray_f32.empty();
}

// === 单 tile ===
void MICycleCheckerCPU::CheckTile(
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
    const MICycleClampConfig& ccfg) const
    {
        if (!ptrDN)
            return;
        cv::Mat geo_err = cv::Mat::zeros(H_, W_, CV_32F);
        cv::Mat pho_err = cv::Mat::zeros(H_, W_, CV_32F);

        // ==== Completeness (local counters per tile) ====
        int64_t comp_mask_total_local = 0;
        int64_t comp_good_local       = 0;
        int64_t comp_good_local_geo   = 0;
        int64_t comp_good_local_photo = 0;
        std::vector<float> geo_v, pho_v, geo_v_masked, pho_v_masked;
        geo_v.reserve(W_*H_);
        pho_v.reserve(W_*H_);
        geo_v_masked.reserve(W_*H_/2);
        pho_v_masked.reserve(W_*H_/2);

        // 纹理掩膜（仅参考图）
        MICycleMaskParams mcfg;
        mcfg.sobel_thresh=8.0;
        mcfg.var_thresh=60.0;
        mcfg.use_and=true;
        mcfg.grad_hist_bins=50;
        mcfg.grad_hist_max=50.0;
        cv::Mat grad_mag, var7x7, mask_good;
        BuildTextureMaskAndViz(in.ref_gray_f32, mcfg, grad_mag, var7x7, mask_good,
                       save_root + "/" + frameName + "/" + key->StrRemoveLOD() + "/texture");

    auto bilinear_disp = [&](const DisparityAndNormalPtr& dn, float fx, float fy)->float{
        if (!dn)
            return std::numeric_limits<float>::quiet_NaN();
        if (fx<0||fy<0||fx>W_-1||fy>H_-1)
            return std::numeric_limits<float>::quiet_NaN();

        int x0=(int)std::floor(fx), y0=(int)std::floor(fy);
        int x1=std::min(x0+1,W_-1), y1=std::min(y0+1,H_-1);
        float ax=fx-x0, ay=fy-y0;
        auto d_at=[&](int xi,int yi)->float{
            const float4 P = dn->ph_cuda[yi*W_ + xi]; return P.w;
        };
        float d00=d_at(x0,y0), d01=d_at(x0,y1), d10=d_at(x1,y0), d11=d_at(x1,y1);
        if (!std::isfinite(d00))
            d00=0.f;
        if (!std::isfinite(d01))
            d01=d00;
        if (!std::isfinite(d10))
            d10=d00;
        if (!std::isfinite(d11))
            d11=d00;
        return (1-ax)*(1-ay)*d00 + (1-ax)*ay*d01 + ax*(1-ay)*d10 + ax*ay*d11;
    };

    auto patch_mean_std = [&](const cv::Mat& img, float fx, float fy, float& m, float& s)->bool{
        int cx = (int)std::round(fx), cy=(int)std::round(fy);
        if (cx<2 || cy<2 || cx>=img.cols-2 || cy>=img.rows-2)
            return false;
        //cv::Rect roi(cx-2, cy-2, 5, 5);
        cv::Rect roi(cx-2, cy-2, 3, 3); // todo:lzd
        cv::Scalar mean, stdv;
        cv::meanStdDev(img(roi), mean, stdv);
        m=(float)mean[0];
        s=(float)stdv[0];
        return std::isfinite(m) && std::isfinite(s) && s>1e-3f;
    };

    auto one = [&](int x,int y,int idx){
        const float d = ptrDN->ph_cuda[idx].w;
        if (!std::isfinite(d))
            return;
        const unsigned int mask = ptrDN->selected_views[idx];
        std::vector<int> vids;
        EnumerateSetBits(mask, vids, max_triplet);
        if (vids.empty()) return;

        const float px=(float)x, py=(float)y;
        float gsum=0.f, psum=0.f;
        int cnt=0;

        // ---- Completeness: denominator via patch_mean_std on the reference pixel ----
        float m_ref = 0.f, s_ref = 0.f;
        const bool ref_mask_ok = patch_mean_std(in.ref_gray_f32, px, py, m_ref, s_ref);
        if (ref_mask_ok) ++comp_mask_total_local;

        // ---- Completeness: per-pixel neighbor pass counter ----
        int cnt_gatepass = 0;
        int cnt_gatepass_geo = 0;
        int cnt_gatepass_photo = 0;
// >>> 新增：记录该像素在“所有邻域视图”里的 phe 最大值（用来替代循环后拿不到的 phe）
    float phe_max_local = 0.f; 
    bool  has_phe_local = false;
    // <<<
        for (int b : vids)
        {
            if (b<0 || b>=(int)in.neigh_gray_f32.size()) continue;
            const cv::Point2f& Bv = in.baselines_px[b];
            const float baseline = std::hypot(Bv.x,Bv.y); if (baseline<1e-6f) continue;
            const float d_x=Bv.x/baseline, d_y=Bv.y/baseline;
            const float scale = baseline / m_baseline_unit;
            //const float scale = 1.0;

            const float qx = px + d_x * d * scale;
            const float qy = py + d_y * d * scale;
            if (qx<0||qy<0||qx>W_-1||qy>H_-1) continue;

            DisparityAndNormalPtr neiDN;
            if (b<(int)in.neigh_keys.size()){
                auto itN = dis_normals.find(in.neigh_keys[b]);
                if (itN != dis_normals.end()) neiDN = itN->second;
            }
            if (!neiDN)
            {
                neiDN = ptrDN;
            }
            const float d_nei = bilinear_disp(neiDN, qx, qy);
            const float phx = qx - d_x * d_nei * scale;
            const float phy = qy - d_y * d_nei * scale;
            const float geo_e = std::hypot(phx - px, phy - py);
            if (!(geo_e < W_*2)) continue;

            float Ir=0.f, In=0.f;
            if (!BilinearSampleF32(in.ref_gray_f32, px, py, Ir))
                continue;
            if (!BilinearSampleF32(in.neigh_gray_f32[b], qx, qy, In))
                continue;

            // 光度残差：优先用5*5局部z-score，最后统一夹紧到3u
            float mr,sr,mn,sn, phe;
            bool used_zscore = false;
            if (patch_mean_std(in.ref_gray_f32, px, py, mr, sr) &&
                patch_mean_std(in.neigh_gray_f32[b], qx, qy, mn, sn))
            {
                // 归一化差值
                float Irn=(Ir-mr)/sr, Inn=(In-mn)/sn;
                phe = std::fabs(Irn-Inn);
                used_zscore = true;
            }
            else
            {
                // 回退：绝对差（不同数据流范围可能是0-255或0-1）
                phe = std::fabs(In - Ir);
            }
            float phe_raw_for_gate = phe;

            // 无论哪个分支，都统一夹紧到3u
            if (std::isfinite(phe))
            {
                phe = std::min(phe, 3.0f);
            }

            // >>> 新增：更新本像素的“单视 phe 最大值”
            phe_max_local = std::max(phe_max_local, phe);
            has_phe_local = true;

            // === Used-values + per-pixel K-neighbor gating ===
            float geo_e_used = geo_e;
            if (ccfg.clamp_geo)
            {
                geo_e_used = std::min(geo_e_used, (float)ccfg.geo_u_px);
            }

            float phe_used;
            if (ccfg.clamp_photo)
            {
                phe_used = std::min(phe, (float)ccfg.photo_u);
            }
            else
            {
                phe_used = phe;
            }

            // Completeness neighbor threshold test (always evaluated)
            const bool pass_geo_for_comp = (ccfg.gate_use_clamped ? (geo_e_used <= (float)ccfg.gate_geo_px)
                                                                  : (geo_e      <= (float)ccfg.gate_geo_px));
            const bool pass_phe_for_comp = (ccfg.gate_use_clamped ? (phe_used   <= (float)ccfg.gate_photo_u)
                                                                  : (phe_raw_for_gate <= (float)ccfg.gate_photo_u));
            if (pass_geo_for_comp && pass_phe_for_comp)
                ++cnt_gatepass;
            if (pass_geo_for_comp)
                ++cnt_gatepass_geo;
            if (pass_phe_for_comp)
                ++cnt_gatepass_photo;

            bool pixel_ok = true;
            if (ccfg.pixel_gate_enabled)
            {
                const bool pass_geo = (ccfg.gate_use_clamped ? (geo_e_used <= (float)ccfg.gate_geo_px)
                                                         : (geo_e      <= (float)ccfg.gate_geo_px));
                const bool pass_phe = (ccfg.gate_use_clamped ? (phe_used   <= (float)ccfg.gate_photo_u)
                                                         : (phe_raw_for_gate <= (float)ccfg.gate_photo_u));
                pixel_ok = (pass_geo && pass_phe);
            }
            if (pixel_ok)
            {
                gsum+=geo_e_used;
                psum+=phe_used;
                ++cnt;
            }
            float& gcell = geo_err.at<float>(y,x);
            float& pcell = pho_err.at<float>(y,x);
            if (cnt>0) { gcell = std::max(gcell, geo_e_used); pcell = std::max(pcell, phe_used); }
            if (!mask_good.empty() && mask_good.at<uchar>(y,x))
            {
            // >>> 修改这里：用 phe_max_local（或用上面的 pcell 也行）来统计掩膜内的单视最大 phe
            #if MCHECK_DEBUG_PHOTO
            if (has_phe_local)
                g_dbg_phe_masked_max = std::max(g_dbg_phe_masked_max, phe_max_local);
            #endif
            // <<<
            }
        }
            // ---- Completeness: count pixel as good if >=K neighbors passed ----
            if (ref_mask_ok && cnt_gatepass >= ccfg.gate_min_good_neighbors)
            {
                ++comp_good_local;
            }
           if (ref_mask_ok && cnt_gatepass_geo >= ccfg.gate_min_good_neighbors)
            {
                ++comp_good_local_geo;
            }
            if (ref_mask_ok && cnt_gatepass_photo >= ccfg.gate_min_good_neighbors)
            {
                ++comp_good_local_photo;
            }

        // === Finalize this pixel: push averages if K-neighbor condition satisfied ===
        if ( (!ccfg.pixel_gate_enabled && cnt > 0) ||
             (ccfg.pixel_gate_enabled && cnt >= ccfg.gate_min_good_neighbors) )
        {
            const float g_avg = (cnt>0) ? (gsum / cnt) : 0.f;
            const float p_avg = (cnt>0) ? (psum / cnt) : 0.f;
            geo_v.push_back(g_avg);
            pho_v.push_back(p_avg);
            if (!mask_good.empty() && mask_good.at<uchar>(y,x))
            {
                geo_v_masked.push_back(g_avg);
                pho_v_masked.push_back(p_avg);
                #if MCHECK_DEBUG_PHOTO
                if (has_phe_local) g_dbg_phe_masked_max = std::max(g_dbg_phe_masked_max, phe_max_local);
                #endif
            }
        }
    };

    for (int y=0;y<H_;++y)
    {
        for (int x=0;x<W_;++x)
        {
            one(x,y,y*W_+x);
        }
    }

    MICycleCheckStats s{};
    s.num_samples = (int)geo_v.size();
    s.mean_geo_err_px = Mean(geo_v);
    s.median_geo_err_px = Percentile(geo_v,0.5);
    s.p90_geo_err_px    = Percentile(geo_v,0.9);
    s.mean_photo_err    = Mean(pho_v);
    double geo_sq=0;
    double pho_sq=0;
    for(float g:geo_v)
        geo_sq+=g*g;
    for(float p:pho_v)
        pho_sq+=p*p;
    if (s.num_samples>0)
    {
        s.rmse_geo_err_px = std::sqrt(geo_sq/s.num_samples);
        s.rmse_photo_err = std::sqrt(pho_sq/s.num_samples);
    }

    int N0 = io_stats.num_samples + s.num_samples;
    if (s.num_samples>0)
    {
        io_stats.mean_geo_err_px=(io_stats.mean_geo_err_px*io_stats.num_samples + s.mean_geo_err_px*s.num_samples)/std::max(1,N0);
        io_stats.mean_photo_err =(io_stats.mean_photo_err *io_stats.num_samples + s.mean_photo_err *s.num_samples)/std::max(1,N0);
        io_stats.num_samples=N0;
        io_stats.rmse_geo_err_px += geo_sq;
        io_stats.rmse_photo_err  += pho_sq;
    }

    double geo_sq_m=0, pho_sq_m=0;
    for(float g:geo_v_masked)
    {
        geo_sq_m+=g*g;
    }
    for(float p:pho_v_masked)
    {
        pho_sq_m+=p*p;
    }
    g_mask_geo_sq_sum += geo_sq_m;
    g_mask_photo_sq_sum += pho_sq_m;
    g_mask_geo_sum += std::accumulate(geo_v_masked.begin(), geo_v_masked.end(), 0.0);
    g_mask_photo_sum += std::accumulate(pho_v_masked.begin(), pho_v_masked.end(), 0.0);
    // === 累加全局
    // ---- Completeness: accumulate tile-local counters to globals ----
    g_comp_mask_total  += comp_mask_total_local;
    g_comp_good_pixels += comp_good_local;
    g_comp_good_pixels_geo += comp_good_local_geo;
    g_comp_good_pixels_photo += comp_good_local_photo;

    // 直方图（mask_good 下的 geo 误差，0.5px 分箱）并更新全局 min/max ===
    for (float gval : geo_v_masked)
    {
        if (!std::isfinite(gval))
            continue;
        if (gval < 0.f)
            continue;
        int bi = (int)std::floor(gval / G_MASK_BIN_W);
        if (bi < 0)
            bi = 0;
        if ((int)g_mask_hist_bins.size() <= bi)
        {
            g_mask_hist_bins.resize(bi+1, 0);
        }

        g_mask_hist_bins[bi] += 1;
        // 全局 min/max
        if (gval < g_mask_geo_min)
            g_mask_geo_min = gval;
        if (gval > g_mask_geo_max)
            g_mask_geo_max = gval;
    }

    // === 每个 tile 的 mask_good 统计日志（包含 min/max/median/p90 与 0.5px 分箱占比） ===
    if (!geo_v_masked.empty())
    {
        // 计算 min/max（精确）、median/p90（精确，基于该 tile 的样本）
        double t_min = std::numeric_limits<double>::infinity(), t_max=0.0;
        for(float g: geo_v_masked)
        {
            if(std::isfinite(g))
            {
                if(g<t_min)
                    t_min=g;
                if(g>t_max)
                    t_max=g;
            }
        }
        std::vector<float> tmp = geo_v_masked;
        double t_median = Percentile(tmp, 0.5);
        double t_p90    = Percentile(tmp, 0.9);

        // 构建 0.5px 分箱占比
        int nb = (int)g_mask_hist_bins.size();
        // 这里按该 tile 的数据再独立统计一份，以 tile 为单位输出占比
        int nb_local = (int)std::ceil((t_max + 1e-9) / G_MASK_BIN_W);
        if (nb_local <= 0)
            nb_local = 1;
        std::vector<int64_t> bins_local(nb_local, 0);
        for(float g: geo_v_masked)
        {
            if(!std::isfinite(g) || g<0)
                continue;
            int bi = (int)std::floor(g / G_MASK_BIN_W);
            if (bi<0)
                bi=0;
            if (bi>=nb_local)
            {
                bins_local.resize(bi+1, 0);
                nb_local=bi+1;
            }
            bins_local[bi] += 1;
        }
        std::ostringstream oss_bins;
        const double denom = (double)geo_v_masked.size();
        for (int i=0;i<nb_local;++i)
        {
            double lo = i * G_MASK_BIN_W;
            double hi = (i+1) * G_MASK_BIN_W;
            double frac = (denom>0)? (bins_local[i] / denom) : 0.0;
            if (frac > 0.001)
                oss_bins << "[" << lo << "," << hi << "):" << std::fixed << std::setprecision(4) << frac;
            if (i!=nb_local-1)
                oss_bins << " ";
        }
        // LOG_ERROR("[CycleCheck-MaskedTile] frame=", frameName
        //           , " tile=", key->StrRemoveLOD()
        //           , " N=", (int)geo_v_masked.size()
        //           , " geo_mean=", Mean(geo_v_masked)
        //           , " geo_rmse=", std::sqrt( std::inner_product(geo_v_masked.begin(), geo_v_masked.end(), geo_v_masked.begin(), 0.0) / std::max<size_t>(1, geo_v_masked.size()) )
        //           , " geo_min=", t_min
        //           , " geo_max=", t_max
        //           , " geo_median=", t_median
        //           , " geo_p90=", t_p90);
        // LOG_ERROR("[CycleCheck-MaskedTile-Hist] frame=", frameName
        //           , " tile=", key->StrRemoveLOD()
        //           , " bin_w=0.5"
        //           , " bins=", oss_bins.str());
    }

    g_mask_count += (int64_t)geo_v_masked.size();
    g_all_count_seen += (int64_t)geo_v.size();

    if (dump_heatmap)
    {
        const std::string dir = save_root + "/" + frameName + "/" + key->StrRemoveLOD();
        boost::filesystem::create_directories(dir);
        DumpHeatMapFixed(dir + "/geo_err_fixed.png",  geo_err, vis_geo_max_px, "px");
        DumpHeatMapFixed(dir + "/photo_err_fixed.png",pho_err, vis_photo_max,  "");
        DumpHeatMap(dir + "/geo_err_auto.png", geo_err);
        DumpHeatMap(dir + "/photo_err_auto.png", pho_err);
        DumpHeatMapFixedMasked(dir + "/geo_err_fixed_masked.png",  geo_err, mask_good, vis_geo_max_px, "px");
        DumpHeatMapFixedMasked(dir + "/photo_err_fixed_masked.png",pho_err, mask_good, vis_photo_max,  "");

        if (dump_hist && !geo_v.empty()){
            cv::Mat hist = RenderHistogramImage(geo_v, hist_bins, hist_geo_max_x, 800, 300);
            cv::imwrite(dir + "/geo_err_hist.png", hist);
        }

        if (!grad_mag.empty()){
            const int B=mcfg.grad_hist_bins; const double GX=mcfg.grad_hist_max;
            std::vector<double> edges, vals, sum(B,0.0); std::vector<int> cnt(B,0);
            for (int y=0;y<H_;++y){
                const float* gptr=grad_mag.ptr<float>(y);
                for (int x=0;x<W_;++x){
                    if (!mask_good.empty() && !mask_good.at<uchar>(y,x)) continue;
                    float G=gptr[x]; if(!std::isfinite(G)) continue;
                    float gg=std::min<float>(G,(float)GX);
                    int bi=std::min(B-1,(int)std::floor(gg/GX*B));
                    sum[bi]+=geo_err.at<float>(y,x); cnt[bi]+=1;
                }
            }
            for (int i=0;i<B;++i){ edges.push_back((i+1)*(GX/B)); vals.push_back(cnt[i]>0? (sum[i]/cnt[i]) : 0.0); }
            cv::imwrite(dir + "/geo_vs_grad_curve.png", RenderBucketCurve(edges, vals, "Geo mean vs gradient (masked)"));
        }
    }
}

// === 全局 ===
void MICycleCheckerCPU::CheckGlobal(const std::string& frameName,
        const QuadTreeTileInfoMap& MLA_info_map,
        QuadTreeProblemMap&  problems_map,
        QuadTreeDisNormalMap& dis_normals,
        MICycleCheckStats& out_stats,
        int max_triplet,
        bool dump_heatmap,
        const std::string& save_root,
        bool dump_hist,
        int  hist_bins,
        double hist_geo_max_x,
        double vis_geo_max_px,
        double vis_photo_max,
        const MICycleClampConfig& ccfg) const
    {
       out_stats = MICycleCheckStats{};
        g_mask_geo_sq_sum=g_mask_photo_sq_sum=0.0;
        g_mask_geo_sum=g_mask_photo_sum=0.0;
        g_mask_count=g_all_count_seen=0;
        g_mask_hist_bins.clear();
        g_mask_geo_min = std::numeric_limits<double>::infinity();
        g_mask_geo_max = 0.0;


        // ---- Completeness: reset global counters ----
        g_comp_mask_total  = 0;
        g_comp_good_pixels = 0;
        g_comp_good_pixels_geo = 0;
        g_comp_good_pixels_photo = 0;

    for (auto kv=problems_map.begin(); kv!=problems_map.end(); ++kv)
        {
            QuadTreeTileKeyPtr key = kv->first;
            auto itDN = dis_normals.find(key);
            if (itDN==dis_normals.end() || !itDN->second)
                continue;

        // 组建微图像的邻域视图信息
        TileInputs ti;
        if (!BuildTileInputsFromMaps(key, MLA_info_map, problems_map, ti))
        {
            LOG_ERROR("[CycleCheck-Tile] frame=", frameName, " tile=", key->StrRemoveLOD(), " cannot build inputs");
            continue;
        }

        CheckTile(frameName, key, problems_map, dis_normals,
                ti, itDN->second,out_stats, max_triplet, dump_heatmap,
                save_root, dump_hist, hist_bins, hist_geo_max_x,
                vis_geo_max_px, vis_photo_max, ccfg);

        // ---- Completeness (global) ----
        double completeness = (g_comp_mask_total > 0)
                              ? (double)g_comp_good_pixels / (double)g_comp_mask_total
                              : 0.0;
        double completeness_geo = (g_comp_mask_total > 0)
                          ? (double)g_comp_good_pixels_geo / (double)g_comp_mask_total
                          : 0.0;
        double completeness_photo = (g_comp_mask_total > 0)
                          ? (double)g_comp_good_pixels_photo / (double)g_comp_mask_total
                          : 0.0;
        // LOG_ERROR("[CycleCheck-Completeness] frame=", frameName,
        //           " K=", ccfg.gate_min_good_neighbors,
        //           " masked_total=", g_comp_mask_total,
        //           " good_pixels=", g_comp_good_pixels,
        //           " completeness=", completeness);
        out_stats.completeness               = completeness;
        out_stats.completeness_geo           = completeness_geo;
        out_stats.completeness_photo         = completeness_photo;
        out_stats.completeness_masked_pixels = g_comp_mask_total;
        out_stats.completeness_good_pixels   = g_comp_good_pixels;
        out_stats.completeness_good_pixels_geo   = g_comp_good_pixels_geo;
        out_stats.completeness_good_pixels_photo   = g_comp_good_pixels_photo;
        }

        if (out_stats.num_samples>0){
            out_stats.rmse_geo_err_px = std::sqrt(out_stats.rmse_geo_err_px / (double)out_stats.num_samples);
            out_stats.rmse_photo_err  = std::sqrt(out_stats.rmse_photo_err  / (double)out_stats.num_samples);
        }

    // 这里 median/p90 用近似（如需精确，可扩展保存全体像素误差再算分位数）
    out_stats.median_geo_err_px = out_stats.mean_geo_err_px;
    out_stats.p90_geo_err_px    = out_stats.mean_geo_err_px;

    LOG_ERROR("[CycleCheck-Global] frame=", frameName
              , " tiles=", problems_map.size()
              , " N=", out_stats.num_samples
              , " geo_mean=", out_stats.mean_geo_err_px
              , " geo_med~=", out_stats.median_geo_err_px
              , " geo_p90~=", out_stats.p90_geo_err_px
              , " geo_rmse=", out_stats.rmse_geo_err_px
              , " photo_mean=", out_stats.mean_photo_err
              , " photo_rmse=", out_stats.rmse_photo_err);

    double masked_geo_mean = (g_mask_count>0)? (g_mask_geo_sum / (double)g_mask_count) : 0.0;
    double masked_photo_mean = (g_mask_count>0)? (g_mask_photo_sum / (double)g_mask_count) : 0.0;
    double masked_geo_rmse = (g_mask_count>0)? std::sqrt(g_mask_geo_sq_sum / (double)g_mask_count) : 0.0;
    double masked_photo_rmse = (g_mask_count>0)? std::sqrt(g_mask_photo_sq_sum / (double)g_mask_count) : 0.0;
    double coverage = (g_all_count_seen>0)? (double)g_mask_count / (double)g_all_count_seen : 0.0;

    LOG_ERROR("[CycleCheck-Masked] frame=", frameName
              , " coverage=", coverage
              , " N=", g_mask_count
              , " geo_mean=", masked_geo_mean
              , " geo_rmse=", masked_geo_rmse
              , " photo_mean=", masked_photo_mean
              , " photo_rmse=", masked_photo_rmse
              , " masked_total=", out_stats.completeness_masked_pixels
              , " good_pixels=", out_stats.completeness_good_pixels
              , " completeness=", out_stats.completeness
              , " completeness_geo=", out_stats.completeness_geo
              , " completeness_photo=", out_stats.completeness_photo);

    // === 基于全局直方图（0.5px 分箱）估计 masked 的中位数与 p90（分位近似），并输出分箱占比 ===
    double masked_med_approx = 0.0, masked_p90_approx = 0.0;
    if (g_mask_count > 0 && !g_mask_hist_bins.empty()){
        int64_t cum = 0;
        const int64_t n = g_mask_count;
        bool med_set=false, p90_set=false;
        for (size_t i=0;i<g_mask_hist_bins.size();++i){
            cum += g_mask_hist_bins[i];
            if (!med_set && cum >= (int64_t)std::ceil(0.5 * n)){
                masked_med_approx = (i + 0.5) * G_MASK_BIN_W;
                med_set = true;
            }
            if (!p90_set && cum >= (int64_t)std::ceil(0.9 * n)){
                masked_p90_approx = (i + 0.5) * G_MASK_BIN_W;
                p90_set = true;
            }
            if (med_set && p90_set) break;
        }
    }
    LOG_ERROR("[CycleCheck-MaskedExt] frame=", frameName
              , " N=", g_mask_count
              , " geo_min=", (std::isfinite(g_mask_geo_min)? g_mask_geo_min : 0.0)
              , " geo_max=", g_mask_geo_max
              , " geo_median~=", masked_med_approx
              , " geo_p90~=", masked_p90_approx);

    if (g_mask_count > 0 && !g_mask_hist_bins.empty()){
        std::ostringstream oss;
        for (size_t i=0;i<g_mask_hist_bins.size();++i){
            double lo = i * G_MASK_BIN_W;
            double hi = (i+1) * G_MASK_BIN_W;
            double frac = (double)g_mask_hist_bins[i] / (double)g_mask_count;
            oss << "[" << lo << "," << hi << "):" << std::fixed << std::setprecision(4) << frac;
            if (i+1<g_mask_hist_bins.size()) oss << " ";
        }
        LOG_ERROR("[CycleCheck-Masked-Hist] frame=", frameName
                  , " bin_w=0.5"
                  , " bins=", oss.str());
    }


    #if MCHECK_DEBUG_PHOTO
{
    const double denom = std::max<int64_t>(1, g_dbg_zscore_cnt + g_dbg_fallback_cnt);
    const double z_ratio = (double)g_dbg_zscore_cnt / denom;
    LOG_ERROR( "[CycleCheck-PhotoDbg] zscore_cnt=" , g_dbg_zscore_cnt
              , " fallback_cnt=" , g_dbg_fallback_cnt
              , " z_ratio=" , z_ratio
              , " phe_max=" , g_dbg_phe_max
              , " phe_masked_max=" , g_dbg_phe_masked_max);
}
#endif

    }

// === 随机匹配可视化（全局） ===
void MICycleCheckerCPU::VisualizeRandomMatchesGlobal(
    const std::string& frameName,
    const QuadTreeTileInfoMap& MLA_info_map,
    const QuadTreeProblemMap&  problems_map,
    const QuadTreeDisNormalMap& dis_normals,
    int tiles_to_pick,
    int points_per_tile,
    int max_triplet,
    const std::string& save_root,
    unsigned int rng_seed,
    bool draw_lines) const
{
    if (tiles_to_pick<=0 || points_per_tile<=0) return;
    std::vector<QuadTreeTileKeyPtr> keys; keys.reserve(problems_map.size());
    for (auto const& kv : problems_map) keys.push_back(kv.first);
    std::mt19937 rng(rng_seed); std::shuffle(keys.begin(), keys.end(), rng);
    if ((int)keys.size()>tiles_to_pick) keys.resize(tiles_to_pick);

    for (auto const& key : keys)
    {
        auto itDN = dis_normals.find(key);
        if (itDN==dis_normals.end() || !itDN->second)
            continue;
        TileInputs ti;
        if (!BuildTileInputsFromMaps(key, MLA_info_map, problems_map, ti))
            continue;
        VisualizeRandomMatchesTile(frameName, key, ti, problems_map, dis_normals,
                                   itDN->second, points_per_tile, max_triplet,
                                   save_root, rng, draw_lines);
    }
}

// === 随机匹配可视化（单 tile）— 严格复用上一轮随机样本 ===
void MICycleCheckerCPU::VisualizeRandomMatchesTile(
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
    bool draw_lines) const
{
    if (!refDN) return; if (in.ref_gray_f32.empty() || in.neigh_gray_f32.empty()) return;

    auto to_bgr=[&](const cv::Mat& g)->cv::Mat{
        cv::Mat g32,g8,bgr; if (g.type()!=CV_32FC1) g.convertTo(g32, CV_32FC1); else g32=g;
        double mn,mx; cv::minMaxLoc(g32,&mn,&mx); if(mx<=mn) mx=mn+1.0;
        g32.convertTo(g8, CV_8U, 255.0/(mx-mn), -mn*255.0/(mx-mn)); cv::cvtColor(g8,bgr,cv::COLOR_GRAY2BGR); return bgr;
    };
    cv::Mat ref_vis = to_bgr(in.ref_gray_f32);
    std::vector<cv::Mat> nei_vis(in.neigh_gray_f32.size());
    for (size_t i=0;i<in.neigh_gray_f32.size();++i) nei_vis[i]=to_bgr(in.neigh_gray_f32[i]);

    std::vector<bool> has_success(in.neigh_gray_f32.size(), false);

    const cv::Scalar COL_REF (255,0,0);     // 左图（参考）蓝点（按你的要求）
    const cv::Scalar COL_NEI (0,0,255);     // 右图（邻域）红点
    const cv::Scalar COL_CYCLE(0,255,255);  // 回映射点（仍标在参考图上以便观察闭环）
    const cv::Scalar COL_LINE (255,200,0);  // 跨图连线（只画在 pair-panel 上）
    const int R=2, TH=2;

    auto bilinear_disp_dn = [&](const DisparityAndNormalPtr& dn, float x, float y)->float{
        if (!dn) return std::numeric_limits<float>::quiet_NaN();
        if (x<0||y<0||x>W_-1||y>H_-1) return std::numeric_limits<float>::quiet_NaN();
        int x0=(int)std::floor(x), y0=(int)std::floor(y);
        int x1=std::min(x0+1,W_-1), y1=std::min(y0+1,H_-1);
        float ax=x-x0, ay=y-y0;
        auto d_at=[&](int xi,int yi)->float{ const float4 P=dn->ph_cuda[yi*W_+xi]; return P.w; };
        float d00=d_at(x0,y0), d01=d_at(x0,y1), d10=d_at(x1,y0), d11=d_at(x1,y1);
        if (!std::isfinite(d00)) d00=0.f; if (!std::isfinite(d01)) d01=d00;
        if (!std::isfinite(d10)) d10=d00; if (!std::isfinite(d11)) d11=d00;
        return (1-ax)*(1-ay)*d00 + (1-ax)*ay*d01 + ax*(1-ay)*d10 + ax*ay*d11;
    };

    std::uniform_int_distribution<int> rx(2, std::max(2, W_-3));
    std::uniform_int_distribution<int> ry(2, std::max(2, H_-3));

    struct PairMatch { cv::Point2f pref; cv::Point2f qnei; int id; };
    std::vector<std::vector<PairMatch>> pair_lists(in.neigh_gray_f32.size());

    int drawn=0; int max_trials=points_per_tile*20;
    for (int t=0; t<max_trials && drawn<points_per_tile; ++t){
        int x=rx(rng), y=ry(rng), idx=y*W_+x;
        float d_r = refDN->ph_cuda[idx].w; if (!std::isfinite(d_r)) continue;
        unsigned int mask = refDN->selected_views[idx]; if (mask==0u) continue;

        // 参考随机点：蓝点 + 编号（只画在参考图）
        cv::circle(ref_vis, cv::Point(x,y), R, COL_REF, TH, cv::LINE_AA);
        cv::putText(ref_vis, std::to_string(drawn), cv::Point(x+3,y-3), 0, 0.4, cv::Scalar(255,255,255),1,cv::LINE_AA);

        std::vector<int> vids; EnumerateSetBits(mask, vids, max_triplet);
        for (int b:vids)
        {
            if (b<0 || b>=(int)in.neigh_gray_f32.size())
                continue;

            const cv::Point2f& Bv=in.baselines_px[b];
            float baseline=std::hypot(Bv.x,Bv.y);
            if (baseline<1e-6f)
                continue;

            float d_x=Bv.x/baseline, d_y=Bv.y/baseline;
            float scale=baseline/m_baseline_unit;
            //float scale=1.0;

            float qx=(float)x + d_x*d_r*scale;
            float qy=(float)y + d_y*d_r*scale;

            // —— 按你的要求：不在邻域图上画“参考点”与“邻域内连线” —— //
            bool inbound=(qx>=1.f && qy>=1.f && qx<=in.neigh_gray_f32[b].cols-2.f && qy<=in.neigh_gray_f32[b].rows-2.f);
            if (inbound){
                cv::Point qp((int)std::round(qx),(int)std::round(qy));
                // 邻域同名点（只画红点 + 编号）
                cv::circle(nei_vis[b], qp, R, COL_NEI, TH, cv::LINE_AA);
                cv::putText(nei_vis[b], std::to_string(drawn), qp+cv::Point(3,-3), 0, 0.4, cv::Scalar(255,255,255),1,cv::LINE_AA);

                // 记录“上一轮随机样本”的对应，用于跨图连线面板
                pair_lists[b].push_back(PairMatch{ cv::Point2f((float)x,(float)y), cv::Point2f(qx,qy), drawn });
                has_success[b]=true;
            }

            // 回映射点仍可标在参考图，便于观察闭环
            DisparityAndNormalPtr neiDN;
            if (b<(int)in.neigh_keys.size())
            {
                auto itN=dis_normals.find(in.neigh_keys[b]);
                if (itN!=dis_normals.end())
                    neiDN=itN->second;
            }
            if (!neiDN)
                neiDN=refDN;
            if (inbound)
            {
                float d_n=bilinear_disp_dn(neiDN, qx, qy);
                float qbx=qx + d_x*d_n*scale, qby=qy + d_y*d_n*scale;
                if (qbx>=1 && qby>=1 && qbx<=W_-2 && qby<=H_-2)
                {
                    // 黄色：回投到参考图像的点
                    //cv::circle(ref_vis, cv::Point((int)std::round(qbx),(int)std::round(qby)), R, COL_CYCLE, TH, cv::LINE_AA);
                }
            }
        }
        ++drawn;
    }

    const std::string dir = save_root + "/" + frameName + "/" + key->StrRemoveLOD();
    boost::filesystem::create_directories(dir);
    cv::imwrite(dir + "/ref_marked.png", ref_vis);
    for (size_t i=0;i<nei_vis.size();++i)
    {
        if (!nei_vis[i].empty())
        {
            // 取得邻域 Key 的行列信息用于命名（使用已有的 StrRemoveLOD，通常包含 r/c）
            std::string nbKeyStr = (i < in.neigh_keys.size() && in.neigh_keys[i])
                                  ? in.neigh_keys[i]->StrRemoveLOD()
                                  : std::string("nei")+std::to_string(i);
            cv::imwrite(dir + "/nei_"+std::to_string(i)+ "__" + nbKeyStr+"_marked.png", nei_vis[i]);
        }
    }

    // 横向拼接：参考 + 有成功匹配的邻域（邻域图里不再有内部连线/蓝点）
    std::vector<cv::Mat> row; row.push_back(ref_vis);
    for (size_t i=0;i<nei_vis.size(); ++i)
        if (i<pair_lists.size() && !pair_lists[i].empty())
            row.push_back(nei_vis[i]);
    int TW=0, TH_tmp=0;
    for(auto& m:row)
    {
        TW+=m.cols;
        TH_tmp=std::max(TH_tmp,m.rows);
    }
    cv::Mat mosaic(TH_tmp, TW, CV_8UC3, cv::Scalar(0,0,0)); int xoff=0;
    for (auto& m:row){ cv::Mat roi=mosaic(cv::Rect(xoff,0,m.cols,m.rows)); m.copyTo(roi); xoff+=m.cols; }
    cv::imwrite(dir + "/mosaic.png", mosaic);

    // —— 成对连线面板（只画跨图连线）+ 文件名备注邻域 Key（含行列号） —— //
    {
        const int gap = 10;
        for (size_t b = 0; b < pair_lists.size(); ++b)
        {
            if (pair_lists[b].empty() || nei_vis[b].empty())
                continue;

            int H = std::max(ref_vis.rows, nei_vis[b].rows);
            int W = ref_vis.cols + gap + nei_vis[b].cols;
            cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(0,0,0));

            ref_vis.copyTo(canvas(cv::Rect(0,0,ref_vis.cols, ref_vis.rows)));
            nei_vis[b].copyTo(canvas(cv::Rect(ref_vis.cols + gap, 0, nei_vis[b].cols, nei_vis[b].rows)));

            for (const auto& pm : pair_lists[b])
            {
                cv::Point lp((int)std::round(pm.pref.x), (int)std::round(pm.pref.y));
                cv::Point rp((int)std::round(pm.qnei.x) + ref_vis.cols + gap,
                             (int)std::round(pm.qnei.y));

                // 左：蓝点；右：红点；只画跨图连线
                cv::circle(canvas, lp, 1, COL_REF, 1, cv::LINE_AA);
                //cv::putText(canvas, std::to_string(pm.id), lp+cv::Point(3,-3), 0, 0.4, cv::Scalar(255,255,255),1,cv::LINE_AA);
                cv::circle(canvas, rp, 1, COL_NEI, 1, cv::LINE_AA);
                //cv::putText(canvas, std::to_string(pm.id), rp+cv::Point(3,-3), 0, 0.4, cv::Scalar(255,255,255),1,cv::LINE_AA);
                if (draw_lines)
                    cv::line(canvas, lp, rp, COL_LINE, 1, cv::LINE_AA);
            }

            // 取得邻域 Key 的行列信息用于命名（使用已有的 StrRemoveLOD，通常包含 r/c）
            std::string nbKeyStr = (b < in.neigh_keys.size() && in.neigh_keys[b])
                                  ? in.neigh_keys[b]->StrRemoveLOD()
                                  : std::string("nei")+std::to_string(b);

            cv::imwrite(dir + "/pair_nei_" + std::to_string(b) + "__" + nbKeyStr + ".png", canvas);
        }

        // 竖向汇总（可选）
        std::vector<cv::Mat> panels;
        for (size_t b=0; b<pair_lists.size(); ++b)
        {
            std::string p = dir + "/pair_nei_" + std::to_string(b) + "__" +
                            ((b<in.neigh_keys.size() && in.neigh_keys[b]) ? in.neigh_keys[b]->StrRemoveLOD() : std::string("nei")+std::to_string(b))
                            + ".png";
            cv::Mat m = cv::imread(p);
            if (!m.empty()) panels.push_back(m);
        }
        if (!panels.empty())
        {
            int Wmax=0, Hsum=0;
            for (auto& m:panels)
            {
                Wmax=std::max(Wmax,m.cols);
                Hsum+=m.rows;
            }
            cv::Mat all(Hsum, Wmax, CV_8UC3, cv::Scalar(0,0,0));
            int yoff=0;
            for (auto& m:panels)
            {
                m.copyTo(all(cv::Rect(0,yoff,m.cols,m.rows)));
                yoff+=m.rows;
            }
            cv::imwrite(dir + "/pair_mosaic.png", all);
        }
    }
}
} // namespace LFMVS