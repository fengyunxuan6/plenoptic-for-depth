/********************************************************************
file base:      Distribution_analyzer.cpp
author:         LZD
created:        2025/08/10
purpose:        视差、虚拟深度等数据的分布情况。
                本头文件定义了浮点数据分布分析的核心接口与数据结构。
                支持在「数据已在内存中」的场景下，直接传入指针/长度或 std::vector<float>。
                同时提供基于 OpenCV 的直方图与 CDF 可视化函数。
*********************************************************************/
#include "Distribution_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Logger.h"

namespace LFMVS
{
    // ================= 在线统计实现 =================
    void OnlineStats::add(double x) {
        // 过滤 NaN / Inf
        if (std::isnan(x)) { nan_count++; return; }
        if (!std::isfinite(x)) { inf_count++; return; }

        // 更新最小/最大
        if (x < minv) minv = x;
        if (x > maxv) maxv = x;

        // Terriberry 单遍更新：同时维护均值、二/三/四阶累积量
        long double n1 = n;
        n++;
        long double delta = x - mean;
        long double delta_n = delta / n;
        long double delta_n2 = delta_n * delta_n;
        long double term1 = delta * delta_n * n1;
        M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6.0L * delta_n2 * M2 - 4.0L * delta_n * M3;
        M3 += term1 * delta_n * (n - 2) - 3.0L * delta_n * M2;
        M2 += term1;
        mean += delta_n;
    }

    bool OnlineStats::valid() const { return n > 1 && std::isfinite((double)mean); }

    double OnlineStats::get_mean() const { return (double)mean; }

    double OnlineStats::sample_variance() const { return (n > 1) ? (double)(M2 / (n - 1)) : NAN; }

    double OnlineStats::sample_std() const { return (n > 1) ? std::sqrt(sample_variance()) : NAN; }

    double OnlineStats::skewness() const {
        if (n < 3 || M2 == 0.0L) return NAN;
        long double s = std::sqrt((long double)n) * M3 / std::pow(M2, 1.5L);
        return (double)s;
    }

    double OnlineStats::excess_kurtosis() const {
        if (n < 4 || M2 == 0.0L) return NAN;
        long double k = (long double)n * M4 / (M2 * M2) - 3.0L;
        return (double)k;
    }

    // ================= P² 分位数实现 =================
    P2Quantile::P2Quantile(double prob)
        : p(prob), initialized(false), count(0) { initbuf.reserve(100); }

    void P2Quantile::add(double x) {
        if (!std::isfinite(x)) return;
        if (!initialized) {
            // 初始化阶段：收集前 5 个样本并排序，建立 5 个标记
            initbuf.push_back(x); count++;
            if (initbuf.size() == 5) {
                std::sort(initbuf.begin(), initbuf.end());
                for (int i=0;i<5;++i) q[i] = initbuf[i];
                for (int i=0;i<5;++i) npos[i] = i+1; // 位置从 1..5
                ndes[0]=1; ndes[1]=1+2*p; ndes[2]=1+4*p; ndes[3]=3+2*p; ndes[4]=5;
                dn[0]=0; dn[1]=p/2; dn[2]=p; dn[3]=(1+p)/2; dn[4]=1;
                initialized = true; initbuf.clear();
            }
            return;
        }

        // 找到 x 落入的区间（q[k] <= x < q[k+1]）
        int k;
        if (x < q[0]) { q[0] = x; k = 0; }
        else if (x >= q[4]) { q[4] = x; k = 3; }
        else { for (k=0; k<4; ++k) if (x < q[k+1]) break; }

        // 更新标记位置与期望位置
        for (int i=0;i<5;++i) { if (i <= k) npos[i] += 1; ndes[i] += dn[i]; }

        // 尝试用抛物线/线性内插调整中间三个标记高度
        for (int i=1;i<=3;++i) {
            double d = ndes[i] - npos[i];
            if ((d >= 1.0 && npos[i+1]-npos[i] > 1.0) || (d <= -1.0 && npos[i-1]-npos[i] < -1.0)) {
                int sgn = (d > 0) ? 1 : -1;
                double qp = q[i] + sgn * ( ((npos[i]-npos[i-1]+sgn)*(q[i+1]-q[i])/(npos[i+1]-npos[i]))
                                          + ((npos[i+1]-npos[i]-sgn)*(q[i]-q[i-1])/(npos[i]-npos[i-1])) ) / (npos[i+1]-npos[i-1]);
                if (qp > q[i-1] && qp < q[i+1]) q[i] = qp;               // 抛物线预测在邻域内，接受
                else q[i] += sgn * (q[i+sgn] - q[i]) / (npos[i+sgn] - npos[i]); // 否则线性插值
                npos[i] += sgn;
            }
        }
        count++;
    }

    double P2Quantile::value() const {
        if (!initialized) {
            // 样本不足 5 个时，返回当前缓冲区的准确分位值
            if (initbuf.empty()) return NAN;
            std::vector<double> tmp = initbuf;
            std::sort(tmp.begin(), tmp.end());
            size_t idx = (size_t)std::floor((tmp.size()-1) * p + 0.5);
            if (idx >= tmp.size()) idx = tmp.size()-1;
            return tmp[idx];
        }
        return q[2]; // P² 中部标记作为分位数估计
    }

    // ================= 直方图实现 =================
    Histogram::Histogram(): lo(0), hi(0), binw(0), bins(0) {}

    // 从 vector 构建
    Histogram Histogram::build(const std::vector<float>& data, double lo, double hi, double iqr,
                               uint64_t valid_n, size_t min_bins, size_t max_bins) {
        return Histogram::build(data.data(), data.size(), lo, hi, iqr, valid_n, min_bins, max_bins);
    }

    // 从指针区间构建（避免复制）
    Histogram Histogram::build(const float* data, size_t n, double lo, double hi, double iqr,
                               uint64_t valid_n, size_t min_bins, size_t max_bins) {
        Histogram H; if (!(hi>lo) || valid_n==0 || data==nullptr || n==0) return H;

        // Freedman–Diaconis 法则估计箱宽；若失败则回退到 Scott 或固定分箱
        double h = 0.0;
        if (iqr > 0) h = 2.0 * iqr * std::pow((double)valid_n, -1.0/3.0);
        if (h <= 0.0 || !std::isfinite(h)) {
            double sigma = (iqr>0) ? (iqr/1.349) : (hi - lo) / 6.0; // Scott 回退
            h = 3.5 * sigma * std::pow((double)valid_n, -1.0/3.0);
            if (h <= 0.0) h = (hi - lo) / 256.0;                    // 最后回退
        }

        size_t bins = (size_t)std::ceil((hi - lo) / h);
        if (bins < min_bins) bins = min_bins;
        if (bins > max_bins) bins = max_bins;

        H.lo = lo; H.hi = hi; H.bins = bins; H.binw = (hi - lo) / (double)bins;
        H.cnts.assign(bins, 0);

        for (size_t i=0;i<n;++i) {
            double x = data[i];
            if (!std::isfinite(x)) continue;
            if (x <= lo) { H.cnts[0]++; continue; }
            if (x >= hi) { H.cnts[bins-1]++; continue; }
            size_t bi = (size_t)((x - lo) / H.binw);
            if (bi >= bins) bi = bins - 1;
            H.cnts[bi]++;
        }
        return H;
    }

    std::pair<double,uint64_t> Histogram::mode() const {
        if (bins==0) return std::make_pair(NAN,(uint64_t)0);
        size_t i0 = (size_t)(std::max_element(cnts.begin(), cnts.end()) - cnts.begin());
        uint64_t c0 = cnts[i0];
        if (bins < 3 || i0==0 || i0==bins-1) {
            double x = lo + (i0 + 0.5) * binw; return std::make_pair(x, c0);
        }
        double cL = (double)cnts[i0-1], cM = (double)cnts[i0], cR = (double)cnts[i0+1];
        double denom = (cL - 2*cM + cR);
        double delta = 0.0; // 亚箱级细化偏移，范围约 [-0.5, 0.5]
        if (std::fabs(denom) > 1e-12) {
            delta = 0.5 * (cL - cR) / denom; if (delta < -0.5) delta = -0.5; if (delta > 0.5) delta = 0.5;
        }
        double center = lo + (i0 + 0.5 + delta) * binw;
        return std::make_pair(center, (uint64_t)cM);
    }

    std::pair<double,double> Histogram::hdi(double p, uint64_t total) const {
        if (bins==0 || total==0) return std::make_pair(NAN, NAN);
        uint64_t need = (uint64_t)std::ceil(p * (double)total);
        size_t L=0; uint64_t sum=0; double bestW = std::numeric_limits<double>::infinity();
        std::pair<size_t,size_t> best(0,0);
        for (size_t R=0; R<bins; ++R) {
            sum += cnts[R];
            while (sum >= need && L<=R) {
                double w = (R - L + 1) * binw;
                if (w < bestW) { bestW = w; best = std::make_pair(L, R); }
                sum -= cnts[L++];
            }
        }
        if (!std::isfinite(bestW)) return std::make_pair(NAN, NAN);
        double left = lo + best.first * binw;
        double right = lo + (best.second + 1) * binw;
        return std::make_pair(left, right);
    }

    // ================= 结果与配置实现 =================
    AnalysisResult::AnalysisResult()
        : N(0), invalid_nan(0), invalid_inf(0),
          minv(NAN), maxv(NAN), mean(NAN), stdv(NAN), skew(NAN), kurt_excess(NAN),
          q1(NAN), median(NAN), q3(NAN), p90(NAN), iqr(NAN), p1(NAN), p99(NAN),
          mode_val(NAN), tukey_lo(NAN), tukey_hi(NAN) {}

    AnalyzerConfig::AnalyzerConfig()
        : exact_quantiles(false), exact_hdi(false), hdi_p(0.85) {}

    // ================= 内部工具：精确分位数与精确 HDI =================
    static double nth_quantile_from_copy(const std::vector<float> &a, double p) {
        if (a.empty() || !(p>=0.0 && p<=1.0)) return NAN;
        std::vector<float> b = a; // 拷贝一份，避免修改原数据
        size_t k = (size_t)std::floor((b.size()-1)*p + 0.5);
        std::nth_element(b.begin(), b.begin()+k, b.end());
        return (double)b[k];
    }

    static std::pair<double,double> exact_hdi_sorted(const std::vector<float>& sorted, double p) {
        if (sorted.empty()) return std::make_pair(NAN, NAN);
        size_t n = sorted.size();
        size_t need = (size_t)std::ceil(p * n);
        double bestW = std::numeric_limits<double>::infinity();
        std::pair<double,double> best(NAN, NAN);
        for (size_t i=0; i+need-1<n; ++i) {
            double L = sorted[i];
            double R = sorted[i + need - 1];
            double w = R - L;
            if (w < bestW) { bestW = w; best = std::make_pair(L, R); }
        }
        return best;
    }

    // ================= 核心分析（指针区间） =================
    AnalysisResult analyze(const float* data, size_t n, const AnalyzerConfig& cfg) {
        AnalysisResult R;
        if (!data || n==0) return R;

        // 1) 单遍在线统计 + P² 分位数估计
        OnlineStats S;
        P2Quantile Pq1(0.25), Pq2(0.5), Pq3(0.75), Pq90(0.90), Pq01(0.01), Pq99(0.99);
        for (size_t i=0;i<n;++i) {
            double x = data[i];
            if (std::isnan(x)) { S.nan_count++; continue; }
            if (!std::isfinite(x)) { S.inf_count++; continue; }
            S.add(x);
            Pq1.add(x); Pq2.add(x); Pq3.add(x); Pq90.add(x); Pq01.add(x); Pq99.add(x);
        }

        R.N = S.n; R.invalid_nan = S.nan_count; R.invalid_inf = S.inf_count;
        if (S.n == 0) return R;

        R.minv = S.minv; R.maxv = S.maxv; R.mean = S.get_mean(); R.stdv = S.sample_std();
        R.skew = S.skewness(); R.kurt_excess = S.excess_kurtosis();

        // 2) 分位数（可选精确）
        if (!cfg.exact_quantiles) {
            R.q1 = Pq1.value(); R.median = Pq2.value(); R.q3 = Pq3.value();
            R.p90 = Pq90.value(); R.p1 = Pq01.value(); R.p99 = Pq99.value();
        } else {
            // 精确分位数需要一次性持有数据；这里复制到临时向量上做 nth_element
            std::vector<float> tmp; tmp.reserve(S.n);
            for (size_t i=0;i<n;++i) if (std::isfinite((double)data[i])) tmp.push_back(data[i]);
            R.q1 = nth_quantile_from_copy(tmp, 0.25);
            R.median = nth_quantile_from_copy(tmp, 0.50);
            R.q3 = nth_quantile_from_copy(tmp, 0.75);
            R.p90 = nth_quantile_from_copy(tmp, 0.90);
            R.p1  = nth_quantile_from_copy(tmp, 0.01);
            R.p99 = nth_quantile_from_copy(tmp, 0.99);
        }
        R.iqr = (std::isfinite(R.q1) && std::isfinite(R.q3)) ? (R.q3 - R.q1) : NAN;

        // 3) 直方图（用于众数与近似 HDI）
        double iqr_for_bin = std::isfinite(R.iqr) ? R.iqr : (R.maxv - R.minv)/2.0;
        Histogram H = Histogram::build(data, n, R.minv, R.maxv, iqr_for_bin, R.N);
        std::pair<double,uint64_t> m = H.mode(); R.mode_val = m.first;

        // 4) HDI：近似（直方图）或精确（排序）
        if (!cfg.exact_hdi) {
            R.hdi_main = H.hdi(cfg.hdi_p, R.N);
            R.hdi_95   = H.hdi(0.95, R.N);
        } else {
            std::vector<float> filtered; filtered.reserve(R.N);
            for (size_t i=0;i<n;++i) if (std::isfinite((double)data[i])) filtered.push_back(data[i]);
            std::sort(filtered.begin(), filtered.end());
            R.hdi_main = exact_hdi_sorted(filtered, cfg.hdi_p);
            R.hdi_95   = exact_hdi_sorted(filtered, 0.95);
        }

        // 5) Tukey 异常值界
        if (std::isfinite(R.q1) && std::isfinite(R.q3)) {
            R.tukey_lo = R.q1 - 1.5 * R.iqr;
            R.tukey_hi = R.q3 + 1.5 * R.iqr;
        }

        return R;
    }

    // ================= 便利重载（vector 引用） =================
    AnalysisResult analyze(const std::vector<float>& data, const AnalyzerConfig& cfg) {
        return analyze(data.data(), data.size(), cfg);
    }

    // ================= 打印结果（中文标签） =================
    void print_result(const AnalysisResult& R, const AnalyzerConfig& cfg) {
        LOG_ERROR("有效样本数: R (NaN: ", R.invalid_nan, ", Inf: ", R.invalid_inf, ")");
        LOG_ERROR("最小/最大: ", R.minv, " / ", R.maxv);
        LOG_ERROR("均值/标准差: ", R.mean, " / ", R.stdv);
        LOG_ERROR("偏度/超额峰度: ", R.skew, " / ", R.kurt_excess);
        LOG_ERROR("Q1/中位数/Q3: ", R.q1, " / ", R.median, " / ", R.q3, "   (IQR=", R.iqr, ")");
        LOG_ERROR("P1/P99: ", R.p1, " / " , R.p99 , "   P90: " , R.p90);
        LOG_ERROR("众数(直方图+抛物线): " , R.mode_val);
        LOG_ERROR("HDI " , (int)(100*cfg.hdi_p) , "%: [" , R.hdi_main.first ,", " , R.hdi_main.second ,"]");
        LOG_ERROR("HDI 95%: [" , R.hdi_95.first , ", " , R.hdi_95.second , "]");
        LOG_ERROR("Tukey 界: [" , R.tukey_lo , ", " , R.tukey_hi , "]");
    }

    // ================= 可视化实现（OpenCV） =================
    static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v);}

    static int draw_axes(cv::Mat& img, int marginL, int marginT, int plotW, int plotH) {
        cv::line(img, cv::Point(marginL, marginT+plotH), cv::Point(marginL+plotW, marginT+plotH), cv::Scalar(0,0,0), 1);
        cv::line(img, cv::Point(marginL, marginT), cv::Point(marginL, marginT+plotH), cv::Scalar(0,0,0), 1);
        return 0;
    }

    cv::Mat draw_histogram(const Histogram& H, const AnalysisResult& R, int width, int height, const AnalyzerConfig& cfg) {
        int marginL=80, marginR=30, marginT=40, marginB=60;
        int W = width, Hh = height; if (W<400) W=400; if (Hh<240) Hh=240;
        cv::Mat img(Hh, W, CV_8UC3, cv::Scalar(255,255,255));
        if (H.bins==0 || !(R.N>0)) {
            cv::putText(img, "no hist data", cv::Point(30, Hh/2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 2);
            return img;
        }
        int plotW = W - marginL - marginR;
        int plotH = Hh - marginT - marginB;
        cv::rectangle(img, cv::Rect(marginL, marginT, plotW, plotH), cv::Scalar(240,240,240), -1);

        uint64_t maxc = 0; for (size_t i=0;i<H.cnts.size();++i) if (H.cnts[i]>maxc) maxc=H.cnts[i];
        if (maxc==0) maxc=1;

        double barPx = (double)plotW / (double)H.bins;
        for (size_t i=0;i<H.bins;++i) {
            double frac = (double)H.cnts[i] / (double)maxc;
            int hpx = (int)std::round(frac * plotH);
            int x0 = (int)std::floor(marginL + i * barPx);
            int x1 = (int)std::floor(marginL + (i+1) * barPx);
            if (x1 <= x0) x1 = x0 + 1;
            cv::rectangle(img, cv::Point(x0, marginT + plotH - hpx), cv::Point(x1, marginT + plotH), cv::Scalar(200,200,255), cv::FILLED);
        }

        draw_axes(img, marginL, marginT, plotW, plotH);

        auto x_to_px = [&](double x){
            if (!(H.hi>H.lo)) return marginL;
            double u = (x - H.lo) / (H.hi - H.lo);
            if (u<0) u=0; if (u>1) u=1;
            return marginL + (int)std::round(u * plotW);
        };

        // 关键位置标注：Q1/Median/Q3/Mode
        struct VLine { double x; cv::Scalar color; const char* label; };
        std::vector<VLine> vls;
        vls.push_back({R.q1, cv::Scalar(0,128,0), "Q1"});
        vls.push_back({R.median, cv::Scalar(0,0,200), "Median"});
        vls.push_back({R.q3, cv::Scalar(0,128,0), "Q3"});
        vls.push_back({R.mode_val, cv::Scalar(200,0,0), "Mode"});
        for (size_t i=0;i<vls.size();++i) {
            int x = x_to_px(vls[i].x);
            cv::line(img, cv::Point(x, marginT), cv::Point(x, marginT+plotH), vls[i].color, 2);
            cv::putText(img, vls[i].label, cv::Point(clampi(x-20, 5, W-60), marginT+20), cv::FONT_HERSHEY_SIMPLEX, 0.5, vls[i].color, 1);
        }

        // HDI 区间以顶部横线标出
        auto draw_hdi = [&](std::pair<double,double> hdi, const cv::Scalar& col, const std::string& label, int yoff){
            if (!std::isfinite(hdi.first) || !std::isfinite(hdi.second)) return;
            int xL = x_to_px(hdi.first), xR = x_to_px(hdi.second);
            int y = marginT + 15 + yoff;
            cv::line(img, cv::Point(xL, y), cv::Point(xR, y), col, 3);
            cv::putText(img, label, cv::Point(clampi((xL+xR)/2 - 40, 5, W-120), y-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
        };
        char buf[64];
        std::snprintf(buf, sizeof(buf), "HDI %.0f%%", 100.0*cfg.hdi_p);
        draw_hdi(R.hdi_main, cv::Scalar(0,100,255), std::string(buf), 0);
        draw_hdi(R.hdi_95,   cv::Scalar(150,150,0),  "HDI 95%", 20);

        // X 轴刻度（min, Q1, median, Q3, max）
        std::vector<std::pair<double,std::string> > ticks;
        std::ostringstream oss; oss.setf(std::ios::fixed); oss<<std::setprecision(3);
        oss.str(""); oss<<R.minv; ticks.push_back(std::make_pair(R.minv, oss.str()));
        oss.str(""); oss<<R.q1;   ticks.push_back(std::make_pair(R.q1, oss.str()));
        oss.str(""); oss<<R.median; ticks.push_back(std::make_pair(R.median, oss.str()));
        oss.str(""); oss<<R.q3;   ticks.push_back(std::make_pair(R.q3, oss.str()));
        oss.str(""); oss<<R.maxv; ticks.push_back(std::make_pair(R.maxv, oss.str()));
        for (size_t i=0;i<ticks.size();++i) {
            int x = x_to_px(ticks[i].first);
            cv::line(img, cv::Point(x, marginT+plotH), cv::Point(x, marginT+plotH+5), cv::Scalar(0,0,0), 1);
            cv::putText(img, ticks[i].second, cv::Point(clampi(x-30, 5, W-80), marginT+plotH+20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(50,50,50), 1);
        }

        cv::putText(img, "Histogram+ quantile/Mode/HDI", cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2);

        return img;
    }

    cv::Mat draw_cdf(const Histogram& H, const AnalysisResult& R, int width, int height) {
        (void)R; // 如需，可在此标注中位数垂线
        int marginL=80, marginR=30, marginT=40, marginB=60;
        int W = width, Hh = height; if (W<400) W=400; if (Hh<240) Hh=240;
        cv::Mat img(Hh, W, CV_8UC3, cv::Scalar(255,255,255));
        if (H.bins==0) { cv::putText(img, "No CDF Data", cv::Point(30, Hh/2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 2); return img; }
        int plotW = W - marginL - marginR; int plotH = Hh - marginT - marginB;
        cv::rectangle(img, cv::Rect(marginL, marginT, plotW, plotH), cv::Scalar(245,245,245), -1);

        // 构造累计比例
        std::vector<double> cdf(H.bins, 0.0);
        uint64_t sum=0; for (size_t i=0;i<H.bins;++i) { sum += H.cnts[i]; cdf[i] = (sum>0) ? (double)sum : 0.0; }
        if (sum==0) sum=1; for (size_t i=0;i<H.bins;++i) cdf[i] /= (double)sum;

        auto x_to_px = [&](double x){
            if (!(H.hi>H.lo)) return marginL;
            double u = (x - H.lo) / (H.hi - H.lo); if (u<0) u=0; if (u>1) u=1; return marginL + (int)std::round(u * plotW);
        };
        auto y_to_px = [&](double y){ if (y<0) y=0; if (y>1) y=1; return marginT + plotH - (int)std::round(y * plotH); };

        // 用折线连接各箱中心点上的 CDF 值
        std::vector<cv::Point> pts; pts.reserve(H.bins);
        for (size_t i=0;i<H.bins;++i) {
            double xc = H.lo + (i + 0.5) * H.binw; int x = x_to_px(xc);
            int y = y_to_px(cdf[i]); pts.push_back(cv::Point(x,y));
        }
        for (size_t i=1;i<pts.size();++i) cv::line(img, pts[i-1], pts[i], cv::Scalar(0,0,180), 2);

        // 坐标轴与水平刻度
        draw_axes(img, marginL, marginT, plotW, plotH);
        for (int i=0;i<=4;++i) {
            double v = 0.25 * i; int y = y_to_px(v);
            cv::line(img, cv::Point(marginL-5,y), cv::Point(marginL,y), cv::Scalar(0,0,0), 1);
            std::ostringstream oss; oss.setf(std::ios::fixed); oss<<std::setprecision(2)<<v;
            cv::putText(img, oss.str(), cv::Point(10, y+4), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(50,50,50), 1);
        }

        cv::putText(img, "CDF", cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2);
        return img;
    }

    // ================= 文件 I/O（可选） =================
    bool load_binary_floats(const std::string& path, std::vector<float>& out) {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        if (!ifs) return false;
        ifs.seekg(0, std::ios::end); std::streampos sz = ifs.tellg(); ifs.seekg(0, std::ios::beg);
        if (sz <= 0) return false;
        size_t n = (size_t)(sz / (std::streamoff)sizeof(float));
        out.resize(n);
        ifs.read(reinterpret_cast<char*>(&out[0]), (std::streamsize)(n*sizeof(float)));
        return ifs.good() || ifs.eof();
    }

    bool load_text_floats(const std::string& path, std::vector<float>& out) {
        std::ifstream ifs(path.c_str()); if (!ifs) return false;
        out.clear(); out.reserve(1000000);
        std::string line; line.reserve(64);
        float v;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            if (iss >> v) out.push_back(v);
        }
        return true;
    }

    // ================= 一键封装类实现 =================
    bool FloatDistributionAnalyzer::run(const float* data, size_t n, const Options& opt,
        AnalysisResult* out_result) const
    {
        if (!data || n==0) return false;

        // 1) 组装分析配置并执行
        AnalyzerConfig cfg; cfg.exact_quantiles = opt.exact_quantiles; cfg.exact_hdi = opt.exact_hdi; cfg.hdi_p = opt.hdi_p;
        AnalysisResult R = analyze(data, n, cfg);

        // 2) 构建直方图
        double iqr_for_bin = std::isfinite(R.iqr) ? R.iqr : (R.maxv - R.minv)/2.0;
        Histogram H = Histogram::build(data, n, R.minv, R.maxv, iqr_for_bin, R.N);

        // 3) 生成图像
        cv::Mat histImg = draw_histogram(H, R, opt.image_width, opt.image_height, cfg);
        cv::Mat cdfImg = draw_cdf(H, R, opt.image_width, opt.image_height);

        // 4) 落盘
        if (opt.save_hist) cv::imwrite(opt.output_prefix + "_hist.png", histImg);
        if (opt.save_cdf)  cv::imwrite(opt.output_prefix + "_cdf.png",  cdfImg);

        // 5) 可选显示窗口（服务器/无界面环境请关闭）
        if (opt.show_windows) {
            cv::imshow("Histogram", histImg);
            cv::imshow("CDF", cdfImg);
            cv::waitKey(0);
        }

        if (out_result) *out_result = R;
        return true;
    }

    // === 新增：FloatDistributionAnalyzer 类内 HDI 过滤绘图实现 ===
cv::Mat FloatDistributionAnalyzer::draw_histogram_hdi_filtered(const float* data, size_t n,
                                        double hdi_p, int width, int height,
                                        const AnalyzerConfig& cfg_for_hdi,
                                        AnalysisResult* out_filtered) const {
    // 0) 基础检查
    if (!data || n==0) {
        return cv::Mat(height>0?height:300, width>0?width:600, CV_8UC3, cv::Scalar(255,255,255));
    }

    // 1) 全量分析以拿到基本分位与 IQR（不改变现有 analyze() 行为）
    AnalysisResult R_full = analyze(data, n, cfg_for_hdi);
    if (!(R_full.N>0)) {
        return cv::Mat(height>0?height:300, width>0?width:600, CV_8UC3, cv::Scalar(255,255,255));
    }

    // 2) 求 HDI p% 区间（与 cfg_for_hdi.exact_hdi 一致：true=精确、false=直方图近似）
    std::pair<double,double> hdi(NAN, NAN);
    if (cfg_for_hdi.exact_hdi) {
        std::vector<float> finite; finite.reserve(R_full.N);
        for (size_t i=0;i<n;++i) { float x=data[i]; if (std::isfinite((double)x)) finite.push_back(x); }
        std::sort(finite.begin(), finite.end());
        hdi = exact_hdi_sorted(finite, hdi_p);
    } else {
        double iqr_for_bin = std::isfinite(R_full.iqr) ? R_full.iqr : (R_full.maxv - R_full.minv)/2.0;
        Histogram Hfull = Histogram::build(data, n, R_full.minv, R_full.maxv, iqr_for_bin, R_full.N);
        hdi = Hfull.hdi(hdi_p, R_full.N);
    }
    double keep_lo = hdi.first, keep_hi = hdi.second;
    if (!std::isfinite(keep_lo) || !std::isfinite(keep_hi) || !(keep_hi>keep_lo)) {
        // 回退：如果 HDI 求解失败，直接绘制全量直方图
        double iqr_for_bin = std::isfinite(R_full.iqr) ? R_full.iqr : (R_full.maxv - R_full.minv)/2.0;
        Histogram H0 = Histogram::build(data, n, R_full.minv, R_full.maxv, iqr_for_bin, R_full.N);
        return draw_histogram(H0, R_full, width, height, cfg_for_hdi);
    }

    // 3) 两遍法：在 [keep_lo, keep_hi] 子集上统计并分箱
    OnlineStats S; P2Quantile Pq1(0.25), Pq2(0.5), Pq3(0.75);
    for (size_t i=0;i<n;++i) {
        double x = data[i]; if (!std::isfinite(x)) continue; if (x < keep_lo || x > keep_hi) continue; S.add(x); Pq1.add(x); Pq2.add(x); Pq3.add(x);
    }
    if (S.n == 0) {
        cv::Mat img(height>0?height:300, width>0?width:600, CV_8UC3, cv::Scalar(255,255,255));
        cv::putText(img, "HDI過濾後無數據", cv::Point(30, img.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 2);
        return img;
    }

    AnalysisResult Rf; Rf.N=S.n; Rf.invalid_nan=0; Rf.invalid_inf=0;
    Rf.minv=S.minv; Rf.maxv=S.maxv; Rf.mean=S.get_mean(); Rf.stdv=S.sample_std();
    Rf.q1=Pq1.value(); Rf.median=Pq2.value(); Rf.q3=Pq3.value();
    Rf.iqr=(std::isfinite(Rf.q1)&&std::isfinite(Rf.q3))?(Rf.q3-Rf.q1):(Rf.maxv-Rf.minv)/2.0;

    double hbin=0.0; if (Rf.iqr>0) hbin=2.0*Rf.iqr*std::pow((double)Rf.N,-1.0/3.0);
    if (hbin<=0.0||!std::isfinite(hbin)) { double sigma=(Rf.iqr>0)?(Rf.iqr/1.349):(Rf.maxv-Rf.minv)/6.0; hbin=3.5*sigma*std::pow((double)Rf.N,-1.0/3.0); if (hbin<=0.0) hbin=(Rf.maxv-Rf.minv)/256.0; }
    size_t bins=(size_t)std::ceil((Rf.maxv-Rf.minv)/hbin); if (bins<64) bins=64; if (bins>65536) bins=65536;

    Histogram Hf; Hf.lo=Rf.minv; Hf.hi=Rf.maxv; Hf.bins=bins; Hf.binw=(Rf.maxv-Rf.minv)/(double)bins; Hf.cnts.assign(bins,0);
    for (size_t i=0;i<n;++i) {
        double x=data[i]; if (!std::isfinite(x)) continue; if (x<keep_lo||x>keep_hi) continue; if (x<=Hf.lo){Hf.cnts[0]++;continue;} if (x>=Hf.hi){Hf.cnts[bins-1]++;continue;} size_t bi=(size_t)((x-Hf.lo)/Hf.binw); if (bi>=bins) bi=bins-1; Hf.cnts[bi]++; }

    Rf.mode_val = Hf.mode().first;            // 过滤后众数
    Rf.hdi_main = std::make_pair(keep_lo, keep_hi); // 主图展示的 HDI p%
    Rf.hdi_95   = Hf.hdi(0.95, Rf.N);         // 过滤后再估 95% HDI（可选）
    if (std::isfinite(Rf.q1) && std::isfinite(Rf.q3)) { Rf.tukey_lo = Rf.q1 - 1.5*Rf.iqr; Rf.tukey_hi = Rf.q3 + 1.5*Rf.iqr; }

    if (out_filtered) *out_filtered = Rf;
    return draw_histogram(Hf, Rf, width, height, cfg_for_hdi);
}

bool FloatDistributionAnalyzer::save_histogram_hdi_filtered(const float* data, size_t n,
                                     double hdi_p, int width, int height,
                                     const AnalyzerConfig& cfg_for_hdi,
                                     const std::string& out_path_png,
                                     AnalysisResult* out_filtered) const {
    cv::Mat img = draw_histogram_hdi_filtered(data, n, hdi_p, width, height, cfg_for_hdi, out_filtered);
    if (img.empty()) return false;
    return cv::imwrite(out_path_png, img);
}
}
