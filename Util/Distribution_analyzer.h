/********************************************************************
file base:      Distribution_analyzer.h
author:         LZD
created:        2025/08/10
purpose:        视差、虚拟深度等数据的分布情况。
                本头文件定义了浮点数据分布分析的核心接口与数据结构。
                支持在「数据已在内存中」的场景下，直接传入指针/长度或 std::vector<float>。
                同时提供基于 OpenCV 的直方图与 CDF 可视化函数。
*********************************************************************/
#ifndef DISTRIBUTION_ANALYZER_H
#define DISTRIBUTION_ANALYZER_H
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include <string>
#include <limits>

#include <opencv2/opencv.hpp>


// 为避免强制所有翻译单元都包含 OpenCV 头，这里仅做前向声明。
namespace cv { class Mat; }

namespace LFMVS
{
    // ---------------- 在线统计量（均值/方差/偏度/峰度、最小/最大） ----------------
    struct OnlineStats {
        uint64_t n = 0;                         // 有效样本数
        long double mean = 0.0L;                // 均值（增量法）
        long double M2 = 0.0L, M3 = 0.0L, M4 = 0.0L; // 用于方差/偏度/峰度的累积量
        double minv = std::numeric_limits<double>::infinity();   // 最小值
        double maxv = -std::numeric_limits<double>::infinity();  // 最大值
        uint64_t nan_count = 0, inf_count = 0;  // NaN/Inf 计数

        // 增量加入一个样本（采用 Terriberry 一遍更新公式）
        void add(double x);

        // 工具函数
        bool valid() const;
        double get_mean() const;
        double sample_variance() const; // 无偏样本方差
        double sample_std() const;      // 样本标准差
        double skewness() const;        // 样本偏度
        double excess_kurtosis() const; // 超额峰度（kurtosis - 3）
    };

    // ---------------- P² 分位数估计器（Jain & Chlamtac, 1985） ----------------
    // 作用：在不排序、不存储全部数据的前提下，对目标分位数（如中位数、四分位数）进行在线估计。
    struct P2Quantile {
        double p;          // 目标分位点，取值 (0,1)
        bool initialized;  // 是否完成初始化
        double q[5];       // 五个标记高度
        double npos[5];    // 五个标记位置（实际）
        double ndes[5];    // 五个标记位置（期望）
        double dn[5];      // 每次观测后期望位置的增量
        std::vector<double> initbuf; // 初始化缓冲区（前 5 个样本）
        uint64_t count;    // 已处理样本数

        explicit P2Quantile(double prob=0.5);
        void add(double x);
        double value() const; // 当前估计值
    };

    // ---------------- 自适应分箱直方图（Freedman–Diaconis 法则） ----------------
    struct Histogram {
        double lo, hi, binw; // 取值范围与每箱宽度
        size_t bins;         // 箱数
        std::vector<uint64_t> cnts; // 各箱计数

        Histogram();

        // 从 std::vector<float> 构建直方图
        static Histogram build(const std::vector<float>& data, double lo, double hi, double iqr,
                               uint64_t valid_n, size_t min_bins=64, size_t max_bins=65536);
        // 从内存指针区间构建直方图（避免复制）
        static Histogram build(const float* data, size_t n, double lo, double hi, double iqr,
                               uint64_t valid_n, size_t min_bins=64, size_t max_bins=65536);

        // 众数估计：返回（众值近似、峰值所在箱的计数），内部用三点抛物线细化
        std::pair<double,uint64_t> mode() const;

        // 近似 HDI（最小宽度 p 覆盖区间）：基于直方图滑窗
        std::pair<double,double> hdi(double p, uint64_t total) const;
    };

    // ---------------- 分析结果 & 配置 ----------------
    struct AnalysisResult {
        // 基本规模与异常计数
        uint64_t N, invalid_nan, invalid_inf;

        // 位置/尺度/形状统计量
        double minv, maxv, mean, stdv, skew, kurt_excess;

        // 分位数与 IQR
        double q1, median, q3, p90, iqr, p1, p99;

        // 众数与集中区间（HDI）
        double mode_val;
        std::pair<double,double> hdi_main, hdi_95;

        // Tukey 异常值界限
        double tukey_lo, tukey_hi;

        AnalysisResult();
    };

    struct AnalyzerConfig {
        bool exact_quantiles; // 是否使用精确分位数（nth_element/排序），否则使用 P² 近似
        bool exact_hdi;       // 是否使用精确 HDI（需排序），否则用直方图近似
        double hdi_p;         // 主要 HDI 覆盖比例（如 0.85）

        AnalyzerConfig();
    };

    // ---------------- 核心分析接口（支持内存数据） ----------------
    // 直接对内存区间进行分析：传入指针与长度（不复制数据）
    AnalysisResult analyze(const float* data, size_t n, const AnalyzerConfig& cfg);

    // 便利重载：接受 std::vector<float>（按引用，不复制）
    AnalysisResult analyze(const std::vector<float>& data, const AnalyzerConfig& cfg);

    // 打印结果（中文标签）
    void print_result(const AnalysisResult& R, const AnalyzerConfig& cfg);

    // ---------------- 可选：文件 I/O 助手（如需从文件加载） ----------------
    bool load_binary_floats(const std::string& path, std::vector<float>& out);
    bool load_text_floats(const std::string& path, std::vector<float>& out);

    // ---------------- OpenCV 可视化 ----------------
    cv::Mat draw_histogram(const Histogram& H, const AnalysisResult& R, int width, int height, const AnalyzerConfig& cfg);
    cv::Mat draw_cdf(const Histogram& H, const AnalysisResult& R, int width, int height);

    // ---------------- 一键封装类：FloatDistributionAnalyzer ----------------
    // 目的：将 main 中的调用收敛为一个接口（run），对内完成分析与可视化并落盘。
    class FloatDistributionAnalyzer {
    public:
        // 运行参数（可按需扩展）
        struct Options {
            bool exact_quantiles = false;   // 是否精确分位数
            bool exact_hdi = false;         // 是否精确 HDI
            double hdi_p = 0.85;            // 主要 HDI 覆盖比例
            int image_width = 1400;         // 输出图像宽
            int image_height = 480;         // 输出图像高
            std::string output_prefix = "out"; // 输出前缀：前缀_hist.png / 前缀_cdf.png
            bool save_hist = true;          // 是否保存直方图
            bool save_cdf = true;           // 是否保存 CDF
            bool show_windows = false;      // 是否弹出窗口显示（服务器环境建议关闭）
        };

        // 从内存数据运行（float* + n），返回 true 表示成功。
        // out_result 可选返回分析结果（如需进一步使用数值）。
        bool run(const float* data, size_t n, const Options& opt, AnalysisResult* out_result=nullptr) const;

        // 便利重载：std::vector<float>
        bool run(const std::vector<float>& data, const Options& opt, AnalysisResult* out_result=nullptr) const {
            return run(data.data(), data.size(), opt, out_result);
        }


        // === 新增：类内便捷接口——按 HDI p% 过滤后绘制直方图（返回图像） ===
        // 说明：不影响 run() 的行为，可独立调用；用于“去除噪点/离群后”的分布展示
        cv::Mat draw_histogram_hdi_filtered(const float* data, size_t n,
                                            double hdi_p, int width, int height,
                                            const AnalyzerConfig& cfg_for_hdi,
                                            /*out*/ AnalysisResult* out_filtered = nullptr) const;

        // 便捷重载（std::vector）
        cv::Mat draw_histogram_hdi_filtered(const std::vector<float>& v,
                                            double hdi_p, int width, int height,
                                            const AnalyzerConfig& cfg_for_hdi,
                                            /*out*/ AnalysisResult* out_filtered = nullptr) const {
            return draw_histogram_hdi_filtered(v.data(), v.size(), hdi_p, width, height, cfg_for_hdi, out_filtered);
        }

        // === 新增：类内便捷接口——绘制并直接写出 PNG ===
        bool save_histogram_hdi_filtered(const float* data, size_t n,
                                         double hdi_p, int width, int height,
                                         const AnalyzerConfig& cfg_for_hdi,
                                         const std::string& out_path_png,
                                         /*out*/ AnalysisResult* out_filtered = nullptr) const;

        // 便捷重载（std::vector）
        bool save_histogram_hdi_filtered(const std::vector<float>& v,
                                         double hdi_p, int width, int height,
                                         const AnalyzerConfig& cfg_for_hdi,
                                         const std::string& out_path_png,
                                         /*out*/ AnalysisResult* out_filtered = nullptr) const {
            return save_histogram_hdi_filtered(v.data(), v.size(), hdi_p, width, height, cfg_for_hdi, out_path_png, out_filtered);
        }
    };

}
#endif //DISTRIBUTION_ANALYZER_H
