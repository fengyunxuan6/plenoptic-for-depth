//
// Created by wdy on 25-10-17.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <cmath>
#include "Behavioralmodel.h"
#include "Util/Logger.h"
#include <random>
#include <ctime>   // 用于 time(nullptr)

namespace LFMVS
{
// 将任意三通道图转换为 CV_32FC3，便于统一取值
  //  cv::Mat Behavioralmodel::toFloat3(const cv::Mat& img)
    cv::Mat toFloat3(const cv::Mat& img)
    {
        CV_Assert(img.channels() == 3);
        cv::Mat out;
        if (img.depth() == CV_32F) {
            out = img;
        } else {
            img.convertTo(out, CV_32F); // 不缩放，保持原始量纲，只做类型转换
        }
        return out;
    }

// 取 (x,y) 的三通道均值，越界返回 NaN
  //  float Behavioralmodel::sampleMean3(const cv::Mat& img32f3, int x, int y)
    float sampleMean3(const cv::Mat& img32f3, int x, int y)
    {
        if (x < 0 || y < 0 || x >= img32f3.cols || y >= img32f3.rows) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const cv::Vec3f v = img32f3.at<cv::Vec3f>(y, x); // 注意先 y 后 x
        return (v[0] + v[1] + v[2]) / 3.0f;
    }


 //   bool Behavioralmodel::ExtractDepthsFromImages(
    /* bool ExtractDepthsFromImages(
             cv::Mat& real_img,                // 已经加载好的图像
             cv::Mat& virt_img,
             std::vector<cv::Point>& coords_real,
             std::vector<cv::Point>& coords_virtual,
             std::vector<float>& realdepth,
             std::vector<float>& virtualdepth)
     {
         realdepth.clear();
         virtualdepth.clear();

         if (real_img.empty() || virt_img.empty()) {
             std::cerr << "[ExtractDepths] Empty input image.\n";
             return false;
         }

         // 辅助函数：提取像素值（带邻域回退）
         auto extractPixelValue = [](const cv::Mat& img, int x, int y) -> float {
             if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
                 return 0.0f;

             auto getPixel = [&](int yy, int xx) -> float {
                 if (xx < 0 || yy < 0 || xx >= img.cols || yy >= img.rows)
                     return 0.0f;

                 int type = img.type() & CV_MAT_DEPTH_MASK;

                 if (img.channels() == 3) {
                     cv::Vec3f pix = img.at<cv::Vec3f>(yy, xx);
                     return pix[0];
                 }

                 switch (type) {
                     case CV_32F: return img.at<float>(yy, xx);
                     case CV_64F: return static_cast<float>(img.at<double>(yy, xx));
                     case CV_16U: return static_cast<float>(img.at<unsigned short>(yy, xx));
                     case CV_8U:  return static_cast<float>(img.at<uchar>(yy, xx));
                     default:     return 0.0f;
                 }
             };

             //  读取像素
             float v = getPixel(y, x);

             // 若为无效值则查找邻域
             if (std::isnan(v) || v <= 0.0f) {
                 const int radius = 1;  // 邻域半径，可调：1=3x3，2=5x5
                 float nearest = 0.0f;
                 float minDist2 = 1e9;

                 for (int dy = -radius; dy <= radius; ++dy) {
                     for (int dx = -radius; dx <= radius; ++dx) {
                         if (dx == 0 && dy == 0) continue;
                         float val = getPixel(y + dy, x + dx);
                         if (!std::isnan(val) && val > 0.0f) {
                             float dist2 = dx * dx + dy * dy;
                             if (dist2 < minDist2) {
                                 minDist2 = dist2;
                                 nearest = val;
                             }
                         }
                     }
                 }

                 // 若找到邻域有效值，则使用它
                 if (nearest > 0.0f)
                     v = nearest;
             }
             return v;
         };

         // 提取 real_img 深度值
         realdepth.reserve(coords_real.size());
         for (const auto& p : coords_real)
             realdepth.push_back(extractPixelValue(real_img, p.x, p.y));

         // 提取 virt_img 深度值
         virtualdepth.reserve(coords_virtual.size());
         for (const auto& p : coords_virtual)
             virtualdepth.push_back(extractPixelValue(virt_img, p.x, p.y));

         return true;
     }*/

    bool ExtractDepthsFromImages(
            cv::Mat& real_img,                // 已经加载好的图像
            cv::Mat& virt_img,
            std::vector<cv::Point>& coords_real,
            std::vector<cv::Point>& coords_virtual,
            std::vector<float>& realdepth,
            std::vector<float>& virtualdepth,
            int& radius)
    {
        realdepth.clear();
        virtualdepth.clear();

    //    const int radius = 50;  // 邻域半径，可调
        if (real_img.empty() || virt_img.empty())
        {
            std::cerr << "[ExtractDepths] Empty input image.\n";
            return false;
        }

        // 辅助函数：提取单点像素值,空则取邻域
        auto extractPixelValue = [](const cv::Mat& img, int x, int y,int radius) -> float {
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
                return 0.0f;

            auto getPixel = [&](int yy, int xx) -> float {
                if (xx < 0 || yy < 0 || xx >= img.cols || yy >= img.rows)
                    return 0.0f;

                int type = img.type() & CV_MAT_DEPTH_MASK;

                if (img.channels() == 3) {
                    cv::Vec3f pix = img.at<cv::Vec3f>(yy, xx);
                    return pix[0];
                }

                switch (type) {
                    case CV_32F: return img.at<float>(yy, xx);
                    case CV_64F: return static_cast<float>(img.at<double>(yy, xx));
                    case CV_16U: return static_cast<float>(img.at<unsigned short>(yy, xx));
                    case CV_8U:  return static_cast<float>(img.at<uchar>(yy, xx));
                    default:     return 0.0f;
                }
            };

            // 读取像素
            float v = getPixel(y, x);

            // 若为无效值则查找邻域
            if (std::isnan(v) || v <= 0.0f) {
                float nearest = 0.0f;
                float minDist2 = 1e9;

                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        float val = getPixel(y + dy, x + dx);
                        if (!std::isnan(val) && val > 0.0f) {
                            float dist2 = dx * dx + dy * dy;
                            if (dist2 < minDist2) {
                                minDist2 = dist2;
                                nearest = val;
                            }
                        }
                    }
                }

                if (nearest > 0.0f)
                    v = nearest;
            }
            return v;
        };

        // 提取 real_img 深度值
        realdepth.reserve(coords_real.size());
        for (const auto& p : coords_real)
            realdepth.push_back(extractPixelValue(real_img, p.x, p.y,radius));

        // 提取 virt_img 深度值
        virtualdepth.reserve(coords_virtual.size());
        const int windowSize = 10;
        const int halfWin = windowSize / 2;     // 5x5窗口中值滤波

        for (const auto& p : coords_virtual)
        {
            if (p.x < halfWin || p.y < halfWin ||
                p.x >= virt_img.cols - halfWin || p.y >= virt_img.rows - halfWin)
            {
                // 边界直接取单点值
                virtualdepth.push_back(extractPixelValue(virt_img, p.x, p.y,radius));
                continue;
            }

            // 收集邻域像素值
            std::vector<float> neighborhood;
            neighborhood.reserve(windowSize * windowSize);
            for (int dy = -halfWin; dy <= halfWin; ++dy)
            {
                for (int dx = -halfWin; dx <= halfWin; ++dx)
                {
                    float val = extractPixelValue(virt_img, p.x + dx, p.y + dy,radius);
                    if (!std::isnan(val) && val > 0.0f)
                        neighborhood.push_back(val);
                }
            }

            if (neighborhood.empty())
            {
                virtualdepth.push_back(extractPixelValue(virt_img, p.x, p.y,radius));
                continue;
            }

            // 排序取中值
            std::nth_element(neighborhood.begin(),
                             neighborhood.begin() + neighborhood.size() / 2,
                             neighborhood.end());
            float medianVal = neighborhood[neighborhood.size() / 2];
            virtualdepth.push_back(medianVal);
        }

        return true;
    }



    //  std::array<double,3> Behavioralmodel::BehavioralModel( std::vector<double>& realdepth, std::vector<double>& virtualdepth)
    std::array<double,3> BehavioralModel( std::vector<float>& realdepth, std::vector<float>& virtualdepth)
        {
            CV_Assert(realdepth.size() == virtualdepth.size());
            const int N = static_cast<int>(realdepth.size());
            CV_Assert(N >= 3); // 至少 3 个点才能估 3 个参数

            cv::Mat X(N, 3, CV_64F); // [u, v, 1]
            cv::Mat y(N, 1, CV_64F); // a_L

            for (int i = 0; i < N; ++i) {
                const double aL = realdepth[i];
                const double v  = virtualdepth[i];
                const double u  = aL * v;           // u = a_L * v

                X.at<double>(i, 0) = u;             // 列 0: u
                X.at<double>(i, 1) = v;             // 列 1: v
                X.at<double>(i, 2) = 1.0;           // 列 2: 常数项
                y.at<double>(i, 0) = aL;            // 目标：a_L
            }

            cv::Mat c; // 3x1
            bool ok = cv::solve(X, y, c, cv::DECOMP_SVD);  // 最小二乘
            if (!ok) {
                throw std::runtime_error("FitBehavioralModel: cv::solve failed.");
            }

            // 按顺序输出 c0, c1, c2
            return { c.at<double>(0,0), c.at<double>(1,0), c.at<double>(2,0) };
        }

    // 用行为模型c0,c1,c2，把虚拟深度转为真实深度
    cv::Mat convertVirtualToRealDepth(cv::Mat& virtualDepth, std::array<double, 3>& coeffs)
    {
        // 参数 c0, c1, c2
        const double c0 = coeffs[0];
        const double c1 = coeffs[1];
        const double c2 = coeffs[2];

        if (virtualDepth.empty())
        {
            std::cerr << "[Error] Virtual depth map is empty!" << std::endl;
            return cv::Mat();
        }

        cv::Mat v;
        virtualDepth.convertTo(v, CV_64F);

        // 计算 zC = (v * c1 + c2) / (1 - v * c0)
        cv::Mat numerator   = v * c1 + c2;
        cv::Mat denominator = 1.0 - v * c0;
        cv::Mat zC;
        cv::divide(numerator, denominator, zC);

        // 去除 NaN、Inf、负值
        for (int y = 0; y < zC.rows; ++y)
        {
            double* row = zC.ptr<double>(y);
            for (int x = 0; x < zC.cols; ++x)
            {
                double& val = row[x];
                if (std::isnan(val) || std::isinf(val) || val < 0)
                    val = 0.0;
            }
        }
        return zC; // 返回真实深度图 CV_64F
    }
 //   计算行为模型转换的真实深度与相机坐标系下的激光雷达真实深度的绝对误差--采样
    void imageDistanceSampling(cv::Mat rdImage_Bea, cv::Mat& refcasd_img,
                             double threshold /* mm */,
                             int searchRadius,      // 虚拟深度可能被过滤，若取值点空，则邻域半径搜索
                             std::vector<cv::Point> coords_virtual,
                             std::vector<cv::Point>& sampledPoints,
                             int windowSize,int samplesPerPoint)
    {
         bool rejectLocalOutliers = true;      // 局部检测开关
         int outlierWindow = 5;     // 过滤噪点局部窗口
         double outlierK = 2.0;     // 局部窗口检测异常值的阈值: outlierK * MAD（中值）
        /* int windowSize = 100;        // 随机采样窗口大小
         int samplesPerPoint = 15;   // 每个坐标点采样数量*/

         if (rdImage_Bea.empty() || refcasd_img.empty())
         {
             std::cerr << "[Error] Input image is empty.\n";
         }
         if (rdImage_Bea.size() != refcasd_img.size())
         {
             std::cerr << "[Error] Image sizes do not match.\n";
         }

         // 统一通道
         if (rdImage_Bea.channels() == 3) {
             std::vector<cv::Mat> ch;
             cv::split(rdImage_Bea, ch);
             rdImage_Bea = ch[0];
         }
         if (refcasd_img.channels() == 3) {
             std::vector<cv::Mat> ch;
             cv::split(refcasd_img, ch);
             refcasd_img = ch[0];
         }

         //   std::vector<cv::Point> coords_virtual = {cv::Point(724,451),cv::Point(2145,645),cv::Point(3835,956),cv::Point(756,2127),cv::Point(3483,1965)};

         // 统一数据类型
         if (rdImage_Bea.type() == CV_64F)
             rdImage_Bea.convertTo(rdImage_Bea, CV_32F);
         if (refcasd_img.type() == CV_32FC1)
             refcasd_img.convertTo(refcasd_img, CV_32F);


         // 局部噪点检测与剔除
         cv::Mat mask_outlier;
         if (rejectLocalOutliers) {
             mask_outlier = buildOutlierMask(refcasd_img, outlierWindow, outlierK);
             mask_outlier.convertTo(mask_outlier, CV_8U);
         }

         int rows = rdImage_Bea.rows;
         int cols = rdImage_Bea.cols;

        // 局部取值（带邻域回退）
        auto getValidRefValue = [&](const cv::Mat& img, int x, int y, int searchRadius) -> float {
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
                return std::numeric_limits<float>::quiet_NaN();

            // 优先取当前像素
            float val = img.at<float>(y, x);
            if (std::isfinite(val) && val > 0.0f)
                return val;

            // 若无效则在邻域内搜索最近的有效点
            float nearest = std::numeric_limits<float>::quiet_NaN();
            float minDist2 = 1e9;

            for (int dy = -searchRadius; dy <= searchRadius; ++dy)
            {
                int yy = y + dy;
                if (yy < 0 || yy >= img.rows) continue;

                for (int dx = -searchRadius; dx <= searchRadius; ++dx)
                {
                    int xx = x + dx;
                    if (xx < 0 || xx >= img.cols) continue;

                    float v = img.at<float>(yy, xx);
                    if (std::isfinite(v) && v > 0.0f)
                    {
                        float dist2 = dx*dx + dy*dy;
                        if (dist2 < minDist2)
                        {
                            minDist2 = dist2;
                            nearest = v;
                        }
                    }
                }
            }

            return nearest;  // 若全无有效点则返回 NaN
        };


        //  误差计算
         if (!coords_virtual.empty())
         {
             std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
             std::uniform_int_distribution<int> dist(-windowSize / 2, windowSize / 2);

             std::vector<float> diffs;  // 所有采样点的绝对误差
             diffs.reserve(coords_virtual.size() * samplesPerPoint);

             for (const auto &p: coords_virtual) {
                 for (int s = 0; s < samplesPerPoint; ++s) {
                     int dx = dist(rng);
                     int dy = dist(rng);
                     int x = p.x + dx;
                     int y = p.y + dy;

                     if (x < 0 || y < 0 || x >= cols || y >= rows)
                         continue;
                     if (rejectLocalOutliers && mask_outlier.at<uchar>(y, x))
                         continue;

                     float valPred = rdImage_Bea.at<float>(y, x);
                  //   float valRef = refcasd_img.at<float>(y, x);
                     float valRef = getValidRefValue(refcasd_img, x, y, searchRadius);

                     if (!std::isfinite(valPred) || !std::isfinite(valRef))
                         continue;

                     float diff = std::abs(valPred - valRef);
                     if (diff < threshold)
                     {
                         diffs.push_back(diff);
                         sampledPoints.emplace_back(x, y);
                         LOG_WARN("Base: (", p.x,",",p.y,"）---Sample:(",x,",",y,")---Diff:",diff);

                         std::cout << "Base (" << p.x << "," << p.y << ")  "
                                   << "Sample (" << x << "," << y << ")  "
                                   << "Pred=" << valPred << "  "
                                   << "Ref=" << valRef << "  "
                                   << "Diff=" << diff << std::endl;
                     }
                 }
             }

             if (diffs.empty())
             {
                 std::cerr << "[WARN] No valid sampled points.\n";
             }

             // ==== 计算四种误差指标 ====
             double chamfer = 0.0, euclidean = 0.0, mean = 0.0, median = 0.0;
             size_t N = diffs.size();

             for (float d: diffs)
             {
                 chamfer += d * d;
                 euclidean += d * d;
                 mean += d;
             }
             chamfer = chamfer / (N + 1e-12);
             euclidean = std::sqrt(euclidean / (N + 1e-12));
             mean = mean / (N + 1e-12);

             std::sort(diffs.begin(), diffs.end());
             median = diffs[diffs.size() / 2];

             LOG_WARN("Euclidean (RMSE): ", euclidean,"  Mean (MAE):",mean,"  Median:",median);

             std::cout << "Random-sampled " << N << " points in local " << windowSize
                       << "x" << windowSize << " window:" << std::endl;
             std::cout << "  Chamfer (MSE):     " << chamfer << std::endl;
             std::cout << "  Euclidean (RMSE):  " << euclidean << std::endl;
             std::cout << "  Mean (MAE):        " << mean << std::endl;
             std::cout << "  Median:            " << median << std::endl;
         }

    }
    // 计算行为模型转换的真实深度与相机坐标系下的激光雷达真实深度的绝对误差
    double imageDistance(cv::Mat rdImage_Bea, cv::Mat& refcasd_img,
                         DistanceType dt, double threshold /* mm */,
                         bool rejectLocalOutliers,
                         bool useLocalNearest)
    {
        int outlierWindow = 5;     // 过滤噪点局部窗口
        double outlierK = 2.0;     // 阈值:outlierK * MAD
        int searchRadius = 2;    // 1：3*3 ; 2：5*5 ; 3: 7*7

        if (rdImage_Bea.empty() || refcasd_img.empty()) {
            std::cerr << "[Error] Input image is empty.\n";
            return -1.0;
        }
        if (rdImage_Bea.size() != refcasd_img.size()) {
            std::cerr << "[Error] Image sizes do not match.\n";
            return -1.0;
        }

        // 统一通道
        if (rdImage_Bea.channels() == 3) {
            std::vector<cv::Mat> ch;
            cv::split(rdImage_Bea, ch);
            rdImage_Bea = ch[0];
        }
        if (refcasd_img.channels() == 3) {
            std::vector<cv::Mat> ch;
            cv::split(refcasd_img, ch);
            refcasd_img = ch[0];
        }

     //   std::vector<cv::Point> coords_virtual = {cv::Point(724,451),cv::Point(2145,645),cv::Point(3835,956),cv::Point(756,2127),cv::Point(3483,1965)};

        // 统一数据类型
        if (rdImage_Bea.type() == CV_64F)
            rdImage_Bea.convertTo(rdImage_Bea, CV_32F);
        if (refcasd_img.type() == CV_32FC1)
            refcasd_img.convertTo(refcasd_img, CV_32F);


        // 局部噪点检测与剔除
        cv::Mat mask_outlier;
        if (rejectLocalOutliers)
        {
            mask_outlier = buildOutlierMask(refcasd_img, outlierWindow, outlierK);
            mask_outlier.convertTo(mask_outlier, CV_8U);
        }

        int rows = rdImage_Bea.rows;
        int cols = rdImage_Bea.cols;

        double dist = 0.0;
        double N = 0.0;

        // 局部最近邻搜索
        auto findLocalNearest = [&](int y, int x, float refDepth) -> float {
            float bestDiff = 1e12;
            for (int dy = -searchRadius; dy <= searchRadius; ++dy)
            {
                int yy = y + dy;
                if (yy < 0 || yy >= rows) continue;

                float* rowPtr = refcasd_img.ptr<float>(yy);
                for (int dx = -searchRadius; dx <= searchRadius; ++dx)
                {
                    int xx = x + dx;
                    if (xx < 0 || xx >= cols) continue;

                    float val = rowPtr[xx];
                    if (!std::isfinite(val)) continue;

                    float diff = std::abs(val - refDepth);
                    if (diff < bestDiff)
                        bestDiff = diff;
                }
            }
            return bestDiff;
        };

        //  误差计算
        if (dt == DistanceType::Chamfer || dt == DistanceType::Euclidean)
        {
#pragma omp parallel for reduction(+:dist) reduction(+:N)
            for (int y = 0; y < rows; ++y)
            {
                float* pa = rdImage_Bea.ptr<float>(y);
                uchar* pm = rejectLocalOutliers ? mask_outlier.ptr<uchar>(y) : nullptr;

                for (int x = 0; x < cols; ++x)
                {
                    if (!std::isfinite(pa[x])) continue;
                    if (rejectLocalOutliers && pm[x]) continue;
                    float diff = 0.0;
                    if (useLocalNearest)
                    {
                        diff = findLocalNearest(y, x, pa[x]);
                    }
                    else
                    {
                        float refVal = refcasd_img.at<float>(y, x);
                        diff = std::abs(pa[x] - refVal);
                    }
                    float d2 = diff * diff;
                    if (d2 < threshold * threshold)
                    {
                        dist += d2; ++N;
                    }
                }
            }
            if (dt == DistanceType::Chamfer)
            {
                // Chamfer平均平方误差 (MSE)
                dist = dist / (N + 1e-12);
            }
            else if (dt == DistanceType::Euclidean)
            {
                // Euclidean均方根误差 (RMSE)
                dist = std::sqrt(dist / (N + 1e-12));
            }
        }
        else if (dt == DistanceType::Mean)
        {
#pragma omp parallel for reduction(+:dist) reduction(+:N)
            for (int y = 0; y < rows; ++y)
            {
                float* pa = rdImage_Bea.ptr<float>(y);
                uchar* pm = rejectLocalOutliers ? mask_outlier.ptr<uchar>(y) : nullptr;

                for (int x = 0; x < cols; ++x)
                {
                    if (!std::isfinite(pa[x])) continue;
                    if (rejectLocalOutliers && pm[x]) continue;

                    float diff = useLocalNearest ?
                                  findLocalNearest(y, x, pa[x]) :
                                  std::abs(pa[x] - refcasd_img.at<float>(y, x));

                    if (diff < threshold) { dist += diff; ++N; }
                }
            }
            dist /= (N + 1e-12);
        }
        else if (dt == DistanceType::Median)
        {
            std::vector<float> diffs;
            diffs.reserve(rows * cols);

            for (int y = 0; y < rows; ++y)
            {
                float* pa = rdImage_Bea.ptr<float>(y);
                uchar* pm = rejectLocalOutliers ? mask_outlier.ptr<uchar>(y) : nullptr;

                for (int x = 0; x < cols; ++x)
                {
                    if (!std::isfinite(pa[x])) continue;
                    if (rejectLocalOutliers && pm[x]) continue;

                    float diff = useLocalNearest ?
                                  findLocalNearest(y, x, pa[x]) :
                                  std::abs(pa[x] - refcasd_img.at<float>(y, x));

                    if (diff < threshold) diffs.push_back(diff);
                }
            }

            if (diffs.empty()) return -1.0;
            std::nth_element(diffs.begin(), diffs.begin() + diffs.size() / 2, diffs.end());
            dist = diffs[diffs.size() / 2];
        }
        else
        {
            std::cerr << "[Error] Unknown DistanceType.\n";
            return -1.0;
        }
        std::cout << static_cast<int>(dt) <<"  dist： " << dist <<  std::endl;
        return dist;
    }





    cv::Mat buildOutlierMask(cv::Mat& img, int ksize, double k)
    {
        CV_Assert(img.type() == CV_32FC1);

        cv::Mat med, absdiff, mad;
        cv::medianBlur(img, med, ksize);      // 局部中位数
        cv::absdiff(img, med, absdiff);
        cv::medianBlur(absdiff, mad, ksize);  // 局部MAD

        mad += 1e-9; // 防止除零
        cv::Mat mask = absdiff > (k * mad);   // mask=1 表示该点为噪点
        return mask;
    }

    bool loadPointsXML(std::string& path,
                       std::vector<cv::Point>& coords_casd,
                       std::vector<cv::Point>& coords_virtual)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open " << path << " for reading\n";
            return false;
        }

        auto readSeq = [&](const cv::FileNode& node, std::vector<cv::Point>& out){
            out.clear();
            if (node.type() != cv::FileNode::SEQ) return;
            for (auto it = node.begin(); it != node.end(); ++it) {
                int x = (int)(*it)["x"];
                int y = (int)(*it)["y"];
                out.emplace_back(x, y);
            }
        };

        readSeq(fs["coords_casd"], coords_casd);
        readSeq(fs["coords_virtual"], coords_virtual);
        fs.release();
        return true;
    }

    cv::Mat drawRandomColorCrosses(const cv::Mat& img, const std::vector<cv::Point>& points)
    {
        int size = 24;
        int thickness = 8;
        int seed = -1;

        CV_Assert(!img.empty());
        CV_Assert(size > 0 && thickness > 0);

        // 1) 确保是三通道彩色图（便于画彩色）
        cv::Mat canvas;
        if (img.channels() == 3 && img.type() == CV_8UC3) {
            canvas = img.clone();
        } else if (img.channels() == 1) {
            // 灰度 -> BGR
            if (img.type() != CV_8UC1) {
                cv::Mat tmp8u;
                img.convertTo(tmp8u, CV_8U, 1.0); // 简单线性转换为8位
                cv::cvtColor(tmp8u, canvas, cv::COLOR_GRAY2BGR);
            } else {
                cv::cvtColor(img, canvas, cv::COLOR_GRAY2BGR);
            }
        } else {
            // 其他类型/深度，先转8位再保证三通道
            cv::Mat tmp8u;
            img.convertTo(tmp8u, CV_8U);
            if (tmp8u.channels() == 1) cv::cvtColor(tmp8u, canvas, cv::COLOR_GRAY2BGR);
            else if (tmp8u.channels() == 3) canvas = tmp8u.clone();
            else {
                // 通道数太奇怪：只取前3个通道
                std::vector<cv::Mat> chs;
                cv::split(tmp8u, chs);
                while (chs.size() < 3) chs.push_back(chs[0]);
                cv::merge(std::vector<cv::Mat>{chs[0], chs[1], chs[2]}, canvas);
            }
        }

        // 2) 随机颜色生成器
        std::mt19937 rng(seed >= 0 ? static_cast<unsigned>(seed)
                                   : static_cast<unsigned>(time(nullptr)));
        std::uniform_int_distribution<int> col(0, 255);

        auto inBounds = [&](int x, int y){
            return x >= 0 && y >= 0 && x < canvas.cols && y < canvas.rows;
        };

        // 3) 逐点画十字（水平+竖直两条线）
        for (const auto& p : points) {
            if (!inBounds(p.x, p.y)) continue;

            // 随机 BGR
            cv::Scalar color(col(rng), col(rng), col(rng));

            // 计算端点并裁剪到图像边界
            cv::Point left (std::max(0, p.x - size), p.y);
            cv::Point right(std::min(canvas.cols - 1, p.x + size), p.y);
            cv::Point up   (p.x, std::max(0, p.y - size));
            cv::Point down (p.x, std::min(canvas.rows - 1, p.y + size));

            cv::line(canvas, left, right, color, thickness, cv::LINE_AA);
            cv::line(canvas, up,   down,  color, thickness, cv::LINE_AA);
        }

        return canvas;
    }


}
