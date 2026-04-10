//
// Created by wdy on 25-10-17.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <cmath>
#include "VirtualToRealDepthFunc.h"
#include "Util/Logger.h"
#include <random>
#include <ctime>   // 用于 time(nullptr)
#include <boost/filesystem/path.hpp>

namespace LFMVS {
    VirtualToRealDepthFunc::VirtualToRealDepthFunc(DepthSolver *pDepthSolver)
            : m_ptrDepthSolver(pDepthSolver) {

    }

    VirtualToRealDepthFunc::~VirtualToRealDepthFunc() {

    }

    void VirtualToRealDepthFunc::SetVirtualToRealDepthType(VTORDType type) {
        m_VirtualToRealDepthType = type;
    }

    VTORDType VirtualToRealDepthFunc::GetVirtualToRealDepthType() {
        return m_VirtualToRealDepthType;
    }

    void VirtualToRealDepthFunc::SetSamplePointSelectType(SamplePointSelectType type) {
        m_SamplePointSelectType = type;
    }

    SamplePointSelectType VirtualToRealDepthFunc::GetSamplePointSelectType() {
        return m_SamplePointSelectType;
    }

    void VirtualToRealDepthFunc::VirtualToRealDepth(QuadTreeProblemMapMap::iterator &itrP) {
        switch (m_VirtualToRealDepthType) {
            case VTORD_Behavioralmodel: {
                VirtualToRealDepthByBM();
            }
                break;
            default:
                break;
        }
    }

    void VirtualToRealDepthFunc::SamplePointSelect() {
        switch (m_SamplePointSelectType) {
            case SPSelectByLocalWindow: {
                SamplePointSelectByLW();
            }
                break;
            default:
                break;
        }
    }

    // TODO 增加bool,ture,在此提供路径；false:在内存中保存
    void VirtualToRealDepthFunc::VirtualToRealDepthByBM() {
        m_strRootPath = m_ptrDepthSolver->GetRootPath();
//        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/m_vIDepth_Raw.tiff";
        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/VD_Raw.tiff";
//        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/VitualDepthMap.tiff";
        std::string refDepthImg_path = m_strRootPath + "/behavior_model/ref-csad-rd.tiff";
        std::string pointsXmlPath = m_strRootPath + "/behavior_model/points.xml";
        std::string focuseImg_Path = m_strRootPath + "/behavior_model/VD.png";

        std::vector<float> refDepthValue;
        std::vector<float> virtualDepthValue;
        m_virtualDepthImage = cv::imread(virtualDepthImg_path, cv::IMREAD_UNCHANGED);  // 读取虚拟深度图
        m_refDepthImage = cv::imread(refDepthImg_path, cv::IMREAD_UNCHANGED);   // 读取激光深度图
//        cv::Mat focuseImage = cv::imread(focuseImg_Path);                  // 读取全聚焦图

        /*// 参数调整 TODO：封装为struct，参数命名修改
        int refSearchRadius = 50;  // 点云真值稀疏，若取值点空，则邻域半径搜索
        int virtualSearchRadius = 50;  // 虚拟深度可能被过滤，若取值点空，则邻域半径搜索
        double outlierDistanceThreshold = 2000;   // 计算误差时异常距离阈值
        int localWindowSize = 100;        // 误差计算：随机采样窗口大小
        int sampleCount = 15;   // 误差计算：每个坐标点采样数量*/
        // todo: 设置枚举方式,选择距离误差计算点坐标的不同采样方式


        std::string vDImg_marked_path = m_strRootPath + "/behavior_model/m_vIDepth_Marked.png";
        std::string output_csv_path = m_strRootPath + "/behavior_model/output_csv.csv";

        // 挑选拟合多个行为模型参数的样本点
        sampleVirtualDepthPointsByRegion(virtualDepthImg_path,
                                 refDepthImg_path,
                                 vDImg_marked_path,
                                 output_csv_path);

        loadPointsXML(pointsXmlPath);

        // 1>: 转换图像格式
        ExtractDepthsFromImages(refDepthValue, virtualDepthValue);

        // 1.5>: 新增不同景深点坐标
//        appendDepthPoints(refDepthValue, virtualDepthValue);
//        cv::Rect roi = cv::Rect(1819,1153,1000,1000);   // 圆盘
        cv::Rect roi = cv::Rect(3810, 140, 2000, 2000);   // 汽车，墙面
        refDepthValue.clear();
        virtualDepthValue.clear();
        appendDepthPoints(refDepthValue, virtualDepthValue, roi);

        // 2>: 计算行为模型
        std::array<double, 3> behaviorModelParams = BehavioralModel(refDepthValue, virtualDepthValue);

        // 3>: 使用行为模型
        cv::Mat realDepthImage = convertVirtualToRealDepth(behaviorModelParams);

        // 4>: 采样
        SamplePointSelect();

        // 5>: 计算绝对误差
        imageDistanceSampling(realDepthImage, m_refDepthImage);

//        // 6>: 在全聚焦图上对误差采样点标记“+”
//        cv::Mat focImgResult = drawRandomColorCrosses(focuseImage, m_samplePoints);
//        cv::imwrite(m_strRootPath +"/behavior_model/VD_BlurFeature_real_color.png", focImgResult);

        LOG_WARN("behaviorModelParams: c0:", behaviorModelParams[0], ",c1:", behaviorModelParams[1], ",c2:",
                 behaviorModelParams[2]);
    }

    bool VirtualToRealDepthFunc::ExtractDepthsFromImages(
            std::vector<float> &refDepthValue,
            std::vector<float> &virtualDepthValue) {
        refDepthValue.clear();
        virtualDepthValue.clear();

        if (m_coordsRef.empty() || m_coordsVirtual.empty()) {
            std::cerr << "[ExtractDepths] Empty input image.\n";
            return false;
        }

        if (m_refDepthImage.empty() || m_virtualDepthImage.empty()) {
            std::cerr << "[ExtractDepths] Empty input image.\n";
            return false;
        }

        // 辅助函数：提取单点像素值,空则取邻域
        auto extractPixelValue = [](const cv::Mat &img, int x, int y, int radius) -> float {
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
                    case CV_32F:
                        return img.at<float>(yy, xx);
                    case CV_64F:
                        return static_cast<float>(img.at<double>(yy, xx));
                    case CV_16U:
                        return static_cast<float>(img.at<unsigned short>(yy, xx));
                    case CV_8U:
                        return static_cast<float>(img.at<uchar>(yy, xx));
                    default:
                        return 0.0f;
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

        // 提取 m_refDepthImage 深度值
        refDepthValue.reserve(m_coordsRef.size());
        for (const auto &p: m_coordsRef)
            refDepthValue.push_back(extractPixelValue(m_refDepthImage, p.x, p.y, SESOptions.refSearchRadius));

        // 提取 m_refDepthImage 深度值
        virtualDepthValue.reserve(m_coordsVirtual.size());
        const int windowSize = 10;
        const int halfWin = windowSize / 2;     // 5x5窗口中值滤波

        for (const auto &p: m_coordsVirtual) {
            if (p.x < halfWin || p.y < halfWin ||
                p.x >= m_virtualDepthImage.cols - halfWin || p.y >= m_virtualDepthImage.rows - halfWin) {
                // 边界直接取单点值
                virtualDepthValue.push_back(
                        extractPixelValue(m_virtualDepthImage, p.x, p.y, SESOptions.virtualSearchRadius));
                continue;
            }

            // 收集邻域像素值
            std::vector<float> neighborhood;
            neighborhood.reserve(windowSize * windowSize);
            for (int dy = -halfWin; dy <= halfWin; ++dy) {
                for (int dx = -halfWin; dx <= halfWin; ++dx) {
                    float val = extractPixelValue(m_virtualDepthImage, p.x + dx, p.y + dy,
                                                  SESOptions.virtualSearchRadius);
                    if (!std::isnan(val) && val > 0.0f)
                        neighborhood.push_back(val);
                }
            }

            if (neighborhood.empty()) {
                virtualDepthValue.push_back(
                        extractPixelValue(m_virtualDepthImage, p.x, p.y, SESOptions.virtualSearchRadius));
                continue;
            }

            // 排序取中值
            std::nth_element(neighborhood.begin(),
                             neighborhood.begin() + neighborhood.size() / 2,
                             neighborhood.end());
            float medianVal = neighborhood[neighborhood.size() / 2];
            virtualDepthValue.push_back(medianVal);
        }

        return true;
    }

    bool VirtualToRealDepthFunc::appendDepthPoints(
            std::vector<float> &refDepthValue,
            std::vector<float> &virtualDepthValue,
            cv::Rect &roi) {
        std::vector<cv::Point> m_addedCoords;
        m_addedCoords.clear();

        if (m_refDepthImage.empty() || m_virtualDepthImage.empty()) {
            std::cerr << "[AppendPairsByRefDepthSlices] Empty input image.\n";
            return false;
        }
        if (m_refDepthImage.size() != m_virtualDepthImage.size()) {
            std::cerr << "[AppendPairsByRefDepthSlices] Image size mismatch.\n";
            return false;
        }

        // ====== 0) 参数：按你的深度单位自行确认 ======
        const float kSliceStep = 100.0f; // 单位=你真实深度图单位；若是mm请改500.0f
        const int stride = 4;          // 扫描步长：2更密，4更快
        const int virtR = SESOptions.virtualSearchRadius;
        const float minValid = 0.0f;

        // ====== 1) 小工具：读像素为float + 判断有效 + 邻域找最近有效(虚拟深度用) ======
        auto isValid = [&](float v) -> bool {
            return std::isfinite(v) && v > minValid;
        };

        auto readFloat = [&](const cv::Mat &img, int x, int y) -> float {
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
                return std::numeric_limits<float>::quiet_NaN();

            const int depth = img.depth();
            const int ch = img.channels();

            if (ch == 1) {
                switch (depth) {
                    case CV_32F:
                        return img.at<float>(y, x);
                    case CV_64F:
                        return static_cast<float>(img.at<double>(y, x));
                    case CV_16U:
                        return static_cast<float>(img.at<uint16_t>(y, x));
                    case CV_8U:
                        return static_cast<float>(img.at<uint8_t>(y, x));
                    default:
                        return std::numeric_limits<float>::quiet_NaN();
                }
            } else {
                // 多通道时取第0通道（和你之前一致）
                if (depth == CV_32F && ch >= 3) {
                    cv::Vec3f v = img.at<cv::Vec3f>(y, x);
                    return v[0];
                }
                return std::numeric_limits<float>::quiet_NaN();
            }
        };

        // 在固定坐标(x,y)上取值；若无效则在半径R内找最近的有效值（坐标不变，只取邻域值）
        auto getWithNeighborhood = [&](const cv::Mat &img, int x, int y, int R) -> float {
            float v0 = readFloat(img, x, y);
            if (isValid(v0)) return v0;

            float best = 0.0f;
            int bestD2 = std::numeric_limits<int>::max();

            for (int dy = -R; dy <= R; ++dy) {
                for (int dx = -R; dx <= R; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    float v = readFloat(img, x + dx, y + dy);
                    if (!isValid(v)) continue;

                    int d2 = dx * dx + dy * dy;
                    if (d2 < bestD2) {
                        bestD2 = d2;
                        best = v;
                    }
                }
            }
            return best; // 若没找到则为0
        };

        // ====== 2) 统计真实深度的bin范围（按 0.5m 分段） ======
        float dmin = std::numeric_limits<float>::infinity();
        float dmax = -std::numeric_limits<float>::infinity();

        // 留边：虚拟深度要做邻域查找，避免越界
        const int margin = std::max(virtR, 1);

        cv::Rect roiClamped = roi;
        if (roiClamped.width <= 0 || roiClamped.height <= 0) {
            roiClamped = cv::Rect(0, 0, m_refDepthImage.cols, m_refDepthImage.rows);
        } else {
            roiClamped &= cv::Rect(0, 0, m_refDepthImage.cols, m_refDepthImage.rows);
        }
        if (roiClamped.empty()) {
            std::cerr << "[AppendPairsByRefDepthSlices] ROI empty after clamp.\n";
            return false;
        }

// 留边后真正可扫的范围（避免邻域搜索越界）
        const int xBeg = roiClamped.x + margin;
        const int xEnd = roiClamped.x + roiClamped.width - margin;
        const int yBeg = roiClamped.y + margin;
        const int yEnd = roiClamped.y + roiClamped.height - margin;

        if (xBeg >= xEnd || yBeg >= yEnd) {
            std::cerr << "[AppendPairsByRefDepthSlices] ROI too small for margin.\n";
            return false;
        }

//        for (int y = margin; y < m_refDepthImage.rows - margin; y += stride)
//        {
//            for (int x = margin; x < m_refDepthImage.cols - margin; x += stride)
        for (int y = yBeg; y < yEnd; y += stride) {
            for (int x = xBeg; x < xEnd; x += stride) {
                float d = readFloat(m_refDepthImage, x, y); // 真实深度：只认本点有效（不建议对真实深度“借邻域”，会破坏对应坐标）
                if (!isValid(d)) continue;
                dmin = std::min(dmin, d);
                dmax = std::max(dmax, d);
            }
        }

        if (!std::isfinite(dmin) || !std::isfinite(dmax) || dmax <= dmin) {
            std::cerr << "[AppendPairsByRefDepthSlices] Not enough valid ref depth.\n";
            return false;
        }

        const int minBin = (int) std::floor(dmin / kSliceStep);
        const int maxBin = (int) std::floor(dmax / kSliceStep);
        const int numBins = maxBin - minBin + 1;

        if (numBins <= 0)
            return false;

        // ====== 3) 标记你“已有 refDepthValue”覆盖了哪些bin（避免重复补同深度段） ======
        std::vector<uint8_t> covered(numBins, 0);
        for (float d: refDepthValue) {
            if (!isValid(d)) continue;
            int b = (int) std::floor(d / kSliceStep);
            int idx = b - minBin;
            if (idx >= 0 && idx < numBins) covered[idx] = 1;
        }

        // ====== 4) 扫图：为每个未覆盖bin挑一个“最接近bin中心”的点，且虚拟深度可取到 ======
        struct Pick {
            bool ok = false;
            float score = std::numeric_limits<float>::infinity();
            cv::Point pt;
            float realD = 0.f;
            float virtD = 0.f;
        };
        std::vector<Pick> best(numBins);

//        for (int y = margin; y < m_refDepthImage.rows - margin; y += stride)
//        {
//            for (int x = margin; x < m_refDepthImage.cols - margin; x += stride)
        for (int y = yBeg; y < yEnd; y += stride) {
            for (int x = xBeg; x < xEnd; x += stride) {
                float realD = readFloat(m_refDepthImage, x, y);
                if (!isValid(realD)) continue;

                int b = (int) std::floor(realD / kSliceStep);
                int idx = b - minBin;
                if (idx < 0 || idx >= numBins) continue;
                if (covered[idx]) continue; // 该深度段你已有点了，不再补

                // 虚拟深度：允许邻域补；补不到就放弃这个像素
                float virtD = getWithNeighborhood(m_virtualDepthImage, x, y, virtR);
                if (!isValid(virtD)) continue;

                float binCenter = ((b + 0.5f) * kSliceStep);
                float score = std::fabs(realD - binCenter);

                if (!best[idx].ok || score < best[idx].score) {
                    best[idx].ok = true;
                    best[idx].score = score;
                    best[idx].pt = cv::Point(x, y);
                    best[idx].realD = realD;
                    best[idx].virtD = virtD;
                }
            }
        }

        // ====== 5) 按bin从近到远追加到 refDepthValue / virtualDepthValue，并记录坐标 ======
        size_t beforeN = refDepthValue.size();

        for (int i = 0; i < numBins; ++i) {
            if (!best[i].ok) continue;

            // 最终再做一次“双方非0有效”确认；有问题就跳过
            if (!isValid(best[i].realD) || !isValid(best[i].virtD))
                continue;

            refDepthValue.push_back(best[i].realD);
            virtualDepthValue.push_back(best[i].virtD);
            m_addedCoords.push_back(best[i].pt);
        }

        // 保证一一对应
        if (refDepthValue.size() != virtualDepthValue.size()) {
            // 回滚到调用前的长度（强一致）
            refDepthValue.resize(beforeN);
            virtualDepthValue.resize(beforeN);
            m_addedCoords.clear();
            std::cerr << "[AppendPairsByRefDepthSlices] Pair size mismatch, rollback.\n";
            return false;
        }

        // 没补到也算成功（说明对应深度段确实缺数据/太稀疏）
        return true;
    }

    std::array<double, 3> VirtualToRealDepthFunc::BehavioralModel(std::vector<float> &refDepthValue,
                                                                  std::vector<float> &virtualDepthValue) {
        CV_Assert(refDepthValue.size() == virtualDepthValue.size());
        const int N = static_cast<int>(refDepthValue.size());
        CV_Assert(N >= 3); // 至少 3 个点才能估 3 个参数

        cv::Mat X(N, 3, CV_64F); // [u, v, 1]
        cv::Mat y(N, 1, CV_64F); // a_L

        for (int i = 0; i < N; ++i) {
            const double aL = refDepthValue[i];
            const double v = virtualDepthValue[i];
            const double u = aL * v;           // u = a_L * v

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
        return {c.at<double>(0, 0), c.at<double>(1, 0), c.at<double>(2, 0)};
    }

    // 用行为模型c0,c1,c2，把虚拟深度转为真实深度
    cv::Mat VirtualToRealDepthFunc::convertVirtualToRealDepth(std::array<double, 3> &behaviorModelParams) {
        // 参数 c0, c1, c2
        const double c0 = behaviorModelParams[0];
        const double c1 = behaviorModelParams[1];
        const double c2 = behaviorModelParams[2];

        if (m_virtualDepthImage.empty()) {
            std::cerr << "[Error] Virtual depth map is empty!" << std::endl;
            return cv::Mat();
        }

        cv::Mat v;
        m_virtualDepthImage.convertTo(v, CV_64F);

        // 计算 zC = (v * c1 + c2) / (1 - v * c0)
        cv::Mat numerator = v * c1 + c2;
        cv::Mat denominator = 1.0 - v * c0;
        cv::Mat zC;
        cv::divide(numerator, denominator, zC);

        // 去除 NaN、Inf、负值
        for (int y = 0; y < zC.rows; ++y) {
            double *row = zC.ptr<double>(y);
            for (int x = 0; x < zC.cols; ++x) {
                double &val = row[x];
                if (std::isnan(val) || std::isinf(val) || val < 0)
                    val = 0.0;
            }
        }
        return zC; // 返回真实深度图 CV_64F
    }

    //   计算行为模型转换的真实深度与相机坐标系下的激光雷达真实深度的绝对误差--采样
    void VirtualToRealDepthFunc::imageDistanceSampling(cv::Mat realDepthImage, cv::Mat refDepthImage) {
        if (m_samplePoints.empty()) {
            printf("采样点：m_samplePoints为空！");
            return;
        }

        int outlierWindow = 5;     // 过滤噪点局部窗口
        double outlierK = 2.0;     // 局部窗口检测异常值的阈值: outlierK * MAD（中值）

        if (realDepthImage.empty() || refDepthImage.empty()) {
            std::cerr << "[Error] Input image is empty.\n";
        }
        if (realDepthImage.size() != refDepthImage.size()) {
            std::cerr << "[Error] Image sizes do not match.\n";
        }

        // 统一通道
        if (realDepthImage.channels() == 3) {
            std::vector<cv::Mat> ch;
            cv::split(realDepthImage, ch);
            realDepthImage = ch[0];
        }
        if (refDepthImage.channels() == 3) {
            std::vector<cv::Mat> ch;
            cv::split(refDepthImage, ch);
            refDepthImage = ch[0];
        }

        // 统一数据类型
        if (realDepthImage.type() == CV_64F)
            realDepthImage.convertTo(realDepthImage, CV_32F);
        if (refDepthImage.type() == CV_32FC1)
            refDepthImage.convertTo(refDepthImage, CV_32F);

        // 局部噪点检测与剔除
        cv::Mat mask_outlier;
        mask_outlier = buildOutlierMask(refDepthImage, outlierWindow, outlierK);
        mask_outlier.convertTo(mask_outlier, CV_8U);

        m_rdImage_rows = realDepthImage.rows;
        m_rdImage_cols = realDepthImage.cols;


        // 局部取值（带邻域回退）
        auto getValidRefValue = [&](const cv::Mat &img, int x, int y, int searchRadius) -> float {
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
                return std::numeric_limits<float>::quiet_NaN();

            // 优先取当前像素
            float val = img.at<float>(y, x);
            if (std::isfinite(val) && val > 0.0f)
                return val;

            // 若无效则在邻域内搜索最近的有效点
            float nearest = std::numeric_limits<float>::quiet_NaN();
            float minDist2 = 1e9;

            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= img.rows) continue;

                for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= img.cols) continue;

                    float v = img.at<float>(yy, xx);
                    if (std::isfinite(v) && v > 0.0f) {
                        float dist2 = dx * dx + dy * dy;
                        if (dist2 < minDist2) {
                            minDist2 = dist2;
                            nearest = v;
                        }
                    }
                }
            }
            return nearest;  // 若全无有效点则返回 NaN
        };

        // 采样点误差计算
        std::vector<float> diffs;
        diffs.reserve(m_coordsVirtual.size() * SESOptions.sampleCount);

        for (int i = 0; i < m_samplePoints.size(); i++) {
            int x = m_samplePoints[i].x;
            int y = m_samplePoints[i].y;
            if (mask_outlier.at<uchar>(y, x))
                continue;

            float valPred = realDepthImage.at<float>(y, x);
            //   float valPred = getValidRefValue(realDepthImage, x, y, SESOptions.virtualSearchRadius);
            float valRef = getValidRefValue(refDepthImage, x, y, SESOptions.virtualSearchRadius);

            if (!std::isfinite(valPred) || !std::isfinite(valRef))
                continue;

            float diff = std::abs(valPred - valRef);
            if (diff < SESOptions.outlierDistanceThreshold) {
                diffs.push_back(diff);
                LOG_WARN("Sample:(", x, ",", y, ")---Diff:", diff);
                std::cout << "Sample (" << x << "," << y << ")  "
                          << "Pred=" << valPred << "  "
                          << "Ref=" << valRef << "  "
                          << "Diff=" << diff << std::endl;

            }
        }

        /*//  误差计算

             std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
             std::uniform_int_distribution<int> dist(-localWindowSize / 2, localWindowSize / 2);

             std::vector<float> diffs;  // 所有采样点的绝对误差
             diffs.reserve(coords_virtual.size() * sampleCount);

             for (const auto &p: coords_virtual) {
                 for (int s = 0; s < sampleCount; ++s) {
                     int dx = dist(rng);
                     int dy = dist(rng);
                     int x = p.x + dx;
                     int y = p.y + dy;

                     if (x < 0 || y < 0 || x >= m_rdImage_cols || y >= m_rdImage_rows)
                         continue;
                     if (rejectLocalOutliers && mask_outlier.at<uchar>(y, x))
                         continue;

                     float valPred = realDepthImage.at<float>(y, x);
                  //   float valRef = refDepthImage.at<float>(y, x);
                     float valRef = getValidRefValue(refDepthImage, x, y, virtualSearchRadius);

                     if (!std::isfinite(valPred) || !std::isfinite(valRef))
                         continue;

                     float diff = std::abs(valPred - valRef);
                     if (diff < outlierDistanceThreshold)
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
             }*/

        if (diffs.empty()) {
            std::cerr << "No valid sampled points.\n";
        }

        //  计算四种误差指标: MSE, RMSE, MAE, Median
        double chamfer = 0.0, euclidean = 0.0, mean = 0.0, median = 0.0;
        size_t N = diffs.size();

        for (float d: diffs) {
            chamfer += d * d;
            euclidean += d * d;
            mean += d;
        }
        chamfer = chamfer / (N + 1e-12);
        euclidean = std::sqrt(euclidean / (N + 1e-12));
        mean = mean / (N + 1e-12);

        std::sort(diffs.begin(), diffs.end());
        median = diffs[diffs.size() / 2];

        LOG_WARN("Euclidean (RMSE): ", euclidean, "  Mean (MAE):", mean, "  Median:", median);

        std::cout << "Random-sampled " << N << " points in local " << SESOptions.localWindowSize
                  << "x" << SESOptions.localWindowSize << " window:" << std::endl;
        std::cout << "  Chamfer (MSE):     " << chamfer << std::endl;
        std::cout << "  Euclidean (RMSE):  " << euclidean << std::endl;
        std::cout << "  Mean (MAE):        " << mean << std::endl;
        std::cout << "  Median:            " << median << std::endl;
    }


    void VirtualToRealDepthFunc::SamplePointSelectByLW() {
        m_samplePoints.clear();
        if (m_coordsVirtual.empty()) {
            printf("m_coordsVirtual is null!");
            return;
        }

        std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
        std::uniform_int_distribution<int> dist(-SESOptions.localWindowSize / 2, SESOptions.localWindowSize / 2);

        for (int i = 0; i < m_coordsVirtual.size(); i++) {
            for (int s = 0; s < SESOptions.sampleCount; ++s) {
                cv::Point p = m_coordsVirtual[i];
                int dx = dist(rng);
                int dy = dist(rng);
                int x = p.x + dx;
                int y = p.y + dy;
                if (x < 0 || y < 0 || x >= m_rdImage_cols || y >= m_rdImage_rows)
                    continue;
                m_samplePoints.emplace_back(x, y);
            }
        }
    }

    cv::Mat VirtualToRealDepthFunc::buildOutlierMask(cv::Mat &img, int ksize, double k) {
        CV_Assert(img.type() == CV_32FC1);

        cv::Mat med, absdiff, mad;
        cv::medianBlur(img, med, ksize);      // 局部中位数
        cv::absdiff(img, med, absdiff);
        cv::medianBlur(absdiff, mad, ksize);  // 局部MAD

        mad += 1e-9; // 防止除零
        cv::Mat mask = absdiff > (k * mad);   // mask=1 表示该点为噪点
        return mask;
    }

    bool VirtualToRealDepthFunc::loadPointsXML(std::string &pointsXmlPath) {
        cv::FileStorage fs(pointsXmlPath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open " << pointsXmlPath << " for reading\n";
            return false;
        }

        auto readSeq = [&](const cv::FileNode &node, std::vector<cv::Point> &out) {
            out.clear();
            if (node.type() != cv::FileNode::SEQ) return;
            for (auto it = node.begin(); it != node.end(); ++it) {
                int x = (int) (*it)["x"];
                int y = (int) (*it)["y"];
                out.emplace_back(x, y);
            }
        };

        readSeq(fs["coords_casd"], m_coordsRef);
        readSeq(fs["coords_virtual"], m_coordsVirtual);
        fs.release();
        return true;
    }

    cv::Mat
    VirtualToRealDepthFunc::drawRandomColorCrosses(const cv::Mat &focuseImage, const std::vector<cv::Point> &points) {
        int size = 24;
        int thickness = 8;
        int seed = -1;

        CV_Assert(!focuseImage.empty());
        CV_Assert(size > 0 && thickness > 0);

        // 1) 确保是三通道彩色图（便于画彩色）
        cv::Mat canvas;
        if (focuseImage.channels() == 3 && focuseImage.type() == CV_8UC3) {
            canvas = focuseImage.clone();
        } else if (focuseImage.channels() == 1) {
            // 灰度 -> BGR
            if (focuseImage.type() != CV_8UC1) {
                cv::Mat tmp8u;
                focuseImage.convertTo(tmp8u, CV_8U, 1.0); // 简单线性转换为8位
                cv::cvtColor(tmp8u, canvas, cv::COLOR_GRAY2BGR);
            } else {
                cv::cvtColor(focuseImage, canvas, cv::COLOR_GRAY2BGR);
            }
        } else {
            // 其他类型/深度，先转8位再保证三通道
            cv::Mat tmp8u;
            focuseImage.convertTo(tmp8u, CV_8U);
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

        auto inBounds = [&](int x, int y) {
            return x >= 0 && y >= 0 && x < canvas.cols && y < canvas.rows;
        };

        // 3) 逐点画十字（水平+竖直两条线）
        for (const auto &p: points) {
            if (!inBounds(p.x, p.y)) continue;

            // 随机 BGR
            cv::Scalar color(col(rng), col(rng), col(rng));

            // 计算端点并裁剪到图像边界
            cv::Point left(std::max(0, p.x - size), p.y);
            cv::Point right(std::min(canvas.cols - 1, p.x + size), p.y);
            cv::Point up(p.x, std::max(0, p.y - size));
            cv::Point down(p.x, std::min(canvas.rows - 1, p.y + size));

            cv::line(canvas, left, right, color, thickness, cv::LINE_AA);
            cv::line(canvas, up, down, color, thickness, cv::LINE_AA);
        }

        return canvas;
    }

    // 挑选拟合多个行为模型参数的样本点
    void VirtualToRealDepthFunc::sampleVirtualDepthPointsByRegion(
            std::string& vdepth_path,
            std::string& gt_path,
            std::string& output_mark_path,
            std::string& output_csv)
    {
        std::vector<std::pair<float,float>> ranges =
                {
                        {4.2f, 4.4f},
                        {4.4f, 4.6f},
                        {4.6f, 4.8f},
                        {4.8f, 5.0f}
                };

        int points_per_range = 20;      // 每个区间的采样点数
        float neighbor_tol = 0.3f;      // 局部邻域内点距离小于该值时，认为该点是同一个点
        int minDist = 20;      // 相邻点之间的最小距离

        float init_value_tol_ratio = 0.01f;
        float max_value_tol_ratio  = 0.05f;
        float value_tol_step_ratio = 0.001f;

        //  区域划分
        int gridCols = 5;      // 划分为n*n
        int gridRows = 5;
        int primaryCellsRange = 5;   // primary：只用前几个格子
        int fallbackExtraCells = 3;    // fallback：额外再放宽几个格子候选
        float min_gt_valid_ratio = 0.00005f;     // 格子中最小有效GT值比例
        int min_range_pixel_count = 20;     // 格子中最小有效vd像素数

        std::vector<cv::Scalar> colors
                {
//                        cv::Scalar(255,0,0),
//                        cv::Scalar(0,255,0),
                        cv::Scalar(0,0,255),
                        cv::Scalar(0,255,255),
                        cv::Scalar(255,0,255),
                        cv::Scalar(255,255,0)
                };

        struct Sample
        {
            int colIndex;
            int rowIndex;
            float vDepth;
            float rDepth;
        };

        struct Candidate
        {
            int x;
            int y;
            float v;
            float gt;
            int gx;
            int gy;
            float score;
            int cell_id;
        };

        struct BehaviorSegmentResult
        {
            float depthMin;
            float depthMax;
            int sampleCount;
            std::array<double, 3> params;
        };

        struct CellInfo
        {
            int id;
            int x0;
            int y0;
            int x1;
            int y1;

            int total_pixels;
            int valid_v_pixels;
            int valid_gt_pixels;

            float valid_gt_ratio;

            int range_pixel_count;
            float range_mean_v;
            float priority_score;
        };

        cv::Mat depth = imread(vdepth_path, cv::IMREAD_UNCHANGED);
        if (depth.empty())
            std::cout<<"无法读取虚拟深度图"<<std:: endl;
        if (depth.channels() > 1)
            extractChannel(depth, depth, 0);
        depth.convertTo(depth, CV_32F);

        cv::Mat gt = imread(gt_path, cv::IMREAD_UNCHANGED);
        if (gt.empty())
            std::cout<<"无法读取GT"<<std:: endl;
        if (gt.channels() > 1)
            extractChannel(gt, gt, 0);
        gt.convertTo(gt, CV_32F);

        cv::Mat vis;
        {
            boost::filesystem::path depth_path(vdepth_path);
            boost::filesystem::path vis_path = depth_path.parent_path() / "fullfocus.png";

            vis = cv::imread(vis_path.string(), cv::IMREAD_COLOR);
            if (vis.empty())
                std::cout<<"无法读取可视化全聚焦图"<<std:: endl;
        }

        std::vector<Sample> samples;
        std::vector<cv::Point> selected_points;
        std::vector<BehaviorSegmentResult> segment_results;

        int w = depth.cols;
        int h = depth.rows;

        int cell_w = std::max(1, (w + gridCols - 1) / gridCols);
        int cell_h = std::max(1, (h + gridRows - 1) / gridRows);

 //----------------------------局部lambda变量(主函数在后面)------------------------------
        // 保证选取的点之间至少有minDist个像素的距离，避免重合
        auto isFarEnough = [&](int x, int y) -> bool
        {
            for (const auto& p : selected_points)
            {
                int dx = p.x - x;
                int dy = p.y - y;
                if (dx * dx + dy * dy < minDist * minDist)
                    return false;
            }
            return true;
        };

        // 获取局部稳定vd点（3*3区域内）
        auto localStabilityScore = [&](int x, int y, float v) -> float
        {
            float acc = 0.0f;
            int cnt = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0) continue;

                    float nv = depth.at<float>(y + dy, x + dx);
                    if (!std::isfinite(nv) || nv <= 0.0f) continue;

                    acc += std::fabs(nv - v);
                    cnt++;
                }
            }
            if (cnt == 0) return 1e9f;
            return acc / cnt;
        };

        auto getCellId = [&](int x, int y) -> int
        {
            int cx = std::min(gridCols - 1, std::max(0, x / cell_w));
            int cy = std::min(gridRows - 1, std::max(0, y / cell_h));
            return cy * gridCols + cx;
        };

        // 把整张图划分为 n*n 个格子，并统计每个格子的基础信息
        auto buildCells = [&]() -> std::vector<CellInfo>
        {
            std::vector<CellInfo> cells;
            cells.reserve(gridCols * gridRows);

            for (int gy = 0; gy < gridRows; ++gy)
            {
                for (int gx = 0; gx < gridCols; ++gx)
                {
                    CellInfo c;
                    c.id = gy * gridCols + gx;
                    c.x0 = gx * cell_w;
                    c.y0 = gy * cell_h;
                    c.x1 = std::min(w, (gx + 1) * cell_w);
                    c.y1 = std::min(h, (gy + 1) * cell_h);

                    c.total_pixels = 0;
                    c.valid_v_pixels = 0;
                    c.valid_gt_pixels = 0;
                    c.valid_gt_ratio = 0.0f;
                    c.range_pixel_count = 0;
                    c.range_mean_v = 0.0f;
                    c.priority_score = 0.0f;

                    for (int y = c.y0; y < c.y1; ++y)
                    {
                        for (int x = c.x0; x < c.x1; ++x)
                        {
                            c.total_pixels++;

                            float vv = depth.at<float>(y, x);
                            if (std::isfinite(vv) && vv > 0.0f)
                                c.valid_v_pixels++;

                            float gv = gt.at<float>(y, x);
                            if (std::isfinite(gv) && gv > 0.0f)
                                c.valid_gt_pixels++;
                        }
                    }

                    if (c.valid_v_pixels > 0)
                        c.valid_gt_ratio = static_cast<float>(c.valid_gt_pixels) / c.valid_v_pixels;
                    else
                        c.valid_gt_ratio = 0.0f;

                    cells.push_back(c);
                }
            }

            return cells;
        };

        // 统计格子区间的均值等，获取格子优先级
        auto updateCellsForRange = [&](std::vector<CellInfo>& cells, float rmin, float rmax)
        {
            for (auto& c : cells)
            {
                c.range_pixel_count = 0;
                c.range_mean_v = 0.0f;
                c.priority_score = -1e9f;

                double v_sum = 0.0;

                for (int y = c.y0; y < c.y1; ++y)
                {
                    for (int x = c.x0; x < c.x1; ++x)
                    {
                        float vv = depth.at<float>(y, x);
                        if (!std::isfinite(vv) || vv <= 0.0f)
                            continue;

                        if (vv >= rmin && vv < rmax)
                        {
                            c.range_pixel_count++;
                            v_sum += vv;
                        }
                    }
                }

                if (c.range_pixel_count > 0)
                    c.range_mean_v = static_cast<float>(v_sum / c.range_pixel_count);

                if (c.range_pixel_count >= min_range_pixel_count &&
                    c.valid_gt_ratio >= min_gt_valid_ratio)
                {
                    c.priority_score =
                            static_cast<float>(c.range_pixel_count) +
                            c.valid_gt_ratio * 1000.0f;
                }
            }
        };

        // 从格子中选择候选拟合点
        auto collectCandidatesInCell = [&](const CellInfo& cell,
                                           float rmin,
                                           float rmax,
                                           std::vector<Candidate>& out_candidates)
        {
            float center_v = 0.5f * (rmin + rmax);

            for (int y = std::max(1, cell.y0); y < std::min(h - 1, cell.y1); ++y)
            {
                for (int x = std::max(1, cell.x0); x < std::min(w - 1, cell.x1); ++x)
                {
                    float v = depth.at<float>(y, x);

                    if (!std::isfinite(v) || v <= 0.0f)
                        continue;

                    if (v < rmin || v >= rmax)
                        continue;

                    if (!isValidPixel(depth, x, y, v, neighbor_tol))
                        continue;

                    float gt_value;
                    int gx, gy;
                    if (!findStableGTValue(gt, x, y, gt_value, gx, gy, 43))
                        continue;

                    float stability = localStabilityScore(x, y, v);
                    float score = std::fabs(v - center_v) * 10.0f + stability;

                    out_candidates.push_back({x, y, v, gt_value, gx, gy, score, cell.id});
                }
            }
        };

        // 被选取参与拟合的点，在全聚焦图像用不同颜色标注
        auto addSampleToVis = [&](int idx, int selected_in_range, const cv::Scalar& color, const Candidate& c)
        {
            cv::circle(vis, cv::Point(c.x, c.y), 6, color, 2);

            std::string label = std::to_string(idx) + "-" + std::to_string(selected_in_range);
            cv::putText(vis, label, cv::Point(c.x + 5, c.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
        };


        auto sortCandidatesDeterministic = [&](std::vector<Candidate>& candidates)
        {
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b)
                      {
                          if (std::fabs(a.score - b.score) > 1e-6f)
                              return a.score < b.score;
                          if (a.cell_id != b.cell_id)
                              return a.cell_id < b.cell_id;
                          if (a.y != b.y)
                              return a.y < b.y;
                          return a.x < b.x;
                      });
        };
//----------------------------------------------------------
        //  主函数开始
        for (int idx = 0; idx < (int)ranges.size(); ++idx)
        {
            float rmin = ranges[idx].first;
            float rmax = ranges[idx].second;

            std::cout << "Range " << rmin << "-" << rmax << std::endl;

            cv::Scalar color = colors[idx % colors.size()];
            int selected_in_range = 0;

            std::vector<Sample> range_samples;

            std::vector<CellInfo> cells = buildCells();
            // Step 1: 统计格子区域的均值，计算格子优先级
            updateCellsForRange(cells, rmin, rmax);

            std::sort(cells.begin(), cells.end(),
                      [](const CellInfo& a, const CellInfo& b)
                      {
                          if (std::fabs(a.priority_score - b.priority_score) > 1e-6f)
                              return a.priority_score > b.priority_score;
                          return a.id < b.id;
                      });

            std::vector<CellInfo> primary_cells;    // 优先选取格子
            std::vector<CellInfo> fallback_cells;   // 候选格子

            for (const auto& c : cells)
            {
                if (c.priority_score <= 0.0f)
                    continue;

                if ((int)primary_cells.size() < primaryCellsRange)
                {
                    primary_cells.push_back(c);
                }
                else if ((int)fallback_cells.size() < fallbackExtraCells)
                {
                    fallback_cells.push_back(c);
                }

                if ((int)primary_cells.size() >= primaryCellsRange &&
                    (int)fallback_cells.size() >= fallbackExtraCells)
                    break;
            }

            std::cout << "  primary cells: ";
            for (const auto& c : primary_cells)
            {
                std::cout << "[id=" << c.id
                          << " rangeCount=" << c.range_pixel_count
                          << " gtRatio=" << c.valid_gt_ratio
                          << " meanV=" << c.range_mean_v
                          << "] ";
            }
            std::cout << std::endl;

            std::cout << "  fallback cells: ";
            for (const auto& c : fallback_cells)
            {
                std::cout << "[id=" << c.id
                          << " rangeCount=" << c.range_pixel_count
                          << " gtRatio=" << c.valid_gt_ratio
                          << " meanV=" << c.range_mean_v
                          << "] ";
            }
            std::cout << std::endl;

            if (primary_cells.empty())
            {
                std::cout << "  no valid primary cells for this range." << std::endl;
                continue;
            }

            // Step 2: primary pick拟合点选取：只从 primary格子里挑
            std::vector<Candidate> primary_candidates;
            primary_candidates.reserve(50000);

            for (const auto& c : primary_cells)
                collectCandidatesInCell(c, rmin, rmax, primary_candidates);

            {
                std::vector<Candidate> deduped;
                deduped.reserve(primary_candidates.size());
                std::set<std::pair<int,int>> used_xy;

                for (const auto& c : primary_candidates)
                {
                    std::pair<int,int> key(c.x, c.y);
                    if (used_xy.insert(key).second)
                        deduped.push_back(c);
                }

                primary_candidates.swap(deduped);
            }

            sortCandidatesDeterministic(primary_candidates);

            std::cout << "  primary candidate pool: " << primary_candidates.size() << std::endl;

            for (const auto& c : primary_candidates)
            {
                if ((int)range_samples.size() >= points_per_range)
                    break;

                if (!isFarEnough(c.x, c.y))
                    continue;

                Sample s{c.x, c.y, c.v, c.gt};
                samples.push_back(s);
                range_samples.push_back(s);
                selected_points.emplace_back(c.x, c.y);
                // 可视化参与拟合的样本点
                addSampleToVis(idx, selected_in_range, color, c);

                std::cout << "    pick: x=" << c.x
                          << " y=" << c.y
                          << " V=" << c.v
                          << " GT=" << c.gt
                          << " cell=" << c.cell_id
                          << " score=" << c.score
                          << std::endl;

                selected_in_range++;
            }

            // fallback pick：只从 fallback_cells 里补（候选备用，暂时不参与拟合）
            if ((int)range_samples.size() < points_per_range && !fallback_cells.empty())
            {
                std::vector<Candidate> fallback_candidates;
                fallback_candidates.reserve(50000);

                for (const auto& c : fallback_cells)
                    collectCandidatesInCell(c, rmin, rmax, fallback_candidates);

                {
                    std::vector<Candidate> deduped;
                    deduped.reserve(fallback_candidates.size());
                    std::set<std::pair<int,int>> used_xy;

                    for (const auto& c : fallback_candidates)
                    {
                        std::pair<int,int> key(c.x, c.y);
                        if (used_xy.insert(key).second)
                            deduped.push_back(c);
                    }

                    fallback_candidates.swap(deduped);
                }

                sortCandidatesDeterministic(fallback_candidates);

                std::cout << "  fallback candidate pool: " << fallback_candidates.size() << std::endl;

                for (const auto& c : fallback_candidates)
                {
                    if ((int)range_samples.size() >= points_per_range)
                        break;

                    if (!isFarEnough(c.x, c.y))
                        continue;

                    Sample s{c.x, c.y, c.v, c.gt};
                    samples.push_back(s);
                    range_samples.push_back(s);
                    selected_points.emplace_back(c.x, c.y);

                    addSampleToVis(idx, selected_in_range, color, c);

                    std::cout << "    fallback pick: x=" << c.x
                              << " y=" << c.y
                              << " V=" << c.v
                              << " GT=" << c.gt
                              << " cell=" << c.cell_id
                              << " score=" << c.score
                              << std::endl;

                    selected_in_range++;
                }
            }

            std::cout << "Range done, selected count: " << selected_in_range << std::endl;
            // Step 3: 分段拟合行为模型参数
            if ((int)range_samples.size() >= 8)
            {
                std::vector<float> refDepthValue;
                std::vector<float> virtualDepthValue;
                refDepthValue.reserve(range_samples.size());
                virtualDepthValue.reserve(range_samples.size());

                for (const auto& s : range_samples)
                {
                    virtualDepthValue.push_back(s.vDepth);
                    refDepthValue.push_back(s.rDepth);
                }

                std::array<double, 3> behaviorModelParams =
                        BehavioralModel(refDepthValue, virtualDepthValue);

                BehaviorSegmentResult seg;
                seg.depthMin = rmin;
                seg.depthMax = rmax;
                seg.sampleCount = (int)range_samples.size();
                seg.params = behaviorModelParams;
                segment_results.push_back(seg);

                std::cout << "  BehavioralModel fitted for range ["
                          << rmin << ", " << rmax << "], sampleCount="
                          << range_samples.size()
                          << ", params=("
                          << behaviorModelParams[0] << ", "
                          << behaviorModelParams[1] << ", "
                          << behaviorModelParams[2] << ")"
                          << std::endl;
            }
            else
            {
                std::cout << "  Skip BehavioralModel for range ["
                          << rmin << ", " << rmax
                          << "], sampleCount=" << range_samples.size()
                          << " (require >= 8 samples)" << std::endl;
            }
        }

        imwrite(output_mark_path, vis);

        // 记录样本点，写入csv文件中
        std::ofstream csv(output_csv);
        csv << "x,y,virtual_depth,real_depth\n";
        for (auto& s : samples)
        {
            csv << s.colIndex << ","
                << s.rowIndex << ","
                << s.vDepth << ","
                << s.rDepth << "\n";
        }
        csv.close();

        // Step 4: 写出拟合结果xml文件
        {
            std::time_t t = std::time(nullptr);
            std::tm tm_now;
            localtime_r(&t, &tm_now);

            char time_buf[32] = {0};
            std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", &tm_now);

            std::string xml_path;
            {
                boost::filesystem::path csv_path(output_csv);
                boost::filesystem::path parent_dir = csv_path.parent_path();
                xml_path = (parent_dir / ("behaviorModelParams_" + std::string(time_buf) + ".xml")).string();
            }

            std::ofstream xml(xml_path);
            if (!xml.is_open())
            {
                std::cerr << "无法写入 XML: " << xml_path << std::endl;
            }
            else
            {
                xml << std::fixed << std::setprecision(6);

                xml << "<?xml version=\"1.0\"?>\n";
                xml << "<opencv_storage>\n";
                xml << "    <BehaviorModelSegments>\n\n";

                for (const auto& seg : segment_results)
                {
                    xml << "        <Segment>\n";
                    xml << "            <DepthMin>" << seg.depthMin << "</DepthMin>\n";
                    xml << "            <DepthMax>" << seg.depthMax << "</DepthMax>\n\n";
                    xml << "            <Param>\n";
                    xml << "                <c0>" << seg.params[0] << "</c0>\n";
                    xml << "                <c1>" << seg.params[1] << "</c1>\n";
                    xml << "                <c2>" << seg.params[2] << "</c2>\n";
                    xml << "            </Param>\n";
                    xml << "        </Segment>\n\n";
                }

                xml << "    </BehaviorModelSegments>\n";
                xml << "</opencv_storage>\n";
                xml.close();

                std::cout << "Behavior model XML saved: " << xml_path << std::endl;
            }
        }

        std::cout << "CSV保存: " << output_csv << std::endl;
        std::cout << "采样完成，总点数: " << samples.size() << std::endl;
    }

    // 获取局部稳定vd像素点
    bool VirtualToRealDepthFunc::isValidPixel(
            cv::Mat& depth, int x, int y, float value, float neighbor_tol)
    {
        int w = depth.cols;
        int h = depth.rows;

        int valid_cnt = 0;
        int consistent_cnt = 0;
        float diff_sum = 0.0f;
        float max_diff = 0.0f;

        float tol = std::max(neighbor_tol, std::fabs(value) * 0.005f);

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;

                int nx = x + dx;
                int ny = y + dy;

                if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                    continue;

                float v = depth.at<float>(ny, nx);
                if (!std::isfinite(v) || v <= 0.0f)
                    continue;

                valid_cnt++;

                float diff = std::fabs(v - value);
                diff_sum += diff;
                max_diff = std::max(max_diff, diff);

                if (diff <= tol)
                    consistent_cnt++;
            }
        }

        if (valid_cnt < 4)
            return false;

        float consistent_ratio = static_cast<float>(consistent_cnt) / valid_cnt;
        float mean_diff = diff_sum / valid_cnt;

        return (consistent_ratio >= 0.5f &&
                mean_diff <= tol * 1.2f &&
                max_diff <= tol * 3.0f);
    }

    bool VirtualToRealDepthFunc::isStableGTPixel(
            cv::Mat& gt,
            int x,
            int y,
            float value,
            float neighbor_tol_ratio)
    {
        if (x < 0 || x >= gt.cols || y < 0 || y >= gt.rows)
            return false;

        if (!std::isfinite(value) || value <= 0.0f)
            return false;

        return true;
    }

    // 获取GT像素值
    bool VirtualToRealDepthFunc::findStableGTValue(
            cv::Mat& gt,
            int vd_x,
            int vd_y,
            float& gt_value,
            int& gt_x,
            int& gt_y,
            int max_radius)
    {
        struct Candidate
        {
            int x;
            int y;
            float value;
            float dist2;
            float weight;
        };

        const float eps = 1e-6f;
        const int min_valid_points = 2;     // 指定区域内最小有效GT点数
        const float value_tol_ratio = 0.2f;   // 相对中位数容差
        const float min_abs_tol = 50.0f;      // 绝对最小容差，按GT单位调

        std::vector<Candidate> candidates;
        candidates.reserve((2 * max_radius + 1) * (2 * max_radius + 1));

        for (int dy = -max_radius; dy <= max_radius; ++dy)
        {
            for (int dx = -max_radius; dx <= max_radius; ++dx)
            {
                int nx = vd_x + dx;
                int ny = vd_y + dy;

                if (nx < 0 || nx >= gt.cols || ny < 0 || ny >= gt.rows)
                    continue;

                float v = gt.at<float>(ny, nx);

                if (!isStableGTPixel(gt, nx, ny, v, 0.0f))
                    continue;

                float dist2 = static_cast<float>(dx * dx + dy * dy);
                float weight = 1.0f / (dist2 + eps);

                candidates.push_back({nx, ny, v, dist2, weight});
            }
        }

        if ((int)candidates.size() < min_valid_points)
            return false;

        // 先求中位数
        std::vector<float> values;
        values.reserve(candidates.size());
        for (const auto& c : candidates)
            values.push_back(c.value);

        std::sort(values.begin(), values.end());
        float median = values[values.size() / 2];

        float tol = std::max(min_abs_tol, std::fabs(median) * value_tol_ratio);

        // 过滤离群值后再加权平均
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        int inlier_count = 0;

        for (const auto& c : candidates)
        {
            if (std::fabs(c.value - median) > tol)
                continue;

            weighted_sum += static_cast<double>(c.value) * c.weight;
            weight_sum += c.weight;
            inlier_count++;
        }

        if (inlier_count < min_valid_points || weight_sum <= eps)
            return false;

        gt_value = static_cast<float>(weighted_sum / weight_sum);

        // 返回最近的inlier点坐标
        bool found_nearest = false;
        float best_dist2 = std::numeric_limits<float>::max();

        for (const auto& c : candidates)
        {
            if (std::fabs(c.value - median) > tol)
                continue;

            if (c.dist2 < best_dist2)
            {
                best_dist2 = c.dist2;
                gt_x = c.x;
                gt_y = c.y;
                found_nearest = true;
            }
        }

        return found_nearest;
    }
}