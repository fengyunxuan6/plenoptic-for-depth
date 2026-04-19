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
#include "LFRefocus.h"
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

    void VirtualToRealDepthFunc::VirtualToRealDepth(QuadTreeProblemMapMap::iterator &itrP)
    {
        switch (m_VirtualToRealDepthType)
        {
            case VTORD_Behavioralmodel:
            {
                VirtualToRealDepthByBM();
            }
                break;
            case VTORD_SegmentBehavioralmodel:
            {
                VirtualToRealDepthBySegBM();
            }
                break;
            case VTORD_SegmentBehavioralmodel_2:
            {
                VirtualToRealDepthBySegBM_2();
            }
                break;
            default:
                break;
        }
    }

    void VirtualToRealDepthFunc::SamplePointSelect()
    {
        switch (m_SamplePointSelectType)
        {
            case SPSelectByLocalWindow:
            {
                SamplePointSelectByLW();
            }
                break;
            case SPSelectByRandom:
            {
                SamplePointSelectByRandom();
            }
                break;
            default:
                break;
        }
    }

    void VirtualToRealDepthFunc::VirtualToRealDepthBySegBM_2()
    {
        m_strRootPath = m_ptrDepthSolver->GetRootPath();
        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/VD_Raw.tiff";
        std::string refDepthImg_path = m_strRootPath + "/behavior_model/ref-csad-rd.tiff";
        std::string focuseImg_Path = m_strRootPath + "/behavior_model/fullfocus.png";
        std::string vDImg_marked_path = m_strRootPath + "/behavior_model/m_vIDepth_Marked.png";
        std::string distanceImage_path = m_strRootPath + "/behavior_model/distanceImage.png";
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME;
        std::string xml_path = strCalibPath +"behaviorModelParamsSegment.xml";

        m_virtualDepthImage = cv::imread(virtualDepthImg_path, cv::IMREAD_UNCHANGED);  // 读取虚拟深度图
        m_refDepthImage = cv::imread(refDepthImg_path, cv::IMREAD_UNCHANGED);   // 读取激光深度图
        cv::Mat focusImage = cv::imread(focuseImg_Path);                  // 读取全聚焦图

        // Step1: 对GT分段
        segmentByGT(m_refDepthImage, focusImage);

        // Step2：挑选GT点
        selectGtPoints();

        // Step3：找对应vd点
        selectVdPoints();

        // Step4：拟合行为模型参数
        fitSegmentsParams(xml_path);

        // Step5：统计误差
        // TODO:拆分函数
        SamplePointSelect();
        cv::Mat realDepthImage = ConvertVdImageToRd(m_virtualDepthImage);
        errorStatisticsImageGTSeg(realDepthImage,focusImage, distanceImage_path);
    }

    void VirtualToRealDepthFunc::segmentByGT(cv::Mat &refDepthImage,cv::Mat &focusImage)
    {
        if(refDepthImage.empty())
        {
            std::cout << "refDepthImage is empty!" << std::endl;
        }

        if (refDepthImage.channels() > 1)
            extractChannel(refDepthImage, refDepthImage, 0);
        refDepthImage.convertTo(refDepthImage, CV_32F);

        float x = 1.0f;

        float maxDepth = 0.0f;
        float minDepth = std::numeric_limits<float>::infinity();

        std::string filename = m_strRootPath + "/group";

        for (int r = 0; r < refDepthImage.rows; ++r)
        {
            for (int c = 0; c < refDepthImage.cols; ++c)
            {
                float gtDepth = refDepthImage.at<float>(r, c);

                if (gtDepth <= 0 || std::isnan(gtDepth) || std::isinf(gtDepth))
                    continue;

                if (gtDepth > maxDepth)
                    maxDepth = gtDepth;
                if (gtDepth < minDepth)
                    minDepth = gtDepth;
            }
        }

        int startBin = static_cast<int>(std::floor(minDepth/1000.0f));
        int endBin  = static_cast<int>(std::ceil(maxDepth/1000.0f));
        int n = static_cast<int>(std::ceil(endBin-startBin));
        samplePointsVector.resize(endBin+1);
        samplePointsVectorFiltered.resize(endBin+1);
        int count = 0;

        for (int r = 0; r < refDepthImage.rows; ++r)
        {
            for (int c = 0; c < refDepthImage.cols; ++c)
            {
                float gtDepth = refDepthImage.at<float>(r, c);

                if (gtDepth <= 0 || std::isnan(gtDepth) || std::isinf(gtDepth))
                    continue;

                int groupIdx = static_cast<int>(gtDepth/1000.0f);

                if (groupIdx < 0 || groupIdx > endBin)
                    continue;

                count++;
                SamplePoint pt;
                pt.colIndex = c;
                pt.rowIndex = r;
                pt.gtDepth = gtDepth;
                samplePointsVector[groupIdx].push_back(pt);    // 输出：n组GT点集合samplePointsVector
            }
        }
        std::cout << "All gt points numbers: " << count << std::endl;
    }

    void VirtualToRealDepthFunc::selectGtPoints()
    {
        // 输入：n组GT点集合samplePointsVector
        if(samplePointsVector.empty())
        {
            std::cout << "n组GT点集合samplePointsVector is empty!" << std::endl;
            return;
        }
        for(int i =0;i < samplePointsVector.size();i++)
        {
            std::vector<SamplePoint>& groupPoints = samplePointsVector[i];
            if (groupPoints.empty())
                continue;

            for(int j = 00; j < groupPoints.size(); j++)
            {
                SamplePoint& pt = groupPoints[j];
                int row = groupPoints[j].rowIndex;
                int col = groupPoints[j].colIndex;

                // 当前值：这里用 gtDepth
                float centerValue = groupPoints[j].gtDepth;
//                std::cout << "当前point: " << groupPoints[j].colIndex << "; " << groupPoints[j].rowIndex << "; "<< groupPoints[j].gtDepth<< std::endl;

                std::vector<float> neighborValues;
                // 自适应窗口参数
                int startRadius = 1;
                int maxRadius = 70;
                int minNeighborCount = 4;

                for (int radius = startRadius; radius <= maxRadius; radius++)
                {
                    int rMin = std::max(0, row - radius);
                    int rMax = std::min(m_refDepthImage.rows - 1, row + radius);
                    int cMin = std::max(0, col - radius);
                    int cMax = std::min(m_refDepthImage.cols - 1, col + radius);

                    // 只扫描当前新增的一圈边界
                    for (int r = rMin; r <= rMax; r++)
                    {
                        for (int c = cMin; c <= cMax; c++)
                        {
                            // 只处理当前半径这一圈，不扫内部旧区域
                            if (r != rMin && r != rMax && c != cMin && c != cMax)
                                continue;

                            if (r == row && c == col)
                                continue;

                            float val = m_refDepthImage.at<float>(r, c);
                            if (val > 0 && std::isfinite(val))
                            {
                                neighborValues.push_back(val);
//                                std::cout << "邻域点radius: " << radius<< "; ("<< c << "; " << r << "); " << val << std::endl;
                            }
                        }
                    }
                    if (neighborValues.size() >= minNeighborCount)
                    {
                        break;
                    }
                }

                int count = neighborValues.size();
                if(count == 0)
                    continue;

                float sumSq = 0.0f;
                for(int k = 0; k < count; k++)
                {
                    float diff = neighborValues[k] - centerValue;
                    sumSq += diff * diff;
                }

                float rmse = std::sqrt(sumSq / count);

                if(rmse < 1000.0f)
                {
                    samplePointsVectorFiltered[i].push_back(pt);
//                    std::cout << "point is ok: RMSE"<< rmse << groupPoints[j].colIndex << "; " << groupPoints[j].rowIndex << "; "<< groupPoints[j].gtDepth<< std::endl;
                } else
                {
//                    std::cout << "point is filted: RMSE"<< rmse << groupPoints[j].colIndex << "; " << groupPoints[j].rowIndex << "; "<< groupPoints[j].gtDepth<< std::endl;
                }
            }
        }
        float step = 100.0f;
        int maxKeepPerSubGroup = 10;
        samplePoints.clear();

        for (int i = 0; i < samplePointsVectorFiltered.size(); i++)
        {
            std::vector<SamplePoint>& groupPoints = samplePointsVectorFiltered[i];
            if (groupPoints.empty())
                continue;

            // 按 gtDepth 从小到大排序
            std::sort(groupPoints.begin(), groupPoints.end(),
                      [](const SamplePoint& a, const SamplePoint& b)
                      {
                          return a.gtDepth < b.gtDepth;
                      });

            // 10个小组，分别对应 [i, i+0.1), [i+0.1, i+0.2), ..., [i+0.9, i+1.0)
            std::vector<std::vector<SamplePoint>> subGroups(10);

            for (int k = 0; k < groupPoints.size(); k++)
            {
                float depth = groupPoints[k].gtDepth;

                // 计算它在第几个 0.1 小段
                int subIndex = static_cast<int>((depth - i) / step);

                // 处理边界，避免越界
                if (subIndex < 0)
                    continue;
                if (subIndex >= 10)
                    subIndex = 9;

                subGroups[subIndex].push_back(groupPoints[k]);
            }

            // 每个0.1小组最多保留10个
            for (int s = 0; s < 10; s++)
            {
                int keepCount = std::min(static_cast<int>(subGroups[s].size()), maxKeepPerSubGroup);

                for (int t = 0; t < keepCount; t++)
                {
                    samplePoints.push_back(subGroups[s][t]);
                }
            }
        }
         // 输出：GT点集合samplePoints
    }

    void VirtualToRealDepthFunc::selectVdPoints()
    {
        // ============================================================
        // AIF 引导的局部表面一致性采样
        //
        // 核心思想：
        // 1) 当前 GT 点(samplePoints[i].rowIndex / colIndex)只是 seed / 引子点
        // 2) 在 vd 图中，结合 AIF(fullfocus.png) 构建“同一局部表面支持域”
        // 3) vd 和 GT 都在这个支持域上做鲁棒统计
        //
        // 局部表面一致性在这里具体体现为：
        // - 空间连通一致性
        // - vd 几何连续性
        // - AIF 外观一致性（Lab颜色差）
        // - AIF 强边界约束（梯度大时不跨边界）
        // ============================================================

        if (samplePoints.empty() || m_virtualDepthImage.empty() || m_refDepthImage.empty())
        {
            std::cout << "samplePoints / m_virtualDepthImage / m_refDepthImage is empty!" << std::endl;
            return;
        }

        // ------------------------------------------------------------
        // 1) 统一输入图像：单通道 float
        // ------------------------------------------------------------
        cv::Mat vdImg, gtImg;

        if (m_virtualDepthImage.channels() > 1)
            extractChannel(m_virtualDepthImage, vdImg, 0);
        else
            vdImg = m_virtualDepthImage.clone();

        if (m_refDepthImage.channels() > 1)
            extractChannel(m_refDepthImage, gtImg, 0);
        else
            gtImg = m_refDepthImage.clone();

        if (vdImg.type() != CV_32FC1)
            vdImg.convertTo(vdImg, CV_32F);

        if (gtImg.type() != CV_32FC1)
            gtImg.convertTo(gtImg, CV_32F);

        // ------------------------------------------------------------
        // 2) 读取 AIF（全聚焦图）作为 guide
        // ------------------------------------------------------------
        std::string aifPath = m_strRootPath + "/behavior_model/fullfocus.png";
        cv::Mat guideColor = cv::imread(aifPath, cv::IMREAD_COLOR);

        bool hasGuide = !guideColor.empty();
        cv::Mat guideGray, guideLab, guideGrad;

        if (hasGuide)
        {
            if (guideColor.size() != vdImg.size())
            {
                cv::resize(guideColor, guideColor, vdImg.size(), 0.0, 0.0, cv::INTER_LINEAR);
            }

            cv::cvtColor(guideColor, guideGray, cv::COLOR_BGR2GRAY);

            cv::Mat guideLab8U;
            cv::cvtColor(guideColor, guideLab8U, cv::COLOR_BGR2Lab);
            guideLab8U.convertTo(guideLab, CV_32FC3);

            cv::Mat blurGray;
            cv::GaussianBlur(guideGray, blurGray, cv::Size(5, 5), 1.0);

            cv::Mat gx, gy, mag;
            cv::Sobel(blurGray, gx, CV_32F, 1, 0, 3);
            cv::Sobel(blurGray, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, mag);

            double gmin = 0.0, gmax = 0.0;
            cv::minMaxLoc(mag, &gmin, &gmax);

            if (gmax > 1e-6)
                mag.convertTo(guideGrad, CV_32F, 255.0 / gmax);
            else
                guideGrad = cv::Mat::zeros(mag.size(), CV_32F);
        }
        else
        {
            std::cout << "[selectVdPoints] Warning: fullfocus.png not found. "
                      << "Fallback to vd-only support selection." << std::endl;
        }

        // ------------------------------------------------------------
        // 3) 参数
        // ------------------------------------------------------------
        const int   startRadius            = 1;
        const int   maxRadius              = 18;
        const int   minWindowValidCount    = 6;
        const int   minSupportCount        = 5;
        const int   gtDilateIter           = 1;
        const int   minGtSupportCount      = 1;

        const float maxSeedDriftPx         = 3.0f;
        const float maxCentroidDriftPx     = 2.0f;

        const float absBandMin             = 0.020f;
        const float madScale               = 2.5f;
        const float minMadFloor            = 0.003f;
        const float localJumpScale         = 1.6f;
        const float maxSupportSpanVD       = 0.12f;
        const float maxSupportSpanGT       = 3000.0f;

        const float maxSeedLabDiff         = 18.0f;
        const float maxGrowLabDiff         = 16.0f;
        const float strongEdgeThresh       = 45.0f;
        const float edgeDiffRelaxFactor    = 0.7f;

        // ------------------------------------------------------------
        // 4) 小工具
        // ------------------------------------------------------------
        auto isValidValue = [](float v) -> bool
        {
            return std::isfinite(v) && v > 0.0f;
        };

        auto robustMedian = [](std::vector<float> vals) -> float
        {
            if (vals.empty())
                return 0.0f;

            size_t mid = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
            float med = vals[mid];

            if (vals.size() % 2 == 0)
            {
                float lower = *std::max_element(vals.begin(), vals.begin() + mid);
                med = 0.5f * (lower + med);
            }
            return med;
        };

        auto quantileFromSorted = [](const std::vector<float>& sortedVals, float q) -> float
        {
            if (sortedVals.empty())
                return 0.0f;
            if (sortedVals.size() == 1)
                return sortedVals[0];

            float pos = q * static_cast<float>(sortedVals.size() - 1);
            int lo = static_cast<int>(std::floor(pos));
            int hi = static_cast<int>(std::ceil(pos));
            float t = pos - static_cast<float>(lo);

            if (lo == hi)
                return sortedVals[lo];

            return sortedVals[lo] * (1.0f - t) + sortedVals[hi] * t;
        };

        auto labDiff = [&](int y1, int x1, int y2, int x2) -> float
        {
            if (!hasGuide)
                return 0.0f;

            const cv::Vec3f& a = guideLab.at<cv::Vec3f>(y1, x1);
            const cv::Vec3f& b = guideLab.at<cv::Vec3f>(y2, x2);

            float d0 = a[0] - b[0];
            float d1 = a[1] - b[1];
            float d2 = a[2] - b[2];
            return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
        };

        struct LocalPixel
        {
            int x;
            int y;
            float vd;
            float dist2Center;
        };

        // ------------------------------------------------------------
        // 5) 初始化输出
        // ------------------------------------------------------------
        for (size_t i = 0; i < samplePoints.size(); ++i)
        {
            samplePoints[i].vDepth = 0.0f;
        }

        int keptCount = 0;
        int rejectNoValid      = 0;
        int rejectNoSeed       = 0;
        int rejectWeakSupport  = 0;
        int rejectDrift        = 0;
        int rejectVDSpan       = 0;
        int rejectGTSupport    = 0;
        int rejectGTSpan       = 0;

        // ------------------------------------------------------------
        // 6) 对每个 GT seed，构建局部表面支持域，然后同时统计 vd / GT
        // ------------------------------------------------------------
        for (size_t i = 0; i < samplePoints.size(); ++i)
        {
            SamplePoint& pt = samplePoints[i];
            const int row = pt.rowIndex;
            const int col = pt.colIndex;

            if (row < 0 || row >= vdImg.rows || col < 0 || col >= vdImg.cols)
            {
                rejectNoValid++;
                continue;
            }

            bool accepted = false;

            for (int radius = startRadius; radius <= maxRadius; ++radius)
            {
                const int rMin = std::max(0, row - radius);
                const int rMax = std::min(vdImg.rows - 1, row + radius);
                const int cMin = std::max(0, col - radius);
                const int cMax = std::min(vdImg.cols - 1, col + radius);

                const int winH = rMax - rMin + 1;
                const int winW = cMax - cMin + 1;

                std::vector<LocalPixel> validPixels;
                validPixels.reserve(winH * winW);

                for (int r = rMin; r <= rMax; ++r)
                {
                    for (int c = cMin; c <= cMax; ++c)
                    {
                        float v = vdImg.at<float>(r, c);
                        if (!isValidValue(v))
                            continue;

                        float dx = static_cast<float>(c - col);
                        float dy = static_cast<float>(r - row);

                        LocalPixel px;
                        px.x = c;
                        px.y = r;
                        px.vd = v;
                        px.dist2Center = dx * dx + dy * dy;
                        validPixels.push_back(px);
                    }
                }

                if (static_cast<int>(validPixels.size()) < minWindowValidCount)
                {
                    if (radius == maxRadius)
                        rejectNoValid++;
                    continue;
                }

                int seedX = -1, seedY = -1;
                float bestSeedScore = std::numeric_limits<float>::infinity();

                float centerVD = vdImg.at<float>(row, col);
                if (isValidValue(centerVD))
                {
                    seedX = col;
                    seedY = row;
                }
                else
                {
                    const float maxSeedDist2 = maxSeedDriftPx * maxSeedDriftPx;

                    for (size_t k = 0; k < validPixels.size(); ++k)
                    {
                        const LocalPixel& cand = validPixels[k];
                        if (cand.dist2Center > maxSeedDist2)
                            continue;

                        float score = cand.dist2Center;

                        if (hasGuide)
                        {
                            float dLab = labDiff(row, col, cand.y, cand.x);
                            if (dLab > maxSeedLabDiff)
                                continue;

                            score += 0.15f * dLab * dLab;
                        }

                        if (score < bestSeedScore)
                        {
                            bestSeedScore = score;
                            seedX = cand.x;
                            seedY = cand.y;
                        }
                    }
                }

                if (seedX < 0 || seedY < 0)
                {
                    if (radius == maxRadius)
                        rejectNoSeed++;
                    continue;
                }

                std::vector<float> seedVals;
                for (int rr = std::max(rMin, seedY - 1); rr <= std::min(rMax, seedY + 1); ++rr)
                {
                    for (int cc = std::max(cMin, seedX - 1); cc <= std::min(cMax, seedX + 1); ++cc)
                    {
                        float v = vdImg.at<float>(rr, cc);
                        if (isValidValue(v))
                            seedVals.push_back(v);
                    }
                }

                if (seedVals.size() < 3)
                    continue;

                float seedMed = robustMedian(seedVals);

                std::vector<float> absDev;
                absDev.reserve(seedVals.size());
                for (size_t k = 0; k < seedVals.size(); ++k)
                    absDev.push_back(std::fabs(seedVals[k] - seedMed));

                float mad = robustMedian(absDev);
                float band = std::max(absBandMin, madScale * std::max(mad, minMadFloor));

                cv::Mat candidateMask = cv::Mat::zeros(winH, winW, CV_8UC1);
                cv::Mat visited = cv::Mat::zeros(winH, winW, CV_8UC1);

                for (size_t k = 0; k < validPixels.size(); ++k)
                {
                    const LocalPixel& cand = validPixels[k];

                    if (std::fabs(cand.vd - seedMed) > band)
                        continue;

                    if (hasGuide)
                    {
                        float dLabSeed = labDiff(seedY, seedX, cand.y, cand.x);
                        if (dLabSeed > maxSeedLabDiff * 1.2f)
                            continue;
                    }

                    int ly = cand.y - rMin;
                    int lx = cand.x - cMin;
                    candidateMask.at<uchar>(ly, lx) = 255;
                }

                int seedLX = seedX - cMin;
                int seedLY = seedY - rMin;

                if (seedLX < 0 || seedLX >= winW || seedLY < 0 || seedLY >= winH)
                    continue;
                if (candidateMask.at<uchar>(seedLY, seedLX) == 0)
                    continue;

                std::vector<cv::Point> stack;
                stack.reserve(validPixels.size());
                stack.push_back(cv::Point(seedLX, seedLY));
                visited.at<uchar>(seedLY, seedLX) = 255;

                std::vector<LocalPixel> support;
                support.reserve(validPixels.size());

                while (!stack.empty())
                {
                    cv::Point cur = stack.back();
                    stack.pop_back();

                    int gx = cMin + cur.x;
                    int gy = rMin + cur.y;
                    float curVD = vdImg.at<float>(gy, gx);

                    LocalPixel sp;
                    sp.x = gx;
                    sp.y = gy;
                    sp.vd = curVD;
                    float dx = static_cast<float>(gx - col);
                    float dy = static_cast<float>(gy - row);
                    sp.dist2Center = dx * dx + dy * dy;
                    support.push_back(sp);

                    float curGrad = hasGuide ? guideGrad.at<float>(gy, gx) : 0.0f;

                    for (int ddy = -1; ddy <= 1; ++ddy)
                    {
                        for (int ddx = -1; ddx <= 1; ++ddx)
                        {
                            if (ddx == 0 && ddy == 0)
                                continue;

                            int nx = cur.x + ddx;
                            int ny = cur.y + ddy;

                            if (nx < 0 || nx >= winW || ny < 0 || ny >= winH)
                                continue;
                            if (visited.at<uchar>(ny, nx))
                                continue;
                            if (candidateMask.at<uchar>(ny, nx) == 0)
                                continue;

                            int gx2 = cMin + nx;
                            int gy2 = rMin + ny;
                            float nextVD = vdImg.at<float>(gy2, gx2);

                            if (std::fabs(nextVD - curVD) > band * localJumpScale)
                                continue;

                            if (hasGuide)
                            {
                                float nextGrad = guideGrad.at<float>(gy2, gx2);
                                float dLabLocal = labDiff(gy, gx, gy2, gx2);

                                if (dLabLocal > maxGrowLabDiff)
                                    continue;

                                if (std::max(curGrad, nextGrad) > strongEdgeThresh &&
                                    dLabLocal > maxGrowLabDiff * edgeDiffRelaxFactor)
                                {
                                    continue;
                                }
                            }

                            visited.at<uchar>(ny, nx) = 255;
                            stack.push_back(cv::Point(nx, ny));
                        }
                    }
                }

                if (static_cast<int>(support.size()) < minSupportCount)
                    continue;

                float meanX = 0.0f, meanY = 0.0f;
                std::vector<float> supportVDVals;
                supportVDVals.reserve(support.size());

                for (size_t k = 0; k < support.size(); ++k)
                {
                    meanX += static_cast<float>(support[k].x);
                    meanY += static_cast<float>(support[k].y);
                    supportVDVals.push_back(support[k].vd);
                }

                meanX /= static_cast<float>(support.size());
                meanY /= static_cast<float>(support.size());

                float centroidDrift = std::sqrt((meanX - col) * (meanX - col) +
                                                (meanY - row) * (meanY - row));
                if (centroidDrift > maxCentroidDriftPx)
                {
                    if (radius == maxRadius)
                        rejectDrift++;
                    continue;
                }

                std::sort(supportVDVals.begin(), supportVDVals.end());
                float vdQ10 = quantileFromSorted(supportVDVals, 0.10f);
                float vdQ90 = quantileFromSorted(supportVDVals, 0.90f);
                float vdSpan = vdQ90 - vdQ10;

                if (vdSpan > maxSupportSpanVD)
                {
                    if (radius == maxRadius)
                        rejectVDSpan++;
                    continue;
                }

                float vdRep = quantileFromSorted(supportVDVals, 0.50f);

                cv::Mat supportMask = cv::Mat::zeros(winH, winW, CV_8UC1);
                for (size_t k = 0; k < support.size(); ++k)
                {
                    int ly = support[k].y - rMin;
                    int lx = support[k].x - cMin;
                    supportMask.at<uchar>(ly, lx) = 255;
                }

                cv::Mat supportMaskGT = supportMask.clone();
                if (gtDilateIter > 0)
                {
                    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
                    cv::dilate(supportMask, supportMaskGT, kernel, cv::Point(-1, -1), gtDilateIter);
                }

                std::vector<float> supportGTVals;
                supportGTVals.reserve(support.size());

                for (int ly = 0; ly < supportMaskGT.rows; ++ly)
                {
                    for (int lx = 0; lx < supportMaskGT.cols; ++lx)
                    {
                        if (supportMaskGT.at<uchar>(ly, lx) == 0)
                            continue;

                        int gx = cMin + lx;
                        int gy = rMin + ly;

                        float gv = gtImg.at<float>(gy, gx);
                        if (!isValidValue(gv))
                            continue;

                        supportGTVals.push_back(gv);
                    }
                }

                if (static_cast<int>(supportGTVals.size()) < minGtSupportCount)
                {
                    if (radius == maxRadius)
                        rejectGTSupport++;
                    continue;
                }

                std::sort(supportGTVals.begin(), supportGTVals.end());

                if (supportGTVals.size() >= 3)
                {
                    float gtQ10 = quantileFromSorted(supportGTVals, 0.10f);
                    float gtQ90 = quantileFromSorted(supportGTVals, 0.90f);
                    float gtSpan = gtQ90 - gtQ10;

                    if (gtSpan > maxSupportSpanGT)
                    {
                        if (radius == maxRadius)
                            rejectGTSpan++;
                        continue;
                    }
                }

                float gtRep = quantileFromSorted(supportGTVals, 0.50f);

                if (!isValidValue(vdRep) || !isValidValue(gtRep))
                    continue;

                pt.vDepth = vdRep;
                pt.gtDepth = gtRep;

                accepted = true;
                keptCount++;
                break;
            }

            if (!accepted)
            {
                // 保持 vDepth = 0，后面 fitSegmentsParams 会自动跳过
            }
        }

        int invalidCount = 0;
        for (size_t i = 0; i < samplePoints.size(); ++i)
        {
            if (!(std::isfinite(samplePoints[i].vDepth) && samplePoints[i].vDepth > 0.0f &&
                  std::isfinite(samplePoints[i].gtDepth) && samplePoints[i].gtDepth > 0.0f))
            {
                invalidCount++;
            }
        }

        std::cout << "[selectVdPoints] total GT seeds     : " << samplePoints.size() << std::endl;
        std::cout << "[selectVdPoints] kept pairs         : " << keptCount << std::endl;
        std::cout << "[selectVdPoints] invalid pairs      : " << invalidCount << std::endl;
        std::cout << "[selectVdPoints] rejectNoValid      : " << rejectNoValid << std::endl;
        std::cout << "[selectVdPoints] rejectNoSeed       : " << rejectNoSeed << std::endl;
        std::cout << "[selectVdPoints] rejectWeakSupport  : " << rejectWeakSupport << std::endl;
        std::cout << "[selectVdPoints] rejectDrift        : " << rejectDrift << std::endl;
        std::cout << "[selectVdPoints] rejectVDSpan       : " << rejectVDSpan << std::endl;
        std::cout << "[selectVdPoints] rejectGTSupport    : " << rejectGTSupport << std::endl;
        std::cout << "[selectVdPoints] rejectGTSpan       : " << rejectGTSpan << std::endl;
    }

    void VirtualToRealDepthFunc::fitSegmentsParams(std::string xml_path)
    {
        if (samplePoints.empty())
        {
            std::cout << "samplePoints is empty!" << std::endl;
            return;
        }

        // ============================================================
        // 新思路：
        // 1) 输入 samplePoints 中的 (gtDepth, vDepth) 已经被 selectVdPoints()
        //    改造成“同一局部表面支持域上的鲁棒代表值”
        // 2) 这里不再按 GT 分段，而改成按 VD(vDepth) 分段
        // 3) 每段边界直接由该段样本的真实 vd min/max 决定
        // 4) 每段拟合前后做一次鲁棒清理，减少坏样本影响
        // ============================================================

        const int   minSamplesPerSegment = 8;
        const int   targetSamplesPerSeg  = 16;
        const float maxVdSpanPerSegment  = 0.18f;
        const float minVdSpanPerSegment  = 0.04f;
        const float residualMadScale     = 3.0f;
        const float minResidualAbsTol    = 800.0f;
        const float segmentBoundaryEps   = 1e-4f;

        auto isValidPair = [](const SamplePoint& p) -> bool
        {
            return std::isfinite(p.vDepth) && std::isfinite(p.gtDepth) &&
                   p.vDepth > 0.0f && p.gtDepth > 0.0f;
        };

        auto robustMedian = [](std::vector<float> vals) -> float
        {
            if (vals.empty())
                return 0.0f;

            size_t mid = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
            float med = vals[mid];

            if (vals.size() % 2 == 0)
            {
                float lower = *std::max_element(vals.begin(), vals.begin() + mid);
                med = 0.5f * (lower + med);
            }
            return med;
        };

        auto quantileFromSorted = [](const std::vector<float>& sortedVals, float q) -> float
        {
            if (sortedVals.empty())
                return 0.0f;
            if (sortedVals.size() == 1)
                return sortedVals[0];

            float pos = q * static_cast<float>(sortedVals.size() - 1);
            int lo = static_cast<int>(std::floor(pos));
            int hi = static_cast<int>(std::ceil(pos));
            float t = pos - static_cast<float>(lo);

            if (lo == hi)
                return sortedVals[lo];
            return sortedVals[lo] * (1.0f - t) + sortedVals[hi] * t;
        };

        auto evalBehaviorModel = [](const std::array<double, 3>& p, float gt, float vd) -> float
        {
            double denom = 1.0 - static_cast<double>(vd) * p[0];
            if (std::fabs(denom) < 1e-12)
                return 0.0f;

            double pred = (static_cast<double>(vd) * p[1] + p[2]) / denom;
            return static_cast<float>(pred);
        };

        std::vector<SamplePoint> validSamples;
        validSamples.reserve(samplePoints.size());

        for (size_t i = 0; i < samplePoints.size(); ++i)
        {
            if (isValidPair(samplePoints[i]))
                validSamples.push_back(samplePoints[i]);
        }

        if (validSamples.size() < 3)
        {
            std::cout << "Not enough valid sample pairs for fitSegmentsParams()." << std::endl;
            return;
        }

        std::sort(validSamples.begin(), validSamples.end(),
                  [](const SamplePoint& a, const SamplePoint& b)
                  {
                      return a.vDepth < b.vDepth;
                  });

        std::vector<std::vector<SamplePoint>> rawSegments;
        rawSegments.clear();

        std::vector<SamplePoint> curSeg;
        curSeg.reserve(targetSamplesPerSeg);

        for (size_t i = 0; i < validSamples.size(); ++i)
        {
            if (curSeg.empty())
            {
                curSeg.push_back(validSamples[i]);
                continue;
            }

            float curMinVD = curSeg.front().vDepth;
            float nextVD   = validSamples[i].vDepth;

            float newSpan = nextVD - curMinVD;

            bool shouldCut = false;

            if (static_cast<int>(curSeg.size()) >= targetSamplesPerSeg &&
                newSpan >= minVdSpanPerSegment)
            {
                shouldCut = true;
            }

            if (newSpan > maxVdSpanPerSegment &&
                static_cast<int>(curSeg.size()) >= minSamplesPerSegment)
            {
                shouldCut = true;
            }

            if (shouldCut)
            {
                rawSegments.push_back(curSeg);
                curSeg.clear();
            }

            curSeg.push_back(validSamples[i]);
        }

        if (!curSeg.empty())
        {
            if (!rawSegments.empty() &&
                static_cast<int>(curSeg.size()) < minSamplesPerSegment)
            {
                rawSegments.back().insert(rawSegments.back().end(), curSeg.begin(), curSeg.end());
            }
            else
            {
                rawSegments.push_back(curSeg);
            }
        }

        {
            std::vector<std::vector<SamplePoint>> merged;
            for (size_t i = 0; i < rawSegments.size(); ++i)
            {
                if (rawSegments[i].empty())
                    continue;

                if (merged.empty())
                {
                    merged.push_back(rawSegments[i]);
                    continue;
                }

                if (static_cast<int>(rawSegments[i].size()) < minSamplesPerSegment)
                {
                    merged.back().insert(merged.back().end(),
                                         rawSegments[i].begin(),
                                         rawSegments[i].end());
                }
                else
                {
                    merged.push_back(rawSegments[i]);
                }
            }
            rawSegments.swap(merged);
        }

        if (rawSegments.empty())
        {
            std::cout << "No vd-domain segments built." << std::endl;
            return;
        }

        std::vector<BehaviorSegmentResult> segment_results;
        segment_results.clear();

        for (size_t s = 0; s < rawSegments.size(); ++s)
        {
            std::vector<SamplePoint>& segPts = rawSegments[s];
            if (segPts.size() < 3)
                continue;

            std::vector<float> gtVals;
            std::vector<float> vdVals;
            gtVals.reserve(segPts.size());
            vdVals.reserve(segPts.size());

            for (size_t i = 0; i < segPts.size(); ++i)
            {
                gtVals.push_back(segPts[i].gtDepth);
                vdVals.push_back(segPts[i].vDepth);
            }

            if (gtVals.size() < 3)
                continue;

            std::array<double, 3> params = BehavioralModel(gtVals, vdVals);

            std::vector<float> residuals;
            residuals.reserve(segPts.size());

            for (size_t i = 0; i < segPts.size(); ++i)
            {
                float pred = evalBehaviorModel(params, segPts[i].gtDepth, segPts[i].vDepth);
                float res  = std::fabs(pred - segPts[i].gtDepth);
                residuals.push_back(res);
            }

            float medRes = robustMedian(residuals);

            std::vector<float> absDev;
            absDev.reserve(residuals.size());
            for (size_t i = 0; i < residuals.size(); ++i)
                absDev.push_back(std::fabs(residuals[i] - medRes));

            float madRes = robustMedian(absDev);
            float resTol = std::max(minResidualAbsTol, residualMadScale * std::max(madRes, 1.0f));

            std::vector<SamplePoint> inliers;
            inliers.reserve(segPts.size());

            for (size_t i = 0; i < segPts.size(); ++i)
            {
                if (residuals[i] <= resTol)
                    inliers.push_back(segPts[i]);
            }

            if (inliers.size() < 3)
                inliers = segPts;

            gtVals.clear();
            vdVals.clear();
            gtVals.reserve(inliers.size());
            vdVals.reserve(inliers.size());

            for (size_t i = 0; i < inliers.size(); ++i)
            {
                gtVals.push_back(inliers[i].gtDepth);
                vdVals.push_back(inliers[i].vDepth);
            }

            if (gtVals.size() < 3)
                continue;

            params = BehavioralModel(gtVals, vdVals);

            float vdMin = std::numeric_limits<float>::infinity();
            float vdMax = 0.0f;

            for (size_t i = 0; i < inliers.size(); ++i)
            {
                vdMin = std::min(vdMin, inliers[i].vDepth);
                vdMax = std::max(vdMax, inliers[i].vDepth);
            }

            if (!(std::isfinite(vdMin) && std::isfinite(vdMax) && vdMax > vdMin))
                continue;

            BehaviorSegmentResult seg;
            seg.vdepthMin = vdMin;
            seg.vdepthMax = vdMax;
            seg.sampleCount = static_cast<int>(inliers.size());
            seg.params = params;
            segment_results.push_back(seg);
        }

        if (segment_results.empty())
        {
            std::cout << "No valid fitted segments." << std::endl;
            return;
        }

        std::sort(segment_results.begin(), segment_results.end(),
                  [](const BehaviorSegmentResult& a, const BehaviorSegmentResult& b)
                  {
                      return a.vdepthMin < b.vdepthMin;
                  });

        for (size_t i = 1; i < segment_results.size(); ++i)
        {
            float leftMax  = segment_results[i - 1].vdepthMax;
            float rightMin = segment_results[i].vdepthMin;

            float mid = 0.5f * (leftMax + rightMin);

            segment_results[i - 1].vdepthMax = mid;
            segment_results[i].vdepthMin = mid;
        }

        for (size_t i = 0; i < segment_results.size(); ++i)
        {
            if (segment_results[i].vdepthMax <= segment_results[i].vdepthMin)
            {
                segment_results[i].vdepthMax = segment_results[i].vdepthMin + segmentBoundaryEps;
            }
        }

        std::ofstream xml(xml_path);
        if (!xml.is_open())
        {
            std::cerr << "无法写入 XML: " << xml_path << std::endl;
            return;
        }

        xml << std::fixed << std::setprecision(6);
        xml << "<?xml version=\"1.0\"?>\n";
        xml << "<opencv_storage>\n";
        xml << "    <BehaviorModelSegments>\n\n";

        for (size_t i = 0; i < segment_results.size(); ++i)
        {
            const BehaviorSegmentResult& seg = segment_results[i];

            xml << "        <Segment>\n";
            xml << "            <DepthMin>" << seg.vdepthMin << "</DepthMin>\n";
            xml << "            <DepthMax>" << seg.vdepthMax << "</DepthMax>\n";
            xml << "            <SampleCount>" << seg.sampleCount << "</SampleCount>\n\n";
            xml << "            <Param>\n";
            xml << "                <c0>" << seg.params[0] << "</c0>\n";
            xml << "                <c1>" << seg.params[1] << "</c1>\n";
            xml << "                <c2>" << seg.params[2] << "</c2>\n";
            xml << "            </Param>\n";
            xml << "        </Segment>\n\n";

            std::cout << "[fitSegmentsParams] Segment " << i
                      << "  vd=[" << seg.vdepthMin << ", " << seg.vdepthMax << "]"
                      << "  sampleCount=" << seg.sampleCount
                      << "  params=(" << seg.params[0] << ", "
                                      << seg.params[1] << ", "
                                      << seg.params[2] << ")"
                      << std::endl;
        }

        xml << "    </BehaviorModelSegments>\n";
        xml << "</opencv_storage>\n";
        xml.close();

        std::cout << "Behavior model XML saved: " << xml_path << std::endl;
    }

    // TODO 增加bool,ture,在此提供路径；false:在内存中保存
    void VirtualToRealDepthFunc::VirtualToRealDepthBySegBM()
    {
        m_strRootPath = m_ptrDepthSolver->GetRootPath();
        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/VD_Raw.tiff";
//        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/VitualDepthMap.tiff";
        std::string refDepthImg_path = m_strRootPath + "/behavior_model/ref-csad-rd.tiff";
        std::string pointsXmlPath = m_strRootPath + "/behavior_model/points.xml";
        std::string focuseImg_Path = m_strRootPath + "/behavior_model/fullfocus.png";

        m_virtualDepthImage = cv::imread(virtualDepthImg_path, cv::IMREAD_UNCHANGED);  // 读取虚拟深度图
        m_refDepthImage = cv::imread(refDepthImg_path, cv::IMREAD_UNCHANGED);   // 读取激光深度图
        cv::Mat focusImage = cv::imread(focuseImg_Path);                  // 读取全聚焦图

        std::string vDImg_marked_path = m_strRootPath + "/behavior_model/m_vIDepth_Marked.png";
        std::string output_csv_path = m_strRootPath + "/behavior_model/output_csv.csv";
        std::string distanceImage_path = m_strRootPath + "/behavior_model/distanceImage.png";

        // Step 1: 挑选拟合多个行为模型参数的样本点
        sampleVirtualDepthPointsByRegion(virtualDepthImg_path,
                                         refDepthImg_path,
                                         vDImg_marked_path,
                                         output_csv_path);

        // Step 2: 使用分段行为模型
        cv::Mat realDepthImage = ConvertVdImageToRd(m_virtualDepthImage);

        // Step 3: 采样
        SamplePointSelect();

        // Step 4: 误差统计
//        errorStatisticsImage(realDepthImage, m_refDepthImage, focusImage, distanceImage_path);
        errorStatisticsImageSeg(realDepthImage, m_refDepthImage, m_virtualDepthImage,focusImage, distanceImage_path);

    }
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


       /* std::string vDImg_marked_path = m_strRootPath + "/behavior_model/m_vIDepth_Marked.png";
        std::string output_csv_path = m_strRootPath + "/behavior_model/output_csv.csv";

        // 挑选拟合多个行为模型参数的样本点
        sampleVirtualDepthPointsByRegion(virtualDepthImg_path,
                                 refDepthImg_path,
                                 vDImg_marked_path,
                                 output_csv_path);*/

        loadPointsXML(pointsXmlPath);

        // 1>: 转换图像格式
        ExtractDepthsFromImages(refDepthValue, virtualDepthValue);

        // 1.5>: 新增不同景深点坐标
//        appendDepthPoints(refDepthValue, virtualDepthValue);
//        cv::Rect roi = cv::Rect(1819,1153,1000,1000);   // 圆盘
       /* cv::Rect roi = cv::Rect(3810, 140, 2000, 2000);   // 汽车，墙面
        refDepthValue.clear();
        virtualDepthValue.clear();
        appendDepthPoints(refDepthValue, virtualDepthValue, roi);*/

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

    cv::Mat VirtualToRealDepthFunc::ConvertVdImageToRd(cv::Mat virtualDepthImage)
    {
        if (virtualDepthImage.empty())
        {
            std::cout << "virtualDepthImage is empty!" << std::endl;
        }

        // 输出一张同尺寸的真实深度图
        cv::Mat realDepthImage = cv::Mat::zeros(virtualDepthImage.size(), CV_32FC1);

        for (int y = 0; y < virtualDepthImage.rows; ++y)
        {
            for (int x = 0; x < virtualDepthImage.cols; ++x)
            {
                float v_depth = virtualDepthImage.at<float>(y, x);

                if (v_depth <= 0.0f || !std::isfinite(v_depth))
                {
                    realDepthImage.at<float>(y, x) = 0.0f;
                    continue;
                }

                LFRefocus refocusImp(m_ptrDepthSolver);
                float r_depth = refocusImp.ConvertVdToRdSegment(v_depth);
                realDepthImage.at<float>(y, x) = r_depth;
            }
        }
        return realDepthImage;
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

        float maxDiffRecorded = 0.0f;

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

    void VirtualToRealDepthFunc::errorStatisticsImage(cv::Mat realDepthImage, cv::Mat refDepthImage, cv::Mat bgImage,std::string errorMapSavePath)
    {
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

        // 误差图
        cv::Mat errorMapFloat = cv::Mat::zeros(realDepthImage.size(), CV_32FC1);
        cv::Mat validMask = cv::Mat::zeros(realDepthImage.size(), CV_8UC1);

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

        float maxDiffRecorded = 0.0f;

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

                // 写入误差图
                errorMapFloat.at<float>(y, x) = diff;
                validMask.at<uchar>(y, x) = 255;

                if (diff > maxDiffRecorded)
                    maxDiffRecorded = diff;

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

        // 误差统计图可视化
        cv::Mat errorMap8U = cv::Mat::zeros(errorMapFloat.size(), CV_8UC1);

        if (maxDiffRecorded > 0)
        {
            // 先把有效采样点写入 8bit 图
            for (int y = 0; y < errorMapFloat.rows; ++y) {
                const float* errPtr = errorMapFloat.ptr<float>(y);
                const uchar* maskPtr = validMask.ptr<uchar>(y);
                uchar* outPtr = errorMap8U.ptr<uchar>(y);

                for (int x = 0; x < errorMapFloat.cols; ++x) {
                    if (maskPtr[x]) {
                        float normVal = errPtr[x] / maxDiffRecorded * 255.0f;
                        normVal = std::max(0.0f, std::min(255.0f, normVal));
                        outPtr[x] = static_cast<uchar>(normVal);
                    }
                }
            }
        }

        // 膨胀
        cv::Mat visErrorMap8U = errorMap8U.clone();
        cv::Mat visMask = validMask.clone();

        // 核大小可调：3x3 ， 5x5 ， 7x7
        int dilateSize = 5; // kernel = 5x5
        cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2 * dilateSize + 1, 2 * dilateSize + 1)
        );

        // 对误差图和 mask 同时膨胀
        cv::dilate(errorMap8U, visErrorMap8U, kernel);
        cv::dilate(validMask, visMask, kernel);

        // 生成伪彩色误差图
        cv::Mat errorColor;
        cv::applyColorMap(visErrorMap8U, errorColor, cv::COLORMAP_JET);

        // 把误差点覆盖到背景图上
        cv::Mat finalVis = bgImage.clone();

        for (int y = 0; y < errorColor.rows; ++y) {
            const uchar* maskPtr = visMask.ptr<uchar>(y);
            const cv::Vec3b* errPtr = errorColor.ptr<cv::Vec3b>(y);
            cv::Vec3b* outPtr = finalVis.ptr<cv::Vec3b>(y);

            for (int x = 0; x < errorColor.cols; ++x) {
                if (maskPtr[x]) {
                    outPtr[x] = errPtr[x];   // 直接覆盖
                }
            }
        }

        // 保存结果
        if (!errorMapSavePath.empty()) {
            cv::imwrite(errorMapSavePath, finalVis);
            std::cout << "Error map saved to: " << errorMapSavePath << std::endl;
        }
    }

    void VirtualToRealDepthFunc::errorStatisticsImageSeg(cv::Mat realDepthImage,
                                                         cv::Mat refDepthImage,
                                                         cv::Mat virtualDepthImage,
                                                         cv::Mat bgImage,
                                                         std::string errorMapSavePath)
    {
        if (m_samplePoints.empty()) {
            std::cerr << "采样点：m_samplePoints为空！" << std::endl;
            return;
        }

        const int outlierWindow = 5;
        const double outlierK = 2.0;

        // ---------------------------
        // 1) 基础检查
        // ---------------------------
        if (realDepthImage.empty() || refDepthImage.empty() || virtualDepthImage.empty()) {
            std::cerr << "[Error] Input image is empty." << std::endl;
            return;
        }

        // ---------------------------
        // 2) 统一通道
        // ---------------------------
        auto toSingleChannel = [](cv::Mat &img) {
            if (img.channels() > 1) {
                std::vector<cv::Mat> ch;
                cv::split(img, ch);
                img = ch[0];
            }
        };

        toSingleChannel(realDepthImage);
        toSingleChannel(refDepthImage);
        toSingleChannel(virtualDepthImage);

        // ---------------------------
        // 3) 统一数据类型到 CV_32FC1
        // ---------------------------
        if (realDepthImage.type() != CV_32FC1) {
            realDepthImage.convertTo(realDepthImage, CV_32F);
        }
        if (refDepthImage.type() != CV_32FC1) {
            refDepthImage.convertTo(refDepthImage, CV_32F);
        }
        if (virtualDepthImage.type() != CV_32FC1) {
            virtualDepthImage.convertTo(virtualDepthImage, CV_32F);
        }

        // ---------------------------
        // 4) 背景图处理
        // ---------------------------
        if (bgImage.empty()) {
            bgImage = cv::Mat::zeros(realDepthImage.size(), CV_8UC3);
        } else {
            if (bgImage.size() != realDepthImage.size()) {
                cv::resize(bgImage, bgImage, realDepthImage.size());
            }

            if (bgImage.channels() == 1) {
                cv::cvtColor(bgImage, bgImage, cv::COLOR_GRAY2BGR);
            } else if (bgImage.channels() == 4) {
                cv::cvtColor(bgImage, bgImage, cv::COLOR_BGRA2BGR);
            }
        }

        // ---------------------------
        // 5) outlier mask
        //    误差是和 ref 比，所以异常点仍建议按 ref 建
        // ---------------------------
        cv::Mat mask_outlier = buildOutlierMask(refDepthImage, outlierWindow, outlierK);
        if (mask_outlier.type() != CV_8UC1) {
            mask_outlier.convertTo(mask_outlier, CV_8U);
        }

        m_rdImage_rows = realDepthImage.rows;
        m_rdImage_cols = realDepthImage.cols;

        // ---------------------------
        // 6) 误差图
        // ---------------------------
        cv::Mat errorMapFloat = cv::Mat::zeros(realDepthImage.size(), CV_32FC1);
        cv::Mat validMask = cv::Mat::zeros(realDepthImage.size(), CV_8UC1);

        // ---------------------------
        // 7) vd 分段配置
        // ---------------------------
        const std::vector<std::pair<float, float>> vdSegments = {
                {3.9f, 4.2f},
                {4.2f, 4.3f},
                {4.3f, 4.4f},
                {4.4f, 4.5f},
                {4.5f, 4.6f},
                {4.6f, 4.8f},
                {4.8f, 5.0f},
                {5.0f, 5.2f}
        };

        // ---------------------------
        // 8) 统计桶
        // ---------------------------
        struct ErrorStatsBucket {
            int count = 0;
            double sum = 0.0;
            double sumSq = 0.0;
            float maxDiff = 0.0f;
            std::vector<float> diffs;

            void add(float d) {
                ++count;
                sum += d;
                sumSq += static_cast<double>(d) * static_cast<double>(d);
                maxDiff = std::max(maxDiff, d);
                diffs.push_back(d);
            }

            double mse() const {
                return count > 0 ? sumSq / static_cast<double>(count) : 0.0;
            }

            double rmse() const {
                return count > 0 ? std::sqrt(sumSq / static_cast<double>(count)) : 0.0;
            }

            double mae() const {
                return count > 0 ? sum / static_cast<double>(count) : 0.0;
            }

            double median() const {
                if (diffs.empty()) return 0.0;

                std::vector<float> tmp = diffs;
                const size_t mid = tmp.size() / 2;
                std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());

                if (tmp.size() % 2 == 1) {
                    return static_cast<double>(tmp[mid]);
                } else {
                    const float upper = tmp[mid];
                    const float lower = *std::max_element(tmp.begin(), tmp.begin() + mid);
                    return 0.5 * static_cast<double>(lower + upper);
                }
            }
        };

        ErrorStatsBucket totalStats;
        std::vector<ErrorStatsBucket> segmentStats(vdSegments.size());

        int outOfDefinedRangeCount = 0;
        float maxDiffRecorded = 0.0f;

        // ---------------------------
        // 9) 通用取有效值（带邻域回退）
        // ---------------------------
        auto getValidValue = [&](const cv::Mat &img, int x, int y, int searchRadius) -> float {
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
                return std::numeric_limits<float>::quiet_NaN();
            }

            float val = img.at<float>(y, x);
            if (std::isfinite(val) && val > 0.0f) {
                return val;
            }

            float nearest = std::numeric_limits<float>::quiet_NaN();
            float minDist2 = std::numeric_limits<float>::max();

            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= img.rows) continue;

                for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= img.cols) continue;

                    const float v = img.at<float>(yy, xx);
                    if (std::isfinite(v) && v > 0.0f) {
                        const float dist2 = static_cast<float>(dx * dx + dy * dy);
                        if (dist2 < minDist2) {
                            minDist2 = dist2;
                            nearest = v;
                        }
                    }
                }
            }
            return nearest;
        };

        // ---------------------------
        // 10) 根据 VD 找分段
        // ---------------------------
        auto findSegmentIndexByVD = [&](float vd) -> int {
            for (size_t i = 0; i < vdSegments.size(); ++i) {
                const float low = vdSegments[i].first;
                const float high = vdSegments[i].second;

                const bool inRange =
                        (i + 1 == vdSegments.size())
                        ? (vd >= low && vd <= high)
                        : (vd >= low && vd < high);

                if (inRange) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        };

        // ---------------------------
        // 11) 采样点统计
        //     分段按 VD
        //     误差按 |GT(=realDepthImage) - REF|
        // ---------------------------
        totalStats.diffs.reserve(m_samplePoints.size());

        for (size_t i = 0; i < m_samplePoints.size(); ++i) {
            const int x = m_samplePoints[i].x;
            const int y = m_samplePoints[i].y;

            if (x < 0 || y < 0 || x >= m_rdImage_cols || y >= m_rdImage_rows) {
                continue;
            }

           /* if (mask_outlier.at<uchar>(y, x)) {
                continue;
            }*/

            // gt / pred
            const float valPred = realDepthImage.at<float>(y, x);

            // ref
            const float valRef = getValidValue(refDepthImage, x, y, SESOptions.virtualSearchRadius);

            // vd：分段依据
            const float valVD = getValidValue(virtualDepthImage, x, y, SESOptions.virtualSearchRadius);

            if (!std::isfinite(valPred) || !std::isfinite(valRef) || !std::isfinite(valVD) ||
                valPred <= 0.0f || valRef <= 0.0f || valVD <= 0.0f) {
                continue;
            }

            const float diff = std::abs(valPred - valRef);
            if (diff >= SESOptions.outlierDistanceThreshold) {
                continue;
            }

            // 总体统计
            totalStats.add(diff);

            // 分段统计：按 VD 分段
            const int segIdx = findSegmentIndexByVD(valVD);
            if (segIdx >= 0) {
                segmentStats[segIdx].add(diff);
            } else {
                ++outOfDefinedRangeCount;
            }

            // 写入误差图
            errorMapFloat.at<float>(y, x) = diff;
            validMask.at<uchar>(y, x) = 255;
            maxDiffRecorded = std::max(maxDiffRecorded, diff);

            std::cout << "Sample (" << x << "," << y << ")  "
                      << "VD=" << valVD << "  "
                      << "Pred=" << valPred << "  "
                      << "Ref=" << valRef << "  "
                      << "Diff=" << diff;

            if (segIdx >= 0) {
                std::cout << "  VDSegment=["
                          << vdSegments[segIdx].first << ", "
                          << vdSegments[segIdx].second
                          << ((segIdx + 1 == static_cast<int>(vdSegments.size())) ? "]" : ")");
            } else {
                std::cout << "  VDSegment=OUT_OF_DEFINED_RANGE";
            }

            std::cout << std::endl;
        }

        if (totalStats.count == 0) {
            std::cerr << "No valid sampled points." << std::endl;
            return;
        }

        // ---------------------------
        // 12) 总体统计输出
        // ---------------------------
        LOG_WARN("Overall -> RMSE:", totalStats.rmse(),
                 "  MAE:", totalStats.mae(),
                 "  Median:", totalStats.median());

        std::cout << "Valid sampled points: " << totalStats.count << std::endl;
        std::cout << "  MSE:     " << totalStats.mse() << std::endl;
        std::cout << "  RMSE:    " << totalStats.rmse() << std::endl;
        std::cout << "  MAE:     " << totalStats.mae() << std::endl;
        std::cout << "  Median:  " << totalStats.median() << std::endl;
//        std::cout << "  MaxDiff: " << totalStats.maxDiff << std::endl;
        std::cout << "  In defined VD segments: "
                  << (totalStats.count - outOfDefinedRangeCount)
                  << " / " << totalStats.count << std::endl;

        // ---------------------------
        // 13) 分段统计输出（按 VD）
        // ---------------------------
        std::cout << "\nSegmented statistics (bucketed by VD):" << std::endl;
        for (size_t i = 0; i < vdSegments.size(); ++i) {
            const float low = vdSegments[i].first;
            const float high = vdSegments[i].second;
            const ErrorStatsBucket &st = segmentStats[i];

            std::cout << "  VD Segment ["
                      << low << ", " << high
                      << ((i + 1 == vdSegments.size()) ? "]" : ")")
                      << "  Count=" << st.count;

            if (st.count == 0) {
                std::cout << "  No valid samples." << std::endl;
                continue;
            }

            const double ratio = 100.0 * static_cast<double>(st.count) /
                                 static_cast<double>(totalStats.count);

            std::cout << "  Ratio=" << ratio << "%"
                      << "  MSE=" << st.mse()
                      << "  RMSE=" << st.rmse()
                      << "  MAE=" << st.mae()
                      << "  Median=" << st.median()
//                      << "  MaxDiff=" << st.maxDiff
                      << std::endl;
        }

        // ---------------------------
        // 14) 误差图可视化
        // ---------------------------
        cv::Mat errorMap8U = cv::Mat::zeros(errorMapFloat.size(), CV_8UC1);

        if (maxDiffRecorded > 0.0f) {
            for (int y = 0; y < errorMapFloat.rows; ++y) {
                const float* errPtr = errorMapFloat.ptr<float>(y);
                const uchar* maskPtr = validMask.ptr<uchar>(y);
                uchar* outPtr = errorMap8U.ptr<uchar>(y);

                for (int x = 0; x < errorMapFloat.cols; ++x) {
                    if (maskPtr[x]) {
                        float normVal = errPtr[x] / maxDiffRecorded * 255.0f;
                        normVal = std::max(0.0f, std::min(255.0f, normVal));
                        outPtr[x] = static_cast<uchar>(normVal);
                    }
                }
            }
        }

        cv::Mat visErrorMap8U = errorMap8U.clone();
        cv::Mat visMask = validMask.clone();

        const int dilateSize = 5;
        cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2 * dilateSize + 1, 2 * dilateSize + 1)
        );

        cv::dilate(errorMap8U, visErrorMap8U, kernel);
        cv::dilate(validMask, visMask, kernel);

        cv::Mat errorColor;
        cv::applyColorMap(visErrorMap8U, errorColor, cv::COLORMAP_JET);

        cv::Mat finalVis = bgImage.clone();
        if (finalVis.type() != CV_8UC3) {
            finalVis.convertTo(finalVis, CV_8UC3);
        }

        for (int y = 0; y < errorColor.rows; ++y) {
            const uchar* maskPtr = visMask.ptr<uchar>(y);
            const cv::Vec3b* errPtr = errorColor.ptr<cv::Vec3b>(y);
            cv::Vec3b* outPtr = finalVis.ptr<cv::Vec3b>(y);

            for (int x = 0; x < errorColor.cols; ++x) {
                if (maskPtr[x]) {
                    outPtr[x] = errPtr[x];
                }
            }
        }

        // ---------------------------
        // 15) 保存结果
        // ---------------------------
        if (!errorMapSavePath.empty()) {
            cv::imwrite(errorMapSavePath, finalVis);
            std::cout << "Error map saved to: " << errorMapSavePath << std::endl;
        }
    }

    void VirtualToRealDepthFunc::errorStatisticsImageGTSeg(cv::Mat realDepthImage,
                                                         cv::Mat bgImage,
                                                         std::string errorMapSavePath)
    {
        if (m_samplePoints.empty())
        {
            std::cerr << "采样点：m_samplePoints为空！" << std::endl;
            return;
        }
        m_virtualDepthImage;
        m_rdImage_cols;

        m_rdImage_rows = realDepthImage.rows;
        m_rdImage_cols = realDepthImage.cols;
        cv::Mat errorMapFloat = cv::Mat::zeros(realDepthImage.size(), CV_32FC1);
        cv::Mat validMask = cv::Mat::zeros(realDepthImage.size(), CV_8UC1);

        // 参数：可按你的数据情况调整
        const int targetNeighborCount_vd = 15;  // 当前点为空时，逐圈找到多少个点后取均值
        const int targetNeighborCount_gt = 5;  // 当前点为空时，逐圈找到多少个点后取均值
        const int maxRadius = 50;            // 最大向外扩展半径

        std::vector<float> absErrors;
        absErrors.reserve(m_samplePoints.size());

        double sumAbsError = 0.0;
        double sumSqError = 0.0;
        int validCount = 0;

        for (int i = 0; i < m_samplePoints.size(); i++)
        {
            cv::Point& pt = m_samplePoints[i];
            if (pt.x < 0 || pt.x >= realDepthImage.cols || pt.y < 0 || pt.y >= realDepthImage.rows)
            {
                continue;
            }

            // 1) 取 realDepthImage 的值(若中心点无效，则逐圈扩展找到 targetNeighborCount_vd 个有效点求均值)
            float realValue = realDepthImage.at<float>(pt.y, pt.x);
            bool realValid = std::isfinite(realValue) && realValue > 0.0f;

            if (!realValid)
            {
                std::vector<float> neighbors;
                neighbors.reserve(targetNeighborCount_vd);

                for (int radius = 1; radius <= maxRadius && neighbors.size() < targetNeighborCount_vd; ++radius)
                {
                    int x0 = std::max(0, pt.x - radius);
                    int x1 = std::min(realDepthImage.cols - 1, pt.x + radius);
                    int y0 = std::max(0, pt.y - radius);
                    int y1 = std::min(realDepthImage.rows - 1, pt.y + radius);

                    for (int yy = y0; yy <= y1 && neighbors.size() < targetNeighborCount_vd; ++yy)
                    {
                        for (int xx = x0; xx <= x1 && neighbors.size() < targetNeighborCount_vd; ++xx)
                        {
                            if (xx == pt.x && yy == pt.y)
                            {
                                continue;
                            }
                            if (std::abs(xx - pt.x) != radius && std::abs(yy - pt.y) != radius)
                            {
                                continue;
                            }

                            float v = realDepthImage.at<float>(yy, xx);
                            if (std::isfinite(v) && v > 0.0f)
                            {
                                neighbors.push_back(v);
                            }
                        }
                    }
                }

                if (neighbors.size() >= targetNeighborCount_vd)
                {
                    float meanValue = 0.0;
                    for (float v : neighbors)
                    {
                        meanValue += v;
                    }
                    meanValue /= neighbors.size();

                    realValue = meanValue;
                    realValid = true;
                }
            }

            if (!realValid)
            {
                continue;
            }

            // 2) 取 refDepthImage 的值(若中心点无效，则逐圈扩展找到 targetNeighborCount_gt 个有效点求均值)
            float refValue = m_refDepthImage.at<float>(pt.y, pt.x);
            bool refValid = std::isfinite(refValue) && refValue > 0.0f;

            if (!refValid)
            {
                std::vector<float> neighbors;
                neighbors.reserve(targetNeighborCount_gt);

                for (int radius = 1; radius <= maxRadius && neighbors.size() < targetNeighborCount_gt; ++radius)
                {
                    int x0 = std::max(0, pt.x - radius);
                    int x1 = std::min(m_refDepthImage.cols - 1, pt.x + radius);
                    int y0 = std::max(0, pt.y - radius);
                    int y1 = std::min(m_refDepthImage.rows - 1, pt.y + radius);

                    for (int yy = y0; yy <= y1 && neighbors.size() < targetNeighborCount_gt; ++yy)
                    {
                        for (int xx = x0; xx <= x1 && neighbors.size() < targetNeighborCount_gt; ++xx)
                        {
                            if (xx == pt.x && yy == pt.y)
                            {
                                continue;
                            }

                            if (std::abs(xx - pt.x) != radius && std::abs(yy - pt.y) != radius)
                            {
                                continue;
                            }

                            float v = m_refDepthImage.at<float>(yy, xx);
                            if (std::isfinite(v) && v > 0.0f)
                            {
                                neighbors.push_back(v);
                            }
                        }
                    }
                }

                if (neighbors.size() >= targetNeighborCount_gt)
                {
                    float meanValue = 0.0;
                    for (float v : neighbors)
                    {
                        meanValue += v;
                    }
                    meanValue /= neighbors.size();

                    refValue = meanValue;
                    refValid = true;
                }
            }

            if (!refValid)
            {
                continue;
            }

            // 3) 计算绝对误差
            float absError = std::fabs(realValue - refValue);

            errorMapFloat.at<float>(pt.y, pt.x) = absError;
            validMask.at<uchar>(pt.y, pt.x) = 255;

            absErrors.push_back(absError);
            sumAbsError += absError;
            sumSqError += absError * absError;
            validCount++;
        }

        if (validCount == 0)
        {
            std::cerr << "没有可用于统计误差的有效采样点！" << std::endl;
            return;
        }

        // 4) 统计 RMSE / MAE / Median
        float mae = sumAbsError / validCount;
        float rmse = std::sqrt(sumSqError / validCount);

        std::sort(absErrors.begin(), absErrors.end());
        float median = 0.0;
        if (absErrors.size() % 2 == 1)
        {
            median = absErrors[absErrors.size() / 2];
        }
        else
        {
            int mid = static_cast<int>(absErrors.size() / 2);
            median = 0.5 * (absErrors[mid - 1] +
                            absErrors[mid]);
        }
        std::cout << "有效采样点数: " << validCount << std::endl;
        std::cout << "RMSE: " << rmse << std::endl;
        std::cout << "MAE: " << mae << std::endl;
        std::cout << "Median: " << median << std::endl;

        cv::Mat visImage;

        // 保证 bgImage 可用于彩色绘制
        if (bgImage.empty())
        {
            visImage = cv::Mat::zeros(realDepthImage.size(), CV_8UC3);
        }
        else
        {
            if (bgImage.size() != realDepthImage.size())
            {
                cv::resize(bgImage, bgImage, realDepthImage.size());
            }

            if (bgImage.type() == CV_8UC1)
            {
                cv::cvtColor(bgImage, visImage, cv::COLOR_GRAY2BGR);
            }
            else if (bgImage.type() == CV_8UC3)
            {
                visImage = bgImage.clone();
            }
            else
            {
                cv::Mat bg8u;
                bgImage.convertTo(bg8u, CV_8U);
                cv::cvtColor(bg8u, visImage, cv::COLOR_GRAY2BGR);
            }
        }

        for (int i = 0; i < m_samplePoints.size(); i++)
        {
            const cv::Point& pt = m_samplePoints[i];

            if (pt.x < 0 || pt.x >= visImage.cols || pt.y < 0 || pt.y >= visImage.rows)
            {
                continue;
            }

            if (validMask.at<uchar>(pt.y, pt.x) == 0)
            {
                continue;
            }

            float absError = errorMapFloat.at<float>(pt.y, pt.x);
            cv::Scalar color;
            if (absError < 1000.0f)
            {
                color = cv::Scalar(255, 0, 0);       // 蓝色
            }
            else if (absError < 3000.0f)
            {
                color = cv::Scalar(0, 255, 255);     // 黄色
            }
            else if (absError < 5000.0f)
            {
                color = cv::Scalar(0, 165, 255);     // 橙黄色
            }
            else
            {
                color = cv::Scalar(0, 0, 255);       // 红色
            }

            // 画圆圈
            cv::circle(visImage, pt, 5, color, 3);

            // 误差文字，单位 mm
            std::string text = std::to_string(static_cast<int>(std::round(absError)));

            // 文字显示在圆圈右上方，避免压住中心点
            cv::Point textOrg(pt.x + 5, pt.y - 5);

            // 防止文字越界到底部/顶部
            if (textOrg.y < 10)
            {
                textOrg.y = pt.y + 12;
            }
            if (textOrg.x > visImage.cols - 40)
            {
                textOrg.x = pt.x - 35;
            }

            // 画黑色描边
            cv::putText(visImage,
                        text,
                        textOrg,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.35,
                        cv::Scalar(0, 0, 0),
                        2,
                        cv::LINE_AA);

            // 再画彩色文字
            cv::putText(visImage,
                        text,
                        textOrg,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                        cv::LINE_AA);
        }

        if (!errorMapSavePath.empty())
        {
            cv::imwrite(errorMapSavePath, visImage);
            std::cout << "误差可视化图已保存: " << errorMapSavePath << std::endl;
        }
    }

    void VirtualToRealDepthFunc::SamplePointSelectByLW()
    {
        m_samplePoints.clear();
        if (m_coordsVirtual.empty())
        {
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

    void VirtualToRealDepthFunc::SamplePointSelectByRandom()
    {
        if (m_virtualDepthImage.empty() || m_refDepthImage.empty())
        {
            return;
        }
        int sampleCount = 500;  // 最大采样点数量

        const int targetNeighborCount_vd = 15;   // 每张图尽量找10个邻域点
        const int targetNeighborCount_gt = 4;   // 每张图尽量找10个邻域点
        const int maxRadius = 50;              // 最大逐圈扩张半径，可按需调
        const float vdRmseThreshold = 0.03f;  // VD图阈值，按你的深度单位调整
        const float refRmseThreshold = 1000.0f; // REF图阈值，按你的深度单位调整

        for (int y = 0; y < m_virtualDepthImage.rows; ++y)
        {
            const float* vPtr = m_virtualDepthImage.ptr<float>(y);
            const float* rPtr = m_refDepthImage.ptr<float>(y);

            for (int x = 0; x < m_virtualDepthImage.cols; ++x)
            {
                float v = vPtr[x];
                float r = rPtr[x];
                if (std::isfinite(v) && std::isfinite(r) && v > 0.0f && r > 0.0f)
                {
                    m_samplePoints.push_back({x, y});
                }
            }
        }

        if (m_samplePoints.empty() || sampleCount >= m_samplePoints.size())
        {
            return;
        }

        std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
        std::shuffle(m_samplePoints.begin(), m_samplePoints.end(), rng);
        m_samplePoints.resize(sampleCount);

        if (sampleCount < m_samplePoints.size())
        {
            m_samplePoints.resize(sampleCount);
        }
        std::vector<cv::Point> stableSamplePoints;
        stableSamplePoints.reserve(m_samplePoints.size());

        for (const auto& pt : m_samplePoints)
        {
            bool vdOk = false;
            bool refOk = false;
            // 1) 检查 VD 图上的局部稳定性
            {
                std::vector<float> neighbors;
                neighbors.reserve(targetNeighborCount_vd);

                float centerValue = m_virtualDepthImage.at<float>(pt.y, pt.x);
                bool centerValid = std::isfinite(centerValue) && centerValue > 0.0f;

                for (int radius = 1; radius <= maxRadius && neighbors.size() < targetNeighborCount_vd; ++radius)
                {
                    int left   = std::max(0, pt.x - radius);
                    int right  = std::min(m_virtualDepthImage.cols - 1, pt.x + radius);
                    int top    = std::max(0, pt.y - radius);
                    int bottom = std::min(m_virtualDepthImage.rows - 1, pt.y + radius);

                    for (int xx = left; xx <= right && neighbors.size() < targetNeighborCount_vd; ++xx)
                    {
                        if (!(xx == pt.x && top == pt.y))
                        {
                            float v = m_virtualDepthImage.at<float>(top, xx);
                            if (std::isfinite(v) && v > 0.0f)
                            {
                                neighbors.push_back(v);
                            }
                        }

                        if (bottom != top && neighbors.size() < targetNeighborCount_vd)
                        {
                            if (!(xx == pt.x && bottom == pt.y))
                            {
                                float v = m_virtualDepthImage.at<float>(bottom, xx);
                                if (std::isfinite(v) && v > 0.0f)
                                {
                                    neighbors.push_back(v);
                                }
                            }
                        }
                    }

                    for (int yy = top + 1; yy <= bottom - 1 && neighbors.size() < targetNeighborCount_vd; ++yy)
                    {
                        if (!(left == pt.x && yy == pt.y))
                        {
                            float v = m_virtualDepthImage.at<float>(yy, left);
                            if (std::isfinite(v) && v > 0.0f)
                            {
                                neighbors.push_back(v);
                            }
                        }

                        if (right != left && neighbors.size() < targetNeighborCount_vd)
                        {
                            if (!(right == pt.x && yy == pt.y))
                            {
                                float v = m_virtualDepthImage.at<float>(yy, right);
                                if (std::isfinite(v) && v > 0.0f)
                                {
                                    neighbors.push_back(v);
                                }
                            }
                        }
                    }
                }

                if (neighbors.size() >= targetNeighborCount_vd)
                {
                    float baseValue = 0.0f;

                    if (centerValid)
                    {
                        baseValue = centerValue;
                    }
                    else
                    {
                        float meanValue = 0.0f;
                        for (float v : neighbors)
                        {
                            meanValue += v;
                        }
                        meanValue /= neighbors.size();
                        baseValue = meanValue;
                    }

                    float sqErrSum = 0.0;
                    for (float v : neighbors)
                    {
                        float diff = v - baseValue;
                        sqErrSum += diff * diff;
                    }

                    float rmse = std::sqrt(sqErrSum / neighbors.size());
                    vdOk = (rmse < vdRmseThreshold && rmse > 0);
                }
            }

            if (!vdOk)
            {
                continue;
            }

            // 2) 检查 REF 图上的局部稳定性
            {
                std::vector<float> neighbors;
                neighbors.reserve(targetNeighborCount_gt);

                float centerValue = m_refDepthImage.at<float>(pt.y, pt.x);
                bool centerValid = std::isfinite(centerValue) && centerValue > 0.0f;

                for (int radius = 1; radius <= maxRadius && neighbors.size() < targetNeighborCount_gt; ++radius)
                {
                    int left   = std::max(0, pt.x - radius);
                    int right  = std::min(m_refDepthImage.cols - 1, pt.x + radius);
                    int top    = std::max(0, pt.y - radius);
                    int bottom = std::min(m_refDepthImage.rows - 1, pt.y + radius);

                    // 上边和下边
                    for (int xx = left; xx <= right && neighbors.size() < targetNeighborCount_gt; ++xx)
                    {
                        if (!(xx == pt.x && top == pt.y))
                        {
                            float v = m_refDepthImage.at<float>(top, xx);
                            if (std::isfinite(v) && v > 0.0f)
                            {
                                neighbors.push_back(v);
                            }
                        }

                        if (bottom != top && neighbors.size() < targetNeighborCount_gt)
                        {
                            if (!(xx == pt.x && bottom == pt.y))
                            {
                                float v = m_refDepthImage.at<float>(bottom, xx);
                                if (std::isfinite(v) && v > 0.0f)
                                {
                                    neighbors.push_back(v);
                                }
                            }
                        }
                    }

                    // 左边和右边（避免重复 corners）
                    for (int yy = top + 1; yy <= bottom - 1 && neighbors.size() < targetNeighborCount_gt; ++yy)
                    {
                        if (!(left == pt.x && yy == pt.y))
                        {
                            float v = m_refDepthImage.at<float>(yy, left);
                            if (std::isfinite(v) && v > 0.0f)
                            {
                                neighbors.push_back(v);
                            }
                        }

                        if (right != left && static_cast<int>(neighbors.size()) < targetNeighborCount_gt)
                        {
                            if (!(right == pt.x && yy == pt.y))
                            {
                                float v = m_refDepthImage.at<float>(yy, right);
                                if (std::isfinite(v) && v > 0.0f)
                                {
                                    neighbors.push_back(v);
                                }
                            }
                        }
                    }
                }

                if (neighbors.size() >= targetNeighborCount_gt)
                {
                    float baseValue = 0.0;

                    if (centerValid)
                    {
                        baseValue = centerValue;
                    }
                    else
                    {
                        float meanValue = 0.0;
                        for (float v : neighbors)
                        {
                            meanValue += v;
                        }
                        meanValue /= neighbors.size();
                        baseValue = meanValue;
                    }

                    float sqErrSum = 0.0;
                    for (float v : neighbors)
                    {
                        float diff = v - baseValue;
                        sqErrSum += diff * diff;
                    }

                    float rmse = std::sqrt(sqErrSum / neighbors.size());
                    refOk = (rmse < refRmseThreshold && rmse > 0);
                }
            }

            if (!refOk)
            {
                continue;
            }
            stableSamplePoints.push_back(pt);
        }
        m_samplePoints.swap(stableSamplePoints);
    }

    cv::Mat VirtualToRealDepthFunc::buildOutlierMask(cv::Mat &img, int ksize, double k)
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
                        {3.9f, 4.2f},
                        {4.2f, 4.3f},
                        {4.3f, 4.4f},
                        {4.4f, 4.5f},
                        {4.5f, 4.6f},
                        {4.6f, 4.8f},
                        {4.8f, 5.0f},
                        {5.0f, 5.2f}
                };

        int points_per_range = 20;      // 每个区间的采样点数
        float neighbor_tol = 0.2f;      // 局部邻域内点距离小于该值时，认为该点是同一个点
        int minDist = 20;      // 相邻点之间的最小距离

        float init_value_tol_ratio = 0.01f;
        float max_value_tol_ratio  = 0.05f;
        float value_tol_step_ratio = 0.001f;

        //  区域划分
        int gridCols = 6;      // 划分为n*n
        int gridRows = 6;
        int primaryCellsRange = 3;   // primary：只用前几个格子
        int fallbackExtraCells = 3;    // fallback：额外再放宽几个格子候选
        float min_gt_valid_ratio = 0.00005f;     // 格子中最小有效GT值比例
        int min_range_pixel_count = 20;     // 格子中最小有效vd像素数

        std::vector<cv::Scalar> colors
                {
                        cv::Scalar(255,0,0),
                        cv::Scalar(0,255,0),
                        cv::Scalar(0,0,255),
                        cv::Scalar(0,255,255),
                        cv::Scalar(255,0,255),
                        cv::Scalar(255,255,0),
                        cv::Scalar(200,255,0),
                        cv::Scalar(100,255,0),
                        cv::Scalar(200,100,0),
                        cv::Scalar(100,100,0),
                        cv::Scalar(255,200,0)
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

        /*struct BehaviorSegmentResult
        {
            float vdepthMin;
            float vdepthMax;
            int sampleCount;
            std::array<double, 3> params;
        };*/

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
                    {
//                        std::cout << "    filter: x=" << x
//                                  << " y=" << y
//                                  << " V=" << v
//                                  << " cell=" << cell.id
////                                  << " reason=isValidPixel_fail"
//                                  << std::endl;
//                        continue;
                    }

                    float gt_value;
                    int gx, gy;
                    if (!findStableGTValue(gt, x, y, gt_value, gx, gy, 20))
                    {
                        /*std::cout << "    filter: x=" << x
                                  << " y=" << y
                                  << " V=" << v
                                  << " cell=" << cell.id
//                                  << " reason=findStableGTValue_fail"
                                  << std::endl;*/
                        continue;
                    }

                    float stability = localStabilityScore(x, y, v);
                    float score = std::fabs(v - center_v) * 10.0f + stability;

                    out_candidates.push_back({x, y, v, gt_value, gx, gy, score, cell.id});
                }
            }
        };

        // 被选取参与拟合的点，在全聚焦图像用不同颜色标注
        auto addSampleToVis = [&](int idx, int selected_in_range, const cv::Scalar& color, const Candidate& c)
        {
            cv::circle(vis, cv::Point(c.x, c.y), 8, color, 2);

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

                // 与图中标注保持一致的编号
                std::string label = std::to_string(idx) + "-" + std::to_string(selected_in_range);

                // 可视化参与拟合的样本点
                addSampleToVis(idx, selected_in_range, color, c);

                std::cout << "    pick[" << label << "]: "
                          << "x=" << c.x
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

                /*for (const auto& c : fallback_candidates)
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
                }*/
            }

            std::cout << "Range done, selected count: " << selected_in_range << std::endl;
            // Step 3: 分段拟合行为模型参数
            if ((int)range_samples.size() >= 3)
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
                seg.vdepthMin = rmin;
                seg.vdepthMax = rmax;
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
                    xml << "            <DepthMin>" << seg.vdepthMin << "</DepthMin>\n";
                    xml << "            <DepthMax>" << seg.vdepthMax << "</DepthMax>\n\n";
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

    // 挑选拟合多个行为模型参数的样本点
    /*void VirtualToRealDepthFunc::sampleVirtualDepthPointsByRegion(
            std::string& vdepth_path,
            std::string& gt_path,
            std::string& output_mark_path,
            std::string& output_csv)
    {
        std::vector<std::pair<float,float>> ranges =
                {
                        {3.9f, 4.2f},
                        {4.2f, 4.25f},
                        {4.25f, 4.3f},
                        {4.3f, 4.35f},
                        {4.35f, 4.4f},
                        {4.4f, 4.45f},
                        {4.45f, 4.5f},
                        {4.5f, 4.6f},
                        {4.6f, 4.8f},
                        {4.8f, 5.0f},
                        {5.0f, 5.2f}
                };

        int points_per_range = 20;      // 每个区间的采样点数
        float neighbor_tol = 0.2f;      // 局部邻域内点距离小于该值时，认为该点是同一个点
        int minDist = 20;               // 相邻点之间的最小距离

        float init_value_tol_ratio = 0.01f;
        float max_value_tol_ratio  = 0.05f;
        float value_tol_step_ratio = 0.001f;

        //  区域划分
        int gridCols = 6;               // 划分为n*n
        int gridRows = 6;
        int primaryCellsRange = 3;      // primary：只用前几个格子
        int fallbackExtraCells = 3;     // fallback：额外再放宽几个格子候选
        float min_gt_valid_ratio = 0.00005f;   // 格子中最小有效GT值比例
        int min_range_pixel_count = 20;        // 格子中最小有效候选数

        // GT 驱动候选的局部统计参数
        int vd_patch_radius = 5;
        int min_vd_neighbors = 3;
        float max_vd_mad = 0.03f;
        float max_vd_span = 0.12f;
        float gt_hist_step_mm = 2000.0f;

        std::vector<cv::Scalar> colors
                {
                        cv::Scalar(255,0,0),
                        cv::Scalar(0,255,0),
                        cv::Scalar(0,0,255),
                        cv::Scalar(0,255,255),
                        cv::Scalar(255,0,255),
                        cv::Scalar(255,255,0),
                        cv::Scalar(200,255,0),
                        cv::Scalar(100,255,0),
                        cv::Scalar(200,100,0),
                        cv::Scalar(100,100,0),
                        cv::Scalar(255,200,0)
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
            float base_score;
            float vd_mad;
            float vd_span;
            int vd_count;
            int cell_id;
        };

        struct BehaviorSegmentResult
        {
            float vdepthMin;
            float vdepthMax;
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

        struct PatchStats
        {
            bool valid;
            int count;
            float median;
            float q10;
            float q90;
            float mad;
        };

        struct GTBandInfo
        {
            bool valid;
            float peak_center;
            float center;
            float low;
            float high;
            int peak_count;
        };

        cv::Mat depth = imread(vdepth_path, cv::IMREAD_UNCHANGED);
        if (depth.empty())
        {
            std::cout << "无法读取虚拟深度图" << std::endl;
            return;
        }
        if (depth.channels() > 1)
            extractChannel(depth, depth, 0);
        depth.convertTo(depth, CV_32F);

        cv::Mat gt = imread(gt_path, cv::IMREAD_UNCHANGED);
        if (gt.empty())
        {
            std::cout << "无法读取GT" << std::endl;
            return;
        }
        if (gt.channels() > 1)
            extractChannel(gt, gt, 0);
        gt.convertTo(gt, CV_32F);

        cv::Mat vis;
        {
            boost::filesystem::path depth_path(vdepth_path);
            boost::filesystem::path vis_path = depth_path.parent_path() / "fullfocus.png";

            vis = cv::imread(vis_path.string(), cv::IMREAD_COLOR);
            if (vis.empty())
                std::cout << "无法读取可视化全聚焦图" << std::endl;
        }

        std::vector<Sample> samples;
        std::vector<cv::Point> selected_points;
        std::vector<BehaviorSegmentResult> segment_results;

        int w = depth.cols;
        int h = depth.rows;

        int cell_w = std::max(1, (w + gridCols - 1) / gridCols);
        int cell_h = std::max(1, (h + gridRows - 1) / gridRows);

        (void)init_value_tol_ratio;
        (void)max_value_tol_ratio;
        (void)value_tol_step_ratio;

//----------------------------局部lambda变量(主函数在后面)------------------------------
        auto getQuantileFromSorted = [&](const std::vector<float>& vals, float q) -> float
        {
            if (vals.empty())
                return 0.0f;
            if (vals.size() == 1)
                return vals[0];

            float pos = q * static_cast<float>(vals.size() - 1);
            int lo = static_cast<int>(std::floor(pos));
            int hi = static_cast<int>(std::ceil(pos));
            float t = pos - static_cast<float>(lo);

            if (lo == hi)
                return vals[lo];
            return vals[lo] * (1.0f - t) + vals[hi] * t;
        };

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

        auto isFinitePositive = [&](float v) -> bool
        {
            return std::isfinite(v) && v > 0.0f;
        };

        auto getCellId = [&](int x, int y) -> int
        {
            int cx = std::min(gridCols - 1, std::max(0, x / cell_w));
            int cy = std::min(gridRows - 1, std::max(0, y / cell_h));
            return cy * gridCols + cx;
        };

        auto computePatchStats = [&](const cv::Mat& src, int x, int y, int radius) -> PatchStats
        {
            PatchStats st;
            st.valid = false;
            st.count = 0;
            st.median = 0.0f;
            st.q10 = 0.0f;
            st.q90 = 0.0f;
            st.mad = 0.0f;

            int x0 = std::max(0, x - radius);
            int x1 = std::min(src.cols - 1, x + radius);
            int y0 = std::max(0, y - radius);
            int y1 = std::min(src.rows - 1, y + radius);

            std::vector<float> vals;
            vals.reserve((2 * radius + 1) * (2 * radius + 1));

            for (int yy = y0; yy <= y1; ++yy)
            {
                for (int xx = x0; xx <= x1; ++xx)
                {
                    float v = src.at<float>(yy, xx);
                    if (!isFinitePositive(v))
                        continue;
                    vals.push_back(v);
                }
            }

            if (vals.empty())
                return st;

            std::sort(vals.begin(), vals.end());
            st.count = static_cast<int>(vals.size());
            st.median = getQuantileFromSorted(vals, 0.5f);
            st.q10 = getQuantileFromSorted(vals, 0.1f);
            st.q90 = getQuantileFromSorted(vals, 0.9f);

            std::vector<float> abs_dev;
            abs_dev.reserve(vals.size());
            for (size_t i = 0; i < vals.size(); ++i)
                abs_dev.push_back(std::fabs(vals[i] - st.median));
            std::sort(abs_dev.begin(), abs_dev.end());
            st.mad = getQuantileFromSorted(abs_dev, 0.5f);
            st.valid = true;
            return st;
        };

        // 从 GT 像素出发，在 GT 点周围取一个鲁棒的 vd 值
        auto findRobustVDValueAroundGT = [&](int gt_x,
                                             int gt_y,
                                             float& vd_value,
                                             float& vd_mad,
                                             float& vd_span,
                                             int& vd_cnt) -> bool
        {
            PatchStats st = computePatchStats(depth, gt_x, gt_y, vd_patch_radius);
            if (!st.valid)
                return false;

            vd_cnt = st.count;
            vd_mad = st.mad;
            vd_span = st.q90 - st.q10;

            if (vd_cnt < min_vd_neighbors)
                return false;
            if (vd_mad > max_vd_mad)
                return false;
            if (vd_span > max_vd_span)
                return false;

            float center_v = depth.at<float>(gt_y, gt_x);
            bool use_center = false;

            if (isFinitePositive(center_v))
            {
                float center_tol = std::max(0.03f, st.mad * 3.0f + 0.02f);
                if (std::fabs(center_v - st.median) <= center_tol &&
                    isValidPixel(depth, gt_x, gt_y, center_v, neighbor_tol))
                {
                    use_center = true;
                }
            }

            vd_value = use_center ? center_v : st.median;
            return true;
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
                    c.priority_score = -1e9f;

                    for (int y = c.y0; y < c.y1; ++y)
                    {
                        for (int x = c.x0; x < c.x1; ++x)
                        {
                            c.total_pixels++;

                            float vv = depth.at<float>(y, x);
                            if (isFinitePositive(vv))
                                c.valid_v_pixels++;

                            float gv = gt.at<float>(y, x);
                            if (isFinitePositive(gv))
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

        // 先一次性构造 GT 驱动候选，后续各个 vd 区间直接从这里筛
        auto buildGlobalCandidates = [&]() -> std::vector<Candidate>
        {
            std::vector<Candidate> out_candidates;
            out_candidates.reserve(50000);

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float gt_value = gt.at<float>(y, x);
                    if (!isFinitePositive(gt_value))
                        continue;

                    float vd_value = 0.0f;
                    float vd_mad = 0.0f;
                    float vd_span = 0.0f;
                    int vd_cnt = 0;

                    if (!findRobustVDValueAroundGT(x, y, vd_value, vd_mad, vd_span, vd_cnt))
                        continue;

                    Candidate c;
                    c.x = x;
                    c.y = y;
                    c.v = vd_value;
                    c.gt = gt_value;
                    c.gx = x;
                    c.gy = y;
                    c.vd_mad = vd_mad;
                    c.vd_span = vd_span;
                    c.vd_count = vd_cnt;
                    c.base_score = vd_mad * 10.0f + vd_span * 3.0f;
                    c.score = c.base_score;
                    c.cell_id = getCellId(x, y);

                    out_candidates.push_back(c);
                }
            }

            return out_candidates;
        };

        auto estimateCleanGTBand = [&](const std::vector<Candidate>& range_candidates) -> GTBandInfo
        {
            GTBandInfo band;
            band.valid = false;
            band.peak_center = 0.0f;
            band.center = 0.0f;
            band.low = 0.0f;
            band.high = 0.0f;
            band.peak_count = 0;

            if (range_candidates.empty())
                return band;

            std::vector<float> gt_values;
            gt_values.reserve(range_candidates.size());
            float min_gt = std::numeric_limits<float>::max();
            float max_gt = 0.0f;

            for (size_t i = 0; i < range_candidates.size(); ++i)
            {
                gt_values.push_back(range_candidates[i].gt);
                min_gt = std::min(min_gt, range_candidates[i].gt);
                max_gt = std::max(max_gt, range_candidates[i].gt);
            }

            std::sort(gt_values.begin(), gt_values.end());
            float median_gt = getQuantileFromSorted(gt_values, 0.5f);

            float hist_start = std::floor(std::max(0.0f, min_gt - gt_hist_step_mm) / gt_hist_step_mm) * gt_hist_step_mm;
            float hist_end = std::ceil((max_gt + gt_hist_step_mm) / gt_hist_step_mm) * gt_hist_step_mm;
            if (hist_end <= hist_start)
                hist_end = hist_start + gt_hist_step_mm;

            int hist_bins = std::max(1, static_cast<int>(std::ceil((hist_end - hist_start) / gt_hist_step_mm)));
            std::vector<int> hist(hist_bins, 0);

            auto toHistBin = [&](float g) -> int
            {
                int idx = static_cast<int>(std::floor((g - hist_start) / gt_hist_step_mm));
                idx = std::max(0, std::min(hist_bins - 1, idx));
                return idx;
            };

            for (size_t i = 0; i < range_candidates.size(); ++i)
                hist[toHistBin(range_candidates[i].gt)]++;

            std::vector<int> peak_bins;
            peak_bins.reserve(hist_bins);
            for (int i = 0; i < hist_bins; ++i)
            {
                if (hist[i] <= 0)
                    continue;

                int left = (i > 0) ? hist[i - 1] : -1;
                int right = (i + 1 < hist_bins) ? hist[i + 1] : -1;
                if (hist[i] >= left && hist[i] >= right)
                    peak_bins.push_back(i);
            }

            int selected_peak = -1;
            if (!peak_bins.empty())
            {
                float best_dist = std::numeric_limits<float>::max();
                int best_count = -1;

                for (size_t i = 0; i < peak_bins.size(); ++i)
                {
                    int bin_id = peak_bins[i];
                    float center = hist_start + (bin_id + 0.5f) * gt_hist_step_mm;
                    float dist = std::fabs(center - median_gt);

                    if (dist < best_dist - 1e-6f)
                    {
                        best_dist = dist;
                        best_count = hist[bin_id];
                        selected_peak = bin_id;
                    }
                    else if (std::fabs(dist - best_dist) <= 1e-6f)
                    {
                        if (hist[bin_id] > best_count)
                        {
                            best_count = hist[bin_id];
                            selected_peak = bin_id;
                        }
                    }
                }
            }
            else
            {
                selected_peak = toHistBin(median_gt);
            }

            if (selected_peak < 0 || selected_peak >= hist_bins)
                return band;

            float peak_center = hist_start + (selected_peak + 0.5f) * gt_hist_step_mm;

            float half_width = 5000.0f;
            if (peak_center >= 50000.0f)
                half_width = 4000.0f;
            else if (peak_center >= 30000.0f)
                half_width = 6000.0f;
            else if (peak_center >= 20000.0f)
                half_width = 5500.0f;
            else
                half_width = 4500.0f;

            float low = std::max(hist_start, peak_center - half_width);
            float high = std::min(hist_end, peak_center + half_width);

            std::vector<float> inlier_gt;
            inlier_gt.reserve(range_candidates.size());
            for (size_t i = 0; i < range_candidates.size(); ++i)
            {
                float g = range_candidates[i].gt;
                if (g >= low && g < high)
                    inlier_gt.push_back(g);
            }

            if (inlier_gt.empty())
                return band;

            std::sort(inlier_gt.begin(), inlier_gt.end());
            float refined_center = getQuantileFromSorted(inlier_gt, 0.5f);

            low = std::max(hist_start, refined_center - half_width);
            high = std::min(hist_end, refined_center + half_width);

            band.valid = true;
            band.peak_center = peak_center;
            band.center = refined_center;
            band.low = low;
            band.high = high;
            band.peak_count = hist[selected_peak];
            return band;
        };

        // 统计格子区间的均值等，获取格子优先级
        auto updateCellsForRange = [&](std::vector<CellInfo>& cells,
                                       const std::vector<Candidate>& range_candidates)
        {
            for (size_t i = 0; i < cells.size(); ++i)
            {
                cells[i].range_pixel_count = 0;
                cells[i].range_mean_v = 0.0f;
                cells[i].priority_score = -1e9f;
            }

            std::vector<double> v_sum(cells.size(), 0.0);

            for (size_t i = 0; i < range_candidates.size(); ++i)
            {
                int id = range_candidates[i].cell_id;
                if (id < 0 || id >= static_cast<int>(cells.size()))
                    continue;

                cells[id].range_pixel_count++;
                v_sum[id] += range_candidates[i].v;
            }

            for (size_t i = 0; i < cells.size(); ++i)
            {
                if (cells[i].range_pixel_count > 0)
                    cells[i].range_mean_v = static_cast<float>(v_sum[i] / cells[i].range_pixel_count);

                if (cells[i].range_pixel_count >= min_range_pixel_count &&
                    cells[i].valid_gt_ratio >= min_gt_valid_ratio)
                {
                    cells[i].priority_score =
                            static_cast<float>(cells[i].range_pixel_count) +
                            cells[i].valid_gt_ratio * 1000.0f;
                }
            }
        };

        // 从格子中选择候选拟合点（这里直接从 GT 驱动的候选池中过滤）
        auto collectCandidatesInCell = [&](const CellInfo& cell,
                                           float rmin,
                                           float rmax,
                                           const GTBandInfo& gt_band,
                                           const std::vector<Candidate>& range_candidates,
                                           std::vector<Candidate>& out_candidates)
        {
            float center_v = 0.5f * (rmin + rmax);

            for (size_t i = 0; i < range_candidates.size(); ++i)
            {
                const Candidate& src = range_candidates[i];
                if (src.cell_id != cell.id)
                    continue;

                Candidate c = src;
                c.score = c.base_score +
                          std::fabs(c.v - center_v) * 10.0f +
                          std::fabs(c.gt - gt_band.center) / 3000.0f;

                out_candidates.push_back(c);
            }
        };

        // 被选取参与拟合的点，在全聚焦图像用不同颜色标注
        auto addSampleToVis = [&](int idx, int selected_in_range, const cv::Scalar& color, const Candidate& c)
        {
            if (vis.empty())
                return;

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

        auto pickCandidatesRoundRobin = [&](const std::vector<CellInfo>& cell_list,
                                            const std::vector<Candidate>& candidate_pool,
                                            int idx,
                                            const cv::Scalar& color,
                                            int& selected_in_range,
                                            std::vector<Sample>& range_samples,
                                            bool is_fallback)
        {
            std::map<int, std::vector<Candidate> > by_cell;
            for (size_t i = 0; i < candidate_pool.size(); ++i)
                by_cell[candidate_pool[i].cell_id].push_back(candidate_pool[i]);

            for (std::map<int, std::vector<Candidate> >::iterator it = by_cell.begin(); it != by_cell.end(); ++it)
                sortCandidatesDeterministic(it->second);

            std::vector<size_t> cursor(cell_list.size(), 0);

            bool progress = true;
            while ((int)range_samples.size() < points_per_range && progress)
            {
                progress = false;

                for (size_t i = 0; i < cell_list.size(); ++i)
                {
                    int cell_id = cell_list[i].id;
                    std::map<int, std::vector<Candidate> >::iterator it = by_cell.find(cell_id);
                    if (it == by_cell.end())
                        continue;

                    std::vector<Candidate>& vec = it->second;
                    while (cursor[i] < vec.size())
                    {
                        const Candidate& c = vec[cursor[i]++];
                        if (!isFarEnough(c.x, c.y))
                            continue;

                        Sample s;
                        s.colIndex = c.x;
                        s.rowIndex = c.y;
                        s.vDepth = c.v;
                        s.rDepth = c.gt;

                        samples.push_back(s);
                        range_samples.push_back(s);
                        selected_points.push_back(cv::Point(c.x, c.y));
                        addSampleToVis(idx, selected_in_range, color, c);

                        if (!is_fallback)
                        {
                            std::cout << "    pick: x=" << c.x
                                      << " y=" << c.y
                                      << " V=" << c.v
                                      << " GT=" << c.gt
                                      << " cell=" << c.cell_id
                                      << " score=" << c.score
                                      << std::endl;
                        }
                        else
                        {
                            std::cout << "    fallback pick: x=" << c.x
                                      << " y=" << c.y
                                      << " V=" << c.v
                                      << " GT=" << c.gt
                                      << " cell=" << c.cell_id
                                      << " score=" << c.score
                                      << std::endl;
                        }

                        selected_in_range++;
                        progress = true;
                        break;
                    }

                    if ((int)range_samples.size() >= points_per_range)
                        break;
                }
            }
        };
//----------------------------------------------------------
        std::vector<Candidate> global_candidates = buildGlobalCandidates();
        std::cout << "GT-driven candidate pool: " << global_candidates.size() << std::endl;

        //  主函数开始
        for (int idx = 0; idx < (int)ranges.size(); ++idx)
        {
            float rmin = ranges[idx].first;
            float rmax = ranges[idx].second;

            std::cout << "Range " << rmin << "-" << rmax << std::endl;

            cv::Scalar color = colors[idx % colors.size()];
            int selected_in_range = 0;

            std::vector<Sample> range_samples;

            // Step 0: 先按 vd 区间收一遍 GT 驱动候选
            std::vector<Candidate> range_candidates_raw;
            range_candidates_raw.reserve(global_candidates.size() / 4 + 1);

            for (size_t i = 0; i < global_candidates.size(); ++i)
            {
                if (global_candidates[i].v >= rmin && global_candidates[i].v < rmax)
                    range_candidates_raw.push_back(global_candidates[i]);
            }

            if (range_candidates_raw.empty())
            {
                std::cout << "  no GT-driven candidates for this range." << std::endl;
                continue;
            }

            GTBandInfo gt_band = estimateCleanGTBand(range_candidates_raw);
            if (!gt_band.valid)
            {
                std::cout << "  failed to estimate GT band for this range." << std::endl;
                continue;
            }

            // Step 1: 在当前 vd 区间里，只保留一个连续 GT 子带，避免 GT 多模态污染
            std::vector<Candidate> range_candidates;
            range_candidates.reserve(range_candidates_raw.size());
            for (size_t i = 0; i < range_candidates_raw.size(); ++i)
            {
                if (range_candidates_raw[i].gt >= gt_band.low &&
                    range_candidates_raw[i].gt < gt_band.high)
                {
                    range_candidates.push_back(range_candidates_raw[i]);
                }
            }

            if (range_candidates.empty())
            {
                std::cout << "  no clean GT-band candidates for this range." << std::endl;
                continue;
            }

            std::cout << "  GT band: [" << gt_band.low << ", " << gt_band.high
                      << "), center=" << gt_band.center
                      << ", peakCenter=" << gt_band.peak_center
                      << ", raw=" << range_candidates_raw.size()
                      << ", clean=" << range_candidates.size()
                      << std::endl;

            std::vector<CellInfo> cells = buildCells();
            // Step 1.5: 统计格子区域的均值，计算格子优先级
            updateCellsForRange(cells, range_candidates);

            std::sort(cells.begin(), cells.end(),
                      [](const CellInfo& a, const CellInfo& b)
                      {
                          if (std::fabs(a.priority_score - b.priority_score) > 1e-6f)
                              return a.priority_score > b.priority_score;
                          return a.id < b.id;
                      });

            std::vector<CellInfo> primary_cells;    // 优先选取格子
            std::vector<CellInfo> fallback_cells;   // 候选格子

            for (size_t i = 0; i < cells.size(); ++i)
            {
                const CellInfo& c = cells[i];
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
            for (size_t i = 0; i < primary_cells.size(); ++i)
            {
                const CellInfo& c = primary_cells[i];
                std::cout << "[id=" << c.id
                          << " rangeCount=" << c.range_pixel_count
                          << " gtRatio=" << c.valid_gt_ratio
                          << " meanV=" << c.range_mean_v
                          << "] ";
            }
            std::cout << std::endl;

            std::cout << "  fallback cells: ";
            for (size_t i = 0; i < fallback_cells.size(); ++i)
            {
                const CellInfo& c = fallback_cells[i];
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

            for (size_t i = 0; i < primary_cells.size(); ++i)
                collectCandidatesInCell(primary_cells[i], rmin, rmax, gt_band, range_candidates, primary_candidates);

            {
                std::vector<Candidate> deduped;
                deduped.reserve(primary_candidates.size());
                std::set<std::pair<int,int> > used_xy;

                for (size_t i = 0; i < primary_candidates.size(); ++i)
                {
                    std::pair<int,int> key(primary_candidates[i].x, primary_candidates[i].y);
                    if (used_xy.insert(key).second)
                        deduped.push_back(primary_candidates[i]);
                }

                primary_candidates.swap(deduped);
            }

            std::cout << "  primary candidate pool: " << primary_candidates.size() << std::endl;
            pickCandidatesRoundRobin(primary_cells, primary_candidates, idx, color, selected_in_range, range_samples, false);

            // fallback pick：只从 fallback_cells 里补
            if ((int)range_samples.size() < points_per_range && !fallback_cells.empty())
            {
                std::vector<Candidate> fallback_candidates;
                fallback_candidates.reserve(50000);

                for (size_t i = 0; i < fallback_cells.size(); ++i)
                    collectCandidatesInCell(fallback_cells[i], rmin, rmax, gt_band, range_candidates, fallback_candidates);

                {
                    std::vector<Candidate> deduped;
                    deduped.reserve(fallback_candidates.size());
                    std::set<std::pair<int,int> > used_xy;

                    for (size_t i = 0; i < fallback_candidates.size(); ++i)
                    {
                        std::pair<int,int> key(fallback_candidates[i].x, fallback_candidates[i].y);
                        if (used_xy.insert(key).second)
                            deduped.push_back(fallback_candidates[i]);
                    }

                    fallback_candidates.swap(deduped);
                }

                std::cout << "  fallback candidate pool: " << fallback_candidates.size() << std::endl;
                pickCandidatesRoundRobin(fallback_cells, fallback_candidates, idx, color, selected_in_range, range_samples, true);
            }

            std::cout << "Range done, selected count: " << selected_in_range << std::endl;
            // Step 3: 分段拟合行为模型参数
            if ((int)range_samples.size() >= 3)
            {
                std::vector<float> refDepthValue;
                std::vector<float> virtualDepthValue;
                refDepthValue.reserve(range_samples.size());
                virtualDepthValue.reserve(range_samples.size());

                for (size_t i = 0; i < range_samples.size(); ++i)
                {
                    virtualDepthValue.push_back(range_samples[i].vDepth);
                    refDepthValue.push_back(range_samples[i].rDepth);
                }

                std::array<double, 3> behaviorModelParams =
                        BehavioralModel(refDepthValue, virtualDepthValue);

                BehaviorSegmentResult seg;
                seg.vdepthMin = rmin;
                seg.vdepthMax = rmax;
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

        if (!vis.empty())
            imwrite(output_mark_path, vis);

        // 记录样本点，写入csv文件中
        std::ofstream csv(output_csv.c_str());
        csv << "x,y,virtual_depth,real_depth\n";
        for (size_t i = 0; i < samples.size(); ++i)
        {
            csv << samples[i].colIndex << ","
                << samples[i].rowIndex << ","
                << samples[i].vDepth << ","
                << samples[i].rDepth << "\n";
        }
        csv.close();

        // Step 4: 写出拟合结果xml文件
        {
            std::time_t t = std::time(NULL);
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

            std::ofstream xml(xml_path.c_str());
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

                for (size_t i = 0; i < segment_results.size(); ++i)
                {
                    const BehaviorSegmentResult& seg = segment_results[i];
                    xml << "        <Segment>\n";
                    xml << "            <DepthMin>" << seg.vdepthMin << "</DepthMin>\n";
                    xml << "            <DepthMax>" << seg.vdepthMax << "</DepthMax>\n\n";
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
    }*/

// 获取局部稳定vd像素点
    /*bool VirtualToRealDepthFunc::isValidPixel(
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

        std::vector<float> neighbors;
        neighbors.reserve(9);
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || nx >= gt.cols || ny < 0 || ny >= gt.rows)
                    continue;

                float v = gt.at<float>(ny, nx);
                if (!std::isfinite(v) || v <= 0.0f)
                    continue;
                neighbors.push_back(v);
            }
        }

        if (neighbors.empty())
            return false;
        if (neighbors.size() == 1)
            return true;

        std::sort(neighbors.begin(), neighbors.end());
        float median = neighbors[neighbors.size() / 2];

        float abs_tol = std::max(300.0f, std::fabs(median) * std::max(0.01f, neighbor_tol_ratio));
        int inlier_cnt = 0;
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            if (std::fabs(neighbors[i] - median) <= abs_tol)
                inlier_cnt++;
        }

        return inlier_cnt >= std::max(1, static_cast<int>(neighbors.size() / 2));
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
        const int min_valid_points = 2;      // 指定区域内最小有效GT点数
        const float value_tol_ratio = 0.05f; // 相对中位数容差
        const float min_abs_tol = 300.0f;    // GT单位为mm时的绝对容差

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
                if (!isStableGTPixel(gt, nx, ny, v, 0.02f))
                    continue;

                float dist2 = static_cast<float>(dx * dx + dy * dy);
                float weight = 1.0f / (dist2 + eps);
                candidates.push_back((Candidate){nx, ny, v, dist2, weight});
            }
        }

        if ((int)candidates.size() < min_valid_points)
            return false;

        std::vector<float> values;
        values.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i)
            values.push_back(candidates[i].value);

        std::sort(values.begin(), values.end());
        float median = values[values.size() / 2];

        float tol = std::max(min_abs_tol, std::fabs(median) * value_tol_ratio);

        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        int inlier_count = 0;

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (std::fabs(candidates[i].value - median) > tol)
                continue;

            weighted_sum += static_cast<double>(candidates[i].value) * candidates[i].weight;
            weight_sum += candidates[i].weight;
            inlier_count++;
        }

        if (inlier_count < min_valid_points || weight_sum <= eps)
            return false;

        gt_value = static_cast<float>(weighted_sum / weight_sum);

        bool found_nearest = false;
        float best_dist2 = std::numeric_limits<float>::max();
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (std::fabs(candidates[i].value - median) > tol)
                continue;

            if (candidates[i].dist2 < best_dist2)
            {
                best_dist2 = candidates[i].dist2;
                gt_x = candidates[i].x;
                gt_y = candidates[i].y;
                found_nearest = true;
            }
        }

        return found_nearest;
    }*/

}
