/********************************************************************
file base:      VirtualToRealDepthFunc.cpp
author:         LZD XYY
created:        2026/03/04
purpose:
*********************************************************************/
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <cmath>

#include "Common/Common.h"
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

    void VirtualToRealDepthFunc::VirtualToRealDepth(QuadTreeProblemMapMap::iterator& itrP)
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
                //VirtualToRealDepthBySegBM_2();
                VirtualToRealDepthBySegBM_new();
            }
                break;
            case VTORD_SegmentBehavioralmodel_Manual:    // 手动
            {
                std::string strFrameName = itrP->first;
                VirtualToRealDepthByManual(strFrameName);
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

void VirtualToRealDepthFunc::VirtualToRealDepthBySegBM_new()
{
    m_strRootPath = m_ptrDepthSolver->GetRootPath();

    const std::string virtualDepthImgPath = m_strRootPath + "/behavior_model/VD_Raw.tiff";
    const std::string refDepthImgPath     = m_strRootPath + "/behavior_model/ref-csad-rd.tiff";
    const std::string focusImgPath        = m_strRootPath + "/behavior_model/fullfocus.png";
    const std::string debugSeedPath       = m_strRootPath + "/behavior_model/vd_seed_pairs_debug.png";
    const std::string distanceImagePath   = m_strRootPath + "/behavior_model/distanceImage_new.png";

    boost::filesystem::path rootPath(m_strRootPath);
    boost::filesystem::path rootPathParent = rootPath.parent_path();
    const std::string strCalibPath = rootPathParent.string() + LF_CALIB_FOLDER_NAME;
    const std::string xmlPath      = strCalibPath + "behaviorModelParamsSegment.xml";

    m_virtualDepthImage = cv::imread(virtualDepthImgPath, cv::IMREAD_UNCHANGED);
    m_refDepthImage     = cv::imread(refDepthImgPath, cv::IMREAD_UNCHANGED);
    cv::Mat focusImage  = cv::imread(focusImgPath, cv::IMREAD_COLOR);

    if (m_virtualDepthImage.empty())
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] cannot read VD image: "
                  << virtualDepthImgPath << std::endl;
        return;
    }
    if (m_refDepthImage.empty())
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] cannot read GT image: "
                  << refDepthImgPath << std::endl;
        return;
    }
    if (focusImage.empty())
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] cannot read focus image: "
                  << focusImgPath << std::endl;
        return;
    }

    if (m_virtualDepthImage.channels() > 1)
        extractChannel(m_virtualDepthImage, m_virtualDepthImage, 0);
    if (m_refDepthImage.channels() > 1)
        extractChannel(m_refDepthImage, m_refDepthImage, 0);

    if (m_virtualDepthImage.type() != CV_32FC1)
        m_virtualDepthImage.convertTo(m_virtualDepthImage, CV_32F);
    if (m_refDepthImage.type() != CV_32FC1)
        m_refDepthImage.convertTo(m_refDepthImage, CV_32F);

    if (focusImage.size() != m_virtualDepthImage.size())
        cv::resize(focusImage, focusImage, m_virtualDepthImage.size(), 0.0, 0.0, cv::INTER_LINEAR);

    samplePoints.clear();
    samplePointsVector.clear();
    samplePointsVectorFiltered.clear();
    samplePointsVectorSorted.clear();
    m_samplePoints.clear();

    struct VDCandidate
    {
        int x;
        int y;
        float vdCenter;
        float vdMedian;
        float vdMad;
        float vdSpan;
        int supportCount;

        float aifGrad;
        float vdEdgeResp;
        float grayMean;
        float grayStd;
        float cornerResp;
        float gtDist;
        float edgeDist;

        float score;
        int cellId;
    };

    struct GTPoint
    {
        int x;
        int y;
        float value;
    };

    struct GTSupportResult
    {
        bool ok;
        std::vector<float> vals;
        std::vector<cv::Point> pts;
        float purity;
        float gap;
        int usedRadius;

        GTSupportResult()
            : ok(false), purity(0.0f), gap(0.0f), usedRadius(-1)
        {}
    };

    auto isValidValue = [](float v) -> bool
    {
        return std::isfinite(v) && v > 0.0f;
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

    auto robustMedian = [&](std::vector<float> vals) -> float
    {
        if (vals.empty())
            return 0.0f;
        std::sort(vals.begin(), vals.end());
        return quantileFromSorted(vals, 0.5f);
    };

    auto computePatchStats = [&](const cv::Mat& src,
                                 int cx, int cy, int radius,
                                 int minCount,
                                 float& median,
                                 float& mad,
                                 float& span,
                                 int& count) -> bool
    {
        std::vector<float> vals;
        vals.reserve((2 * radius + 1) * (2 * radius + 1));

        int x0 = std::max(0, cx - radius);
        int x1 = std::min(src.cols - 1, cx + radius);
        int y0 = std::max(0, cy - radius);
        int y1 = std::min(src.rows - 1, cy + radius);

        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                float v = src.at<float>(y, x);
                if (isValidValue(v))
                    vals.push_back(v);
            }
        }

        count = static_cast<int>(vals.size());
        if (count < minCount)
            return false;

        std::sort(vals.begin(), vals.end());
        median = quantileFromSorted(vals, 0.5f);
        float q10 = quantileFromSorted(vals, 0.1f);
        float q90 = quantileFromSorted(vals, 0.9f);
        span = q90 - q10;

        std::vector<float> absDev;
        absDev.reserve(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
            absDev.push_back(std::fabs(vals[i] - median));
        std::sort(absDev.begin(), absDev.end());
        mad = quantileFromSorted(absDev, 0.5f);

        return true;
    };

    auto computeGrayPatchStats = [&](const cv::Mat& gray,
                                     int cx, int cy, int radius,
                                     float& meanVal,
                                     float& stdVal) -> bool
    {
        int x0 = std::max(0, cx - radius);
        int x1 = std::min(gray.cols - 1, cx + radius);
        int y0 = std::max(0, cy - radius);
        int y1 = std::min(gray.rows - 1, cy + radius);

        cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        if (roi.width <= 0 || roi.height <= 0)
            return false;

        cv::Scalar meanS, stdS;
        cv::meanStdDev(gray(roi), meanS, stdS);
        meanVal = static_cast<float>(meanS[0]);
        stdVal  = static_cast<float>(stdS[0]);
        return true;
    };

    cv::Mat guideGray;
    cv::cvtColor(focusImage, guideGray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(guideGray, guideGray, cv::Size(5, 5), 1.0);

    cv::Mat gx, gy, guideGrad;
    cv::Sobel(guideGray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(guideGray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, guideGrad);
    {
        double gmax = 0.0;
        cv::minMaxLoc(guideGrad, nullptr, &gmax);
        if (gmax > 1e-6)
            guideGrad.convertTo(guideGrad, CV_32F, 255.0 / gmax);
        else
            guideGrad = cv::Mat::zeros(guideGrad.size(), CV_32F);
    }

    cv::Mat cornerResp;
    cv::cornerMinEigenVal(guideGray, cornerResp, 3, 3);
    {
        double cmax = 0.0;
        cv::minMaxLoc(cornerResp, nullptr, &cmax);
        if (cmax > 1e-12)
            cornerResp.convertTo(cornerResp, CV_32F, 255.0 / cmax);
        else
            cornerResp = cv::Mat::zeros(cornerResp.size(), CV_32F);
    }

    // ------------------------------------------------------------
    // VD 局部突变图：很多“深度边缘”在 AIF 梯度上不够强，但在 VD 上会跳变
    // ------------------------------------------------------------
    cv::Mat vdEdgeResp = cv::Mat::zeros(m_virtualDepthImage.size(), CV_32FC1);
    for (int y = 1; y < m_virtualDepthImage.rows - 1; ++y)
    {
        for (int x = 1; x < m_virtualDepthImage.cols - 1; ++x)
        {
            float v = m_virtualDepthImage.at<float>(y, x);
            if (!isValidValue(v))
                continue;

            float maxDiff = 0.0f;
            int validNbr = 0;

            static const int dx4[4] = {1, -1, 0, 0};
            static const int dy4[4] = {0, 0, 1, -1};

            for (int k = 0; k < 4; ++k)
            {
                int xx = x + dx4[k];
                int yy = y + dy4[k];
                float nv = m_virtualDepthImage.at<float>(yy, xx);
                if (!isValidValue(nv))
                    continue;

                maxDiff = std::max(maxDiff, std::fabs(nv - v));
                validNbr++;
            }

            if (validNbr >= 2)
                vdEdgeResp.at<float>(y, x) = maxDiff;
        }
    }

    // 归一化到 0~255，便于与 guideGrad 融合
    {
        double vmax = 0.0;
        cv::minMaxLoc(vdEdgeResp, nullptr, &vmax);
        if (vmax > 1e-8)
            vdEdgeResp.convertTo(vdEdgeResp, CV_32F, 255.0 / vmax);
        else
            vdEdgeResp = cv::Mat::zeros(vdEdgeResp.size(), CV_32F);
    }

    // ------------------------------------------------------------
    // 预提取所有有效 GT 点
    // ------------------------------------------------------------
    std::vector<GTPoint> gtPoints;
    gtPoints.reserve(10000);
    for (int y = 0; y < m_refDepthImage.rows; ++y)
    {
        for (int x = 0; x < m_refDepthImage.cols; ++x)
        {
            float gv = m_refDepthImage.at<float>(y, x);
            if (isValidValue(gv))
            {
                GTPoint p;
                p.x = x;
                p.y = y;
                p.value = gv;
                gtPoints.push_back(p);
            }
        }
    }
    std::cout << "[VirtualToRealDepthBySegBM_new] valid GT points: "
              << gtPoints.size() << std::endl;

    if (gtPoints.empty())
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] no valid GT points." << std::endl;
        return;
    }

    // ------------------------------------------------------------
    // GT 距离图
    // ------------------------------------------------------------
    cv::Mat gtValidMask = cv::Mat::zeros(m_refDepthImage.size(), CV_8UC1);
    for (size_t i = 0; i < gtPoints.size(); ++i)
        gtValidMask.at<uchar>(gtPoints[i].y, gtPoints[i].x) = 255;

    cv::Mat invGtMask;
    cv::bitwise_not(gtValidMask, invGtMask);

    cv::Mat gtDistMap;
    cv::distanceTransform(invGtMask, gtDistMap, cv::DIST_L2, 3);
    gtDistMap.convertTo(gtDistMap, CV_32F);

    // ------------------------------------------------------------
    // 联合边缘图：AIF 边缘 + VD 边缘
    // ------------------------------------------------------------
    const float aifEdgeThresh = 48.0f;
    const float vdEdgeThresh  = 22.0f;   // 归一化后的阈值，先保守一些

    cv::Mat strongEdgeMask = cv::Mat::zeros(guideGrad.size(), CV_8UC1);
    for (int y = 0; y < guideGrad.rows; ++y)
    {
        const float* gptr = guideGrad.ptr<float>(y);
        const float* vptr = vdEdgeResp.ptr<float>(y);
        uchar* mptr = strongEdgeMask.ptr<uchar>(y);

        for (int x = 0; x < guideGrad.cols; ++x)
        {
            if (gptr[x] > aifEdgeThresh || vptr[x] > vdEdgeThresh)
                mptr[x] = 255;
        }
    }

    cv::Mat edgeDistMap;
    if (cv::countNonZero(strongEdgeMask) > 0)
    {
        cv::Mat invStrongEdgeMask;
        cv::bitwise_not(strongEdgeMask, invStrongEdgeMask);
        cv::distanceTransform(invStrongEdgeMask, edgeDistMap, cv::DIST_L2, 3);
        edgeDistMap.convertTo(edgeDistMap, CV_32F);
    }
    else
    {
        edgeDistMap = cv::Mat(m_virtualDepthImage.size(), CV_32F, cv::Scalar(9999.0f));
    }

    // ------------------------------------------------------------
    // 参数
    // ------------------------------------------------------------
    const int   candidateStride         = 2;
    const int   candidatePatchRadius    = 2;
    const int   texturePatchRadius      = 2;
    const int   candidateMinCount       = 3;

    const float candidateMaxMad         = 0.080f;
    const float candidateMaxSpan        = 0.300f;
    const float candidateMaxGrad        = 140.0f;

    const float candidateDarkMeanReject = 22.0f;
    const float candidateDarkStdReject  = 6.0f;
    const float candidateMinGrayStd     = 3.0f;
    const float candidateMinCornerResp  = 1.5f;
    const float candidateMaxGtDist      = 24.0f;

    // 比上一版再收紧一些：seed 远离边
    const float candidateMinEdgeDist    = 7.0f;

    const int   gridCols                = 6;
    const int   gridRows                = 6;
    const int   seedsPerBin             = 24;
    const int   vdBinCount              = 8;

    const int   supportSearchRadius     = 8;
    const int   supportMaxPoints        = 10;   // 更紧凑
    const int   supportMinPoints        = 4;
    const float supportBandAbsMin       = 0.04f;
    const float supportBandMadScale     = 3.5f;
    const float supportMinMadFloor      = 0.006f;

    const float strongEdgeThresh        = 55.0f;
    const float maxGrayJump             = 28.0f;

    const float supportMinGrayStd       = 4.0f;
    const float supportMinCornerResp    = 1.5f;

    const float supportMaxEdgeRatio     = 0.28f;
    const float supportNearEdgeDist     = 3.5f;
    const float supportMaxNearEdgeRatio = 0.30f;

    // 线状 support 判定再严格些
    const float minSupportShapeRatio    = 0.18f;
    const float minSupportFillRatio     = 0.08f;

    // seed 与 support 质心偏太远，也说明 support 更像“沿边挂着”
    const float maxSupportCentroidDrift = 4.5f;

    const std::vector<int> gtSearchRadii = {8, 16, 28, 40, 56};
    const int   minGtCount              = 2;
    const float gtMadAbsTol             = 1800.0f;
    const float gtHistBinWidth          = 2500.0f;

    const float minGtPeakPurity         = 0.45f;
    const float minGtPeakGap            = 0.04f;
    const int   softMaxGtSearchRadius   = 40;

    const float maxGTCentroidOffset     = 7.0f;

    const int   minRepresentativePairsForFit = 6;

    auto cellIdOf = [&](int x, int y) -> int
    {
        int cw = std::max(1, (m_virtualDepthImage.cols + gridCols - 1) / gridCols);
        int ch = std::max(1, (m_virtualDepthImage.rows + gridRows - 1) / gridRows);
        int cx = std::min(gridCols - 1, std::max(0, x / cw));
        int cy = std::min(gridRows - 1, std::max(0, y / ch));
        return cy * gridCols + cx;
    };

    // ------------------------------------------------------------
    // Step 1: VD 候选池
    // ------------------------------------------------------------
    std::vector<VDCandidate> candidatePool;
    candidatePool.reserve((m_virtualDepthImage.rows / candidateStride) *
                          (m_virtualDepthImage.cols / candidateStride));

    float vdMinAll = std::numeric_limits<float>::infinity();
    float vdMaxAll = 0.0f;

    int rejectSeedNearEdge = 0;

    for (int y = candidatePatchRadius; y < m_virtualDepthImage.rows - candidatePatchRadius; y += candidateStride)
    {
        for (int x = candidatePatchRadius; x < m_virtualDepthImage.cols - candidatePatchRadius; x += candidateStride)
        {
            float center = m_virtualDepthImage.at<float>(y, x);
            if (!isValidValue(center))
                continue;

            float median = 0.0f, mad = 0.0f, span = 0.0f;
            int count = 0;
            if (!computePatchStats(m_virtualDepthImage, x, y,
                                   candidatePatchRadius,
                                   candidateMinCount,
                                   median, mad, span, count))
            {
                continue;
            }

            float grad = guideGrad.at<float>(y, x);

            float grayMean = 0.0f, grayStd = 0.0f;
            if (!computeGrayPatchStats(guideGray, x, y, texturePatchRadius, grayMean, grayStd))
                continue;

            float corner = cornerResp.at<float>(y, x);
            float gtDist = gtDistMap.at<float>(y, x);
            float edgeDist = edgeDistMap.at<float>(y, x);
            float vdEdge = vdEdgeResp.at<float>(y, x);

            if (gtDist > candidateMaxGtDist)
                continue;

            if (edgeDist < candidateMinEdgeDist)
            {
                ++rejectSeedNearEdge;
                continue;
            }

            if (mad > candidateMaxMad) continue;
            if (span > candidateMaxSpan) continue;
            if (grad > candidateMaxGrad) continue;

            if (grayMean < candidateDarkMeanReject && grayStd < candidateDarkStdReject)
                continue;

            if (grayStd < candidateMinGrayStd && corner < candidateMinCornerResp)
                continue;

            float texturePenalty = 0.0f;
            if (grayStd < 10.0f) texturePenalty += (10.0f - grayStd) * 0.05f;
            if (corner < 6.0f)   texturePenalty += (6.0f - corner) * 0.04f;
            if (grayMean < 30.0f) texturePenalty += (30.0f - grayMean) * 0.01f;

            texturePenalty += gtDist * 0.06f;
            texturePenalty += 2.5f / (edgeDist + 1.0f);
            texturePenalty += vdEdge * 0.01f;

            VDCandidate c;
            c.x = x;
            c.y = y;
            c.vdCenter = center;
            c.vdMedian = median;
            c.vdMad = mad;
            c.vdSpan = span;
            c.supportCount = count;
            c.aifGrad = grad;
            c.vdEdgeResp = vdEdge;
            c.grayMean = grayMean;
            c.grayStd = grayStd;
            c.cornerResp = corner;
            c.gtDist = gtDist;
            c.edgeDist = edgeDist;
            c.score = mad * 6.0f + span * 3.0f + grad * 0.01f + texturePenalty;
            c.cellId = cellIdOf(x, y);
            candidatePool.push_back(c);

            vdMinAll = std::min(vdMinAll, median);
            vdMaxAll = std::max(vdMaxAll, median);
        }
    }

    std::cout << "[VirtualToRealDepthBySegBM_new] VD candidate pool size: "
              << candidatePool.size() << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectSeedNearEdge(candidate): "
              << rejectSeedNearEdge << std::endl;

    if (candidatePool.size() < 3 || !(std::isfinite(vdMinAll) && std::isfinite(vdMaxAll) && vdMaxAll > vdMinAll))
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] no usable VD candidate pool." << std::endl;
        return;
    }

    // ------------------------------------------------------------
    // Step 2: 按 VD 分段选 seed
    // ------------------------------------------------------------
    struct VDBin
    {
        float low;
        float high;
        std::vector<VDCandidate> candidates;
    };

    std::vector<VDBin> bins(vdBinCount);
    const float step = (vdMaxAll - vdMinAll) / static_cast<float>(vdBinCount);
    for (int i = 0; i < vdBinCount; ++i)
    {
        bins[i].low = vdMinAll + step * static_cast<float>(i);
        bins[i].high = (i == vdBinCount - 1)
                       ? (vdMaxAll + 1e-6f)
                       : (vdMinAll + step * static_cast<float>(i + 1));
    }

    for (size_t i = 0; i < candidatePool.size(); ++i)
    {
        float v = candidatePool[i].vdMedian;
        for (int b = 0; b < vdBinCount; ++b)
        {
            bool inRange = (b == vdBinCount - 1)
                           ? (v >= bins[b].low && v <= bins[b].high)
                           : (v >= bins[b].low && v < bins[b].high);
            if (inRange)
            {
                bins[b].candidates.push_back(candidatePool[i]);
                break;
            }
        }
    }

    std::vector<VDCandidate> selectedSeeds;
    selectedSeeds.reserve(vdBinCount * seedsPerBin);
    std::vector<cv::Point> selectedSeedPts;

    auto isFarEnough = [&](int x, int y, int minDistPx) -> bool
    {
        for (size_t i = 0; i < selectedSeedPts.size(); ++i)
        {
            int dx = selectedSeedPts[i].x - x;
            int dy = selectedSeedPts[i].y - y;
            if (dx * dx + dy * dy < minDistPx * minDistPx)
                return false;
        }
        return true;
    };

    for (int b = 0; b < vdBinCount; ++b)
    {
        if (bins[b].candidates.empty())
            continue;

        std::map<int, std::vector<VDCandidate> > byCell;
        for (size_t i = 0; i < bins[b].candidates.size(); ++i)
            byCell[bins[b].candidates[i].cellId].push_back(bins[b].candidates[i]);

        for (std::map<int, std::vector<VDCandidate> >::iterator it = byCell.begin(); it != byCell.end(); ++it)
        {
            std::sort(it->second.begin(), it->second.end(),
                      [](const VDCandidate& a, const VDCandidate& b)
                      {
                          if (std::fabs(a.score - b.score) > 1e-6f)
                              return a.score < b.score;
                          if (a.y != b.y) return a.y < b.y;
                          return a.x < b.x;
                      });
        }

        std::vector<int> cellOrder;
        for (std::map<int, std::vector<VDCandidate> >::iterator it = byCell.begin(); it != byCell.end(); ++it)
            cellOrder.push_back(it->first);
        std::sort(cellOrder.begin(), cellOrder.end());

        std::map<int, size_t> cursor;
        for (size_t i = 0; i < cellOrder.size(); ++i)
            cursor[cellOrder[i]] = 0;

        int picked = 0;
        bool progress = true;
        while (picked < seedsPerBin && progress)
        {
            progress = false;
            for (size_t ci = 0; ci < cellOrder.size() && picked < seedsPerBin; ++ci)
            {
                int cid = cellOrder[ci];
                std::vector<VDCandidate>& vec = byCell[cid];
                while (cursor[cid] < vec.size())
                {
                    const VDCandidate& cand = vec[cursor[cid]++];
                    if (!isFarEnough(cand.x, cand.y, 12))
                        continue;

                    selectedSeeds.push_back(cand);
                    selectedSeedPts.push_back(cv::Point(cand.x, cand.y));
                    ++picked;
                    progress = true;
                    break;
                }
            }
        }
    }

    std::cout << "[VirtualToRealDepthBySegBM_new] VD seed count: "
              << selectedSeeds.size() << std::endl;

    if (selectedSeeds.size() < 3)
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] not enough VD seeds." << std::endl;
        return;
    }

    auto crossesStrongEdge = [&](int x0, int y0, int x1, int y1) -> bool
    {
        const int steps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
        if (steps <= 1)
            return false;

        float prevGray = static_cast<float>(guideGray.at<uchar>(y0, x0));

        for (int s = 1; s < steps; ++s)
        {
            float t = static_cast<float>(s) / static_cast<float>(steps);
            int x = static_cast<int>(std::round((1.0f - t) * x0 + t * x1));
            int y = static_cast<int>(std::round((1.0f - t) * y0 + t * y1));
            x = std::max(0, std::min(guideGrad.cols - 1, x));
            y = std::max(0, std::min(guideGrad.rows - 1, y));

            float g = guideGrad.at<float>(y, x);
            float vg = vdEdgeResp.at<float>(y, x);
            float curGray = static_cast<float>(guideGray.at<uchar>(y, x));
            float dGray = std::fabs(curGray - prevGray);

            if ((g > strongEdgeThresh && dGray > maxGrayJump) || vg > 24.0f)
                return true;

            prevGray = curGray;
        }
        return false;
    };

    auto distancePointToSupport = [&](int x, int y, const std::vector<cv::Point>& supportPts, int& nearestIdx) -> float
    {
        float bestD2 = std::numeric_limits<float>::infinity();
        nearestIdx = -1;
        for (size_t i = 0; i < supportPts.size(); ++i)
        {
            float dx = static_cast<float>(x - supportPts[i].x);
            float dy = static_cast<float>(y - supportPts[i].y);
            float d2 = dx * dx + dy * dy;
            if (d2 < bestD2)
            {
                bestD2 = d2;
                nearestIdx = static_cast<int>(i);
            }
        }
        return bestD2;
    };

    auto collectGTSupportNearVDSupport = [&](const std::vector<cv::Point>& vdSupportPts) -> GTSupportResult
    {
        GTSupportResult res;

        if (vdSupportPts.empty() || gtPoints.empty())
            return res;

        int minX = std::numeric_limits<int>::max();
        int minY = std::numeric_limits<int>::max();
        int maxX = 0;
        int maxY = 0;
        float cx = 0.0f, cy = 0.0f;

        for (size_t i = 0; i < vdSupportPts.size(); ++i)
        {
            minX = std::min(minX, vdSupportPts[i].x);
            minY = std::min(minY, vdSupportPts[i].y);
            maxX = std::max(maxX, vdSupportPts[i].x);
            maxY = std::max(maxY, vdSupportPts[i].y);
            cx += static_cast<float>(vdSupportPts[i].x);
            cy += static_cast<float>(vdSupportPts[i].y);
        }
        cx /= static_cast<float>(vdSupportPts.size());
        cy /= static_cast<float>(vdSupportPts.size());

        struct GTCandidate
        {
            int x;
            int y;
            float value;
            float nearD2;
            float centroidD2;
            float weight;
        };

        for (size_t rid = 0; rid < gtSearchRadii.size(); ++rid)
        {
            int R = gtSearchRadii[rid];

            int bx0 = std::max(0, minX - R);
            int by0 = std::max(0, minY - R);
            int bx1 = std::min(m_refDepthImage.cols - 1, maxX + R);
            int by1 = std::min(m_refDepthImage.rows - 1, maxY + R);

            std::vector<GTCandidate> candidates;
            candidates.reserve(128);

            for (size_t gi = 0; gi < gtPoints.size(); ++gi)
            {
                const GTPoint& gp = gtPoints[gi];
                if (gp.x < bx0 || gp.x > bx1 || gp.y < by0 || gp.y > by1)
                    continue;

                int nearestIdx = -1;
                float nearD2 = distancePointToSupport(gp.x, gp.y, vdSupportPts, nearestIdx);
                if (nearD2 > static_cast<float>(R * R))
                    continue;
                if (nearestIdx < 0)
                    continue;

                const cv::Point& anchor = vdSupportPts[nearestIdx];
                if (crossesStrongEdge(anchor.x, anchor.y, gp.x, gp.y))
                    continue;

                float dcx = static_cast<float>(gp.x) - cx;
                float dcy = static_cast<float>(gp.y) - cy;
                float centroidD2 = dcx * dcx + dcy * dcy;

                float w = 1.0f / (1.0f + 0.25f * nearD2 + 0.05f * centroidD2);

                GTCandidate c;
                c.x = gp.x;
                c.y = gp.y;
                c.value = gp.value;
                c.nearD2 = nearD2;
                c.centroidD2 = centroidD2;
                c.weight = w;
                candidates.push_back(c);
            }

            if (static_cast<int>(candidates.size()) < minGtCount)
                continue;

            float minGT = std::numeric_limits<float>::infinity();
            float maxGT = 0.0f;
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                minGT = std::min(minGT, candidates[i].value);
                maxGT = std::max(maxGT, candidates[i].value);
            }

            if (!(std::isfinite(minGT) && std::isfinite(maxGT) && maxGT >= minGT))
                continue;

            float histStart = std::floor(minGT / gtHistBinWidth) * gtHistBinWidth;
            float histEnd   = std::ceil((maxGT + gtHistBinWidth) / gtHistBinWidth) * gtHistBinWidth;
            int histBins = std::max(1, static_cast<int>(std::ceil((histEnd - histStart) / gtHistBinWidth)));

            std::vector<double> hist(histBins, 0.0);
            auto toBin = [&](float v) -> int
            {
                int idx = static_cast<int>(std::floor((v - histStart) / gtHistBinWidth));
                idx = std::max(0, std::min(histBins - 1, idx));
                return idx;
            };

            double totalScore = 0.0;
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                hist[toBin(candidates[i].value)] += candidates[i].weight;
                totalScore += candidates[i].weight;
            }

            int bestBin = 0;
            double bestScore = hist[0];
            double secondScore = -1.0;
            for (int b = 1; b < histBins; ++b)
            {
                if (hist[b] > bestScore)
                {
                    secondScore = bestScore;
                    bestScore = hist[b];
                    bestBin = b;
                }
                else if (hist[b] > secondScore)
                {
                    secondScore = hist[b];
                }
            }

            float purity = (totalScore > 1e-12) ? static_cast<float>(bestScore / totalScore) : 0.0f;
            float gap = (bestScore > 1e-12)
                        ? static_cast<float>((bestScore - std::max(0.0, secondScore)) / bestScore)
                        : 0.0f;

            float binLow = histStart + bestBin * gtHistBinWidth;
            float binHigh = binLow + gtHistBinWidth;

            std::vector<float> peakVals;
            std::vector<cv::Point> peakPts;
            peakVals.reserve(candidates.size());
            peakPts.reserve(candidates.size());

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (candidates[i].value >= binLow && candidates[i].value < binHigh)
                {
                    peakVals.push_back(candidates[i].value);
                    peakPts.push_back(cv::Point(candidates[i].x, candidates[i].y));
                }
            }

            if (static_cast<int>(peakVals.size()) < minGtCount)
                continue;

            std::vector<float> sortedPeak = peakVals;
            std::sort(sortedPeak.begin(), sortedPeak.end());
            float med = quantileFromSorted(sortedPeak, 0.5f);

            std::vector<float> absDev;
            absDev.reserve(sortedPeak.size());
            for (size_t i = 0; i < sortedPeak.size(); ++i)
                absDev.push_back(std::fabs(sortedPeak[i] - med));
            std::sort(absDev.begin(), absDev.end());
            float mad = quantileFromSorted(absDev, 0.5f);
            float tol = std::max(gtMadAbsTol, 3.0f * std::max(mad, 1.0f));

            std::vector<float> finalVals;
            std::vector<cv::Point> finalPts;
            finalVals.reserve(peakVals.size());
            finalPts.reserve(peakPts.size());

            for (size_t i = 0; i < peakVals.size(); ++i)
            {
                if (std::fabs(peakVals[i] - med) <= tol)
                {
                    finalVals.push_back(peakVals[i]);
                    finalPts.push_back(peakPts[i]);
                }
            }

            if (static_cast<int>(finalVals.size()) >= minGtCount)
            {
                res.ok = true;
                res.vals.swap(finalVals);
                res.pts.swap(finalPts);
                res.purity = purity;
                res.gap = gap;
                res.usedRadius = R;
                return res;
            }
        }

        return res;
    };

    cv::Mat debugVis = focusImage.clone();

    int rejectSupportTooSmall = 0;
    int rejectNoGTSupport = 0;
    int rejectEdgeCross = 0;
    int rejectWeakTexture = 0;
    int rejectSupportEdge = 0;
    int rejectAmbiguousGT = 0;
    int rejectSupportLineLike = 0;
    int rejectGTCentroidOffset = 0;
    int rejectSupportCentroidDrift = 0;
    int rejectSupportLowFill = 0;

    // ------------------------------------------------------------
    // Step 3: seed -> support -> GT support -> representative pair
    // ------------------------------------------------------------
    for (size_t i = 0; i < selectedSeeds.size(); ++i)
    {
        const VDCandidate& seed = selectedSeeds[i];

        float band = std::max(supportBandAbsMin,
                              supportBandMadScale * std::max(seed.vdMad, supportMinMadFloor));

        struct Neighbor
        {
            int x;
            int y;
            float vd;
            float dist2;
        };

        std::vector<Neighbor> neighbors;
        neighbors.reserve((2 * supportSearchRadius + 1) * (2 * supportSearchRadius + 1));

        int x0 = std::max(0, seed.x - supportSearchRadius);
        int x1 = std::min(m_virtualDepthImage.cols - 1, seed.x + supportSearchRadius);
        int y0 = std::max(0, seed.y - supportSearchRadius);
        int y1 = std::min(m_virtualDepthImage.rows - 1, seed.y + supportSearchRadius);

        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                float v = m_virtualDepthImage.at<float>(y, x);
                if (!isValidValue(v))
                    continue;
                if (std::fabs(v - seed.vdMedian) > band)
                    continue;

                if (crossesStrongEdge(seed.x, seed.y, x, y))
                {
                    ++rejectEdgeCross;
                    continue;
                }

                float d2 = static_cast<float>((x - seed.x) * (x - seed.x) + (y - seed.y) * (y - seed.y));
                Neighbor n;
                n.x = x;
                n.y = y;
                n.vd = v;
                n.dist2 = d2;
                neighbors.push_back(n);
            }
        }

        if (neighbors.empty())
        {
            ++rejectSupportTooSmall;
            continue;
        }

        std::sort(neighbors.begin(), neighbors.end(),
                  [](const Neighbor& a, const Neighbor& b)
                  {
                      return a.dist2 < b.dist2;
                  });

        std::vector<cv::Point> supportPts;
        std::vector<float> supportVDVals;
        std::vector<float> supportGrayVals;
        float sumCorner = 0.0f;
        int edgeCount = 0;
        int nearEdgeCount = 0;

        supportPts.reserve(std::min(static_cast<int>(neighbors.size()), supportMaxPoints));
        supportVDVals.reserve(std::min(static_cast<int>(neighbors.size()), supportMaxPoints));
        supportGrayVals.reserve(std::min(static_cast<int>(neighbors.size()), supportMaxPoints));

        for (size_t k = 0; k < neighbors.size() && static_cast<int>(supportPts.size()) < supportMaxPoints; ++k)
        {
            supportPts.push_back(cv::Point(neighbors[k].x, neighbors[k].y));
            supportVDVals.push_back(neighbors[k].vd);

            float g = static_cast<float>(guideGray.at<uchar>(neighbors[k].y, neighbors[k].x));
            supportGrayVals.push_back(g);

            sumCorner += cornerResp.at<float>(neighbors[k].y, neighbors[k].x);

            if (guideGrad.at<float>(neighbors[k].y, neighbors[k].x) > strongEdgeThresh ||
                vdEdgeResp.at<float>(neighbors[k].y, neighbors[k].x) > 24.0f)
            {
                edgeCount++;
            }

            if (edgeDistMap.at<float>(neighbors[k].y, neighbors[k].x) < supportNearEdgeDist)
                nearEdgeCount++;
        }

        if (static_cast<int>(supportPts.size()) < supportMinPoints)
        {
            ++rejectSupportTooSmall;
            continue;
        }

        std::sort(supportVDVals.begin(), supportVDVals.end());
        float vdRep = quantileFromSorted(supportVDVals, 0.5f);

        float supportGrayMean = 0.0f;
        for (size_t k = 0; k < supportGrayVals.size(); ++k)
            supportGrayMean += supportGrayVals[k];
        supportGrayMean /= static_cast<float>(supportGrayVals.size());

        float supportGrayStd = 0.0f;
        for (size_t k = 0; k < supportGrayVals.size(); ++k)
        {
            float d = supportGrayVals[k] - supportGrayMean;
            supportGrayStd += d * d;
        }
        supportGrayStd = std::sqrt(supportGrayStd / std::max<size_t>(1, supportGrayVals.size()));

        float meanCorner = sumCorner / static_cast<float>(supportPts.size());
        float edgeRatio = static_cast<float>(edgeCount) / static_cast<float>(supportPts.size());
        float nearEdgeRatio = static_cast<float>(nearEdgeCount) / static_cast<float>(supportPts.size());

        if ((supportGrayMean < 18.0f && supportGrayStd < 4.0f) ||
            (supportGrayStd < supportMinGrayStd && meanCorner < supportMinCornerResp))
        {
            ++rejectWeakTexture;
            continue;
        }

        if (edgeRatio > supportMaxEdgeRatio || nearEdgeRatio > supportMaxNearEdgeRatio)
        {
            ++rejectSupportEdge;
            continue;
        }

        float meanX = 0.0f, meanY = 0.0f;
        int minSX = std::numeric_limits<int>::max();
        int minSY = std::numeric_limits<int>::max();
        int maxSX = 0;
        int maxSY = 0;
        for (size_t k = 0; k < supportPts.size(); ++k)
        {
            meanX += static_cast<float>(supportPts[k].x);
            meanY += static_cast<float>(supportPts[k].y);
            minSX = std::min(minSX, supportPts[k].x);
            minSY = std::min(minSY, supportPts[k].y);
            maxSX = std::max(maxSX, supportPts[k].x);
            maxSY = std::max(maxSY, supportPts[k].y);
        }
        meanX /= static_cast<float>(supportPts.size());
        meanY /= static_cast<float>(supportPts.size());

        float supportCentroidDrift = std::sqrt((meanX - seed.x) * (meanX - seed.x) +
                                               (meanY - seed.y) * (meanY - seed.y));
        if (supportCentroidDrift > maxSupportCentroidDrift)
        {
            ++rejectSupportCentroidDrift;
            continue;
        }

        // bbox 填充率：沿边一串的点常常 bbox 很大，但点数很少
        float bboxArea = static_cast<float>((maxSX - minSX + 1) * (maxSY - minSY + 1));
        float fillRatio = bboxArea > 0.0f ? static_cast<float>(supportPts.size()) / bboxArea : 1.0f;
        if (fillRatio < minSupportFillRatio && bboxArea>=25.0f)
        {
            ++rejectSupportLowFill;
        }

        // 协方差形状约束：线状 support 要剔除
        if (supportPts.size() >= 4)
        {
            double varXX = 0.0;
            double varYY = 0.0;
            double varXY = 0.0;

            for (size_t k = 0; k < supportPts.size(); ++k)
            {
                double dx = static_cast<double>(supportPts[k].x) - meanX;
                double dy = static_cast<double>(supportPts[k].y) - meanY;
                varXX += dx * dx;
                varYY += dy * dy;
                varXY += dx * dy;
            }

            varXX /= static_cast<double>(supportPts.size());
            varYY /= static_cast<double>(supportPts.size());
            varXY /= static_cast<double>(supportPts.size());

            double trace = varXX + varYY;
            double det = varXX * varYY - varXY * varXY;
            double tmp = trace * trace * 0.25 - det;
            if (tmp < 0.0) tmp = 0.0;
            tmp = std::sqrt(tmp);

            double lambda1 = trace * 0.5 + tmp;
            double lambda2 = trace * 0.5 - tmp;
            double shapeRatio = (lambda1 > 1e-9) ? (lambda2 / lambda1) : 1.0;

            if (shapeRatio < minSupportShapeRatio)
            {
                ++rejectSupportLineLike;
                continue;
            }
        }

        GTSupportResult gtRes = collectGTSupportNearVDSupport(supportPts);
        if (!gtRes.ok)
        {
            ++rejectNoGTSupport;
            continue;
        }

        float gtMeanX = 0.0f, gtMeanY = 0.0f;
        for (size_t k = 0; k < gtRes.pts.size(); ++k)
        {
            gtMeanX += static_cast<float>(gtRes.pts[k].x);
            gtMeanY += static_cast<float>(gtRes.pts[k].y);
        }
        gtMeanX /= static_cast<float>(gtRes.pts.size());
        gtMeanY /= static_cast<float>(gtRes.pts.size());

        float gtCentroidOffset = std::sqrt((gtMeanX - meanX) * (gtMeanX - meanX) +
                                           (gtMeanY - meanY) * (gtMeanY - meanY));
        if (gtCentroidOffset > maxGTCentroidOffset)
        {
            ++rejectGTCentroidOffset;
            continue;
        }

        if (gtRes.purity < minGtPeakPurity ||
            (gtRes.gap < minGtPeakGap && gtRes.purity < 0.65f) ||
            (gtRes.usedRadius > softMaxGtSearchRadius && gtRes.purity < 0.70f))
        {
            ++rejectAmbiguousGT;
            continue;
        }

        std::sort(gtRes.vals.begin(), gtRes.vals.end());
        float gtRep = quantileFromSorted(gtRes.vals, 0.5f);

        if (!isValidValue(vdRep) || !isValidValue(gtRep))
            continue;

        SamplePoint sp;
        sp.colIndex = seed.x;
        sp.rowIndex = seed.y;
        sp.vDepth = vdRep;
        sp.gtDepth = gtRep;
        samplePoints.push_back(sp);

        cv::circle(debugVis, cv::Point(seed.x, seed.y), 4, cv::Scalar(0, 0, 255), 2);
        for (size_t k = 0; k < supportPts.size(); ++k)
            cv::circle(debugVis, supportPts[k], 1, cv::Scalar(0, 255, 255), -1);
        for (size_t k = 0; k < gtRes.pts.size(); ++k)
            cv::circle(debugVis, gtRes.pts[k], 2, cv::Scalar(0, 255, 0), -1);
    }

    std::cout << "[VirtualToRealDepthBySegBM_new] representative pair count: "
              << samplePoints.size() << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectSupportTooSmall: "
              << rejectSupportTooSmall << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectNoGTSupport: "
              << rejectNoGTSupport << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectEdgeCross: "
              << rejectEdgeCross << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectWeakTexture: "
              << rejectWeakTexture << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectSupportEdge: "
              << rejectSupportEdge << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectSupportLineLike: "
              << rejectSupportLineLike << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectSupportCentroidDrift: "
              << rejectSupportCentroidDrift << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectSupportLowFill: "
              << rejectSupportLowFill << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectGTCentroidOffset: "
              << rejectGTCentroidOffset << std::endl;
    std::cout << "[VirtualToRealDepthBySegBM_new] rejectAmbiguousGT: "
              << rejectAmbiguousGT << std::endl;

    if (!debugVis.empty())
        cv::imwrite(debugSeedPath, debugVis);

    if (samplePoints.size() < static_cast<size_t>(minRepresentativePairsForFit))
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] representative pairs are too few for reliable fitting: "
                  << samplePoints.size() << " < " << minRepresentativePairsForFit << std::endl;
        return;
    }

    fitSegmentsParams_new(xmlPath);

    cv::Mat realDepthImage = ConvertVdImageToRd(m_virtualDepthImage);
    if (realDepthImage.empty())
    {
        std::cout << "[VirtualToRealDepthBySegBM_new] ConvertVdImageToRd failed." << std::endl;
        return;
    }

    SamplePointSelect();
    errorStatisticsImageSeg(realDepthImage,
                            m_refDepthImage,
                            m_virtualDepthImage,
                            focusImage,
                            distanceImagePath);

    std::cout << "[VirtualToRealDepthBySegBM_new] done." << std::endl;
}

void VirtualToRealDepthFunc::fitSegmentsParams_new(std::string xml_path)
{
    if (samplePoints.empty())
    {
        std::cout << "samplePoints is empty!" << std::endl;
        return;
    }

    // ============================================================
    // 新思路：
    // 1) 输入 samplePoints 中的 (gtDepth, vDepth) 已经是局部支持域上的代表值
    // 2) 按 VD(vDepth) 分段，而不是按 GT 分段
    // 3) 每段边界直接由该段有效样本的 vd min/max 决定
    // 4) 拟合前后都做一次鲁棒清理
    // 5) 对每个段增加 GT 单峰性检查，避免“同一 vd 段内 GT 多模态”
    // ============================================================

    // ---------------- 参数：可根据你的数据继续调 ----------------
    const int   minTotalValidPairs   = 12;     // 全局最少有效样本对
    const int   minSamplesPerSegment = 8;      // 每段最少样本数
    const int   targetSamplesPerSeg  = 16;     // 期望每段样本数
    const float maxVdSpanPerSegment  = 0.18f;  // 每段最大 vd 跨度
    const float minVdSpanPerSegment  = 0.04f;  // 段太窄则不稳定
    const float residualMadScale     = 3.0f;   // 拟合残差鲁棒阈值倍数
    const float minResidualAbsTol    = 800.0f; // GT 残差绝对下限（单位按你的 GT，一般 mm）
    const float segmentBoundaryEps   = 1e-4f;  // 防止 min==max

    // GT 单峰性检查
    const float gtHistBinWidth       = 2500.0f;
    const float minGtPeakPurity      = 0.58f;
    const float minGtPeakGap         = 0.10f;

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

    auto evalBehaviorModel = [](const std::array<double, 3>& p, float vd) -> float
    {
        // gt = (vd * c1 + c2) / (1 - vd * c0)
        double denom = 1.0 - static_cast<double>(vd) * p[0];
        if (std::fabs(denom) < 1e-12)
            return 0.0f;

        double pred = (static_cast<double>(vd) * p[1] + p[2]) / denom;
        return static_cast<float>(pred);
    };

    auto isSegmentGTUnimodal = [&](const std::vector<SamplePoint>& pts,
                                   float& purity,
                                   float& gap) -> bool
    {
        purity = 1.0f;
        gap = 1.0f;

        if (pts.size() < 3)
            return false;

        float minGT = std::numeric_limits<float>::infinity();
        float maxGT = 0.0f;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            minGT = std::min(minGT, pts[i].gtDepth);
            maxGT = std::max(maxGT, pts[i].gtDepth);
        }

        if (!(std::isfinite(minGT) && std::isfinite(maxGT) && maxGT >= minGT))
            return false;

        float histStart = std::floor(minGT / gtHistBinWidth) * gtHistBinWidth;
        float histEnd   = std::ceil((maxGT + gtHistBinWidth) / gtHistBinWidth) * gtHistBinWidth;
        int histBins = std::max(1, static_cast<int>(std::ceil((histEnd - histStart) / gtHistBinWidth)));

        std::vector<int> hist(histBins, 0);

        auto toBin = [&](float v) -> int
        {
            int idx = static_cast<int>(std::floor((v - histStart) / gtHistBinWidth));
            idx = std::max(0, std::min(histBins - 1, idx));
            return idx;
        };

        for (size_t i = 0; i < pts.size(); ++i)
            hist[toBin(pts[i].gtDepth)]++;

        int bestBin = 0;
        int secondBin = -1;
        int bestCount = hist[0];
        int secondCount = -1;
        int totalCount = 0;

        for (int i = 0; i < histBins; ++i)
            totalCount += hist[i];

        for (int i = 1; i < histBins; ++i)
        {
            if (hist[i] > bestCount)
            {
                secondCount = bestCount;
                secondBin = bestBin;
                bestCount = hist[i];
                bestBin = i;
            }
            else if (hist[i] > secondCount)
            {
                secondCount = hist[i];
                secondBin = i;
            }
        }

        purity = totalCount > 0 ? static_cast<float>(bestCount) / totalCount : 0.0f;
        gap = bestCount > 0 ? static_cast<float>(bestCount - std::max(0, secondCount)) / bestCount : 0.0f;

        return (purity >= minGtPeakPurity) && (gap >= minGtPeakGap);
    };

    // ------------------------------------------------------------
    // 1) 收集有效样本
    // ------------------------------------------------------------
    std::vector<SamplePoint> validSamples;
    validSamples.reserve(samplePoints.size());

    for (size_t i = 0; i < samplePoints.size(); ++i)
    {
        if (isValidPair(samplePoints[i]))
            validSamples.push_back(samplePoints[i]);
    }

    if (validSamples.size() < static_cast<size_t>(minTotalValidPairs))
    {
        std::cout << "Not enough valid sample pairs for fitSegmentsParams(): "
                  << validSamples.size() << " < " << minTotalValidPairs << std::endl;
        return;
    }

    // 按 vd 升序排列
    std::sort(validSamples.begin(), validSamples.end(),
              [](const SamplePoint& a, const SamplePoint& b)
              {
                  return a.vDepth < b.vDepth;
              });

    // ------------------------------------------------------------
    // 2) 在 vd 域做贪心分段
    // ------------------------------------------------------------
    std::vector<std::vector<SamplePoint> > rawSegments;
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
        float newSpan  = nextVD - curMinVD;

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

    // 再做一次“太碎小段并入前段”
    {
        std::vector<std::vector<SamplePoint> > merged;
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

    // ------------------------------------------------------------
    // 3) 对每段做：
    //    GT 单峰性检查 -> 初拟合 -> 残差清理 -> 再拟合
    // ------------------------------------------------------------
    std::vector<BehaviorSegmentResult> segment_results;
    segment_results.clear();

    for (size_t s = 0; s < rawSegments.size(); ++s)
    {
        std::vector<SamplePoint>& segPts = rawSegments[s];
        if (segPts.size() < 3)
            continue;

        float gtPurity = 1.0f;
        float gtGap = 1.0f;
        if (!isSegmentGTUnimodal(segPts, gtPurity, gtGap))
        {
            std::cout << "[fitSegmentsParams] skip ambiguous segment " << s
                      << "  sampleCount=" << segPts.size()
                      << "  gtPurity=" << gtPurity
                      << "  gtGap=" << gtGap << std::endl;
            continue;
        }

        // ---- 3.1 第一次拟合 ----
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

        // ---- 3.2 按残差做一次鲁棒清理 ----
        std::vector<float> residuals;
        residuals.reserve(segPts.size());

        for (size_t i = 0; i < segPts.size(); ++i)
        {
            float pred = evalBehaviorModel(params, segPts[i].vDepth);
            float res  = std::fabs(pred - segPts[i].gtDepth);
            residuals.push_back(res);
        }

        float medRes = robustMedian(residuals);

        std::vector<float> absDev;
        absDev.reserve(residuals.size());
        for (size_t i = 0; i < residuals.size(); ++i)
            absDev.push_back(std::fabs(residuals[i] - medRes));

        float madRes = robustMedian(absDev);
        float resTol = std::max(minResidualAbsTol,
                                residualMadScale * std::max(madRes, 1.0f));

        std::vector<SamplePoint> inliers;
        inliers.reserve(segPts.size());

        for (size_t i = 0; i < segPts.size(); ++i)
        {
            if (residuals[i] <= resTol)
                inliers.push_back(segPts[i]);
        }

        // 清理后太少，则放弃该段
        if (inliers.size() < static_cast<size_t>(minSamplesPerSegment))
        {
            std::cout << "[fitSegmentsParams] skip weak segment " << s
                      << " after residual filtering, inliers=" << inliers.size() << std::endl;
            continue;
        }

        // 再做一次 GT 单峰性检查，防止 residual 清理后段内形态变化
        gtPurity = 1.0f;
        gtGap = 1.0f;
        if (!isSegmentGTUnimodal(inliers, gtPurity, gtGap))
        {
            std::cout << "[fitSegmentsParams] skip ambiguous segment " << s
                      << " after residual filtering"
                      << "  inliers=" << inliers.size()
                      << "  gtPurity=" << gtPurity
                      << "  gtGap=" << gtGap << std::endl;
            continue;
        }

        // ---- 3.3 再拟合一次 ----
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

        // ---- 3.4 真实段边界：直接来自本段 vd min/max ----
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

        std::cout << "[fitSegmentsParams] keep segment " << s
                  << "  vd=[" << vdMin << ", " << vdMax << "]"
                  << "  sampleCount=" << inliers.size()
                  << "  gtPurity=" << gtPurity
                  << "  gtGap=" << gtGap
                  << "  params=(" << params[0] << ", "
                                  << params[1] << ", "
                                  << params[2] << ")"
                  << std::endl;
    }

    if (segment_results.empty())
    {
        std::cout << "No valid fitted segments." << std::endl;
        return;
    }

    // ------------------------------------------------------------
    // 4) 段边界排序 + 中点拼接，避免缝隙与重叠
    // ------------------------------------------------------------
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

    // ------------------------------------------------------------
    // 5) 写 XML
    // ------------------------------------------------------------
    std::ofstream xml(xml_path.c_str());
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
//        errorStatisticsImage(m_realDepthImage, m_refDepthImage, focusImage, distanceImage_path);
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
            //   float valPred = getValidRefValue(m_realDepthImage, x, y, SESOptions.virtualSearchRadius);
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

                     float valPred = m_realDepthImage.at<float>(y, x);
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
            //   float valPred = getValidRefValue(m_realDepthImage, x, y, SESOptions.virtualSearchRadius);
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

                     float valPred = m_realDepthImage.at<float>(y, x);
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
        //     误差按 |GT(=m_realDepthImage) - REF|
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

            // 1) 取 m_realDepthImage 的值(若中心点无效，则逐圈扩展找到 targetNeighborCount_vd 个有效点求均值)
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


    ManualUIState& GetManualUIState()
    {
        static ManualUIState state;
        return state;
    }

    cv::Mat ToSingleFloat(const cv::Mat& src)
    {
        cv::Mat out;
        if (src.empty())
            return out;

        if (src.channels() > 1)
            cv::extractChannel(src, out, 0);
        else
            out = src.clone();

        if (out.type() != CV_32FC1)
            out.convertTo(out, CV_32F);

        return out;
    }

    float ReadDepthValue(const cv::Mat& img, int x, int y)
    {
        if (img.empty())
            return 0.0f;
        if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
            return 0.0f;

        float v = img.at<float>(y, x);
        if (!std::isfinite(v) || v <= 0.0f)
            return 0.0f;
        return v;
    }

    std::vector<cv::Point> CollectCirclePoints(const cv::Point& center, int radius, const cv::Size& size)
    {
        std::vector<cv::Point> pts;
        if (size.width <= 0 || size.height <= 0)
            return pts;

        int xMin = std::max(0, center.x - radius);
        int xMax = std::min(size.width - 1, center.x + radius);
        int yMin = std::max(0, center.y - radius);
        int yMax = std::min(size.height - 1, center.y + radius);

        pts.reserve((2 * radius + 1) * (2 * radius + 1));

        for (int y = yMin; y <= yMax; ++y)
        {
            for (int x = xMin; x <= xMax; ++x)
            {
                int dx = x - center.x;
                int dy = y - center.y;
                if (dx * dx + dy * dy <= radius * radius)
                    pts.emplace_back(x, y);
            }
        }
        return pts;
    }

    double ComputeLocalRMSE(const cv::Mat& depthImage,
                            const std::vector<cv::Point>& circlePoints,
                            const cv::Point& center,
                            std::vector<float>& validValues)
    {
        validValues.clear();

        for (size_t i = 0; i < circlePoints.size(); ++i)
        {
            float v = ReadDepthValue(depthImage, circlePoints[i].x, circlePoints[i].y);
            if (v > 0.0f)
                validValues.push_back(v);
        }

        if (validValues.empty())
            return 0.0;

        float refValue = ReadDepthValue(depthImage, center.x, center.y);
        if (refValue <= 0.0f)
        {
            double mean = 0.0;
            for (size_t i = 0; i < validValues.size(); ++i)
                mean += validValues[i];
            refValue = static_cast<float>(mean / std::max<size_t>(1, validValues.size()));
        }

        double sumSq = 0.0;
        for (size_t i = 0; i < validValues.size(); ++i)
        {
            double diff = static_cast<double>(validValues[i]) - static_cast<double>(refValue);
            sumSq += diff * diff;
        }

        return std::sqrt(sumSq / std::max<size_t>(1, validValues.size()));
    }

    double ComputeMeanFromValidValues(const std::vector<float>& values)
    {
        if (values.empty())
            return 0.0;

        double sum = 0.0;
        for (size_t i = 0; i < values.size(); ++i)
            sum += values[i];

        return sum / static_cast<double>(values.size());
    }

    double ComputeHoverRefMean(const cv::Mat& refDepth32F,
                               const std::vector<cv::Point>& circlePoints)
    {
        double sum = 0.0;
        int count = 0;

        for (size_t i = 0; i < circlePoints.size(); ++i)
        {
            float ref = ReadDepthValue(refDepth32F, circlePoints[i].x, circlePoints[i].y);
            if (ref > 0.0f)
            {
                sum += ref;
                count++;
            }
        }

        if (count == 0)
            return 0.0;

        return sum / static_cast<double>(count);
    }

    cv::Mat BuildRefFocusImage(const cv::Mat& focusImage,
                               const cv::Mat& refDepth32F,
                               float specialDepthMin,
                               float specialDepthMax)
    {
        cv::Mat out;
        if (focusImage.channels() == 1)
            cv::cvtColor(focusImage, out, cv::COLOR_GRAY2BGR);
        else
            out = focusImage.clone();

        for (int y = 0; y < refDepth32F.rows; ++y)
        {
            for (int x = 0; x < refDepth32F.cols; ++x)
            {
                float d = refDepth32F.at<float>(y, x);
                if (d > 0.0f && std::isfinite(d))
                {
                    // 默认颜色：红色
                    cv::Scalar color(0, 0, 255);

                    // 指定深度段：特殊颜色（例如绿色）
                    if (d >= specialDepthMin && d <= specialDepthMax)
                    {
                        color = cv::Scalar(0, 255, 0);
                    }

                    cv::circle(out, cv::Point(x, y), 5, color, -1);
                }
            }
        }
        return out;
    }

void RefreshInfoPanel()
    {
        ManualUIState& state = GetManualUIState();

        const int panelWidth = 1400;
        const int panelHeight = 1100;
        const int scrollBarWidth = 18;
        const int scrollBarMargin = 8;
        const int trackTop = 165;
        const int trackBottomMargin = 20;
        const int contentRightSafe = panelWidth - scrollBarWidth - scrollBarMargin * 3;

        int maxPointLinesToShow = 10;

        int estimatedContentHeight = 190 +
                                     static_cast<int>(state.selections.size()) * ((maxPointLinesToShow + 4) * 22 + 30);
        estimatedContentHeight = std::max(panelHeight, estimatedContentHeight + 40);

        cv::Mat content(estimatedContentHeight, panelWidth, CV_8UC3, cv::Scalar(255, 255, 255));

        cv::putText(content, "Manual Pair Selection", cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);

        cv::putText(content,
                    "Controls: Left click add circle | Right click or Z/U undo | Mouse wheel on image = zoom | Middle drag = pan | InfoPanel wheel / drag scrollbar = scroll",
                    cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(20, 20, 20), 1);

        cv::putText(content, "Left column: VD values / VD RMSE", cv::Point(20, 110),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(160, 0, 0), 2);
        cv::putText(content, "Right column: REF values / REF RMSE", cv::Point(720, 110),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 100, 0), 2);

        std::string summary =
                "Selections: " + std::to_string(state.selections.size()) +
                "   Kept pairs(mean vd, mean ref): " + std::to_string(state.pairRecords.size()) +
                "   Circle radius: " + std::to_string(state.circleRadius) +
                "   ScrollPx: " + std::to_string(state.infoScrollPx);
        cv::putText(content, summary, cv::Point(20, 145),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(40, 40, 40), 1);

        int y = 185;

        for (int i = 0; i < static_cast<int>(state.selections.size()); ++i)
        {
            const ManualSelectionRecord& rec = state.selections[i];
            int validPairCount = static_cast<int>(rec.pairEnd - rec.pairBegin);

            std::string title =
                    "#" + std::to_string(i + 1) +
                    " p=(" + std::to_string(rec.center.x) + "," + std::to_string(rec.center.y) + ")" +
                    " src=" + rec.sourceWindow +
                    " n=" + std::to_string(rec.circlePoints.size()) +
                    " kept=" + std::to_string(validPairCount);

            cv::putText(content, title, cv::Point(20, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 0, 0), 1);
            y += 26;

            int detailStartY = y;

            for (int k = 0; k < static_cast<int>(rec.vdPointLines.size()) && k < maxPointLinesToShow; ++k)
            {
                cv::putText(content, rec.vdPointLines[k], cv::Point(40, detailStartY + k * 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 80), 1);
            }

            for (int k = 0; k < static_cast<int>(rec.refPointLines.size()) && k < maxPointLinesToShow; ++k)
            {
                cv::putText(content, rec.refPointLines[k], cv::Point(720, detailStartY + k * 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 80), 1);
            }

            if (static_cast<int>(rec.vdPointLines.size()) > maxPointLinesToShow)
            {
                std::string moreLeft =
                        "... " + std::to_string(rec.vdPointLines.size() - maxPointLinesToShow) + " more vd points";
                cv::putText(content, moreLeft, cv::Point(40, detailStartY + maxPointLinesToShow * 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(120, 120, 120), 1);
            }

            if (static_cast<int>(rec.refPointLines.size()) > maxPointLinesToShow)
            {
                std::string moreRight =
                        "... " + std::to_string(rec.refPointLines.size() - maxPointLinesToShow) + " more ref points";
                cv::putText(content, moreRight, cv::Point(720, detailStartY + maxPointLinesToShow * 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(120, 120, 120), 1);
            }

            y = detailStartY + (maxPointLinesToShow + 1) * 22;

            std::string leftRmse =
                    "VD  : count=" + std::to_string(rec.vdValues.size()) +
                    "  mean=" + std::string(cv::format("%.6f", ComputeMeanFromValidValues(rec.vdValues))) +
                    "  rmse=" + std::string(cv::format("%.6f", rec.vdRmse));

            std::string rightRmse =
                    "REF : count=" + std::to_string(rec.refValues.size()) +
                    "  mean=" + std::string(cv::format("%.6f", ComputeMeanFromValidValues(rec.refValues))) +
                    "  rmse=" + std::string(cv::format("%.6f", rec.refRmse));

            cv::putText(content, leftRmse, cv::Point(40, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(160, 0, 0), 1);

            cv::putText(content, rightRmse, cv::Point(720, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(0, 120, 0), 1);

            y += 42;

            cv::line(content, cv::Point(20, y - 18), cv::Point(contentRightSafe, y - 18),
                     cv::Scalar(220, 220, 220), 1);

            y += 10;
        }

        int contentHeight = std::max(panelHeight, y + 20);
        if (content.rows != contentHeight)
            content = content.rowRange(0, contentHeight).clone();

        state.infoContentHeight = contentHeight;

        int maxScrollPx = std::max(0, contentHeight - panelHeight);
        state.infoScrollPx = std::max(0, std::min(maxScrollPx, state.infoScrollPx));

        state.infoPanel = cv::Mat(panelHeight, panelWidth, CV_8UC3, cv::Scalar(255, 255, 255));
        content(cv::Rect(0, state.infoScrollPx, panelWidth, panelHeight)).copyTo(state.infoPanel);

        state.infoScrollTrackRect = cv::Rect();
        state.infoScrollThumbRect = cv::Rect();

        if (maxScrollPx > 0)
        {
            int trackHeight = panelHeight - trackTop - trackBottomMargin;
            int trackX = panelWidth - scrollBarWidth - scrollBarMargin;
            int trackY = trackTop;

            state.infoScrollTrackRect = cv::Rect(trackX, trackY, scrollBarWidth, trackHeight);

            double visibleRatio = static_cast<double>(panelHeight) / static_cast<double>(contentHeight);
            int thumbHeight = std::max(60, static_cast<int>(std::round(trackHeight * visibleRatio)));
            thumbHeight = std::min(trackHeight, thumbHeight);

            int movableRange = std::max(1, trackHeight - thumbHeight);
            int thumbY = trackY + static_cast<int>(
                    std::round(movableRange * static_cast<double>(state.infoScrollPx) / static_cast<double>(maxScrollPx)));

            state.infoScrollThumbRect = cv::Rect(trackX, thumbY, scrollBarWidth, thumbHeight);

            cv::rectangle(state.infoPanel, state.infoScrollTrackRect, cv::Scalar(235, 235, 235), cv::FILLED);
            cv::rectangle(state.infoPanel, state.infoScrollTrackRect, cv::Scalar(180, 180, 180), 1);

            cv::rectangle(state.infoPanel, state.infoScrollThumbRect, cv::Scalar(150, 150, 150), cv::FILLED);
            cv::rectangle(state.infoPanel, state.infoScrollThumbRect, cv::Scalar(100, 100, 100), 1);
        }

        cv::imshow("InfoPanel", state.infoPanel);
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    void InitManualImageViewState(ManualImageViewState& viewState, const cv::Size& size)
    {
        viewState.scale = 1.0;
        viewState.center = cv::Point2d((size.width - 1) * 0.5, (size.height - 1) * 0.5);
        viewState.middleDragging = false;
        viewState.lastMouse = cv::Point(-1, -1);
    }

    void ClampManualImageViewState(ManualImageViewState& viewState, const cv::Size& size)
    {
        if (size.width <= 0 || size.height <= 0)
            return;

        if (viewState.center.x < 0.0 || viewState.center.y < 0.0)
        {
            viewState.center = cv::Point2d((size.width - 1) * 0.5, (size.height - 1) * 0.5);
        }

        if (viewState.scale <= 1.0)
        {
            viewState.center = cv::Point2d((size.width - 1) * 0.5, (size.height - 1) * 0.5);
            return;
        }

        int roiW = std::max(1, static_cast<int>(std::round(size.width / viewState.scale)));
        int roiH = std::max(1, static_cast<int>(std::round(size.height / viewState.scale)));

        double halfW = roiW * 0.5;
        double halfH = roiH * 0.5;

        double minCx = halfW;
        double maxCx = std::max(minCx, static_cast<double>(size.width) - halfW);
        double minCy = halfH;
        double maxCy = std::max(minCy, static_cast<double>(size.height) - halfH);

        viewState.center.x = std::max(minCx, std::min(maxCx, viewState.center.x));
        viewState.center.y = std::max(minCy, std::min(maxCy, viewState.center.y));
    }

    cv::Rect BuildManualViewRoi(const ManualImageViewState& viewState, const cv::Size& size)
    {
        int roiW = std::max(1, static_cast<int>(std::round(size.width / std::max(1.0, viewState.scale))));
        int roiH = std::max(1, static_cast<int>(std::round(size.height / std::max(1.0, viewState.scale))));

        int x = static_cast<int>(std::round(viewState.center.x - roiW * 0.5));
        int y = static_cast<int>(std::round(viewState.center.y - roiH * 0.5));

        x = std::max(0, std::min(x, std::max(0, size.width - roiW)));
        y = std::max(0, std::min(y, std::max(0, size.height - roiH)));

        return cv::Rect(x, y, roiW, roiH);
    }

    void RenderManualZoomedImage(const cv::Mat& src,
                                 const ManualImageViewState& viewState,
                                 cv::Mat& dst)
    {
        if (src.empty())
        {
            dst.release();
            return;
        }

        dst = cv::Mat(src.size(), src.type(), cv::Scalar::all(30));

        if (viewState.scale <= 1.0)
        {
            int scaledW = std::max(1, static_cast<int>(std::round(src.cols * viewState.scale)));
            int scaledH = std::max(1, static_cast<int>(std::round(src.rows * viewState.scale)));

            cv::Mat small;
            cv::resize(src, small, cv::Size(scaledW, scaledH), 0.0, 0.0, cv::INTER_AREA);

            int offX = (dst.cols - scaledW) / 2;
            int offY = (dst.rows - scaledH) / 2;

            cv::Rect pasteRoi(offX, offY, scaledW, scaledH);
            small.copyTo(dst(pasteRoi));
        }
        else
        {
            ManualImageViewState tmpState = viewState;
            ClampManualImageViewState(tmpState, src.size());
            cv::Rect roi = BuildManualViewRoi(tmpState, src.size());

            cv::Mat cropped = src(roi);
            cv::resize(cropped, dst, src.size(), 0.0, 0.0, cv::INTER_LINEAR);
        }

        std::string scaleText = "Scale: " + std::string(cv::format("%.2fx", viewState.scale));
        cv::putText(dst, scaleText, cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    bool MapDisplayPointToManualImage(const cv::Point& displayPt,
                                      const cv::Size& imageSize,
                                      const ManualImageViewState& viewState,
                                      cv::Point& imagePt)
    {
        if (imageSize.width <= 0 || imageSize.height <= 0)
            return false;

        if (displayPt.x < 0 || displayPt.y < 0 ||
            displayPt.x >= imageSize.width || displayPt.y >= imageSize.height)
            return false;

        if (viewState.scale <= 1.0)
        {
            int scaledW = std::max(1, static_cast<int>(std::round(imageSize.width * viewState.scale)));
            int scaledH = std::max(1, static_cast<int>(std::round(imageSize.height * viewState.scale)));
            int offX = (imageSize.width - scaledW) / 2;
            int offY = (imageSize.height - scaledH) / 2;

            if (displayPt.x < offX || displayPt.x >= offX + scaledW ||
                displayPt.y < offY || displayPt.y >= offY + scaledH)
                return false;

            double localX = static_cast<double>(displayPt.x - offX) / std::max(1, scaledW - 1);
            double localY = static_cast<double>(displayPt.y - offY) / std::max(1, scaledH - 1);

            int srcX = static_cast<int>(std::round(localX * (imageSize.width - 1)));
            int srcY = static_cast<int>(std::round(localY * (imageSize.height - 1)));

            srcX = std::max(0, std::min(imageSize.width - 1, srcX));
            srcY = std::max(0, std::min(imageSize.height - 1, srcY));

            imagePt = cv::Point(srcX, srcY);
            return true;
        }
        else
        {
            ManualImageViewState tmpState = viewState;
            ClampManualImageViewState(tmpState, imageSize);
            cv::Rect roi = BuildManualViewRoi(tmpState, imageSize);

            double localX = static_cast<double>(displayPt.x) / std::max(1, imageSize.width - 1);
            double localY = static_cast<double>(displayPt.y) / std::max(1, imageSize.height - 1);

            int srcX = roi.x + static_cast<int>(std::round(localX * (roi.width - 1)));
            int srcY = roi.y + static_cast<int>(std::round(localY * (roi.height - 1)));

            srcX = std::max(0, std::min(imageSize.width - 1, srcX));
            srcY = std::max(0, std::min(imageSize.height - 1, srcY));

            imagePt = cv::Point(srcX, srcY);
            return true;
        }
    }

    void ZoomManualImageView(ManualImageViewState& viewState,
                             const cv::Size& imageSize,
                             const cv::Point& displayPt,
                             int wheelDelta)
    {
        if (imageSize.width <= 0 || imageSize.height <= 0)
            return;

        cv::Point imagePt;
        bool hasAnchor = MapDisplayPointToManualImage(displayPt, imageSize, viewState, imagePt);

        double zoomFactor = (wheelDelta > 0) ? 1.20 : (1.0 / 1.20);
        double newScale = viewState.scale * zoomFactor;
        newScale = std::max(viewState.minScale, std::min(viewState.maxScale, newScale));

        if (std::fabs(newScale - viewState.scale) < 1e-12)
            return;

        viewState.scale = newScale;

        if (hasAnchor)
        {
            viewState.center = cv::Point2d(imagePt.x, imagePt.y);
        }

        ClampManualImageViewState(viewState, imageSize);
    }

    void PanManualImageView(ManualImageViewState& viewState,
                            const cv::Size& imageSize,
                            const cv::Point& currentMouse)
    {
        if (viewState.scale <= 1.0)
            return;

        if (viewState.lastMouse.x < 0 || viewState.lastMouse.y < 0)
        {
            viewState.lastMouse = currentMouse;
            return;
        }

        ManualImageViewState tmpState = viewState;
        ClampManualImageViewState(tmpState, imageSize);
        cv::Rect roi = BuildManualViewRoi(tmpState, imageSize);

        double dxWin = static_cast<double>(currentMouse.x - viewState.lastMouse.x);
        double dyWin = static_cast<double>(currentMouse.y - viewState.lastMouse.y);

        double dxImg = dxWin * roi.width / std::max(1, imageSize.width);
        double dyImg = dyWin * roi.height / std::max(1, imageSize.height);

        viewState.center.x -= dxImg;
        viewState.center.y -= dyImg;
        viewState.lastMouse = currentMouse;

        ClampManualImageViewState(viewState, imageSize);
    }

    void BuildAnnotatedManualImages(cv::Mat& refAnnotated,
                                    cv::Mat& virtualAnnotated)
    {
        ManualUIState& state = GetManualUIState();

        refAnnotated = state.refFocusBase.clone();
        virtualAnnotated = state.virtualColorBase.clone();

        for (size_t i = 0; i < state.selections.size(); ++i)
        {
            const ManualSelectionRecord& rec = state.selections[i];
            cv::Scalar color = (i + 1 == state.selections.size()) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);

            cv::circle(refAnnotated, rec.center, state.circleRadius, color, 2);
            cv::circle(refAnnotated, rec.center, 3, cv::Scalar(255, 0, 0), -1);

            cv::circle(virtualAnnotated, rec.center, state.circleRadius, color, 2);
            cv::circle(virtualAnnotated, rec.center, 3, cv::Scalar(255, 0, 0), -1);

            std::string idx = std::to_string(static_cast<int>(i + 1));
            cv::putText(refAnnotated, idx, rec.center + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            cv::putText(virtualAnnotated, idx, rec.center + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (state.hoverValid)
        {
            std::string hoverText =
                    "hover ref mean = " + std::string(cv::format("%.3f", state.hoverRefMean)) +
                    "  at (" + std::to_string(state.hoverPoint.x) + "," + std::to_string(state.hoverPoint.y) + ")";
            cv::putText(refAnnotated, hoverText, cv::Point(30, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(255, 0, 255), 4);
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////
    void RefreshManualWindows()
    {
        ManualUIState& state = GetManualUIState();

        cv::Mat refAnnotated, virtualAnnotated;
        BuildAnnotatedManualImages(refAnnotated, virtualAnnotated);

        RenderManualZoomedImage(refAnnotated, state.focusViewState, state.refFocusShow);
        RenderManualZoomedImage(virtualAnnotated, state.virtualViewState, state.virtualColorShow);

        RefreshInfoPanel();

        cv::imshow("FocusImage", state.refFocusShow);
        cv::imshow("VirtualDepthColor", state.virtualColorShow);
        cv::imshow("InfoPanel", state.infoPanel);
    }

    void UndoLastManualSelection()
    {
        ManualUIState& state = GetManualUIState();
        if (state.selections.empty())
            return;

        ManualSelectionRecord rec = state.selections.back();
        state.selections.pop_back();

        if (rec.pairBegin < state.pairRecords.size())
            state.pairRecords.resize(rec.pairBegin);

        RefreshManualWindows();
    }

    void HandleManualClick(int x, int y, const std::string& sourceWindow)
    {
        ManualUIState& state = GetManualUIState();

        if (state.refDepth32F.empty() || state.virtualDepth32F.empty())
            return;
        if (x < 0 || y < 0 || x >= state.refDepth32F.cols || y >= state.refDepth32F.rows)
            return;

        ManualSelectionRecord rec;
        rec.center = cv::Point(x, y);
        rec.sourceWindow = sourceWindow;
        rec.circlePoints = CollectCirclePoints(rec.center, state.circleRadius, state.refDepth32F.size());

        rec.vdRmse = ComputeLocalRMSE(state.virtualDepth32F, rec.circlePoints, rec.center, rec.vdValues);
        rec.refRmse = ComputeLocalRMSE(state.refDepth32F, rec.circlePoints, rec.center, rec.refValues);

        rec.pairBegin = state.pairRecords.size();
        rec.vdPointLines.clear();
        rec.refPointLines.clear();

        for (size_t i = 0; i < rec.circlePoints.size(); ++i)
        {
            const cv::Point& p = rec.circlePoints[i];
            float vd = ReadDepthValue(state.virtualDepth32F, p.x, p.y);
            float ref = ReadDepthValue(state.refDepth32F, p.x, p.y);

            if (vd > 0.0f)
            {
                std::string vdLine =
                        "(" + std::to_string(p.x) + "," + std::to_string(p.y) + ")" +
                        "  vd=" + std::to_string(vd);
                rec.vdPointLines.push_back(vdLine);
            }

            if (ref > 0.0f)
            {
                std::string refLine =
                        "(" + std::to_string(p.x) + "," + std::to_string(p.y) + ")" +
                        "  ref=" + std::to_string(ref);
                rec.refPointLines.push_back(refLine);
            }
        }

        bool vdRmseOk = (rec.vdRmse > 0.0 && rec.vdRmse <= 0.06);
        bool refRmseOk = (rec.refRmse > 0.0 && rec.refRmse <= 1000.0);

        if (vdRmseOk && refRmseOk && !rec.vdValues.empty() && !rec.refValues.empty())
        {
            ManualPairRecord pairRec;
            pairRec.selectionIndex = static_cast<int>(state.selections.size());
            pairRec.pt = rec.center;
            pairRec.vdMean = static_cast<float>(ComputeMeanFromValidValues(rec.vdValues));
            pairRec.refMean = static_cast<float>(ComputeMeanFromValidValues(rec.refValues));
            pairRec.vdRmse = rec.vdRmse;
            pairRec.refRmse = rec.refRmse;
            state.pairRecords.push_back(pairRec);
        }

        rec.pairEnd = state.pairRecords.size();
        state.selections.push_back(rec);

        RefreshManualWindows();
    }

 void OnMouseFocus(int event, int x, int y, int flags, void* userdata)
    {
        (void)userdata;

        ManualUIState& state = GetManualUIState();

        if (event == cv::EVENT_MOUSEWHEEL)
        {
            int delta = cv::getMouseWheelDelta(flags);
            if (delta != 0)
            {
                ZoomManualImageView(state.focusViewState,
                                    state.refDepth32F.size(),
                                    cv::Point(x, y),
                                    delta);
                RefreshManualWindows();
            }
            return;
        }

        if (event == cv::EVENT_MBUTTONDOWN)
        {
            state.focusViewState.middleDragging = true;
            state.focusViewState.lastMouse = cv::Point(x, y);
            return;
        }

        if (event == cv::EVENT_MBUTTONUP)
        {
            state.focusViewState.middleDragging = false;
            state.focusViewState.lastMouse = cv::Point(-1, -1);
            return;
        }

        if (event == cv::EVENT_MOUSEMOVE && state.focusViewState.middleDragging)
        {
            PanManualImageView(state.focusViewState,
                               state.refDepth32F.size(),
                               cv::Point(x, y));
            RefreshManualWindows();
            return;
        }

        if (event == cv::EVENT_MOUSEMOVE)
        {
            cv::Point imgPt;
            if (!MapDisplayPointToManualImage(cv::Point(x, y),
                                              state.refDepth32F.size(),
                                              state.focusViewState,
                                              imgPt))
                return;

            auto now = std::chrono::steady_clock::now();

            if (!state.hoverTimeInitialized)
            {
                state.lastHoverUpdateTime = now;
                state.hoverTimeInitialized = true;
            }

            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - state.lastHoverUpdateTime).count();

            if (elapsedMs < 3000)
                return;

            state.lastHoverUpdateTime = now;
            state.hoverPoint = imgPt;

            std::vector<cv::Point> hoverCircle =
                    CollectCirclePoints(state.hoverPoint, state.circleRadius, state.refDepth32F.size());

            state.hoverRefMean = ComputeHoverRefMean(state.refDepth32F, hoverCircle);
            state.hoverValid = true;

            RefreshManualWindows();
            return;
        }

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            cv::Point imgPt;
            if (MapDisplayPointToManualImage(cv::Point(x, y),
                                             state.refDepth32F.size(),
                                             state.focusViewState,
                                             imgPt))
            {
                HandleManualClick(imgPt.x, imgPt.y, "Focus");
            }
        }
        else if (event == cv::EVENT_RBUTTONDOWN)
        {
            UndoLastManualSelection();
        }
    }

    void OnMouseVirtual(int event, int x, int y, int flags, void* userdata)
    {
        (void)userdata;

        ManualUIState& state = GetManualUIState();

        if (event == cv::EVENT_MOUSEWHEEL)
        {
            int delta = cv::getMouseWheelDelta(flags);
            if (delta != 0)
            {
                ZoomManualImageView(state.virtualViewState,
                                    state.virtualDepth32F.size(),
                                    cv::Point(x, y),
                                    delta);
                RefreshManualWindows();
            }
            return;
        }

        if (event == cv::EVENT_MBUTTONDOWN)
        {
            state.virtualViewState.middleDragging = true;
            state.virtualViewState.lastMouse = cv::Point(x, y);
            return;
        }

        if (event == cv::EVENT_MBUTTONUP)
        {
            state.virtualViewState.middleDragging = false;
            state.virtualViewState.lastMouse = cv::Point(-1, -1);
            return;
        }

        if (event == cv::EVENT_MOUSEMOVE && state.virtualViewState.middleDragging)
        {
            PanManualImageView(state.virtualViewState,
                               state.virtualDepth32F.size(),
                               cv::Point(x, y));
            RefreshManualWindows();
            return;
        }

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            cv::Point imgPt;
            if (MapDisplayPointToManualImage(cv::Point(x, y),
                                             state.virtualDepth32F.size(),
                                             state.virtualViewState,
                                             imgPt))
            {
                HandleManualClick(imgPt.x, imgPt.y, "Virtual");
            }
        }
        else if (event == cv::EVENT_RBUTTONDOWN)
        {
            UndoLastManualSelection();
        }
    }

void OnMouseInfoPanel(int event, int x, int y, int flags, void* userdata)
    {
        (void)userdata;

        ManualUIState& state = GetManualUIState();

        int panelHeight = state.infoPanel.empty() ? 1100 : state.infoPanel.rows;
        int maxScrollPx = std::max(0, state.infoContentHeight - panelHeight);

        if (event == cv::EVENT_MOUSEWHEEL)
        {
            int delta = cv::getMouseWheelDelta(flags);
            if (delta > 0)
                state.infoScrollPx = std::max(0, state.infoScrollPx - 120);
            else if (delta < 0)
                state.infoScrollPx = std::min(maxScrollPx, state.infoScrollPx + 120);

            RefreshManualWindows();
            return;
        }

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            cv::Point pt(x, y);

            if (state.infoScrollThumbRect.contains(pt))
            {
                state.infoScrollDragging = true;
                state.infoScrollDragOffsetY = y - state.infoScrollThumbRect.y;
                return;
            }

            if (state.infoScrollTrackRect.contains(pt) &&
                state.infoScrollTrackRect.height > 0 &&
                state.infoScrollThumbRect.height > 0)
            {
                int trackTop = state.infoScrollTrackRect.y;
                int trackHeight = state.infoScrollTrackRect.height;
                int thumbHeight = state.infoScrollThumbRect.height;
                int movableRange = std::max(1, trackHeight - thumbHeight);

                int targetThumbTop = y - thumbHeight / 2;
                int localTop = std::max(0, std::min(movableRange, targetThumbTop - trackTop));

                if (maxScrollPx > 0)
                {
                    state.infoScrollPx = static_cast<int>(
                            std::round(static_cast<double>(localTop) / static_cast<double>(movableRange) *
                                       static_cast<double>(maxScrollPx)));
                }
                else
                {
                    state.infoScrollPx = 0;
                }

                RefreshManualWindows();
                return;
            }
        }

        if (event == cv::EVENT_MOUSEMOVE && state.infoScrollDragging)
        {
            if (state.infoScrollTrackRect.height > 0 &&
                state.infoScrollThumbRect.height > 0)
            {
                int trackTop = state.infoScrollTrackRect.y;
                int trackHeight = state.infoScrollTrackRect.height;
                int thumbHeight = state.infoScrollThumbRect.height;
                int movableRange = std::max(1, trackHeight - thumbHeight);

                int thumbTop = y - state.infoScrollDragOffsetY;
                int localTop = std::max(0, std::min(movableRange, thumbTop - trackTop));

                if (maxScrollPx > 0)
                {
                    state.infoScrollPx = static_cast<int>(
                            std::round(static_cast<double>(localTop) / static_cast<double>(movableRange) *
                                       static_cast<double>(maxScrollPx)));
                }
                else
                {
                    state.infoScrollPx = 0;
                }

                RefreshManualWindows();
            }
            return;
        }

        if (event == cv::EVENT_LBUTTONUP)
        {
            state.infoScrollDragging = false;
            state.infoScrollDragOffsetY = 0;
            return;
        }
    }

    void SaveManualPairsAndScatter()
    {
        ManualUIState& state = GetManualUIState();

        {
            std::ofstream ofs(state.outputTxtPath.c_str());
            ofs << "# selection_index x y vd_mean ref_mean_mm vd_rmse ref_rmse\n";
            for (size_t i = 0; i < state.pairRecords.size(); ++i)
            {
                const ManualPairRecord& rec = state.pairRecords[i];
                ofs << rec.selectionIndex << " "
                    << rec.pt.x << " "
                    << rec.pt.y << " "
                    << rec.vdMean << " "
                    << rec.refMean << " "
                    << rec.vdRmse << " "
                    << rec.refRmse << "\n";
            }
            ofs.close();
        }

        cv::Mat scatter(900, 1200, CV_8UC3, cv::Scalar(255, 255, 255));

        if (state.pairRecords.empty())
        {
            cv::putText(scatter, "No valid mean (vd, ref) pairs collected.",
                        cv::Point(120, 450), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 0, 0), 2);
            cv::imwrite(state.outputScatterPath, scatter);
            return;
        }

        float minVd = std::numeric_limits<float>::max();
        float maxVd = std::numeric_limits<float>::lowest();
        float minRef = std::numeric_limits<float>::max();
        float maxRef = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < state.pairRecords.size(); ++i)
        {
            minVd = std::min(minVd, state.pairRecords[i].vdMean);
            maxVd = std::max(maxVd, state.pairRecords[i].vdMean);
            minRef = std::min(minRef, state.pairRecords[i].refMean);
            maxRef = std::max(maxRef, state.pairRecords[i].refMean);
        }

        if (std::fabs(maxVd - minVd) < 1e-6f)
            maxVd = minVd + 1.0f;
        if (std::fabs(maxRef - minRef) < 1e-6f)
            maxRef = minRef + 1.0f;

        const int left = 100;
        const int right = 1100;
        const int top = 80;
        const int bottom = 780;

        cv::line(scatter, cv::Point(left, bottom), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);
        cv::line(scatter, cv::Point(left, bottom), cv::Point(left, top), cv::Scalar(0, 0, 0), 2);

        for (int i = 0; i <= 5; ++i)
        {
            int x = left + (right - left) * i / 5;
            int y = bottom - (bottom - top) * i / 5;

            cv::line(scatter, cv::Point(x, bottom - 5), cv::Point(x, bottom + 5), cv::Scalar(0, 0, 0), 1);
            cv::line(scatter, cv::Point(left - 5, y), cv::Point(left + 5, y), cv::Scalar(0, 0, 0), 1);

            float vdTick = minVd + (maxVd - minVd) * static_cast<float>(i) / 5.0f;
            float refTick = minRef + (maxRef - minRef) * static_cast<float>(5 - i) / 5.0f;

            cv::putText(scatter, std::to_string(vdTick), cv::Point(x - 20, bottom + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
            cv::putText(scatter, std::to_string(refTick), cv::Point(20, y + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
        }

        for (size_t i = 0; i < state.pairRecords.size(); ++i)
        {
            float vd = state.pairRecords[i].vdMean;
            float ref = state.pairRecords[i].refMean;

            int px = left + static_cast<int>((vd - minVd) / (maxVd - minVd) * (right - left));
            int py = bottom - static_cast<int>((ref - minRef) / (maxRef - minRef) * (bottom - top));

            px = std::max(left, std::min(right, px));
            py = std::max(top, std::min(bottom, py));

            cv::circle(scatter, cv::Point(px, py), 3, cv::Scalar(255, 0, 0), -1);
        }

        cv::putText(scatter, "VD-REF 2D Distribution (Mean Values)", cv::Point(320, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);
        cv::putText(scatter, "x: vd mean (unitless)", cv::Point(450, 860),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);
        cv::putText(scatter, "y: ref mean (mm)", cv::Point(20, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);

        cv::imwrite(state.outputScatterPath, scatter);
    }

    void FitManualBehaviorModelAndWriteXml(VirtualToRealDepthFunc* self)
    {
        if (self == nullptr)
        {
            std::cerr << "FitManualBehaviorModelAndWriteXml: self is null." << std::endl;
            return;
        }

        ManualUIState& state = GetManualUIState();

        std::vector<float> refDepthValue;
        std::vector<float> virtualDepthValue;
        refDepthValue.reserve(state.pairRecords.size());
        virtualDepthValue.reserve(state.pairRecords.size());

        int minGtIndex = -1;
        int maxGtIndex = -1;

        for (size_t i = 0; i < state.pairRecords.size(); ++i)
        {
            const ManualPairRecord& rec = state.pairRecords[i];

            if (!std::isfinite(rec.refMean) || !std::isfinite(rec.vdMean) ||
                rec.refMean <= 0.0f || rec.vdMean <= 0.0f)
            {
                continue;
            }

            refDepthValue.push_back(rec.refMean);
            virtualDepthValue.push_back(rec.vdMean);

            int curIdx = static_cast<int>(refDepthValue.size()) - 1;

            if (minGtIndex < 0 || refDepthValue[curIdx] < refDepthValue[minGtIndex])
                minGtIndex = curIdx;

            if (maxGtIndex < 0 || refDepthValue[curIdx] > refDepthValue[maxGtIndex])
                maxGtIndex = curIdx;
        }

        if (refDepthValue.size() != virtualDepthValue.size() || refDepthValue.size() < 3)
        {
            std::cerr << "FitManualBehaviorModelAndWriteXml: valid pair count is not enough. "
                      << "refDepthValue.size() = " << refDepthValue.size()
                      << ", virtualDepthValue.size() = " << virtualDepthValue.size()
                      << std::endl;
            return;
        }

        std::array<double, 3> behaviorModelParams = self->BehavioralModel(refDepthValue, virtualDepthValue);

        float depthMin = virtualDepthValue[maxGtIndex]; // 最大gt对应的vd
        float depthMax = virtualDepthValue[minGtIndex]; // 最小gt对应的vd

        boost::filesystem::path root_path(self->m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME;
        std::string xml_path = strCalibPath + "behaviorModelParamsSegment.xml";

        std::ofstream xml(xml_path.c_str());
        if (!xml.is_open())
        {
            std::cerr << "无法写入 XML: " << xml_path << std::endl;
            return;
        }

        xml << std::fixed << std::setprecision(6);
        xml << "<?xml version=\"1.0\"?>\n";
        xml << "<opencv_storage>\n";
        xml << "    <BehaviorModelSegments>\n\n";

        xml << "        <Segment>\n";
        xml << "            <DepthMin>" << depthMin << "</DepthMin>\n";
        xml << "            <DepthMax>" << depthMax << "</DepthMax>\n\n";
        xml << "            <Param>\n";
        xml << "                <c0>" << behaviorModelParams[0] << "</c0>\n";
        xml << "                <c1>" << behaviorModelParams[1] << "</c1>\n";
        xml << "                <c2>" << behaviorModelParams[2] << "</c2>\n";
        xml << "            </Param>\n";
        xml << "        </Segment>\n\n";

        xml << "    </BehaviorModelSegments>\n";
        xml << "</opencv_storage>\n";
        xml.close();

        std::cout << "Manual behavior model XML saved: " << xml_path << std::endl;
        std::cout << "Manual behavior model params: "
                  << "c0=" << behaviorModelParams[0] << ", "
                  << "c1=" << behaviorModelParams[1] << ", "
                  << "c2=" << behaviorModelParams[2] << std::endl;
        std::cout << "DepthMin(max gt -> vd): " << depthMin << std::endl;
        std::cout << "DepthMax(min gt -> vd): " << depthMax << std::endl;
    }

void VirtualToRealDepthFunc::VirtualToRealDepthByManual(std::string& strFrameName)
    {
        m_strRootPath = m_ptrDepthSolver->GetRootPath();

        std::string strCommonPath = m_strRootPath + LF_DEPTH_INTRA_NAME + strFrameName + MVS_RESULT_DATA_FOLDER_NAME;
        std::string virtualDepthImg_path =  strCommonPath + strFrameName + "_" + LF_VIRTUALDEPTHMAP_NAME + ".tiff";
        std::string virtualDepthColorImg_path = strCommonPath + strFrameName + "_" + LF_VIRTUALDEPTHMAP_COLOR_NAME + ".png";
        std::string refDepthImg_path = m_strRootPath + LF_BEHAIRMODEL_FOLDER_NAME + "ref-csad-rd.tiff";
        std::string focuseImg_Path = strCommonPath + strFrameName + "_" + LF_FOUCSIMAGE_NAME + ".png";
        std::string distanceImage_path = m_strRootPath + "/behavior_model/distanceImage.png";

        ManualUIState& state = GetManualUIState();
        state = ManualUIState();

        m_virtualDepthImage = cv::imread(virtualDepthImg_path, cv::IMREAD_UNCHANGED);
        m_refDepthImage = cv::imread(refDepthImg_path, cv::IMREAD_UNCHANGED);
        cv::Mat focusImage = cv::imread(focuseImg_Path, cv::IMREAD_COLOR);
        cv::Mat virtualDepthColorImg = cv::imread(virtualDepthColorImg_path, cv::IMREAD_COLOR);

        if (m_virtualDepthImage.empty() || m_refDepthImage.empty() ||
            focusImage.empty() || virtualDepthColorImg.empty())
        {
            std::cout << "manual mode input image is empty!" << std::endl;
            return;
        }

        state.refDepth32F = ToSingleFloat(m_refDepthImage);
        state.virtualDepth32F = ToSingleFloat(m_virtualDepthImage);

        if (state.refDepth32F.empty() || state.virtualDepth32F.empty())
        {
            std::cout << "manual mode depth image preprocess failed!" << std::endl;
            return;
        }

        if (focusImage.size() != state.refDepth32F.size() ||
            virtualDepthColorImg.size() != state.refDepth32F.size())
        {
            std::cout << "manual mode image size mismatch!" << std::endl;
            std::cout << "focusImage: " << focusImage.cols << "x" << focusImage.rows << std::endl;
            std::cout << "virtualDepthColorImg: " << virtualDepthColorImg.cols << "x" << virtualDepthColorImg.rows << std::endl;
            std::cout << "refDepthImage: " << state.refDepth32F.cols << "x" << state.refDepth32F.rows << std::endl;
            return;
        }

        float highlightDepthMin = 16000.0f;
        float highlightDepthMax = 20000.0f;

        state.refFocusBase = BuildRefFocusImage(focusImage, state.refDepth32F, highlightDepthMin, highlightDepthMax);
        state.virtualColorBase = virtualDepthColorImg.clone();

        InitManualImageViewState(state.focusViewState, state.refFocusBase.size());
        InitManualImageViewState(state.virtualViewState, state.virtualColorBase.size());

        state.refFocusShow = state.refFocusBase.clone();
        state.virtualColorShow = state.virtualColorBase.clone();
        state.infoPanel = cv::Mat(1100, 1400, CV_8UC3, cv::Scalar(255, 255, 255));

        state.outputTxtPath = m_strRootPath + "/behavior_model/manual_vd_ref_pairs.txt";
        state.outputScatterPath = m_strRootPath + "/behavior_model/manual_vd_ref_scatter.png";
        state.outputFocusMarkedPath = m_strRootPath + "/behavior_model/manual_ref_focus_marked.png";
        state.outputVirtualMarkedPath = m_strRootPath + "/behavior_model/manual_virtual_marked.png";

        cv::namedWindow("FocusImage", cv::WINDOW_NORMAL);
        cv::namedWindow("VirtualDepthColor", cv::WINDOW_NORMAL);
        cv::namedWindow("InfoPanel", cv::WINDOW_NORMAL);

        cv::moveWindow("FocusImage", 50, 50);
        cv::moveWindow("VirtualDepthColor", 1050, 50);
        cv::moveWindow("InfoPanel", 50, 850);

        cv::resizeWindow("FocusImage", 950, 750);
        cv::resizeWindow("VirtualDepthColor", 950, 750);
        cv::resizeWindow("InfoPanel", 1400, 1100);

        cv::setMouseCallback("FocusImage", OnMouseFocus, nullptr);
        cv::setMouseCallback("VirtualDepthColor", OnMouseVirtual, nullptr);
        cv::setMouseCallback("InfoPanel", OnMouseInfoPanel, nullptr);

        RefreshManualWindows();

        while (true)
        {
            int key = cv::waitKey(30);
            if (key == 27) // ESC
            {
                break;
            }
            else if (key == 'z' || key == 'u' || key == 8)
            {
                UndoLastManualSelection();
            }
            else if (key == 'r')
            {
                InitManualImageViewState(state.focusViewState, state.refFocusBase.size());
                InitManualImageViewState(state.virtualViewState, state.virtualColorBase.size());
                RefreshManualWindows();
            }
        }

        SaveManualPairsAndScatter();

        // Step2: 用 manual 选出来并通过 RMSE 过滤后的均值点对参与拟合
        FitManualBehaviorModelAndWriteXml(this);

        cv::Mat refMarkedFull, virtualMarkedFull;
        BuildAnnotatedManualImages(refMarkedFull, virtualMarkedFull);

        cv::imwrite(state.outputFocusMarkedPath, refMarkedFull);
        cv::imwrite(state.outputVirtualMarkedPath, virtualMarkedFull);

        std::cout << "manual pairs txt saved: " << state.outputTxtPath << std::endl;
        std::cout << "manual scatter saved: " << state.outputScatterPath << std::endl;
        std::cout << "manual ref focus marked image saved: " << state.outputFocusMarkedPath << std::endl;
        std::cout << "manual virtual marked image saved: " << state.outputVirtualMarkedPath << std::endl;

        cv::destroyWindow("FocusImage");
        cv::destroyWindow("VirtualDepthColor");
        cv::destroyWindow("InfoPanel");

        SamplePointSelect();
        m_realDepthImage = ConvertVdImageToRd(m_virtualDepthImage);
        errorStatisticsImageGTSeg(m_realDepthImage, focusImage, distanceImage_path);
    }

    bool LoadBehaviorModelSegmentsFromXml(const std::string& xmlPath,
                                          std::vector<BehaviorModelSegment>& segments)
    {
        segments.clear();

        cv::FileStorage fs(xmlPath, cv::FileStorage::READ);
        if (!fs.isOpened())
            return false;

        cv::FileNode node = fs["BehaviorModelSegments"];
        if (node.empty())
            return false;

        for (auto it = node.begin(); it != node.end(); ++it)
        {
            BehaviorModelSegment seg;
            (*it)["DepthMin"] >> seg.depthMin;
            (*it)["DepthMax"] >> seg.depthMax;

            cv::FileNode param = (*it)["Param"];
            param["c0"] >> seg.params[0];
            param["c1"] >> seg.params[1];
            param["c2"] >> seg.params[2];

            segments.push_back(seg);
        }

        return !segments.empty();
    }

    float ConvertVdToRdBySegments(float vDepth,
                                  const std::vector<BehaviorModelSegment>& segments)
    {
        if (!(vDepth > 0.0f) || !std::isfinite(vDepth))
            return 0.0f;

        double z = 0.0;

        for (size_t i = 0; i < segments.size(); ++i)
        {
            const BehaviorModelSegment& seg = segments[i];

            if (vDepth >= seg.depthMin && vDepth < seg.depthMax)
            {
                double c0 = seg.params[0];
                double c1 = seg.params[1];
                double c2 = seg.params[2];

                double v = static_cast<double>(vDepth);
                double denom = 1.0 - v * c0;

                if (std::fabs(denom) < 1.0e-12)
                    return 0.0f;

                z = (v * c1 + c2) / denom;
                break;
            }
        }

        if (!std::isfinite(z) || z < 0.0)
            return 0.0f;

        return static_cast<float>(z);
    }

    float ComputeMeanInRectValidOnly(const cv::Mat& depth32F,
                                     const cv::Rect& rect,
                                     int& validCount)
    {
        validCount = 0;

        if (depth32F.empty())
            return 0.0f;

        cv::Rect imgRect(0, 0, depth32F.cols, depth32F.rows);
        cv::Rect roi = rect & imgRect;
        if (roi.width <= 0 || roi.height <= 0)
            return 0.0f;

        double sum = 0.0;
        for (int y = roi.y; y < roi.y + roi.height; ++y)
        {
            for (int x = roi.x; x < roi.x + roi.width; ++x)
            {
                float v = ReadDepthValue(depth32F, x, y);
                if (v > 0.0f)
                {
                    sum += v;
                    validCount++;
                }
            }
        }

        if (validCount == 0)
            return 0.0f;

        return static_cast<float>(sum / static_cast<double>(validCount));
    }

    cv::Rect BuildBoundingRectFromPoints(const std::vector<cv::Point>& pts)
    {
        if (pts.empty())
            return cv::Rect();

        return cv::boundingRect(pts);
    }

    void RefreshValidationWindow()
    {
        ValidationUIState& state = GetValidationUIState();

        state.focusShow = state.focusBase.clone();

        // 先画历史记录
        for (size_t i = 0; i < state.records.size(); ++i)
        {
            const ValidationBoxRecord& rec = state.records[i];

            // 画四边形
            for (int k = 0; k < 4; ++k)
            {
                cv::line(state.focusShow,
                         rec.corners[k],
                         rec.corners[(k + 1) % 4],
                         cv::Scalar(0, 255, 255), 2);
            }

            // 画包围盒
            cv::rectangle(state.focusShow, rec.bbox, cv::Scalar(0, 255, 0), 2);

            // 角点
            for (size_t k = 0; k < rec.corners.size(); ++k)
            {
                cv::circle(state.focusShow, rec.corners[k], 2, cv::Scalar(0, 0, 255), -1);
            }

            std::string line1 =
                    "vd_M=" + std::to_string(rec.vdMean) +
                    "  rd=" + std::to_string(rec.rdEst);

            std::string line2;
            if (rec.hasRefMean)
            {
                line2 =
                        "ref_M=" + std::to_string(rec.refMean) +
                        "  absErr=" + std::to_string(rec.absErr);
            }

            int textX = std::max(30, rec.bbox.x);
            int textY = std::max(250, rec.bbox.y - 25);

            cv::putText(state.focusShow, line1, cv::Point(textX, textY),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

            if (!line2.empty())
            {
                cv::putText(state.focusShow, line2, cv::Point(textX, textY - 28),
                            cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 255), 2);
            }

            std::string idx = "#" + std::to_string(static_cast<int>(i + 1));
            cv::putText(state.focusShow, idx, rec.bbox.tl() + cv::Point(4, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 2);
        }

        // 再画当前正在点的点和临时线
        for (size_t i = 0; i < state.currentPoints.size(); ++i)
        {
            cv::circle(state.focusShow, state.currentPoints[i], 5, cv::Scalar(0, 0, 255), -1);

            std::string idx = std::to_string(static_cast<int>(i + 1));
            cv::putText(state.focusShow, idx, state.currentPoints[i] + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

            if (i > 0)
            {
                cv::line(state.focusShow, state.currentPoints[i - 1], state.currentPoints[i],
                         cv::Scalar(255, 255, 0), 2);
            }
        }

        std::string tip =
                "Validate Mode: Left click 4 points | Z/U/Backspace undo | C clear | ESC save&quit";
        cv::putText(state.focusShow, tip, cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(20, 20, 20), 2);

        cv::imshow("FocusImage_Validate", state.focusShow);
    }

    void CommitValidationBoxFromCurrentPoints()
    {
        ValidationUIState& state = GetValidationUIState();

        if (state.currentPoints.size() != 4)
            return;

        ValidationBoxRecord rec;
        rec.corners = state.currentPoints;
        rec.bbox = BuildBoundingRectFromPoints(rec.corners);

        rec.vdMean = ComputeMeanInRectValidOnly(state.vd32F, rec.bbox, rec.vdValidCount);
        rec.rdEst = ConvertVdToRdBySegments(rec.vdMean, state.segments);

        if (!state.ref32F.empty())
        {
            rec.refMean = ComputeMeanInRectValidOnly(state.ref32F, rec.bbox, rec.refValidCount);
            if (rec.refValidCount > 0)
            {
                rec.hasRefMean = true;
                rec.absErr = std::fabs(rec.refMean - rec.rdEst);
            }
        }

        state.records.push_back(rec);
        state.currentPoints.clear();

        RefreshValidationWindow();
    }

    void UndoValidationLastStep()
    {
        ValidationUIState& state = GetValidationUIState();

        if (!state.currentPoints.empty())
        {
            state.currentPoints.pop_back();
        }
        else if (!state.records.empty())
        {
            state.records.pop_back();
        }

        RefreshValidationWindow();
    }

    void OnMouseValidateFocus(int event, int x, int y, int flags, void* userdata)
    {
        (void)flags;
        (void)userdata;

        ValidationUIState& state = GetValidationUIState();

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            if (x < 0 || y < 0 || x >= state.focusBase.cols || y >= state.focusBase.rows)
                return;

            if (state.currentPoints.size() < 4)
            {
                state.currentPoints.emplace_back(x, y);
                RefreshValidationWindow();

                if (state.currentPoints.size() == 4)
                {
                    CommitValidationBoxFromCurrentPoints();
                }
            }
        }
        else if (event == cv::EVENT_RBUTTONDOWN)
        {
            UndoValidationLastStep();
        }
    }

    void SaveValidationResults()
    {
        ValidationUIState& state = GetValidationUIState();

        cv::imwrite(state.outputImagePath, state.focusShow);

        std::ofstream ofs(state.outputTxtPath.c_str());
        if (!ofs.is_open())
            return;

        ofs << "# idx x0 y0 x1 y1 x2 y2 x3 y3 "
               "bbox_x bbox_y bbox_w bbox_h "
               "vd_mean rd_est ref_mean abs_err vd_valid_count ref_valid_count\n";

        for (size_t i = 0; i < state.records.size(); ++i)
        {
            const ValidationBoxRecord& rec = state.records[i];

            ofs << (i + 1) << " ";

            for (int k = 0; k < 4; ++k)
            {
                ofs << rec.corners[k].x << " " << rec.corners[k].y << " ";
            }

            ofs << rec.bbox.x << " "
                << rec.bbox.y << " "
                << rec.bbox.width << " "
                << rec.bbox.height << " "
                << rec.vdMean << " "
                << rec.rdEst << " ";

            if (rec.hasRefMean)
                ofs << rec.refMean << " " << rec.absErr << " ";
            else
                ofs << 0.0f << " " << 0.0f << " ";

            ofs << rec.vdValidCount << " "
                << rec.refValidCount << "\n";
        }

        ofs.close();
    }

    void VirtualToRealDepthFunc::ValidateBehaviorModelByBox()
    {
        m_strRootPath = m_ptrDepthSolver->GetRootPath();

        std::string virtualDepthImg_path = m_strRootPath + "/behavior_model/VD_Raw.tiff";
        std::string refDepthImg_path = m_strRootPath + "/behavior_model/ref-csad-rd.tiff";
        std::string focuseImg_Path = m_strRootPath + "/behavior_model/fullfocus.png";

        ValidationUIState& state = GetValidationUIState();
        state = ValidationUIState();

        cv::Mat vdRaw = cv::imread(virtualDepthImg_path, cv::IMREAD_UNCHANGED);
        cv::Mat refRaw = cv::imread(refDepthImg_path, cv::IMREAD_UNCHANGED);
        cv::Mat focusImage = cv::imread(focuseImg_Path, cv::IMREAD_COLOR);

        if (vdRaw.empty() || focusImage.empty())
        {
            std::cout << "ValidateBehaviorModelByBox: input image is empty!" << std::endl;
            return;
        }

        state.vd32F = ToSingleFloat(vdRaw);
        state.ref32F = ToSingleFloat(refRaw);

        if (state.vd32F.empty())
        {
            std::cout << "ValidateBehaviorModelByBox: vd image preprocess failed!" << std::endl;
            return;
        }

        if (focusImage.size() != state.vd32F.size())
        {
            std::cout << "ValidateBehaviorModelByBox: image size mismatch!" << std::endl;
            return;
        }

        // 读取分段行为模型
        boost::filesystem::path root_path(m_strRootPath);
        boost::filesystem::path root_path_parent = root_path.parent_path();
        std::string strCalibPath = root_path_parent.string() + LF_CALIB_FOLDER_NAME;
        std::string xmlPath = strCalibPath + "behaviorModelParamsSegment.xml";

        if (!LoadBehaviorModelSegmentsFromXml(xmlPath, state.segments))
        {
            std::cout << "ValidateBehaviorModelByBox: failed to load behavior model xml: "
                      << xmlPath << std::endl;
            return;
        }

        state.focusBase = focusImage.clone();
        state.focusShow = state.focusBase.clone();

        state.outputImagePath = m_strRootPath + "/behavior_model/validate_box_result.png";
        state.outputTxtPath = m_strRootPath + "/behavior_model/validate_box_result.txt";

        cv::namedWindow("FocusImage_Validate", cv::WINDOW_NORMAL);
        cv::moveWindow("FocusImage_Validate", 100, 100);
        cv::setMouseCallback("FocusImage_Validate", OnMouseValidateFocus, nullptr);

        RefreshValidationWindow();

        while (true)
        {
            int key = cv::waitKey(30);

            if (key == 27) // ESC
            {
                break;
            }
            else if (key == 'z' || key == 'u' || key == 8)
            {
                UndoValidationLastStep();
            }
            else if (key == 'c')
            {
                state.currentPoints.clear();
                state.records.clear();
                RefreshValidationWindow();
            }
        }

        SaveValidationResults();

        std::cout << "validate image saved: " << state.outputImagePath << std::endl;
        std::cout << "validate txt saved: " << state.outputTxtPath << std::endl;

        cv::destroyWindow("FocusImage_Validate");
    }
}
