//
// Created by wdy on 25-10-17.
//

#ifndef ACMP_VIRTUALTOREALDEPTHFUNC_H
#define ACMP_VIRTUALTOREALDEPTHFUNC_H
#include "DepthSolver.h"

namespace LFMVS
{
    enum VTORDType
    {
        VTORD_Behavioralmodel,
        VTORD_SegmentBehavioralmodel,
        VTORD_SegmentBehavioralmodel_2,
        VTORD_SegmentBehavioralmodel_Manual
    };

    enum SamplePointSelectType
    {
        SPSelectByLocalWindow,
        SPSelectByGlobal,
        SPSelectByRandom,
    };

    class VirtualToRealDepthFunc
    {
        friend DepthSolver;
    public:
        VirtualToRealDepthFunc(DepthSolver* pDepthSolver);
        ~VirtualToRealDepthFunc();

    public:
        void SetVirtualToRealDepthType(VTORDType type);
        VTORDType GetVirtualToRealDepthType();

        void SetSamplePointSelectType(SamplePointSelectType type);
        SamplePointSelectType GetSamplePointSelectType();

        void VirtualToRealDepth(QuadTreeProblemMapMap::iterator& itrP);

        cv::Mat ConvertVdImageToRd(cv::Mat virtualDepthImage);

        void SamplePointSelect();
        std::array<double,3> BehavioralModel(std::vector<float>& refDepthValue, std::vector<float>& virtualDepthValue);

    public:
        struct Sample
        {
            int colIndex;
            int rowIndex;
            float vDepth;
            float rDepth;
        };

        struct SamplePoint
        {
            int colIndex;
            int rowIndex;
            float vDepth;
            float gtDepth;
        };

        struct DepthSegment
        {
            float gtStart;                     // 区间起点
            float gtEnd;                       // 区间终点
            float vdLower;                     // gtStart 对应的 vd
            float vdUpper;                     // gtEnd 对应的 vd
            std::vector<SamplePoint> points;   // 该区间内的所有点
        };

        struct BehaviorSegmentResult
        {
            float vdepthMin;
            float vdepthMax;
            int sampleCount;
            std::array<double, 3> params;
        };
    private:

        void VirtualToRealDepthByBM();
        void VirtualToRealDepthBySegBM();
        void VirtualToRealDepthBySegBM_2();
        void VirtualToRealDepthBySegBM_new();
        void VirtualToRealDepthByManual(std::string& strFrameName);

        enum class DistanceType { Chamfer, Euclidean, Mean, Median};

        bool ExtractDepthsFromImages(
                std::vector<float>& refDepthValue,
                std::vector<float>& virtualDepthValue);

        bool appendDepthPoints(
                std::vector<float>& refDepthValue,
                std::vector<float>& virtualDepthValue,
                cv::Rect& roi);

        cv::Mat convertVirtualToRealDepth( std::array<double, 3>& behaviorModelParams);

        void imageDistanceSampling(cv::Mat realDepthImage, cv::Mat refDepthImage);
        void errorStatisticsImage(cv::Mat realDepthImage, cv::Mat refDepthImage, cv::Mat bgImage, std::string errorMapSavePath);
        void errorStatisticsImageSeg(cv::Mat realDepthImage, cv::Mat refDepthImage, cv::Mat virtualDepthImage,cv::Mat bgImage, std::string errorMapSavePath);
        void errorStatisticsImageGTSeg(cv::Mat realDepthImage, cv::Mat bgImage, std::string errorMapSavePath);

        void SamplePointSelectByLW();
        void SamplePointSelectByRandom();

        cv::Mat buildOutlierMask(cv::Mat& img, int ksize, double k);

        bool loadPointsXML(std::string& pointsXmlPath);

        cv::Mat drawRandomColorCrosses(const cv::Mat& focuseImage, const std::vector<cv::Point>& points);

//        void sampleVirtualDepthPoints(std::string& vdepth_path,
//                                      std::string& gt_path,
//                                      std::string& output_mark_path,
//                                      std::string& output_csv);

        void sampleVirtualDepthPointsByRegion(std::string& vdepth_path,
                                      std::string& gt_path,
                                      std::string& output_mark_path,
                                      std::string& output_csv);

        bool isValidPixel(cv::Mat& depth, int x, int y, float value, float neighbor_tol);

        bool isStableGTPixel(cv::Mat &gt, int x, int y, float value, float neighbor_tol_ratio);

        bool findStableGTValue(cv::Mat& gt, int vd_x, int vd_y, float& gt_value, int& gt_x, int& gt_y, int max_radius);

        void segmentByGT(cv::Mat &m_refDepthImage,cv::Mat &focusImage);

        void selectGtPoints();

        void selectVdPoints();

        void fitSegmentsParams(std::string xml_path);
        void fitSegmentsParams_new(std::string xml_path);

        void ValidateBehaviorModelByBox();

    public:
        struct SampleErrorStatsOptions
                {
                    int refSearchRadius = 50;  // 点云真值稀疏，若取值点空，则邻域半径搜索
                    int virtualSearchRadius = 50;  // 虚拟深度可能被过滤，若取值点空，则邻域半径搜索
                    double outlierDistanceThreshold = 50000;   // 计算误差时异常距离阈值   /* mm */
                    int localWindowSize = 100;        // 误差计算：随机采样窗口大小
                    int sampleCount = 5;   // 误差计算：每个坐标点采样数量
                };

    public:
        std::string                 m_strRootPath;

    private:
        DepthSolver*                m_ptrDepthSolver;

        VTORDType                   m_VirtualToRealDepthType;
        SamplePointSelectType       m_SamplePointSelectType;

        SampleErrorStatsOptions     SESOptions;

        cv::Mat                     m_virtualDepthImage;
        cv::Mat                     m_refDepthImage;
        cv::Mat                     m_realDepthImage;

        int                         m_rdImage_rows;
        int                         m_rdImage_cols;

        std::vector<cv::Point>      m_samplePoints;
        std::vector<cv::Point>      m_coordsVirtual;
        std::vector<cv::Point>      m_coordsRef;

        std::vector<SamplePoint> samplePoints;
        std::vector<std::vector<SamplePoint>> samplePointsVector;

        std::vector<std::vector<SamplePoint>> samplePointsVectorFiltered;
        std::vector<std::vector<SamplePoint>> samplePointsVectorSorted;

    };

}

namespace
{
    struct ManualPairRecord
    {
        int selectionIndex = -1;
        cv::Point pt;          // 鼠标点击中心点
        float vdMean = 0.0f;   // 圆内 VD 均值
        float refMean = 0.0f;  // 圆内 REF 均值
        double vdRmse = 0.0;
        double refRmse = 0.0;
    };

    struct ManualSelectionRecord
    {
        cv::Point center;
        std::string sourceWindow;
        std::vector<cv::Point> circlePoints;
        std::vector<float> vdValues;
        std::vector<float> refValues;

        std::vector<std::string> vdPointLines;   // 左列显示
        std::vector<std::string> refPointLines;  // 右列显示

        double vdRmse = 0.0;
        double refRmse = 0.0;
        size_t pairBegin = 0;
        size_t pairEnd = 0;
    };

    struct ManualUIState
    {
        cv::Mat refDepth32F;
        cv::Mat virtualDepth32F;

        cv::Mat refFocusBase;
        cv::Mat virtualColorBase;

        cv::Mat refFocusShow;
        cv::Mat virtualColorShow;
        cv::Mat infoPanel;

        std::string outputTxtPath;
        std::string outputScatterPath;
        std::string outputFocusMarkedPath;
        std::string outputVirtualMarkedPath;

        int circleRadius =50;

        std::vector<ManualSelectionRecord> selections;
        std::vector<ManualPairRecord> pairRecords;

        cv::Point hoverPoint = cv::Point(-1, -1);
        double hoverRefMean = 0.0;
        bool hoverValid = false;

        int infoScrollOffset = 0;

        std::chrono::steady_clock::time_point lastHoverUpdateTime = std::chrono::steady_clock::now();
        bool hoverTimeInitialized = false;
    };

    struct BehaviorModelSegment
    {
        double depthMin = 0.0;
        double depthMax = 0.0;
        std::array<double, 3> params = {0.0, 0.0, 0.0};
    };

    struct ValidationBoxRecord
    {
        std::vector<cv::Point> corners;   // 4个点
        cv::Rect bbox;                    // 包围盒

        float vdMean = 0.0f;
        float rdEst = 0.0f;

        bool hasRefMean = false;
        float refMean = 0.0f;
        float absErr = 0.0f;

        int vdValidCount = 0;
        int refValidCount = 0;
    };

    struct ValidationUIState
    {
        cv::Mat focusBase;
        cv::Mat focusShow;

        cv::Mat vd32F;
        cv::Mat ref32F;

        std::vector<BehaviorModelSegment> segments;

        std::vector<cv::Point> currentPoints;          // 当前正在点的4个点
        std::vector<ValidationBoxRecord> records;      // 历史验证框

        std::string outputImagePath;
        std::string outputTxtPath;
    };

    ValidationUIState& GetValidationUIState()
    {
        static ValidationUIState state;
        return state;
    }


}

#endif //ACMP_VIRTUALTOREALDEPTHFUNC_H
