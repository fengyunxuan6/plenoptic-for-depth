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
        VTORD_SegmentBehavioralmodel_2
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

        enum class DistanceType { Chamfer, Euclidean, Mean, Median};

        bool ExtractDepthsFromImages(
                std::vector<float>& refDepthValue,
                std::vector<float>& virtualDepthValue);

        bool appendDepthPoints(
                std::vector<float>& refDepthValue,
                std::vector<float>& virtualDepthValue,
                cv::Rect& roi);

        std::array<double,3> BehavioralModel(std::vector<float>& refDepthValue, std::vector<float>& virtualDepthValue);

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

    public:
        struct SampleErrorStatsOptions
                {
                    int refSearchRadius = 50;  // 点云真值稀疏，若取值点空，则邻域半径搜索
                    int virtualSearchRadius = 50;  // 虚拟深度可能被过滤，若取值点空，则邻域半径搜索
                    double outlierDistanceThreshold = 50000;   // 计算误差时异常距离阈值   /* mm */
                    int localWindowSize = 100;        // 误差计算：随机采样窗口大小
                    int sampleCount = 5;   // 误差计算：每个坐标点采样数量
                };

    private:
        DepthSolver*                m_ptrDepthSolver;
        std::string                 m_strRootPath;

        VTORDType                   m_VirtualToRealDepthType;
        SamplePointSelectType       m_SamplePointSelectType;

        SampleErrorStatsOptions     SESOptions;

        cv::Mat                     m_virtualDepthImage;
        cv::Mat                     m_refDepthImage;
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




#endif //ACMP_VIRTUALTOREALDEPTHFUNC_H
