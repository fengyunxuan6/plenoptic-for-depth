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
    };

    enum SamplePointSelectType
    {
        SPSelectByLocalWindow,
        SPSelectByGlobal,
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

        void SamplePointSelect();

    public:
        struct Sample
        {
            int colIndex;
            int rowIndex;
            float vDepth;
            float rDepth;
        };

    private:

        void VirtualToRealDepthByBM();

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

       void SamplePointSelectByLW();

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

        bool findStableGTValue(
                        cv::Mat& gt,
                        int vd_x,
                        int vd_y,
                        float& gt_value,
                        int& gt_x,
                        int& gt_y,
                        int max_radius);

    public:
        struct SampleErrorStatsOptions
                {
                    int refSearchRadius = 50;  // 点云真值稀疏，若取值点空，则邻域半径搜索
                    int virtualSearchRadius = 50;  // 虚拟深度可能被过滤，若取值点空，则邻域半径搜索
                    double outlierDistanceThreshold = 2000;   // 计算误差时异常距离阈值   /* mm */
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

    };

}




#endif //ACMP_VIRTUALTOREALDEPTHFUNC_H
