/********************************************************************
file base:      LFRefocus.h
author:         LZD
created:        2025/03/12
purpose:        数字重聚焦、焦点堆栈、焦平面等生成
                焦点堆栈（Focal Stack)是一种通过合成多张不同焦平面的图像来扩展景深的技术，
                可利用焦点堆栈原理+视差匹配结果进行重聚焦，得到全聚焦图像+虚拟深度图
*********************************************************************/
#ifndef LFMVS_LFREFOCUS_H
#define LFMVS_LFREFOCUS_H

#include "DepthSolver.h"

namespace LFMVS
{
    //////////////////////////////////////////////////////////////////////////////
    // 假设这个像素的结构体如下(用自定义结构体合适还是用 cv::PointMap 合适？)
    struct pixel_point
    {
        int num; // 存储像素信息：微图像序列索引号+在微图像上的坐标
        int x;
        int y;
        bool is_visited= false; // 像素点是否被遍历,true则代表已被遍历 （或许a这个标识符单独存放成一个矩阵）

        bool operator==(const pixel_point& c_pixel_point) const
        {
            return x==c_pixel_point.x && y==c_pixel_point.y && num==c_pixel_point.num;
        }
    };
    typedef std::vector<pixel_point> PixelPointVec;
    typedef std::vector<PixelPointVec> PixelPointVecVec;
    typedef std::vector<PixelPointVecVec> PixelPointVecVecVec;

    struct PointProperty
    {
        float       virtualDepth;   // 存储第二次排序后的虚拟深度值
        float       distence;   // 基线
        float       disp;   // 视差
        uchar       color0;         // 存储RGB的三个通道值
        uchar       color1;
        uchar       color2;
        float       metricDepth;    // 真实深度，不确定此时是否需要存储
        float       pointRichScore;
        QuadTreeTileKeyPtr srcImageKey;     // 所在微图像的key
    };
    typedef std::map<std::pair<int, int>,PointProperty> PointMap;     //存放微图像每个像素点的最终结果值

    struct VIPointProperty
    {
        int                     virtualImageX;
        int                     virtualImageY;
        int                     srcImageX;
        int                     srcImageY;
        QuadTreeTileKeyPtr      srcImageKey;     // 像素点所在微图像的key
        float                   score;                        //对该点的综合评分
        float                   virtualDepth;
        float                   vd_Media;
        cv::Vec3b               rgb_Media;
        cv::Vec3b               lab;

    };
    typedef std::map<std::pair<int, int>,std::vector<VIPointProperty>> CoVIPointMap; // 虚拟深度图中同一位置的同名点
    typedef std::map<std::pair<int, int>,VIPointProperty> VIPointMap;

    struct Coeffs
    {
        double c0, c1, c2;
    };

    //////////////////////////////////////////////////////////////////////////////
    // 重聚焦类
    class LFRefocus
    {
        friend DepthSolver;
    public:
        LFRefocus(DepthSolver* pDepthSolver);

        ~LFRefocus();

    public: // 接口
        // 最优方式
        void AIFImageCompositeForMIA(QuadTreeProblemMapMap::iterator& itrFrame);

        // 加权平均
        void AIFImageCompositeForMIA_Weights(QuadTreeProblemMapMap::iterator& itrFrame);

        // 融合虚拟深度：重投影
        void FuseVirtualDepth_BackProject(QuadTreeProblemMapMap::iterator& itrFrame);

        // 融合虚拟深度：重投影+八叉树
        void FuseVirtualDepth_BackProject_OctreeVoxel(QuadTreeProblemMapMap::iterator& itrFrame);

        void FuseVirtualDepth_BackProject_OctreeVoxel_Copy(QuadTreeProblemMapMap::iterator& itrFrame);

    private:

        // 将微图像上的视差转换为虚像平面上的深度
        float ComputeVirtualDepth(const double disparity, const LightFieldParams& params);
        cv::Vec3f ComputeVirtual3D(float virtualDepth, int p_col, int p_row, int mi_cols, int mi_rows,
                                    cv::Point2f center_at_mla);
        cv::Vec2f BackProject2MI(cv::Vec3f virtual_Point, cv::Point2f center_at_mla, cv::Vec2f center_mi_src);

        // 虚拟深度转真实深度
        void Virtual2ObjectDepth(std::vector<PointList>& PointCloud_Object, const std::vector<PointList>& PointCloud_Virtual);

        // 筛选同名点
        void FilterMatchingPoints(std::vector<VIPointProperty>& vIPointVec, QuadTreeProblemMap& problem_map);

        VIPointProperty DenoisByColorConsist(std::vector<VIPointProperty>& vIPointVec, QuadTreeProblemMap& problem_map);

        void ImageDenoising(QuadTreeProblemMap& problem_map);

        // 填充像素点颜色，并生成虚拟深度图
        void GenerateVirtualImage(VIPointMap& vIPointMap, QuadTreeProblemMap& problem_map,LightFieldParams& params);

        void GenerateVIByColorConsist(VIPointMap& vIPointMap, QuadTreeProblemMap& problem_map,LightFieldParams& params);

        // 空洞填充
        void HoleFilling(cv::Mat& virtualImage, VIPointMap& vIPointMap, QuadTreeProblemMap& problem_map);

       // void CollectPointNeigInfo(int64_t p_index, Vec3f& p_coord, NeighborCorrInfoMap& neigInfoMap);
        bool WritePointAndNeigInfo();

        // 随机采样n个点
        std::vector<cv::Point> samplePoints(int width, int height, int N, int seed);

        // 读取行为模型参数
        bool LoadCoeffsFromXml(const std::string& xmlPath,std::array<double, 3>& behaviorModelParams);

        float ConvertVdToRd(float v_depth,std::array<double, 3>& behaviorModelParams);

        float ConvertVdToRdSegment(float v_depth);

    private:
        DepthSolver*                m_ptrDepthSolver;

        int                         m_RawImage_Width;   // 微透镜图像的宽（单位像素）
        int                         m_RawImage_Height;

        std::string                 m_strSavePath;
        cv::Mat                     m_VirtualDepthMap;          // 存储虚拟深度
        cv::Mat                     m_AIF_Color;         // all-in-focus image


        int                         m_virtualImageX;
        int                         m_virtualImageY;
        CoVIPointMap                m_coVIPointMap;         // 虚拟深度图中同一位置的同名点
        VIPointProperty             m_vIPoint;              // m_AIF_Color
        VIPointMap                  m_vIPointMap;           // m_vIPointMap，去除重复值后的全聚焦图Map
        cv::Mat                     m_vIDepthGray;          // 存储虚拟深度灰度图
        cv::Mat                     m_vIDepth;              // 存储归一化后的 1/虚拟深度
        cv::Mat                     m_vIDepth_Reciprocal;   // 存储 1/虚拟深度
        cv::Mat                     m_rIDepth_mm;           // 存储真实深度
        cv::Mat                     m_rIDepth_m;            // 存储真实深度
        std::vector<VIPointProperty> m_vIPointVec;
        std::vector<cv::Vec3b>      m_labVec;

     //   PointPacketMap              m_PointPackMap;
    };
}
#endif //LFMVS_LFREFOCUS_H
