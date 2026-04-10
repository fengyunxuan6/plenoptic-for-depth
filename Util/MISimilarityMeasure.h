/********************************************************************
file base:      MISimilarMeasure.cpp
author:         LZD
created:        2025/04/25
purpose:        对微图像进行模糊程度的量化
*********************************************************************/

#ifndef ACMP_MISIMILARITYMEASURE_H
#define ACMP_MISIMILARITYMEASURE_H

#endif //ACMP_MISIMILARITYMEASURE_H

#include "DepthSolver.h"

namespace LFMVS
{
    class MISimilarityMeasure
    {
        friend DepthSolver;
    public:

        MISimilarityMeasure(DepthSolver* pDepthSolver);

        ~MISimilarityMeasure();

    public:
        void SetSimilarityScoreType(SimilarityScoreType type);

        SimilarityScoreType GetSimilarityScoreType();

        // 判断邻域微图像的相似性
        void MeasureSimilarityForMI(QuadTreeProblemMapMap::iterator& itrFrame);

        // 判断两个微图像的相似性
        float MeasureSimilarity(cv::Mat& srcImage, cv::Mat& neighImage, QuadTreeTileKeyPtr ptrSrcKey,QuadTreeTileKeyPtr ptrNeighKey);

        // 切割纹理图，并存储到 MLA_problem中
        void Slice_RichnessMLAImage(QuadTreeProblemMapMap::iterator& itrp);

        // 获取每个微图像的邻域图像集合,集合包含两圈邻域微图像
        // TODO 暂时截至到参照中心微图像的第三圈邻域，以后按需设置截至条件为重叠度
        void CollectMIANeighImagesByCircle(MLA_Problem& problem);
        QuadTreeTileKeyPtrVec CollectMIANeighImagesByBaseline(MLA_Problem& problem,LightFieldParams& lf_Params);

    private:
        //SSIM 衡量结构相似性
        double MeasureSimilarityBySSIM(cv::Mat srcImage,cv::Mat neighborImage);

        //衡量两个物体轮廓的相似性,使用opencv的函数接口
        double MeasureSimilarityByHu(cv::Mat srcImage,cv::Mat neighborImage,QuadTreeTileKeyPtr KeyPtr,QuadTreeTileKeyPtr Neigh_Key);

        //把轮廓当作复数序列，做傅里叶变换
        double MeasureSimilarityByFD(cv::Mat srcImage,cv::Mat neighborImage);
        std::vector<double> calcFourierDescriptors(cv::Mat image,int number);

        //豪斯多夫距离，提取两幅图像的轮廓特征点集，再计算对应点的豪斯多夫距离
        void MeasureSimilarityByHausdorff();

        //形状上下文，精确匹配，但计算量大
        void MeasureSimilarityByShapeContext();

        double MeasureSimilarityByRichness(cv::Mat srcImage,cv::Mat neighborImage);

        double MeasureSimilarityByShift(cv::Mat srcImage,cv::Mat neighborImage,QuadTreeTileKeyPtr KeyPtr,QuadTreeTileKeyPtr Neigh_Key);

    private: // 测试代码

        void TestDrawCircleNeighLines(QuadTreeTileKeyPtr ptrCenterKey, QuadTreeTileKeyPtrVec& circle_Keys);


    private:
        //定义变量
        LightFieldParams                        m_Params;

        DepthSolver*                            m_ptrDepthSolver;
        std::string                             m_strRootPath;
        QuadTreeTileKeyPtrVec                   m_NeigKeyPtrVec;

        QuadTreeProblemMapMap                   m_MIA_problem_map_map;

        SimilarityScoreType                     m_SimilarityScoreType;

        QuadTreeTileKeyPtrVec                   candidatePtrKeyVec;
        QuadTreeTileKeyPtrVec                   ptrKeyVec_1;
        QuadTreeTileKeyPtrVec                   ptrKeyVec_2;
        QuadTreeTileKeyPtrVec                   ptrKeyVec_3;
        QuadTreeTileKeyPtrVec                   ptrKeyVec_4;

        QuadTreeTileKeyPtrVec                   thresholdNeighKeyPtrVec;

    };


}