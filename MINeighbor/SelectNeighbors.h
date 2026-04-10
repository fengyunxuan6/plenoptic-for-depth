/********************************************************************
file base:      SelectNeighbors.h
author:         LZD
created:        2025/01/13
purpose:
*********************************************************************/
#ifndef LFMVS_SELECTNEIGHBORS_H
#define LFMVS_SELECTNEIGHBORS_H

#include "DepthSolver.h"

#include "Util/MISimilarityMeasure.h"

namespace LFMVS
{
    class SelectNeighbors
    {
    public:
        SelectNeighbors(DepthSolver* pDepthSolver);

        ~SelectNeighbors();

    public:
        // 邻域选择--融合用
        void CollectMIANeighImagesForRefocus(QuadTreeProblemMapMap::iterator& itrFrame);

        // 邻域选择--视差匹配用
        void CollectMIANeighImagesForMatch(QuadTreeProblemMapMap::iterator& itrFrame);

        void SelectNeighborsForMIA(QuadTreeProblemMapMap::iterator& itrPM, eSelectNeighborsType eSelectType);

    private:
        void CollectCircleNeighborsKey(QuadTreeTileKeyPtr ptrKey, int current_circle_index,
    QuadTreeTileKeyPtrCircles& candidateNeighKeysMap, QuadTreeTileKeysMapFast& candidateNeighKeysFast);

        void ComputeNearestCircleKeys(QuadTreeTileKeyPtrVec& centerKeys, QuadTreeTileKeyPtrVec& neareastKeys,
            QuadTreeTileKeysMapFast& candidateNeighKeysFast);

        // 同行遍历
        void ConfirmNearestNeighKeysByHorizontal(QuadTreeTileKeyPtr ptrKey, cv::Point2f& fCenter,
                            QuadTreeTileKeyPtr ptrNeighKey,
                            QuadTreeTileKeyPtrVec& neareastKeys,
                            QuadTreeTileKeysMapFast& candidateNeighKeysFast, float fDelt);

        bool ConfirmNearestNeighKeys(QuadTreeTileKeyPtr ptrKey, cv::Point2f& fCenter,
                                    QuadTreeTileKeyPtr ptrNeighKey,
                                    QuadTreeTileKeyPtrVec& neareastKeys,
                                    QuadTreeTileKeysMapFast& candidateNeighKeysFast, float fDelt);

        // 融合用
        void ComputeCandidateNeighborScoresForRefocus(MLA_Problem& curr_problem,
                                                      int circle_index, QuadTreeTileKeyPtrVec& candidate_circle_Keys,
                                                      QuadTreeProblemMapMap::iterator& itrFrame);
        //视差匹配用
        bool ComputeCandidateNeighborScoresForMatch(MLA_Problem& curr_problem,
                                                      int circle_index, QuadTreeTileKeyPtrVec& candidate_circle_Keys,
                                                      QuadTreeProblemMapMap::iterator& itrFrame);

        void ComputeNeighborScoresForRefocus(MLA_Problem& curr_problem,
                                             int circle_index,
                                             QuadTreeProblemMapMap::iterator& itrFrame, float base);
        void ComputeNeighborScoresForMatch(MLA_Problem& curr_problem,
                                   int circle_index,
                                   QuadTreeProblemMapMap::iterator& itrFrame,float base);

        void KmeanNeighborKey(NeighScoreMap& m_NeighScoreMap);
        bool RansacNeighborKey(NeighScoreMap& neighScoreMap);

    private:
        // 选择邻域微图像: 固定位置算法
        void FixedPositionAlgorithm(MLA_Problem& problem);
        void SelectFixedPositionMIByMode1(MLA_Problem& problem);

        // 选择邻域微图像: 特征点提取与匹配算法
        void FeatureSimilartyMeasureAlgorithm(MLA_Problem& problem, MLA_InfoPtr ptrInfo);
        void SelectMIByCircles(MLA_Problem& problem, MLA_InfoPtr ptrInfo, int iCircle_num,
                            QuadTreeTileKeyPtrCircles& keyPtrCircles);

        // 选择邻域微图像: 梯度灰度局部块匹配算法
        void GradientSimilartyMeasureAlgorithm(MLA_Problem& problem);

        void CollectValidNeighborKey(MLA_Problem& problem, int x_offset, int y_offset);
        void SortMIANeighborKeyByDistance(MLA_Problem& problem);

    private: // 测试
        void TestSelectNeighborsMIC(QuadTreeProblemMapMap::iterator& itrPM);

    private:
        std::shared_ptr<DepthSolver>            m_ptrDepthSolver;
        std::shared_ptr<MISimilarityMeasure>    m_ptrImage_SM;
        bool                                    m_bTest;
        float                                   m_MaxBaseline;
    };
}
#endif // LFMVS_SELECTNEIGHBORS_H
