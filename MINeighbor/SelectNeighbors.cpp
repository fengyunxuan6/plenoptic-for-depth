/********************************************************************
file base:      SelectNeighbors.cpp
author:         LZD
created:        2025/01/13
purpose:
*********************************************************************/
#include "SelectNeighbors.h"

#include "random"
#include "boost/filesystem.hpp"
#include "Util/Logger.h"

namespace LFMVS
{
    SelectNeighbors::SelectNeighbors(DepthSolver* pDepthSolver)
    {
        m_ptrDepthSolver = std::make_shared<DepthSolver>(*pDepthSolver);
        m_bTest = false;
    }

    SelectNeighbors::~SelectNeighbors()
    {
        m_ptrDepthSolver = NULL;
    }

    // 邻域选择--融合用
    void SelectNeighbors::CollectMIANeighImagesForRefocus(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("SN: CollectMIANeighImagesForRefocus, Begin");
        // 初始化粗略相似性测量的对象
        m_ptrImage_SM = std::make_shared<MISimilarityMeasure>(m_ptrDepthSolver.get());
        m_ptrImage_SM->SetSimilarityScoreType(SST_Hu);

        // 为每个微图像构建邻域微图像集合
        QuadTreeProblemMap& problem_map = itrFrame->second;
// #pragma omp parallel for
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++)
        {
            QuadTreeTileKeyPtr ptrKey = itr->first;
            MLA_Problem& problem = itr->second;
            QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
            QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(ptrKey);
            if (itrInfo == MLA_info_map.end())
                continue;

            MLA_InfoPtr ptrInfo = itrInfo->second;
            m_MaxBaseline = 0;
            if(ptrKey->GetTileX() == 0 && ptrKey->GetTileY() == 9) // 测试用
            {
                int kk=1;
            }

            // 条件1：此处应该根据纹理丰富性判断当前微图像是否应该作为下一步（视差匹配）的数据源，
            // 若不应该，则无需为其找邻居图像集合，直接处理下一张微图像
            cv::Mat srcImage = problem.m_Image_gray.clone();
            // if ((float)problem.m_RichnessValue/(srcImage.rows*srcImage.cols) < g_Invalid_Match_ratio)
            // {
            //     problem.m_bNeedMatch = false; // 无需视差匹配
            //     continue;
            // }

            // 由内向外，逐圈选择best邻域图像
            int circle_index = 0;
            QuadTreeTileKeyPtrCircles candidateNeighKeysMap;
            QuadTreeTileKeysMapFast candidateNeighKeysFast; // 为了快速查找
            candidateNeighKeysFast[ptrKey] = ptrKey;
            QuadTreeTileKeyPtrVec selectedNeighKeysMap;
            bool bStop = false;
            const int maxCircle = 3;
            while (!bStop && circle_index < maxCircle)
            {
                if(circle_index > 0)
                {
                  bStop = RansacNeighborKey(problem.m_NeighScoreMapForRefocus);
                }

                if (bStop)
                    break;
                // step1: 搜集第circle_index圈的候选邻域图像集合
                CollectCircleNeighborsKey(ptrKey, circle_index, candidateNeighKeysMap, candidateNeighKeysFast);
                // step2: 计算该圈上所有微图像作为邻居的score
                QuadTreeTileKeyPtrVec& candidate_circle_Keys = candidateNeighKeysMap[circle_index] ;
                ComputeCandidateNeighborScoresForRefocus(problem, circle_index, candidate_circle_Keys, itrFrame);
                circle_index++;
            }

            problem.m_NeigDistance_range_forMatch.y = m_MaxBaseline;
           // float  m_MaxBaseline = 288;

            ComputeNeighborScoresForRefocus(problem, circle_index, itrFrame, m_MaxBaseline);

            // 查找完当前微图像的所有邻域图像集合，此时根据score对邻居“best”(可匹配性)排序
            problem.SortNeighScoreForRefocus();

            // test
            bool bTestWrite = false;
            if (bTestWrite)
            {

                if (ptrKey->GetTileX()==56 && ptrKey->GetTileY()==9)
                {
                    problem.WriteNeighbosInfoForRefocus();
                    m_ptrDepthSolver->TestRawImageTilekeyWithSortNeighForRefocus(bTestWrite, ptrKey, problem);

                    //m_ptrDepthSolver->TestRawImageTilekeyWithCircleLine(bTestWrite, ptrKey, candidateNeighKeysMap);
                }
            }
        }
        LOG_ERROR("SN: CollectMIANeighImagesForRefocus, End");
    }

    // 邻域选择--视差匹配用
    void SelectNeighbors::CollectMIANeighImagesForMatch(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        LOG_ERROR("SN: CollectMIANeighImagesForMatch, Begin");

        // 初始化粗略相似性测量的对象
        m_ptrImage_SM = std::make_shared<MISimilarityMeasure>(m_ptrDepthSolver.get());
        m_ptrImage_SM->SetSimilarityScoreType(SST_Hu);

        // 为每个微图像构建邻域微图像集合
        QuadTreeProblemMap& problem_map = itrFrame->second;
#pragma omp parallel for schedule(dynamic)
        for (int id_problem = 0; id_problem < problem_map.size(); id_problem++)
        {
            QuadTreeProblemMap::iterator itrP = problem_map.begin();
            std::advance(itrP, id_problem);

            QuadTreeTileKeyPtr ptrKey = itrP->first;
            MLA_Problem& problem = itrP->second;
            QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
            QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(ptrKey);
            if (itrInfo == MLA_info_map.end())
                continue;

            MLA_InfoPtr ptrInfo = itrInfo->second;

            // 条件1：此处应该根据纹理丰富性判断当前微图像是否应该作为下一步（视差匹配）的数据源，
            // 若不应该，则无需为其找邻居图像集合，直接处理下一张微图像
            cv::Mat refImage = problem.m_Image_gray.clone();
            // if ((float)problem.m_RichnessValue/(refImage.rows*refImage.cols) < g_Invalid_Match_ratio)
            // {
            //     problem.m_bNeedMatch = false;
            //     continue;
            // }

            // 由内向外，逐圈选择best邻域图像
            int circle_index = 0;
            QuadTreeTileKeyPtrCircles candidateNeighKeysMap;
            QuadTreeTileKeysMapFast candidateNeighKeysFast; // 为了快速查找
            candidateNeighKeysFast[ptrKey] = ptrKey;
            QuadTreeTileKeyPtrVec selectedNeighKeysMap;
            bool bStop = false;
            const int maxCircle = 5;
            while (!bStop && circle_index<maxCircle)
            {
                if (circle_index==2)
                {
                    problem.RansacNeighborKeyForMatch(m_ptrDepthSolver->GetLightFieldParams());
                }

                // step1: 搜集第circle_index圈的候选邻域图像集合
                CollectCircleNeighborsKey(ptrKey, circle_index, candidateNeighKeysMap, candidateNeighKeysFast);
                // step2: 计算该圈上所有微图像作为邻居的score
                QuadTreeTileKeyPtrVec& candidate_circle_Keys = candidateNeighKeysMap[circle_index] ;
                bStop = ComputeCandidateNeighborScoresForMatch(problem, circle_index, candidate_circle_Keys, itrFrame);

                if (circle_index>1 && bStop)
                    break;
                circle_index++;
            }
            ComputeNeighborScoresForMatch(problem, circle_index, itrFrame, m_MaxBaseline);

            // 查找完当前微图像的所有邻域图像集合，此时根据score对邻居“best”(可匹配性)排序
            problem.SortNeighScoreForMatch();

            // test
            bool bTestWrite = false;
            if (bTestWrite)
            {
#pragma omp critical
                {
                    problem.WriteNeighbosInfoForMatch();
                    // if (ptrKey->GetTileX()==47 && ptrKey->GetTileY()==12)
                    // {
                    //     m_ptrDepthSolver->TestRawImageTilekeyWithSortNeighForMatch(bTestWrite, ptrKey, problem);
                    //     //m_ptrDepthSolver->TestRawImageTilekeyWithCircleLine(bTestWrite, ptrKey, candidateNeighKeysMap);
                    // }
                }
            }
        }
        LOG_ERROR("SN: CollectMIANeighImagesForMatch, End");
    }

    void SelectNeighbors::ComputeNearestCircleKeys(QuadTreeTileKeyPtrVec& centerKeys,
        QuadTreeTileKeyPtrVec& neareastKeys,
        QuadTreeTileKeysMapFast& candidateNeighKeysFast)
    {
        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();
        float fDelt = 0.15;
        fDelt *= lf_Params.baseline;

        for (int i = 0; i < centerKeys.size(); i++)
        {
            QuadTreeTileKeyPtr ptrKey = centerKeys.at(i);

            // test
            if (ptrKey->GetTileX()==50 && ptrKey->GetTileY()==49)
            {
                int kk=0;
            }

            QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
            QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(ptrKey);
            if (itrInfo == MLA_info_map.end())
            {
                continue;
            }
            MLA_InfoPtr ptrInfo = itrInfo->second;
            cv::Point2f& fCenter = ptrInfo->GetCenter();

            // 上面一行
            int y_center_up = ptrKey->GetTileY()-1;
            if (y_center_up > -1)
            {
                int x_center_up = ptrKey->GetTileX();
                QuadTreeTileKeyPtr ptrKey_neig_up =  QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                    x_center_up, y_center_up);

                // 找到了上行的起始key
                if( ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig_up,
                    neareastKeys, candidateNeighKeysFast, fDelt) )
                {
                    ConfirmNearestNeighKeysByHorizontal(ptrKey, fCenter, ptrKey_neig_up,
                                                        neareastKeys, candidateNeighKeysFast, fDelt);
                }
                else
                {
                    // 先往左找
                    int up_x_center_left = ptrKey->GetTileX()-1;
                    QuadTreeTileKeyPtr ptrKey_neig_up_left = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                        up_x_center_left, y_center_up);
                    if( ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig_up_left,
    neareastKeys, candidateNeighKeysFast, fDelt) )
                    {
                        ConfirmNearestNeighKeysByHorizontal(ptrKey, fCenter,
                            ptrKey_neig_up_left, neareastKeys, candidateNeighKeysFast, fDelt);
                    }
                    else
                    {
                        // 再往右找
                        int up_x_center_right = ptrKey->GetTileX()+1;
                        QuadTreeTileKeyPtr ptrKey_neig_up_right = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                            up_x_center_right, y_center_up);
                        if( ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig_up_right,
        neareastKeys, candidateNeighKeysFast, fDelt) )
                        {
                            ConfirmNearestNeighKeysByHorizontal(ptrKey, fCenter,
                                ptrKey_neig_up_right, neareastKeys, candidateNeighKeysFast, fDelt);
                        }
                        // else
                        // {
                        //     LOG_WARN("Not find neigh up1, cur-key: " , ptrKey->StrRemoveLOD().c_str()
                        //     , "Neig key: ", ptrKey_neig_up_right->StrRemoveLOD().c_str());
                        // }
                    }
                }
            }

            // 自身一行
            {
                // 向左遍历
                int x_left_step = -1;
                int x_left = ptrKey->GetTileX();
                int y_left = ptrKey->GetTileY();
                bool bSearch_left = true;
                while (bSearch_left)
                {
                    x_left += x_left_step;
                    if (x_left <= -1)
                    {
                        bSearch_left = false;
                        break;
                    }

                    QuadTreeTileKeyPtr ptrKey_neig =  QuadTreeTileKey::CreateInstance(TileKey_None, 0,
    x_left, y_left);
                    if(!ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig,
        neareastKeys, candidateNeighKeysFast, fDelt) )
                    {
                        bSearch_left = false;
                    }
                }

                // 向右遍历
                int x_right_step = 1;
                int x_right = ptrKey->GetTileX();
                int y_right = ptrKey->GetTileY();
                bool bSearch_right = true;
                while (bSearch_right)
                {
                    x_right += x_right_step;
                    if (x_right >= lf_Params.mla_u_size)
                    {
                        bSearch_right = false;
                        break;
                    }

                    QuadTreeTileKeyPtr ptrKey_neig = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
    x_right, y_right);
                    if(!ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig,
neareastKeys, candidateNeighKeysFast, fDelt) )
                    {
                        bSearch_right = false;
                    }
                }
            }

            // 下面一行
            int y_center_down = ptrKey->GetTileY() + 1;
            if (y_center_down < lf_Params.mla_v_size)
            {
                int x_center_down = ptrKey->GetTileX();
                QuadTreeTileKeyPtr ptrKey_neig_down =  QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                    x_center_down, y_center_down);

                // 找到了上行的起始key
                if( ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig_down,
                    neareastKeys, candidateNeighKeysFast, fDelt) )
                {
                    ConfirmNearestNeighKeysByHorizontal(ptrKey, fCenter, ptrKey_neig_down,
                                                        neareastKeys, candidateNeighKeysFast, fDelt);
                }
                else
                {
                    // 先往左找
                    int down_x_center_left = ptrKey->GetTileX()-1;
                    QuadTreeTileKeyPtr ptrKey_neig_down_left =  QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                        down_x_center_left, y_center_down);
                    if( ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig_down_left,
    neareastKeys, candidateNeighKeysFast, fDelt) )
                    {
                        ConfirmNearestNeighKeysByHorizontal(ptrKey, fCenter,
                            ptrKey_neig_down_left, neareastKeys, candidateNeighKeysFast, fDelt);
                    }
                    else
                    {
                        // 再往右找
                        int down_x_center_right = ptrKey->GetTileX()+1;
                        QuadTreeTileKeyPtr ptrKey_neig_down_right = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
                            down_x_center_right, y_center_down);
                        if( ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig_down_right,
        neareastKeys, candidateNeighKeysFast, fDelt) )
                        {
                            ConfirmNearestNeighKeysByHorizontal(ptrKey, fCenter,
                                ptrKey_neig_down_right, neareastKeys, candidateNeighKeysFast, fDelt);
                        }
                        // else
                        // {
                        //     LOG_WARN("Not find neigh down1, cur-key: ", ptrKey->StrRemoveLOD().c_str(),
                        //         "Neig key: ", ptrKey_neig_down_right->StrRemoveLOD().c_str());
                        // }
                    }
                }
            }
        }
    }

    void SelectNeighbors::CollectCircleNeighborsKey(QuadTreeTileKeyPtr ptrKey,
        int current_circle_index,
        QuadTreeTileKeyPtrCircles& candidateNeighKeysMap,
        QuadTreeTileKeysMapFast& candidateNeighKeysFast)
    {
        QuadTreeTileKeyPtrVec centerKeys;
        QuadTreeTileKeyPtrVec neareastNeighKeys;
        if (current_circle_index == 0)
        {
            centerKeys.push_back(ptrKey);
        }
        else
        {
            centerKeys = candidateNeighKeysMap[current_circle_index-1];
        }

        ComputeNearestCircleKeys(centerKeys, neareastNeighKeys, candidateNeighKeysFast);
        // 压入
        candidateNeighKeysMap[current_circle_index] = neareastNeighKeys;
    }

    bool SelectNeighbors::ConfirmNearestNeighKeys(QuadTreeTileKeyPtr ptrKey, cv::Point2f& fCenter,
                        QuadTreeTileKeyPtr ptrNeighKey,
                        QuadTreeTileKeyPtrVec& neareastKeys,
                        QuadTreeTileKeysMapFast& candidateNeighKeysFast, float fDelt)
    {
        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        QuadTreeTileInfoMap::iterator itrInfo_Neig = MLA_info_map.find(ptrNeighKey);
        if (itrInfo_Neig != MLA_info_map.end())
        {
            MLA_InfoPtr ptrInfo_Neig = itrInfo_Neig->second;
            cv::Point2f& fCenter_neig = ptrInfo_Neig->GetCenter();

            // 计算距离，判断是否符合条件
            float dist = std::sqrt(std::pow(fCenter.x-fCenter_neig.x,2) +
                             std::pow(fCenter.y-fCenter_neig.y,2));

            if (abs(dist-lf_Params.baseline) < fDelt)
            {
                if (candidateNeighKeysFast.find(ptrNeighKey) == candidateNeighKeysFast.end())
                {
                    candidateNeighKeysFast[ptrNeighKey] = ptrNeighKey;
                    neareastKeys.push_back(ptrNeighKey);
                }
                return true;
            }
            return false;
        }
        return false;
    }

    // 候选邻域--融合用
    void SelectNeighbors::ComputeCandidateNeighborScoresForRefocus(MLA_Problem& curr_problem,
                                                                   int circle_index,
                                                                   QuadTreeTileKeyPtrVec& candidate_circle_Keys,
                                                                   QuadTreeProblemMapMap::iterator& itrFrame)
    {
        // 计算score时，需考虑的因素包括：基线、粗略相似性（类似sift)、模糊程度值

        QuadTreeProblemMap& problem_map = itrFrame->second;
        QuadTreeTileKeyPtr ptrKey = curr_problem.m_ptrKey;
        cv::Mat srcImage = curr_problem.m_Image_gray.clone();

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(curr_problem.m_ptrKey);
        MLA_InfoPtr ptrInfo = itrInfo->second;
        float center_X = ptrInfo->GetCenter().x;
        float center_Y = ptrInfo->GetCenter().y;

        //筛选fSimilarityScore异常值
        for (int i = 0; i < candidate_circle_Keys.size(); i++)
        {
            QuadTreeTileKeyPtr ptrNeighKey = candidate_circle_Keys.at(i);
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeighKey);

            MLA_Problem& problemNeigh = itrNP->second;
            if (problemNeigh.m_Image_gray.empty())
                continue;
            cv::Mat image_Neigh = problemNeigh.m_Image_gray.clone();

            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_BlureValue = problemNeigh.m_BlurenessValue;

            // 计算(光照一致性:illuminationConsist) 并存储到结构体里
            float fSimilarityScore = m_ptrImage_SM->MeasureSimilarity(srcImage,image_Neigh,ptrKey,ptrNeighKey);
            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Similarity = fSimilarityScore;
            cv::Mat srcImage = curr_problem.m_Image_gray.clone();
            float fPhotographicTerm = fSimilarityScore / (srcImage.rows*srcImage.cols);
            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].fPhotographicTerm = fPhotographicTerm;

            // 计算src与neigh基线并存储
            QuadTreeTileInfoMap::iterator neighItrInfo = MLA_info_map.find(ptrNeighKey);
            MLA_InfoPtr neighPtrInfo = neighItrInfo->second;
            float neighCenter_X = neighPtrInfo->GetCenter().x;
            float neighCenter_Y = neighPtrInfo->GetCenter().y;
            float baseline = std::sqrt(std::pow(neighCenter_X-center_X,2)+std::pow(neighCenter_Y-center_Y,2));
            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Baseline = baseline;
            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Circle_index = circle_index;
        }


        /* for (int i = 0; i < candidate_circle_Keys.size(); i++)
          {
              QuadTreeTileKeyPtr ptrNeighKey = candidate_circle_Keys.at(i);
              QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeighKey);

              if (itrNP == problem_map.end())
                  continue;

             // 计算src与neigh基线并存储
            //  QuadTreeTileInfoMap::iterator neighItrInfo = MLA_info_map.find(ptrNeighKey);
           //   MLA_InfoPtr neighPtrInfo = neighItrInfo->second;
           //   float neighCenter_X = neighPtrInfo->GetCenter().x;
           //   float neighCenter_Y = neighPtrInfo->GetCenter().y;
           //   float baseline = std::sqrt(std::pow(neighCenter_X-center_X,2)+std::pow(neighCenter_Y-center_Y,2));
           //   curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Baseline = baseline;

              MLA_Problem& problemNeigh = itrNP->second;
              cv::Mat& image_Neigh = problemNeigh.m_Image_gray;

              // 子项A：函数(光照一致性:illuminationConsist) 并存储到结构体里
              *//*float fSimilarityScore = m_ptrImage_SM->MeasureSimilarity(srcImage,image_Neigh,ptrKey,ptrNeighKey);
             curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Similarity = fSimilarityScore;*//*
             float fSimilarityScore = curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Similarity;
             float fPhotographicTerm = fSimilarityScore / (srcImage.rows*srcImage.cols);

             // 子项B：函数（几何一致性:geometricConsist）
            // 最大不超过1
             float baseline = curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Baseline;
             float m = 0.001;
             //float fGeometricTerm = std::exp(-m * baseline); // 使用指数衰减函数计算几何一致性
               float fGeometricTerm = std::exp(-m * baseline); // 使用指数衰减函数计算几何一致性


             // 子项C：（模糊一致性:blurConsist）
             float fScaleTerm = 1.0;
             unsigned long blurenessValue = curr_problem.m_BlurenessValue;
             unsigned long blurenessValue_neigh = problemNeigh.m_BlurenessValue;
             float r = 0.0;
             if(blurenessValue != 0) // 防止分母为0
             {
                 r = static_cast<float>(blurenessValue_neigh) / blurenessValue;
             }
             if(0.8 < r < 1.2)
             {
                 fScaleTerm = 1.0;
             }
             else if(r > 1.2)
             {
                 fScaleTerm = 1.2 / r;
             }
             else
             {
                 fScaleTerm = r;
             }

             // 计算最终得分:m_Score，该值越大，则作为邻居图像就越好
             // 权重：基线权重最大
             float a = 0.25;
             float b = 0.50;
             float c = 0.25;
             float score = a * fPhotographicTerm + b * fGeometricTerm + c * fScaleTerm;
             curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Score = score;  // 存储最终得分
             curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Circle_index = circle_index;
         }*/

        // 根据该圈的邻居计算的指标值，判断是否停止继续向外扩散找邻域图像
        //     float a_weight=0.4;
        //     float total_score = a_weight*fSimilarityScore + (1-a_weight)*B;
        //
        //     if(fSimilarityScore < 0.3*(srcImage.rows*srcImage.cols))
        //     {
        //         thresholdNeighKeyPtrVec.push_back(ptrNeighKey);
        //     }        //     float a_weight=0.4;
        //     float total_score = a_weight*fSimilarityScore + (1-a_weight)*B;
        //
        //     if(fSimilarityScore < 0.3*(srcImage.rows*srcImage.cols))
        //     {
        //         thresholdNeighKeyPtrVec.push_back(ptrNeighKey);
        //     }
        //
        // if (thresholdNeighKeyPtrVec.size() > candidate_circle_Keys.size()*0.8)
        // {
        //     bProcess = false;
        // }
    }

    //候选邻域--视差匹配用
    bool SelectNeighbors::ComputeCandidateNeighborScoresForMatch(MLA_Problem& curr_problem,
                                                int circle_index,
                                                QuadTreeTileKeyPtrVec& candidate_circle_Keys,
                                                QuadTreeProblemMapMap::iterator& itrFrame)
    {
        QuadTreeProblemMap& problem_map = itrFrame->second;
        QuadTreeTileKeyPtr ptrKey = curr_problem.m_ptrKey;
        cv::Mat srcImage = curr_problem.m_Image_gray.clone();

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(curr_problem.m_ptrKey);
        MLA_InfoPtr ptrInfo = itrInfo->second;
        float center_X = ptrInfo->GetCenter().x;
        float center_Y = ptrInfo->GetCenter().y;

        // 筛选fSimilarityScore异常值
        int outlier_num = 0;
        for (int i = 0; i < candidate_circle_Keys.size(); i++)
        {
            QuadTreeTileKeyPtr ptrNeighKey = candidate_circle_Keys.at(i);
            QuadTreeProblemMap::iterator itrQP = problem_map.find(ptrNeighKey);
            if (itrQP == problem_map.end())
            {
                //std::cout<<"neig_problem not exist, key: "<<ptrNeighKey->StrRemoveLOD().c_str()<<std::endl;
                continue;
            }

            MLA_Problem& problemNeigh = itrQP->second;
            // if ((float)problemNeigh.m_RichnessValue/(problemNeigh.m_Image_gray.rows*problemNeigh.m_Image_gray.cols) < 0.15)
            // {
            //     continue;
            // }
            cv::Mat image_Neigh = problemNeigh.m_Image_gray.clone();

            // 计算(光照一致性:illuminationConsist)
            // 计算src与neigh基线并存储
            float fSimilarityScore = m_ptrImage_SM->MeasureSimilarity(srcImage,image_Neigh,ptrKey,ptrNeighKey);
            QuadTreeTileInfoMap::iterator neighItrInfo = MLA_info_map.find(ptrNeighKey);
            if (neighItrInfo == MLA_info_map.end())
                continue;
            MLA_InfoPtr ptrNgInfo = neighItrInfo->second;
            float neighCenter_X = ptrNgInfo->GetCenter().x;
            float neighCenter_Y = ptrNgInfo->GetCenter().y;
            float baseline = std::sqrt(std::pow(neighCenter_X-center_X,2)+std::pow(neighCenter_Y-center_Y,2));

            // 异常点剔除
            bool bOutlier = curr_problem.OutlierImpByRANSAC(m_ptrDepthSolver->GetLightFieldParams(), baseline, fSimilarityScore);
            if (bOutlier)
            {
                outlier_num++;
            }
            else
            {
                curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].m_BlureValue = problemNeigh.m_BlurenessValue;
                curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].m_Similarity = fSimilarityScore;
                float fPhotographicTerm = fSimilarityScore / (srcImage.rows*srcImage.cols);
                curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].fPhotographicTerm = fPhotographicTerm;
                curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].m_Baseline = baseline;
                curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].m_Circle_index = circle_index;

                // 统计最大最小值
                if (curr_problem.m_NeigDistance_range_forMatch.x > baseline)
                {
                    curr_problem.m_NeigDistance_range_forMatch.x = baseline;
                }
                if (curr_problem.m_NeigDistance_range_forMatch.y < baseline)
                {
                    curr_problem.m_NeigDistance_range_forMatch.y = baseline;
                }

                if (curr_problem.m_PhotographicValue_range_forMatch.x > fPhotographicTerm)
                {
                    curr_problem.m_PhotographicValue_range_forMatch.x = fPhotographicTerm;
                }
                if (curr_problem.m_PhotographicValue_range_forMatch.y < fPhotographicTerm)
                {
                    curr_problem.m_PhotographicValue_range_forMatch.y = fPhotographicTerm;
                }
            }
        }
        if (outlier_num >= 2)
            return true;
        return false;
    }

    //计算score--融合用
    void SelectNeighbors::ComputeNeighborScoresForRefocus(MLA_Problem& curr_problem, int circle_index, QuadTreeProblemMapMap::iterator& itrFrame, float base)
    {
        // 计算score时，需考虑的因素包括：基线、粗略相似性（类似sift)、模糊程度值

        QuadTreeProblemMap& problem_map = itrFrame->second;
        QuadTreeTileKeyPtr ptrKey = curr_problem.m_ptrKey;
        cv::Mat srcImage = curr_problem.m_Image_gray.clone();

        /*QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(curr_problem.m_ptrKey);
        MLA_InfoPtr ptrInfo = itrInfo->second;
        float center_X = ptrInfo->GetCenter().x;
        float center_Y = ptrInfo->GetCenter().y;*/

        //对相似度值归一化
        float minPhotographicValue = std::numeric_limits<float>::max();
        float maxPhotographicValue = std::numeric_limits<float>::lowest();
        for (NeighScoreMap::iterator itrNeigh = curr_problem.m_NeighScoreMapForRefocus.begin(); itrNeigh != curr_problem.m_NeighScoreMapForRefocus.end(); itrNeigh ++)
        {
            float fPhotographicValue = itrNeigh->second.fPhotographicTerm;
            if (fPhotographicValue < minPhotographicValue) {
                minPhotographicValue = fPhotographicValue;
            }
            if (fPhotographicValue > maxPhotographicValue) {
                maxPhotographicValue = fPhotographicValue;
            }
        }
        float range = maxPhotographicValue - minPhotographicValue;
        if (range == 0.0f) {
            return; // 所有值相同，无需归一化
        }
        for (NeighScoreMap::iterator itrNeigh = curr_problem.m_NeighScoreMapForRefocus.begin(); itrNeigh != curr_problem.m_NeighScoreMapForRefocus.end(); itrNeigh ++)
        {
            itrNeigh->second.fPhotographicTerm = (itrNeigh->second.fPhotographicTerm - minPhotographicValue) / range;
        }


        for (NeighScoreMap::iterator itrNeigh = curr_problem.m_NeighScoreMapForRefocus.begin(); itrNeigh != curr_problem.m_NeighScoreMapForRefocus.end(); itrNeigh ++)
        {
            QuadTreeTileKeyPtr ptrNeighKey = itrNeigh->first;
            QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeighKey);

            if (itrNP == problem_map.end())
                continue;

            /*// 计算src与neigh基线并存储
             QuadTreeTileInfoMap::iterator neighItrInfo = MLA_info_map.find(ptrNeighKey);
             MLA_InfoPtr neighPtrInfo = neighItrInfo->second;
             float neighCenter_X = neighPtrInfo->GetCenter().x;
             float neighCenter_Y = neighPtrInfo->GetCenter().y;
             float baseline = std::sqrt(std::pow(neighCenter_X-center_X,2)+std::pow(neighCenter_Y-center_Y,2));
             curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Baseline = baseline;*/

            MLA_Problem& problemNeigh = itrNP->second;
            cv::Mat image_Neigh = problemNeigh.m_Image_gray.clone();

            // 子项A：函数(光照一致性:illuminationConsist) 并存储到结构体里
            /*float fSimilarityScore = m_ptrImage_SM->MeasureSimilarity(srcImage,image_Neigh,ptrKey,ptrNeighKey);
            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Similarity = fSimilarityScore;*/
           // float fSimilarityScore = curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Similarity;
           // float fPhotographicTerm = fSimilarityScore / (srcImage.rows*srcImage.cols);
            float fPhotographicTerm = curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].fPhotographicTerm;

            //增加一个相似度筛选
            /*if(fPhotographicTerm < 0.3)
            {
                curr_problem.m_NeighScoreMapForRefocus.erase(ptrNeighKey);
                std::cout<<"删除："<<ptrNeighKey<<"--"<<fPhotographicTerm<<std::endl;
                continue;
            }*/

            // 子项B：函数（几何一致性:geometricConsist）
            // 最大不超过1
            float baseline = curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Baseline;
           // float m = 0.001;
            // float fGeometricTerm = std::exp(-m * baseline); // 使用指数衰减函数计算几何一致性
            // float fGeometricTerm = std::exp(-m * baseline); // 使用指数衰减函数计算几何一致性
            // base:满足要求的最大基线；baseline：当前基线长度；自适应几何一致性
            float m = 0.000005;
            float n = 27;
            int h = 0;
            if((baseline >= base))
                h = 1;
            else
                h = 0;
            float  fGeometricTerm = std::exp(-m * std::pow(baseline - base ,2)* (1 + n * h)) ;


            // 子项C：（模糊一致性:blurConsist）
            float fScaleTerm = 1.0;
            unsigned long blurenessValue = curr_problem.m_BlurenessValue;
            unsigned long blurenessValue_neigh = problemNeigh.m_BlurenessValue;
            float r = 0.0;
            if(blurenessValue != 0) // 防止分母为0
            {
                r = static_cast<float>(blurenessValue_neigh) / blurenessValue;
            }
            if(0.8 < r < 1.2)
            {
                fScaleTerm = 1.0;
            }
            else if(r > 1.2)
            {
                fScaleTerm = 1.2 / r;
            }
            else
            {
                fScaleTerm = r;
            }

            // 计算最终得分:m_Score，该值越大，则作为邻居图像就越好
            // 权重：基线权重最大
            float a = 0.50;
            float b = 0.30;
            float c = 0.20;
            float score = 0;
         //   score =  fSimilarityScore ;
            score = a * fPhotographicTerm + b * fGeometricTerm + c * fScaleTerm;

            curr_problem.m_NeighScoreMapForRefocus[ptrNeighKey].m_Score = score;  // 存储最终得分

           // std::cout<<ptrNeighKey->GetTileX()<<"-"<<ptrNeighKey->GetTileY()<<":("<<baseline<<"-"<<score<<")"<<std::endl;
        }
    }

    // 计算score--视差匹配用
    void SelectNeighbors::ComputeNeighborScoresForMatch(MLA_Problem& curr_problem,
                    int circle_index,QuadTreeProblemMapMap::iterator& itrFrame,float base)
     {
         // 计算score时，需考虑的因素包括：基线、粗略相似性（类似sift)、模糊程度值

         QuadTreeProblemMap& problem_map = itrFrame->second;
         QuadTreeTileKeyPtr ptrKey = curr_problem.m_ptrKey;
        // 归一化参数
        curr_problem.ItemsNormalization();

         for (NeighScoreMap::iterator itrNeigh = curr_problem.m_NeighScoreMapForMatch.begin(); itrNeigh != curr_problem.m_NeighScoreMapForMatch.end(); itrNeigh ++)
         {
             QuadTreeTileKeyPtr ptrNeighKey = itrNeigh->first;
             QuadTreeProblemMap::iterator itrNP = problem_map.find(ptrNeighKey);

             if (itrNP == problem_map.end())
                 continue;

             MLA_Problem& problemNeigh = itrNP->second;
             cv::Mat image_Neigh = problemNeigh.m_Image_gray.clone();

             // 子项A：基线长度
             float fBaselineTerm = curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].m_Baseline_normalization;

             // 子项B：函数(光照一致性:illuminationConsist) 并存储到结构体里
             float fPhotographicTerm = curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].fPhotographicTerm;

             // 子项C：（模糊一致性:blurConsist）
             float fBlurScaleTerm = 1.0;
             unsigned long blurenessValue = curr_problem.m_BlurenessValue;
             unsigned long blurenessValue_neigh = problemNeigh.m_BlurenessValue;
             float r = 0.0;
             if(blurenessValue != 0) // 防止分母为0
             {
                 r = static_cast<float>(blurenessValue_neigh) / blurenessValue;
             }
             if(0.8 < r < 1.2)
             {
                 fBlurScaleTerm = 1.0;
             }
             else if(r > 1.2)
             {
                 fBlurScaleTerm = 1.2 / r;
             }
             else
             {
                 fBlurScaleTerm = r;
             }

             // 计算最终得分:m_Score，该值越大，则作为邻居图像就越好
             // 权重：基线权重最大 (1.0, 0.0, 0.0)
             float a = 1.00; // 0.50 todo:lzd 0902
             float b = 0.00; // 0.30
             float c = 0.00; // 0.20
             // float a = 0.50; // 0.50 todo:lzd 0902
             // float b = 0.30; // 0.30
             // float c = 0.20; // 0.20

             float score = a*(1.0-fBaselineTerm) + b*fPhotographicTerm + c*fBlurScaleTerm;
             // float score = a*(fBaselineTerm) + b*fPhotographicTerm + c*fBlurScaleTerm;
             curr_problem.m_NeighScoreMapForMatch[ptrNeighKey].m_Score = score;  // 存储最终得分
         }
     }

    bool SelectNeighbors::RansacNeighborKey(NeighScoreMap& neighScoreMap)
    {
        bool bStop = false;
        if(neighScoreMap.size() < 2)
            return true;

        struct RansacStruct
        {
            float baseline;
            float similarity;
            QuadTreeTileKeyPtr neighKey;
        };
        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();
        float errorThreshold = 0.1;  // 控制误差容忍度
        float baselineThreshold = 2 * (lf_Params.baseline * 1.05);  //选择拟合的数量,并允许一定的误差
        std::vector<RansacStruct> ransacMap;
        std::vector<QuadTreeTileKeyPtr> keyVec;

        for (auto itrNeigh = neighScoreMap.begin(); itrNeigh != neighScoreMap.end(); ++itrNeigh)
        {
            keyVec.push_back(itrNeigh->first);
        }

        for (size_t i = 0; i < keyVec.size(); ++i)
        {
            QuadTreeTileKeyPtr neighKey = keyVec[i];
            float baseline = neighScoreMap[neighKey].m_Baseline;
            float similarity = neighScoreMap[neighKey].m_Similarity;
            ransacMap.push_back({baseline, similarity, neighKey });
        }

        // 构建点集用于拟合
        std::vector<cv::Point2f> points;
        for (const auto& item : ransacMap)
        {
            if(item.baseline<baselineThreshold)
                points.emplace_back(item.baseline, item.similarity);
        }

        // 使用 OpenCV 拟合一条线性趋势线
        /*cv::Vec4f lineParams;
        cv::fitLine(points, lineParams, cv::DIST_L2, 0, 0.01, 0.01);

        float vx = lineParams[0], vy = lineParams[1];
        float x0 = lineParams[2], y0 = lineParams[3];
        float a = vy / vx;
        float b = y0 - a * x0;*/

        // ---------- 简单线性回归 ----------
        float sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        for (const auto& p : points) {
            sum_x += p.x;
            sum_y += p.y;
            sum_xx += p.x * p.x;
            sum_xy += p.x * p.y;
        }

        int N = points.size();
        float denominator = N * sum_xx - sum_x * sum_x;
        if (fabs(denominator) < 1e-6) {
            return true;
        }

        float a = (N * sum_xy - sum_x * sum_y) / denominator;  // slope
        float b = (sum_y - a * sum_x) / N;

        for (size_t i = 0; i < ransacMap.size(); ++i)
        {
            float baseline = ransacMap[i].baseline;
            float similarity = ransacMap[i].similarity;

            float predicted = a * baseline + b;
            float error = fabs(similarity - predicted);
            error /= similarity;

            // 基线大，且similarity 明显大于预测值，则剔除
            float baselineCritical = 1 * (lf_Params.baseline * 1.05);  //选择拟合的数量,并允许一定的误差
            if (baseline > baselineCritical && error > errorThreshold)
            {
                bStop = true;
                neighScoreMap.erase(ransacMap[i].neighKey);
                m_MaxBaseline = ransacMap[i].baseline;
                // std::cout << "[剔除异常点] " << ransacMap[i].neighKey->GetTileX() << "-"
                //           << ransacMap[i].neighKey->GetTileY()
                //           << " similarity: (" << similarity << ")"
                //           << " 预测值: " << predicted <<"error/similarity:"<<error<< std::endl;
            }
        }
        return bStop;
    }

    void SelectNeighbors::ConfirmNearestNeighKeysByHorizontal(QuadTreeTileKeyPtr ptrKey,
                                                              cv::Point2f& fCenter,
                                                              QuadTreeTileKeyPtr ptrNeighKey,
                                                              QuadTreeTileKeyPtrVec& neareastKeys,
                                                              QuadTreeTileKeysMapFast& candidateNeighKeysFast, float fDelt)
    {
        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();
        int tile_y = ptrNeighKey->GetTileY();

        // 向左遍历
        int x_left_step = -1;
        int tile_x_left = ptrNeighKey->GetTileX();
        bool bSearch_left = true;
        while (bSearch_left)
        {
            tile_x_left += x_left_step; // 逐个遍历
            if (tile_x_left <= -1)
            {
                bSearch_left = false;
                break;
            }
            QuadTreeTileKeyPtr ptrKey_neig = QuadTreeTileKey::CreateInstance(TileKey_None, 0,
tile_x_left, tile_y);

            if(!ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig,
                    neareastKeys, candidateNeighKeysFast, fDelt) )
            {
                bSearch_left = false;
            }
        }

        // 向右遍历
        int x_right_step = 1;
        int tile_x_right = ptrNeighKey->GetTileX();
        bool bSearch_right = true;
        while (bSearch_right)
        {
            tile_x_right += x_right_step;
            if (tile_x_right >= lf_Params.mla_u_size)
            {
                bSearch_right = false;
                break;
            }

            QuadTreeTileKeyPtr ptrKey_neig =  QuadTreeTileKey::CreateInstance(TileKey_None, 0,
tile_x_right, tile_y);

            if(!ConfirmNearestNeighKeys(ptrKey, fCenter, ptrKey_neig,
                    neareastKeys, candidateNeighKeysFast, fDelt) )
            {
                bSearch_right = false;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    void SelectNeighbors::SelectNeighborsForMIA(QuadTreeProblemMapMap::iterator& itrPM,
                                                eSelectNeighborsType eSelectType)
    {
        if (m_ptrDepthSolver == nullptr)
            return;
        std::map<std::string, cv::Mat>& RawImageMap = m_ptrDepthSolver->GetRawImageMap();
        QuadTreeProblemMapMap& MLAProblemsMap = m_ptrDepthSolver->GetMIAProblemsMapMap();
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        // 当前要选择邻居的微图像集合
        std::string strNameLessExt = itrPM->first;
        QuadTreeProblemMap& problem_map = itrPM->second;

        // 遍历微透镜
        for (QuadTreeTileInfoMap::iterator itr = MLA_info_map.begin(); itr != MLA_info_map.end(); itr++)
        {
            QuadTreeTileKeyPtr ptrKey = itr->first;
            MLA_InfoPtr ptrInfo = itr->second;
            if (ptrInfo->IsAbandonByArea() == true)
                continue;
            QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
            if (itrP == problem_map.end())
                continue;

            MLA_Problem& problem = itrP->second;
            switch (eSelectType)
            {
            case eSNT_FixedPosition: // 固定位置的邻居
                {
                    FixedPositionAlgorithm(problem);
                }
                break;
            case eSNT_Features:
                {
                    FeatureSimilartyMeasureAlgorithm(problem, ptrInfo);
                }
                break;
            case eSNT_Gradients:
                {
                    GradientSimilartyMeasureAlgorithm(problem);
                }
                break;
            default:
                break;
            }
        }

        m_bTest = true;
        if (m_bTest)
        {
            TestSelectNeighborsMIC(itrPM);
        }
    }

    void SelectNeighbors::FixedPositionAlgorithm(MLA_Problem& problem)
    {
        /* 选择固定位置的邻居，规则如下：
           1. 邻居总数为6
           2. 根据前微透镜图像所处的阵列的行列位置，分为一般和特殊两类。其中，特殊类又分为4个边和4个角
        */

        // step1: 按照固定模式1，选择微图像作为邻居
        SelectFixedPositionMIByMode1(problem);
        // step2: 对搜集到邻居进行排序
        SortMIANeighborKeyByDistance(problem); // 按距离(在一定程度上反映的是重叠度)
        // step3: 确定哪些problem不需要深度估计
        if (m_ptrDepthSolver)
        {
            m_ptrDepthSolver->ConfirmProblemForEstimation(problem);
        }
    }

    void SelectNeighbors::SelectFixedPositionMIByMode1(MLA_Problem& problem)
    {
        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
        int32_t tile_X = ptrKey->GetTileX();
        int32_t tile_Y = ptrKey->GetTileY();

        LightFieldParams& lf_Params= m_ptrDepthSolver->GetLightFieldParams();

        /* 注： key的编号坐标为：x轴从左到右为正方向且依次增大， y轴从上到下为正方向且依次增大*/
        if (tile_X > 1 && tile_X < lf_Params.mla_u_size-2 && tile_Y > 0 && tile_Y < lf_Params.mla_v_size-1)
        {
            // 邻居：右
            CollectValidNeighborKey(problem, 1, 0);
            // 邻居：左
            CollectValidNeighborKey(problem, -1, 0);
            // 邻居：右2
            CollectValidNeighborKey(problem, 2, 0);
            // 邻居：上
            CollectValidNeighborKey(problem, 0, -1);
            // 邻居 : 上，左二
            CollectValidNeighborKey(problem, -2, -1);
            // 邻居：下
            CollectValidNeighborKey(problem, 0, 1);
            // 邻居 : 左二 lzd
            CollectValidNeighborKey(problem, -2, 0);
            // 邻居 : 上，左一 lzd
            CollectValidNeighborKey(problem, -1, -1);
            // 邻居 : 上，右一 lzd
            CollectValidNeighborKey(problem, 1, -1);
            // 邻居 : 下，左一 lzd
            CollectValidNeighborKey(problem, -1, 1);
            // 邻居 : 下，右一 lzd
            CollectValidNeighborKey(problem, 1, 1);
            // 邻居 : 下，右二 lzd
            CollectValidNeighborKey(problem, 2, 1);
        }
        else
        {
            if (tile_X == 0 && tile_Y == 0) // 左上角微图像
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：右下
                CollectValidNeighborKey(problem, 1, 1);
                // 邻居：右2 下2
                CollectValidNeighborKey(problem, 2, 2);
                // 邻居： 下2
                CollectValidNeighborKey(problem, 0, 2);
            }
            else if ((tile_X == 1 && tile_Y == 0) || ((tile_X != lf_Params.mla_u_size-2 ||
                tile_X != lf_Params.mla_u_size-1) && tile_Y == 0))
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：左下
                CollectValidNeighborKey(problem, -1, 1);
                // 邻居：右下
                CollectValidNeighborKey(problem, 1, 1);
            }
            else if (tile_X == 0 && tile_Y == lf_Params.mla_v_size-1)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居：右上
                CollectValidNeighborKey(problem, 1, -1);
                // 邻居：右2上2
                CollectValidNeighborKey(problem, 2, -2);
                // 邻居：上2
                CollectValidNeighborKey(problem, 0, -2);
            }
            else if (tile_X == 1 && tile_Y == lf_Params.mla_v_size-1)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居：左上
                CollectValidNeighborKey(problem, -1, -1);
                // 邻居：右上
                CollectValidNeighborKey(problem, 1, -1);
            }
            else if (tile_X == lf_Params.mla_u_size-1 && tile_Y == lf_Params.mla_v_size-1)
            {
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居: 上，左二
                CollectValidNeighborKey(problem, -2, -1);
                // 邻居：左2
                CollectValidNeighborKey(problem, -2, 0);
                // 邻居：上2
                CollectValidNeighborKey(problem, 0, -2);
                // 邻居：左上
                CollectValidNeighborKey(problem, -1, -1);
            }
            else if (tile_X == lf_Params.mla_u_size-2 && tile_Y == lf_Params.mla_v_size-1)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居: 上，左二
                CollectValidNeighborKey(problem, -2, -1);
                // 邻居：左上
                CollectValidNeighborKey(problem, -1, -1);
                // 邻居：右上
                CollectValidNeighborKey(problem, 1, -1);
            }
            else if (tile_X == lf_Params.mla_u_size-1 && tile_Y == 0)
            {
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：左2 下2
                CollectValidNeighborKey(problem, -2, 2);
                // 邻居：左2
                CollectValidNeighborKey(problem, -2, 0);
                // 邻居： 下2
                CollectValidNeighborKey(problem, 0, 2);
                // 邻居：左下
                CollectValidNeighborKey(problem, -1, 1);
            }
            else if (tile_X == lf_Params.mla_u_size-2 && tile_Y == 0)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：左2
                CollectValidNeighborKey(problem, -2, 0);
                // 邻居：左下
                CollectValidNeighborKey(problem, -1, 1);
                // 邻居：右下
                CollectValidNeighborKey(problem, 1, 1);
            }
            else if (tile_X == 0)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：右上
                CollectValidNeighborKey(problem, 1, -1);
                // 邻居：右下
                CollectValidNeighborKey(problem, 1, 1);
            }
            else if (tile_X == 1)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：左上
                CollectValidNeighborKey(problem, -1, -1);
            }
            else if (tile_Y == lf_Params.mla_v_size-1)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：右2
                CollectValidNeighborKey(problem, 2, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居 : 上，左二
                CollectValidNeighborKey(problem, -2, -1);
                // 邻居：右上
                CollectValidNeighborKey(problem, 1, -1);
            }
            else if (tile_X == lf_Params.mla_u_size-1)
            {
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居 : 上，左二
                CollectValidNeighborKey(problem, -2, -1);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：左上
                CollectValidNeighborKey(problem, -1, -1);
                // 邻居：左下
                CollectValidNeighborKey(problem, -1, 1);
            }
            else if (tile_X == lf_Params.mla_u_size-2)
            {
                // 邻居：右
                CollectValidNeighborKey(problem, 1, 0);
                // 邻居：左
                CollectValidNeighborKey(problem, -1, 0);
                // 邻居：上
                CollectValidNeighborKey(problem, 0, -1);
                // 邻居 : 上，左二
                CollectValidNeighborKey(problem, -2, -1);
                // 邻居：下
                CollectValidNeighborKey(problem, 0, 1);
                // 邻居：右下
                CollectValidNeighborKey(problem, 1, 1);
            }
        }
    }

    void SelectNeighbors::FeatureSimilartyMeasureAlgorithm(MLA_Problem& problem, MLA_InfoPtr ptrInfo)
    {
        bool bStop = false;
        int iCircle_num = 0;
        QuadTreeTileKeyPtrCircles keyPtrCircles;

        // 以参考微图像为中心点，由内向外按照圆圈扩散，选择一圈候选邻居然后进行特征提取与匹配
        while (bStop)
        {
            SelectMIByCircles(problem, ptrInfo, iCircle_num, keyPtrCircles);

            bStop = true;
        }

    }

    void SelectNeighbors::SelectMIByCircles(MLA_Problem& problem, MLA_InfoPtr ptrInfo,
                                int iCircle_num, QuadTreeTileKeyPtrCircles& keyPtrCircles)
    {
        cv::Point2f& center = ptrInfo->GetCenter();

        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
        int32_t tile_X = ptrKey->GetTileX();
        int32_t tile_Y = ptrKey->GetTileY();

        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();
        float radiance_radius = iCircle_num*lf_Params.baseline;


    }

    void SelectNeighbors::GradientSimilartyMeasureAlgorithm(MLA_Problem& problem)
    {
        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
        int32_t tile_X = ptrKey->GetTileX();
        int32_t tile_Y = ptrKey->GetTileY();

        return;
    }

    void SelectNeighbors::CollectValidNeighborKey(MLA_Problem& problem, int x_offset, int y_offset)
    {
        if (!m_ptrDepthSolver)
            return;
        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey; // 当前微图像的key
        if (!ptrKey)
            return;

        // 创建邻居图像的key
        QuadTreeTileKeyPtr ptrKey_Neighbor = ptrKey->CreateNeighborKey(x_offset, y_offset);

        // 判断有效的微透镜集合是否存在ptrKey_Neighbor对应的微透镜
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itr = MLA_info_map.find(ptrKey_Neighbor);
        if (itr != MLA_info_map.end())
        {
            MLA_InfoPtr ptrInfo = itr->second;
            if (ptrInfo->IsAbandonByArea() == false)
            {
                problem.m_NeigKeyPtrVec.push_back(ptrKey_Neighbor);
            }
        }
    }

    void SelectNeighbors::SortMIANeighborKeyByDistance(MLA_Problem& problem)
    {
        if (!m_ptrDepthSolver)
            return;

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileKeyPtr ptrCurKey = problem.m_ptrKey;

        QuadTreeTileInfoMap::iterator itr = MLA_info_map.find(ptrCurKey);
        if (itr == MLA_info_map.end())
        {
            std::cout<<"SNKBD: current key not find! " << ptrCurKey->GetTileX()
            << ", " << ptrCurKey->GetTileY() <<std::endl;
            return;
        }
        problem.CreateResTileKeysFromNeiKeyVec(itr->second->GetCenter(), MLA_info_map);
    }

    void SelectNeighbors::TestSelectNeighborsMIC(QuadTreeProblemMapMap::iterator& itrPM)
    {
        if (!m_ptrDepthSolver)
            return;
        cv::Mat backGround_Image = m_ptrDepthSolver->GetWhiteImage().clone();
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        // 当前要选择邻居的微图像集合
        std::string strNameLessExt = itrPM->first;
        QuadTreeProblemMap& problems_map = itrPM->second;

        // 生成随机数
        int count = 20; // 数量
        std::random_device rd;
        std::mt19937 gen(rd());

        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();

        // 设置参考点和邻居点的颜色
        cv::Scalar refColor(0, 0, 255); // 红色
        cv::Scalar neighborColor(255, 0, 0); // 黄色
        for (int i=0; i<count; i++)
        {
            std::uniform_int_distribution<> dis_tileX(1, lf_Params.mla_u_size-1);
            int tile_x = dis_tileX(gen);

            std::uniform_int_distribution<> dis_tileY(1, lf_Params.mla_v_size-1);
            int tile_y = dis_tileY(gen);

            QuadTreeTileKeyPtr ptrKey = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_x, tile_y);
            QuadTreeTileInfoMap::iterator itr = MLA_info_map.find(ptrKey);
            if (itr == MLA_info_map.end())
                continue;
            QuadTreeProblemMap::iterator itrP = problems_map.find(ptrKey);
            if (itrP == problems_map.end())
                continue;

            MLA_InfoPtr ptrInfo = itr->second;
            MLA_Problem& problem = itrP->second;

            float radius = 15;
            // 绘制参考点圆（红色）
            cv::circle(backGround_Image, ptrInfo->GetCenter(), radius, refColor, cv::FILLED);
            // 绘制邻居点圆（黄色）
            Res_image_KeyVec& res_img_vec = problem.m_Res_Image_KeyVec;
            for (size_t idx = 0; idx < res_img_vec.size(); idx++)
            {
                Res_image_Key& res_img = res_img_vec[idx];
                QuadTreeTileKeyPtr ptrNeigKey = res_img.m_ptrKey;
                if (!ptrNeigKey)
                {
                    std::cout<<"ptrNeigKey is Empty!"<<std::endl;
                    continue;
                }
                QuadTreeTileInfoMap::iterator itrN = MLA_info_map.find(ptrNeigKey);
                if (itrN == MLA_info_map.end())
                    continue;
                MLA_InfoPtr ptrNeigInfo = itrN->second;
                cv::circle(backGround_Image, ptrNeigInfo->GetCenter(), radius, neighborColor, cv::FILLED);
            }
        }

        // 写出图像
        std::string m_strRawImagePath = m_ptrDepthSolver->GetRootPath() + LF_DEPTH_INTRA_NAME;
        boost::filesystem::path path(m_strRawImagePath);
        std::string strPath = path.parent_path().string();
        std::string strTestNeigPath = strPath + "/" + strNameLessExt + "/TestNeig.png";
        bool bSave = cv::imwrite(strTestNeigPath, backGround_Image);
        if (bSave)
        {
            std::cout << "TestNeig Image saved successfully" << std::endl;
        }
    }
}
