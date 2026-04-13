/********************************************************************
file base:      LFDepth.cpp
author:         LZD
created:        2024/05/13
purpose:
*********************************************************************/
#include "LFDepthInfo.h"
#include "Common/QuadTree.h"
#include "Util/Logger.h"

namespace LFMVS
{
    namespace
    {
        static inline float SafeDenomCPU(float v)
        {
            if (fabsf(v) < 1e-6f)
                return (v >= 0.0f) ? 1e-6f : -1e-6f;
            return v;
        }

        // 与 CUDA 版 DisparityPlane(p, plane_hypothesis) 保持一致
        static inline float3 DisparityPlaneCPU(const int2 p, const float4& plane_hypothesis)
        {
            const float nz = SafeDenomCPU(plane_hypothesis.z);

            float3 plane;
            plane.x = -plane_hypothesis.x / nz; // alpha
            plane.y = -plane_hypothesis.y / nz; // beta
            plane.z = (plane_hypothesis.x * p.x +
                       plane_hypothesis.y * p.y +
                       plane_hypothesis.z * plane_hypothesis.w) / nz; // gamma
            return plane;
        }

        // 把 disparity plane: d(x,y)=alpha*x+beta*y+gamma
        // 重新编码回当前工程使用的 plane_hypothesis = normal + anchor disparity
        static inline float4 EncodePlaneHypothesisFromDispPlaneCPU(const int2 anchor,
                                                                   const float alpha,
                                                                   const float beta,
                                                                   const float gamma)
        {
            float nx = -alpha;
            float ny = -beta;
            float nz =  1.0f;

            float norm = sqrtf(nx * nx + ny * ny + nz * nz);
            if (norm < 1e-6f)
                norm = 1e-6f;

            nx /= norm;
            ny /= norm;
            nz /= norm;

            const float d_anchor = alpha * anchor.x + beta * anchor.y + gamma;
            return make_float4(nx, ny, nz, d_anchor);
        }
    }

    void MLA_Problem::SetM_NeigKeyPtrVec(QuadTreeTileKeyPtrVec& NeigKeyPtrVec)
    {
        m_NeigKeyPtrVec = NeigKeyPtrVec;
    }

    void MLA_Problem::ComputeBlurenessValue()
    {
        m_BlurenessValue = 0;

        int cols = m_Image_Blureness_Bianry.cols;
        int rows = m_Image_Blureness_Bianry.rows;
#pragma omp parallel for schedule(dynamic)
        for (int col = 0; col < cols; col++)
        {
            for (int row = 0; row < rows; row++)
            {
                uint8 v = m_Image_Blureness_Bianry.at<uint8>(row, col);
                if (v == 255)
                {
                    #pragma omp critical
                    m_BlurenessValue++;
                }
            }
        }
    }

    void MLA_Problem::ComputeRichnessValue()
    {
        m_RichnessValue = 0;

        int cols = m_Image_Richness.cols;
        int rows = m_Image_Richness.rows;
#pragma omp parallel for schedule(dynamic)
        for (int col = 0; col < cols; col++)
        {
            for (int row = 0; row < rows; row++)
            {
                uint8 v = m_Image_Richness.at<uint8>(row, col);
                if (v == 255)
                {
#pragma omp critical
                    m_RichnessValue++;
                }
            }
        }
    }

    void MLA_Problem::SortNeighScoreForRefocus()
    {
        std::vector<std::pair<QuadTreeTileKeyPtr, float>> scoreVec;
        for(NeighScoreMap::iterator itrScore = m_NeighScoreMapForRefocus.begin(); itrScore != m_NeighScoreMapForRefocus.end(); ++itrScore)
        {
            QuadTreeTileKeyPtr ptrNeighKey = itrScore->first;
            float mScore = itrScore->second.m_Score;
            scoreVec.push_back(std::make_pair(ptrNeighKey,mScore));
        }
        std::sort(scoreVec.begin(),scoreVec.end(),[](const auto& a,const auto& b)
        {
            return a.second>b.second;
        });

        for(int i = 0;i < scoreVec.size();i++)
        {
            QuadTreeTileKeyPtr ptrKey = scoreVec[i].first;
            m_NeighsSortVecForRefocus.push_back(ptrKey);

            // 存储best邻居的排序
            NeighScoreMap::iterator itrScore = m_NeighScoreMapForRefocus.find(ptrKey);
            if (itrScore != m_NeighScoreMapForRefocus.end())
            {
                Neigh_Score& n_score = itrScore->second;
                n_score.m_SortIndex = i;
            }
        }
    }

    void MLA_Problem::RansacNeighborKeyForMatch(LightFieldParams& lf_Params)
    {
        if(m_NeighScoreMapForMatch.size() < 2)
            return;

        struct RansacStruct
        {
            float baseline;
            float similarity;
            QuadTreeTileKeyPtr neighKey;
        };

        float baselineThreshold = 2 * (lf_Params.baseline * 1.05);  //选择拟合的数量,并允许一定的误差
        std::vector<RansacStruct> ransacMap;

        for (NeighScoreMap::iterator itrNeigh = m_NeighScoreMapForMatch.begin(); itrNeigh != m_NeighScoreMapForMatch.end(); ++itrNeigh)
        {
            QuadTreeTileKeyPtr neighKey = itrNeigh->first;
            float baseline = itrNeigh->second.m_Baseline;
            float similarity = itrNeigh->second.m_Similarity;
            ransacMap.push_back({baseline, similarity, neighKey});
        }

        // 构建点集用于拟合
        std::vector<cv::Point2f> points;
        for (const auto& item : ransacMap)
        {
            if(item.baseline < baselineThreshold)
            {
                points.emplace_back(item.baseline, item.similarity);
            }
        }

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
        if (fabs(denominator) < 1e-6)
        {
            std::cout<<"ransac error"<<std::endl;
            return;
        }

        m_ransac_ab_forMatch.x = (N * sum_xy - sum_x * sum_y) / denominator;  // slope
        m_ransac_ab_forMatch.y = (sum_y - m_ransac_ab_forMatch.x * sum_x) / N;
        m_bComputRANSAC = true;
    }

    bool MLA_Problem::OutlierImpByRANSAC(LightFieldParams& lf_Params, float baseline, float similarity)
    {
        if (!m_bComputRANSAC)
        {
            return false;
        }

        float errorThreshold = 0.1;  // 控制误差容忍度

        float predicted = m_ransac_ab_forMatch.x * baseline + m_ransac_ab_forMatch.y;
        float error = fabs(similarity - predicted);
        error /= similarity;

        // 基线大，且similarity 明显大于预测值，则剔除
        float baselineCritical = 1 * (lf_Params.baseline * 1.05);  //选择拟合的数量,并允许一定的误差
        if (baseline > baselineCritical && error > errorThreshold)
        {
            return true;
        }
        return false;
    }

    // 排序--视差匹配用
    void MLA_Problem::SortNeighScoreForMatch()
    {
        std::vector<std::pair<QuadTreeTileKeyPtr, float>> scoreVec;
        for(NeighScoreMap::iterator itrScore = m_NeighScoreMapForMatch.begin(); itrScore != m_NeighScoreMapForMatch.end(); ++itrScore)
        {
            QuadTreeTileKeyPtr ptrNeighKey = itrScore->first;
            float mScore = itrScore->second.m_Score;
            scoreVec.push_back(std::make_pair(ptrNeighKey,mScore));
        }
        std::sort(scoreVec.begin(),scoreVec.end(),[](const auto& a,const auto& b)
        {
            return a.second>b.second;
        });

        for(int i = 0;i < scoreVec.size();i++)
        {
            QuadTreeTileKeyPtr ptrKey = scoreVec[i].first;
            m_NeighsSortVecForMatch.push_back(ptrKey);

            // 存储best邻居的排序
            NeighScoreMap::iterator itrScore = m_NeighScoreMapForMatch.find(ptrKey);
            if (itrScore != m_NeighScoreMapForMatch.end())
            {
                Neigh_Score& n_score = itrScore->second;
                n_score.m_SortIndex = i;
            }
        }
    }

    NeighScoreMap& MLA_Problem::GetNeighScoreMapForRefocus()
    {
        return m_NeighScoreMapForRefocus;
    }

    QuadTreeTileKeyPtrVec& MLA_Problem::GetSortedNeighScoreForRefocus()
    {
        return m_NeighsSortVecForRefocus;
    }

    NeighScoreMap& MLA_Problem::GetNeighScoreMapForMatch()
    {
        return m_NeighScoreMapForMatch;
    }

    QuadTreeTileKeyPtrVec& MLA_Problem::GetSortedNeighScoreForMatch()
    {
        return m_NeighsSortVecForMatch;
    }

    void MLA_Problem::WriteNeighbosInfoForRefocus()
    {
        LOG_WARN("Current key= ", m_ptrKey->Str().c_str(), ", neighbors size= ", m_NeighsSortVecForRefocus.size());
        LOG_WARN("Neighbor: (Order, key), (score, Circle_index, BlureValue, Similarity, Baseline)");
        for (int i = 0; i < m_NeighsSortVecForRefocus.size(); i++)
        {
            QuadTreeTileKeyPtr ptrNeighKey = m_NeighsSortVecForRefocus.at(i);
            Neigh_Score& neigh_score = m_NeighScoreMapForRefocus[ptrNeighKey];
            LOG_INFO("(", i, ", ", ptrNeighKey->StrRemoveLOD().c_str(), ")");
            LOG_INFO("(", neigh_score.m_Score, ", ", neigh_score.m_Circle_index, ", ",
                neigh_score.m_BlureValue, ", ", neigh_score.m_Similarity, ", ", neigh_score.m_Baseline);
        }
    }

    void MLA_Problem::WriteNeighbosInfoForMatch()
    {
        LOG_WARN("Current key= ", m_ptrKey->Str().c_str(), ", neighbors size= ", m_NeighsSortVecForMatch.size());
        LOG_WARN("Neighbor: (Order, key), (score, Circle_index, BlureValue, Similarity, Baseline)");
        for (int i = 0; i < m_NeighsSortVecForMatch.size(); i++)
        {
            QuadTreeTileKeyPtr ptrNeighKey = m_NeighsSortVecForMatch.at(i);
            Neigh_Score& neigh_score = m_NeighScoreMapForMatch[ptrNeighKey];
            LOG_INFO("(", i, ", ", ptrNeighKey->StrRemoveLOD().c_str(), ")");
            LOG_INFO("(", neigh_score.m_Score, ", ", neigh_score.m_Circle_index, ", ",
                neigh_score.m_BlureValue, ", ", neigh_score.m_Similarity, ", ", neigh_score.m_Baseline);
        }
    }

    void MLA_Problem::WriteNeighbosInfo_old()
    {
        LOG_WARN("Current key= ", m_ptrKey->Str().c_str(), ", neighbors size= ", m_Res_Image_KeyVec.size());
        LOG_WARN("Neighbor: (Order, key)");
        for (int i = 0; i < m_Res_Image_KeyVec.size(); i++)
        {
            QuadTreeTileKeyPtr ptrNeighKey = m_Res_Image_KeyVec.at(i).m_ptrKey;

            // TODO-0625
            //m_NeighsSortVecForMatch.push_back(m_Res_Image_KeyVec.at(i).m_ptrKey);
            LOG_INFO("(", i, ",", ptrNeighKey->StrRemoveLOD().c_str(), ")");
        }
        // TODO-0625
        //m_Res_Image_KeyVec.clear();
    }

    ////////////////////////////////////////////////////

    Res_image_Key::Res_image_Key(LFMVS::QuadTreeTileKeyPtr ptrKey, float baseline)
    {
        m_ptrKey = ptrKey;
        Base_line = baseline;
        m_iLevel = 0;
        res_number = 0;
        NCC_pointNum_valid = 0;
        NCC_Average_Score_valid = 0.0;
    }

    void Res_image_Key::ReSet()
    {
        m_ptrKey.reset();
        m_iLevel = 0;
        res_number = 0;
        NCC_pointNum_valid = 0;
        NCC_Average_Score_valid = 0.0;
    }

    void Res_image_Key::ComputeValidNCCInfo()
    {
        for (int i = 0; i < Ncc_grade.size(); i++)
        {
            if (Ncc_grade[i] > 0.8)
            {
                NCC_pointNum_valid++;
                NCC_Average_Score_valid += Ncc_grade[i];
            }
        }
        if (NCC_pointNum_valid > 0)
        {
            NCC_Average_Score_valid /= NCC_pointNum_valid;
        }
    }
    ////////////////////////////////////////////////////
    MLA_Problem::MLA_Problem()
    : m_Variance(0.0)
    , m_Standard_deviation(0.0)
    , m_bGarbage(false)
    , m_RichnessValue(0)
    , m_BlurenessValue(0)
    , m_bNeedMatch(true)
    {
        m_ransac_ab_forMatch.x = 0;
        m_ransac_ab_forMatch.y = 0;
        m_bComputRANSAC = false;

        m_NeigDistance_range_forMatch.x = 0.0;
        m_NeigDistance_range_forMatch.y = 0.0;
        m_PhotographicValue_range_forMatch.x = 0.0;
        m_PhotographicValue_range_forMatch.y = 0.0;
    }

    MLA_Problem::~MLA_Problem()
    {
        Release();
    }

    void MLA_Problem::Release()
    {
        res_img.clear();
        m_Res_Image_KeyVec.clear();
        number.clear();
        m_NeigKeyPtrVec.clear();
        result_vec.clear();

        // 释放图像数据
        m_Image_gray.release();
        m_Image_rgb.release();
        m_Image_Blureness_Bianry.release();
        m_Image_Blureness.release();
        m_Image_Richness.release();

        // 释放邻居评分相关数据
        m_NeighScoreMapForRefocus.clear();
        m_NeighsSortVecForRefocus.clear();
        m_NeighScoreMapForMatch.clear();
        m_NeighsSortVecForMatch.clear();

        // 释放其他容器
        m_NeigDistance_range_forMatch = make_float2(0.0f, 0.0f);
        m_PhotographicValue_range_forMatch = make_float2(0.0f, 0.0f);
        m_ransac_ab_forMatch = make_float2(0.0f, 0.0f);
        m_bComputRANSAC = false;

        m_ptrKey.reset();
    }

    float MLA_Problem::GetPhotographicValueRangeforMatch()
    {
        return m_PhotographicValue_range_forMatch.y-m_PhotographicValue_range_forMatch.x;
    }

    float MLA_Problem::ComputePhotoRatio(float value)
    {
        float range = GetPhotographicValueRangeforMatch();
        if (range == 0.0f)
            return 0.0;
        float ratio = abs(value-m_PhotographicValue_range_forMatch.x)/range;
        return ratio;
    }

    float MLA_Problem::ComputeBaselineRatio(float value)
    {
        float range = m_NeigDistance_range_forMatch.y-m_NeigDistance_range_forMatch.x;
        if (range == 0.0f)
            return 0.0;
        float ratio = abs(value-m_NeigDistance_range_forMatch.x)/range;
        return ratio;
    }

    void MLA_Problem::ItemsNormalization()
    {
        for (NeighScoreMap::iterator itrNeigh = m_NeighScoreMapForMatch.begin();
            itrNeigh != m_NeighScoreMapForMatch.end(); ++itrNeigh)
        {
            itrNeigh->second.fPhotographicTerm = ComputePhotoRatio(itrNeigh->second.fPhotographicTerm);
            itrNeigh->second.m_Baseline_normalization = ComputeBaselineRatio(itrNeigh->second.m_Baseline);
        }
    }


    void MLA_Problem::CreateResTileKeysFromNeiKeyVec(cv::Point2f& current_center_coord,
                                                     QuadTreeTileInfoMap& MLA_info_map)
    {
        for (int i = 0; i < m_NeigKeyPtrVec.size(); i++)
        {
            QuadTreeTileKeyPtr ptrNeigKey = m_NeigKeyPtrVec[i];
            QuadTreeTileInfoMap::iterator itr_neig = MLA_info_map.find(ptrNeigKey);
            if (itr_neig == MLA_info_map.end())
            {
                std::cout << 'MLA_Images_SortByLength error' << std::endl;
                continue;
            }

            cv::Point2f& neig_center_coord = itr_neig->second->GetCenter();
            float length = sqrt(pow((neig_center_coord.y - current_center_coord.y), 2) +
                                pow((neig_center_coord.x - current_center_coord.x), 2));

            Res_image_Key res_neighbour(ptrNeigKey, length);
            m_Res_Image_KeyVec.push_back(res_neighbour);
        }
        m_NeigKeyPtrVec.clear();
        sort(m_Res_Image_KeyVec.begin(), m_Res_Image_KeyVec.end(), CompareR_Tilekey);
    }

    void MLA_Problem::Compute_avg_std_key(int iLevel)
    {
        int valid_count = 0;
        float avg_NCC_PointNum = 0.0;
        float avg_NCC_Average_Score = 0.0;
        float std_NCC_PointNum = 0.0;
        float std_NCC_Average_Score = 0.0;

        // 平均值
        for (int i =0; i < m_Res_Image_KeyVec.size(); i++)
        {
            Res_image_Key& res_img= m_Res_Image_KeyVec[i];
            if (res_img.NCC_pointNum_valid == 0)
            {
                continue;
            }

            if (res_img.m_iLevel == iLevel)
            {
                valid_count++;
                avg_NCC_PointNum += float(res_img.NCC_pointNum_valid);
                avg_NCC_Average_Score += res_img.NCC_Average_Score_valid;
            }
        }

        if (valid_count == 0)
        {
            result_vec.push_back(0.0);
            result_vec.push_back(0.0);
        }
        else
        {
            result_vec.push_back(avg_NCC_PointNum / float(valid_count));
            result_vec.push_back(avg_NCC_Average_Score / float(valid_count));
        }

        // 方差
        for (int i =0; i < m_Res_Image_KeyVec.size(); i++)
        {
            Res_image_Key& res_img= m_Res_Image_KeyVec[i];
            if (res_img.NCC_pointNum_valid == 0)
            {
                continue;
            }

            if (res_img.m_iLevel == iLevel)
            {
                std_NCC_PointNum += (float(res_img.NCC_pointNum_valid) - result_vec[0]) *
                                    (float(res_img.NCC_pointNum_valid) - result_vec[0]);
                std_NCC_Average_Score += (res_img.NCC_Average_Score_valid - result_vec[1]) *
                                         (res_img.NCC_Average_Score_valid - result_vec[1]);
            }
        }

        if (valid_count == 0)
        {
            result_vec.push_back(0.0);
            result_vec.push_back(0.0);
        }
        else
        {
            // 标准差
            result_vec.push_back(sqrt(std_NCC_PointNum / float(valid_count)));
            result_vec.push_back(sqrt(std_NCC_Average_Score / float(valid_count)));
        }
    }

    void MLA_Problem::ComputeScoreByNCC(int iLevel)
    {
        std::vector<Res_image_Key> res_img_tilekey_tmp;
        for (int i =0; i < m_Res_Image_KeyVec.size(); i++)
        {
            Res_image_Key& res_img= m_Res_Image_KeyVec[i];
            if (iLevel != res_img.m_iLevel)
                continue;

            Res_image_Key res_img_tmp = res_img;
            if (res_img_tmp.Ncc_grade.empty())
            {
                res_img_tmp.score = -10.0;
                continue;
            }
            res_img_tmp.score = 0.5* ((res_img_tmp.NCC_pointNum_valid - result_vec[0]) / result_vec[2]) +
                            0.5* ((res_img_tmp.NCC_Average_Score_valid - result_vec[1]) / result_vec[3]);
            res_img_tilekey_tmp.push_back(res_img_tmp);
        }

        sort(res_img_tilekey_tmp.begin(), res_img_tilekey_tmp.end(), Compare_Res);
        for (int i = 0; i < 3; i++)
        {
            m_NeigKeyPtrVec.push_back(res_img_tilekey_tmp[i].m_ptrKey);
        }
    }

       ///////////////////////////////////////////////////////////////////////
    DisparityAndNormal::DisparityAndNormal(LightFieldParams& params)
        : m_Params(params)
    {
        row = 0;
        col = 0;
        m_StereoStage = eSS_ACMH_Begin;

        c_cuda = new float[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        d_cuda = new float[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        ph_cuda = new float4[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        disp_v_cuda = new float4[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        selected_views = new unsigned int[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        neighbor_Patch_info = new int3[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        neighbor_PGR_info = new int3[m_Params.mi_width_for_match * m_Params.mi_height_for_match];
        m_iDelete_count = 0;
    }

    DisparityAndNormal::~DisparityAndNormal()
    {
        Release();
    }

    void DisparityAndNormal::Release()
    {
        m_iDelete_count++;
        // std::cout<<"~DN: tile_key is: " << m_ptrKey.StrRemoveLOD().c_str() << std::endl;
        // std::cout<<"delte_count is: " << m_iDelete_count << std::endl;

        m_StereoStage = eSS_ACMH_Begin;
        dis.clear();
        nor.clear();
        if (ph_cuda != nullptr)
        {
            delete[] ph_cuda;
            ph_cuda = nullptr;
        }
        if (disp_v_cuda != nullptr)
        {
            delete[] disp_v_cuda;
            disp_v_cuda = nullptr;
        }
        if (c_cuda != nullptr)
        {
            delete[] c_cuda;
            c_cuda = nullptr;
        }
        if (d_cuda != nullptr)
        {
            delete[] d_cuda;
            d_cuda = nullptr;
        }
        if (selected_views != nullptr)
        {
            delete[] selected_views;
            selected_views = nullptr;
        }
        if (neighbor_Patch_info != nullptr)
        {
            delete[] neighbor_Patch_info;
            neighbor_Patch_info = nullptr;
        }
        if (neighbor_PGR_info != nullptr)
        {
            delete[] neighbor_PGR_info;
            neighbor_PGR_info = nullptr;
        }
    }

    bool DisparityAndNormal::IsBroken(const int2 p, const int mi_height,
                                      const int mi_width, const int propagation_Graph_size)
    {
        if (p.x<propagation_Graph_size || p.x>=mi_width-propagation_Graph_size ||
            p.y<propagation_Graph_size || p.y>=mi_height-propagation_Graph_size)
        {
            return true;
        }
        return false;
    }

    void DisparityAndNormal::CollectPropagationGraphLackedPixels(
        const int2 center,
        const int mi_width,
        const int mi_height,
        MLA_Problem& problem,
        QuadTreeTileInfoMap& mla_info_map,
        std::map<QuadTreeTileKeyPtr, std::shared_ptr<DisparityAndNormal>, QuadTreeTileKeyMapCmpLess>& disNormals_map,
        Proxy_DisPlane* proxy_dis_plane)
    {
        const int p_ref = center.y * mi_width + center.x;
        const int3 pgr_info = neighbor_PGR_info[p_ref];

        // 必须取当前像素自己的 PGR 信息，原来直接用 neighbor_PGR_info->x/y/z 是错的
        if (pgr_info.x <= 0)
            return;

        // 当前逻辑仍然只在边界像素上做 repair
        if (!(center.x < 3 || center.x >= (mi_width - 3) ||
              center.y < 3 || center.y >= (mi_height - 3)))
        {
            return;
        }

        MLA_InfoPtr mla_ref_info = mla_info_map[m_ptrKey];
        if (mla_ref_info == NULL)
            return;

        const cv::Point2f mla_ref_cv_center = mla_ref_info->GetCenter();

        // 当前像素对应的“借用邻域微图像”编号（去掉自身占位）
        const int neig_mi_id = pgr_info.x - 1;
        if (neig_mi_id < 0 || neig_mi_id >= (int)problem.m_NeighsSortVecForMatch.size())
            return;

        QuadTreeTileKeyPtr ptrNeigKey = problem.m_NeighsSortVecForMatch[neig_mi_id];
        if (!ptrNeigKey)
            return;

        QuadTreeDisNormalMap::iterator itr_DN = disNormals_map.find(ptrNeigKey);
        if (itr_DN == disNormals_map.end())
        {
            std::cout << "Neighbor DN not found: " << ptrNeigKey->StrRemoveLOD().c_str() << std::endl;
            return;
        }
        DisparityAndNormalPtr ptrDN_neig = itr_DN->second;

        MLA_InfoPtr mla_src_info = mla_info_map[ptrNeigKey];
        if (mla_src_info == NULL)
            return;

        const cv::Point2f mla_src_cv_center = mla_src_info->GetCenter();
        const float2 delta_u_v = make_float2(
            mla_ref_cv_center.x - mla_src_cv_center.x,
            mla_ref_cv_center.y - mla_src_cv_center.y);

        // 当前像素在邻域微图像中的对应点（由第一阶段记录）
        const int2 neig_correspond_coord = make_int2(pgr_info.y, pgr_info.z);
        if (neig_correspond_coord.x < 0 || neig_correspond_coord.x >= mi_width ||
            neig_correspond_coord.y < 0 || neig_correspond_coord.y >= mi_height)
        {
            return;
        }

        const int p_neig = neig_correspond_coord.y * mi_width + neig_correspond_coord.x;
        const float4 plane_hypothesis_neig = ptrDN_neig->ph_cuda[p_neig];

        // 关键修正：
        // 不能直接把 plane_hypothesis_neig.xyz 当 alpha/beta/gamma
        // 必须先在邻域锚点 q 上转成 disparity plane
        const float3 disp_plane_neig = DisparityPlaneCPU(neig_correspond_coord, plane_hypothesis_neig);
        const float alpha = disp_plane_neig.x;
        const float beta  = disp_plane_neig.y;
        const float gamma = disp_plane_neig.z;

        float denom = 1.0f + alpha * delta_u_v.x + beta * delta_u_v.y;
        denom = SafeDenomCPU(denom);

        // 按你原来的跨视图平面变换公式
        const float aR = alpha / denom;
        const float bR = beta  / denom;
        const float cR = gamma / denom;

        // 邻近八个像素
        const int2 positions[8] =
        {
            make_int2(center.x,   center.y - 1), // up_near
            make_int2(center.x,   center.y - 3), // up_far
            make_int2(center.x,   center.y + 1), // down_near
            make_int2(center.x,   center.y + 3), // down_far
            make_int2(center.x - 1, center.y),   // left_near
            make_int2(center.x - 3, center.y),   // left_far
            make_int2(center.x + 1, center.y),   // right_near
            make_int2(center.x + 3, center.y)    // right_far
        };

        for (int i = 0; i < 8; ++i)
        {
            const int2 p = positions[i];

            // 只给越界位置补 proxy plane
            if (p.x < 0 || p.x >= mi_width || p.y < 0 || p.y >= mi_height)
            {
                // 再编码回 plane_hypothesis 形式，而不是直接 make_float4(aR,bR,cR,dR)
                proxy_dis_plane->plane[i] = EncodePlaneHypothesisFromDispPlaneCPU(p, aR, bR, cR);
            }
        }
    }
}
