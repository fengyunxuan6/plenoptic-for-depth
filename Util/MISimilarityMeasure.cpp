/********************************************************************
file base:      MISimilarMeasure.h
author:         LZD
created:        2025/04/25
purpose:        对微图像进行模糊程度的量化
*********************************************************************/

#include <numeric>
#include "MISimilarityMeasure.h"

#include "MVStereo/LFDepthInfo.h"

#include "boost/filesystem.hpp"

#include "Util/Logger.h"

namespace LFMVS
{
    MISimilarityMeasure::MISimilarityMeasure(DepthSolver* pDepthSolver)
    {
        m_ptrDepthSolver = pDepthSolver;
    }

    MISimilarityMeasure::~MISimilarityMeasure()
    {
        m_ptrDepthSolver = NULL;
    }

    void MISimilarityMeasure::SetSimilarityScoreType(SimilarityScoreType type)
    {
        m_SimilarityScoreType = type;
    }

    SimilarityScoreType MISimilarityMeasure::GetSimilarityScoreType()
    {
        return m_SimilarityScoreType;
    }


    void MISimilarityMeasure::CollectMIANeighImagesByCircle(MLA_Problem& problem)
    {
        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
        int32_t tile_X = ptrKey->GetTileX();
        int32_t tile_Y = ptrKey->GetTileY();

        for(int dx=-2;dx<3;dx++)
        {
            for (int dy=-2;dy<3;dy++)
            {

                if(tile_X%2==0)   //tile_X 行为偶数
                {
                    if(abs(dx)==2 && dy==1) continue;
                }
                else              //tile_X 行不为偶数
                {
                    if(abs(dx)==1 && dy==-2) continue;
                }
                if(dx==0 && dy==0) continue;
                //  if(std::abs(dx) + std::abs(dy) > 4) continue;
                if(abs(dx)==2 && abs(dy)==2) continue;
                int32_t tile_now_x = tile_X + dx;
                int32_t tile_now_y = tile_Y + dy;
                if(tile_now_x < 1 || tile_now_y < 0) continue;
                QuadTreeTileKeyPtr ptrKey_even_row = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_now_x, tile_now_y);
                problem.m_NeigKeyPtrVec.push_back(ptrKey_even_row);
            }
        }
    }
    QuadTreeTileKeyPtrVec MISimilarityMeasure::CollectMIANeighImagesByBaseline(MLA_Problem& problem,LightFieldParams& lf_Params)
    {
        QuadTreeTileKeyPtrVec ptrKeyVec;
        candidatePtrKeyVec = QuadTreeTileKeyPtrVec();
        ptrKeyVec_1 = QuadTreeTileKeyPtrVec();
        ptrKeyVec_2 = QuadTreeTileKeyPtrVec();
        ptrKeyVec_3 = QuadTreeTileKeyPtrVec();
        ptrKeyVec_4 = QuadTreeTileKeyPtrVec();

        QuadTreeTileKeyPtr ptrKey = problem.m_ptrKey;
        int32_t tile_X = ptrKey->GetTileX();
        int32_t tile_Y = ptrKey->GetTileY();
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(problem.m_ptrKey);
        MLA_InfoPtr ptrInfo = itrInfo->second;
        float center_X = ptrInfo->GetCenter().x;
        float center_Y = ptrInfo->GetCenter().y;

        for(int dx=-4;dx<5;dx++)
        {
            for (int dy = -4; dy < 5; dy++)
            {
                if(dx==0 && dy==0) continue;
                int32_t tile_now_x = tile_X + dx;
                int32_t tile_now_y = tile_Y + dy;
                if(tile_now_x < 0 || tile_now_y < 0 || (tile_now_x == 0 && tile_now_y == 0)) continue;
                QuadTreeTileKeyPtr ptrKey_even_row = QuadTreeTileKey::CreateInstance(TileKey_None, 0, tile_now_x, tile_now_y);
                QuadTreeTileInfoMap::iterator itrInfo_now = MLA_info_map.find(ptrKey_even_row);
                if (itrInfo_now == MLA_info_map.end())
                {
                    continue;
                }
                MLA_InfoPtr ptrInfo_now = itrInfo_now->second;
                float center_X_now = ptrInfo_now->GetCenter().x;
                float center_Y_now = ptrInfo_now->GetCenter().y;
                float distance = std::sqrt(std::pow(center_X_now-center_X,2)+std::pow(center_Y_now-center_Y,2));
                /*if(tile_now_x == 57 && tile_now_y==11)
                    std::cout<<"("<<tile_X<<"_"<<tile_Y<<"("<<center_X<<","<<center_Y<<")"<<","<<tile_now_x<<"_"<<tile_now_y<<"("<<center_X_now<<","<<center_Y_now<<")"<<"), distance="<<distance << ", "<<std::endl;*/
                if (distance > (lf_Params.baseline*4+lf_Params.baseline*0.1))
                {
                    continue;
                }
                else
                {
                    ptrKeyVec.push_back(ptrKey_even_row);

                    if ((lf_Params.baseline*0.9)<distance && distance<(lf_Params.baseline*1.1))
                    {
                        problem.m_NeighScoreMapForRefocus[ptrKey_even_row].m_Baseline=lf_Params.baseline;
                        ptrKeyVec_1.push_back(ptrKey_even_row);
                    }
                    else if((lf_Params.baseline*1.1)<distance && distance<(lf_Params.baseline*2.1))
                    {
                        problem.m_NeighScoreMapForRefocus[ptrKey_even_row].m_Baseline= lf_Params.baseline * 2;
                        ptrKeyVec_2.push_back(ptrKey_even_row);
                    }
                    else if((lf_Params.baseline*2.1)<distance && distance<(lf_Params.baseline*3.1))
                    {
                        problem.m_NeighScoreMapForRefocus[ptrKey_even_row].m_Baseline= lf_Params.baseline * 3;
                        ptrKeyVec_3.push_back(ptrKey_even_row);
                    }
                    else if((lf_Params.baseline*3.1)<distance && distance<(lf_Params.baseline*4.1))
                    {
                        problem.m_NeighScoreMapForRefocus[ptrKey_even_row].m_Baseline= lf_Params.baseline * 4;
                        ptrKeyVec_4.push_back(ptrKey_even_row);
                    }
                }
            }
        }
         //   problem.SetM_NeigKeyPtrVec(ptrKeyVec);
        std::sort(ptrKeyVec_1.begin(), ptrKeyVec_1.end(),
        [](const QuadTreeTileKeyPtr& a, const QuadTreeTileKeyPtr& b) {
        if (a->GetTileX() != b->GetTileX()) return a->GetTileX() < b->GetTileX();
        return a->GetTileY() < b->GetTileY();
        });

        std::sort(ptrKeyVec_2.begin(), ptrKeyVec_1.end(),
        [](const QuadTreeTileKeyPtr& a, const QuadTreeTileKeyPtr& b) {
        if (a->GetTileX() != b->GetTileX()) return a->GetTileX() < b->GetTileX();
        return a->GetTileY() < b->GetTileY();
        });

        std::sort(ptrKeyVec_3.begin(), ptrKeyVec_1.end(),
        [](const QuadTreeTileKeyPtr& a, const QuadTreeTileKeyPtr& b) {
        if (a->GetTileX() != b->GetTileX()) return a->GetTileX() < b->GetTileX();
        return a->GetTileY() < b->GetTileY();
        });

        std::sort(ptrKeyVec_4.begin(), ptrKeyVec_1.end(),
        [](const QuadTreeTileKeyPtr& a, const QuadTreeTileKeyPtr& b) {
        if (a->GetTileX() != b->GetTileX()) return a->GetTileX() < b->GetTileX();
        return a->GetTileY() < b->GetTileY();
        });

        candidatePtrKeyVec.insert(candidatePtrKeyVec.end(), ptrKeyVec_1.begin(), ptrKeyVec_1.end());
        candidatePtrKeyVec.insert(candidatePtrKeyVec.end(), ptrKeyVec_2.begin(), ptrKeyVec_1.end());
        candidatePtrKeyVec.insert(candidatePtrKeyVec.end(), ptrKeyVec_3.begin(), ptrKeyVec_1.end());
        candidatePtrKeyVec.insert(candidatePtrKeyVec.end(), ptrKeyVec_4.begin(), ptrKeyVec_1.end());
      // return ptrKeyVec;
        return candidatePtrKeyVec;

    }

    float MISimilarityMeasure::MeasureSimilarity(cv::Mat& srcImage, cv::Mat& neighImage,
                                                 QuadTreeTileKeyPtr ptrSrcKey,
                                                 QuadTreeTileKeyPtr ptrNeighKey)
    {
        float fSimilarityScore = 0.0;

        if (neighImage.empty())
        {
            LOG_ERROR("Neig_image not exist, neig_key: ", ptrNeighKey->StrRemoveLOD().c_str());
            return 0;
        }

        srcImage.convertTo(srcImage, CV_8U);
        neighImage.convertTo(neighImage, CV_8U);
        cv::threshold(srcImage,srcImage,128,255,cv::THRESH_BINARY);
        cv::threshold(neighImage,neighImage,128,255,cv::THRESH_BINARY);

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(ptrSrcKey);
        if (itrInfo == MLA_info_map.end())
            return fSimilarityScore;
        QuadTreeTileInfoMap::iterator itrInfo_Neigh = MLA_info_map.find(ptrNeighKey);
        if (itrInfo_Neigh == MLA_info_map.end())
            return fSimilarityScore;

        MLA_InfoPtr ptrInfo = itrInfo->second;
        float center_X = ptrInfo->GetCenter().x;
        float center_Y = ptrInfo->GetCenter().y;
        MLA_InfoPtr ptrInfo_Neigh = itrInfo_Neigh->second;
        float Neigh_center_X = ptrInfo_Neigh->GetCenter().x;
        float Neigh_center_Y = ptrInfo_Neigh->GetCenter().y;

        float dx = Neigh_center_X - center_X;
        float dy = Neigh_center_Y - center_Y;

        if(std::abs(dx)< 5)
            dx = 0;
        if(std::abs(dy) < 5)
            dy = 0;

        float norm = std::sqrt(dx*dx + dy*dy);
        float stepSize = 2.0f; // 步长
        float dx_step = (dx/norm)*stepSize;
        float dy_step = (dy/norm)*stepSize;

        float shift_x = 0.0f;
        float shift_y = 0.0f;
        int img_width = srcImage.cols;
        int img_height = srcImage.rows;
        int dx_step_counts;
        int dy_step_counts;
        dx_step_counts = img_width/std::abs(dx_step);
        dy_step_counts = img_height/std::abs(dy_step);
        if(dx_step == 0)
        {
            dx_step_counts = 0;
        }
        else if (dy_step == 0)
        {
            dy_step_counts = 0;
        }
        int max_steps = std::ceil(std::max(dx_step_counts,dy_step_counts));
        std::vector<int> match_counts(max_steps);
        match_counts.reserve(max_steps);

        // TODO： 可以分多个段，然后同时计算，最后取结果的最大值
        #pragma omp parallel for schedule(dynamic,20)
        for(int step = 0; step < max_steps; step++)
        {
            if(std::abs(shift_x)>=img_width || std::abs(shift_y)>=img_height)
                continue;

            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, shift_x, 0, 1, shift_y);
            cv::Mat shifted_neighborImage;
            cv::warpAffine(neighImage, shifted_neighborImage, M, neighImage.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

            // 计算重叠区域掩码
            cv::Mat overlap_image = cv::Mat::zeros(srcImage.size(),CV_8UC1);
            float tx = shift_x;
            float ty = shift_y;
            int x1 = static_cast<int>(std::max(0.0f,tx));
            int y1 = static_cast<int>(std::max(0.0f,ty));
            int x2 = static_cast<int>(std::floor(std::min(static_cast<float>(img_width-1),tx+neighImage.cols-1)));
            int y2 = static_cast<int>(std::floor(std::min(static_cast<float>(img_height-1),ty+neighImage.rows-1)));
            if(x2 >= x1 && y2 >= y1)
            {
                cv::Rect overlab_rect(x1,y1,x2-x1+1,y2-y1+1);
                overlap_image(overlab_rect) =255;
            }

            cv::Mat mask_equal;
            cv::compare(srcImage, shifted_neighborImage, mask_equal, cv::CMP_EQ);
            cv::Mat final_mask;
            cv::bitwise_and(mask_equal,overlap_image,final_mask);
            int count = cv::countNonZero(final_mask);
            match_counts[step] = count;
            shift_x = step * dx_step;
            shift_y = step * dy_step;
        }

        if(match_counts.empty())
        {
            fSimilarityScore = 0;
        }
        else
        {
            std::vector<int>::iterator max_match = std::max_element(match_counts.begin(),match_counts.end());
            int max_match_counts = *max_match;
            fSimilarityScore = static_cast<double>(max_match_counts);
        }
        return fSimilarityScore;
    }

    void MISimilarityMeasure::MeasureSimilarityForMI(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        std::string strFrameName = itrFrame->first;
        QuadTreeProblemMap &problem_map = itrFrame->second;
        double SimilarityScore;

        for(QuadTreeProblemMap::iterator itr=problem_map.begin(); itr !=problem_map.end(); itr++)
        {
            QuadTreeTileKeyPtr KeyPtr = itr->first;
            MLA_Problem &Problem = itr->second;
            cv::Mat m_Image_richness = Problem.m_Image_Richness;
            cv::Mat m_Image_gray = Problem.m_Image_gray;

            QuadTreeTileKeyPtrVec m_NeigKeyPtrVec = Problem.m_NeigKeyPtrVec;
            if (m_NeigKeyPtrVec.empty())
                continue;
            //调试检验
            int32_t tile_X = KeyPtr->GetTileX();
            int32_t tile_Y = KeyPtr->GetTileY();
            bool kk=0;
            if(tile_X==14 && tile_Y==0)
            {
                kk =1;
                std::string m_Image_richness_path = "/home/wdy/work/test/xyytest/2600w_1128/scene1/scene/depth_intra/8/MLA_images/56_9_Richness.png";
                cv::Mat m_Image_richness_1 = cv::imread(m_Image_richness_path, cv::IMREAD_COLOR);
                cv::cvtColor(m_Image_richness_1, m_Image_richness_1, cv::COLOR_BGR2GRAY);
                cv::Mat binary = cv::Mat::zeros(67,67,CV_8UC1);
                cv::Mat m_Image_richness_8UC ;
                cv::normalize(m_Image_richness_1,m_Image_richness_8UC,0,1,cv::NORM_MINMAX);
                m_Image_richness_8UC.convertTo(m_Image_richness_8UC,CV_8UC1,255.0);
                cv::threshold(m_Image_richness_8UC,binary,128,255,cv::THRESH_BINARY);
                //cv::imshow("binary_Image",binary);
                imwrite("/home/wdy/work/test/xyytest/neighborContoursImageTest/56_9_richness_gray.png", binary);
                cv::Mat m_Image_56_9 = cv::imread("/home/wdy/work/test/xyytest/neighborContoursImageTest/56_9.png", cv::IMREAD_COLOR);
                cv::cvtColor(m_Image_56_9, m_Image_56_9, cv::COLOR_BGR2GRAY);
                imwrite("/home/wdy/work/test/xyytest/neighborContoursImageTest/56_9_gray.png", m_Image_56_9);
            }
            for(int i=0;i<m_NeigKeyPtrVec.size();i++)
            {
                QuadTreeTileKeyPtr Neigh_Key = m_NeigKeyPtrVec[i];
                QuadTreeProblemMap::iterator NeighItr = problem_map.find(Neigh_Key);
                MLA_Problem Neigh_Problem = NeighItr->second;
          //      cv::Mat m_Neigh_Image_richness = Neigh_Problem.m_Image_Richness;
                cv::Mat Image_Neigh_gray = Neigh_Problem.m_Image_gray;
                cv::Mat richness_MI; // 微图像的纹理丰富性
                cv::Mat richness_Neigh_MI;
                switch (m_SimilarityScoreType)
                {
                    case SST_SSIM:
                    {
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, KeyPtr,richness_MI);
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, Neigh_Key,richness_Neigh_MI);
                        SimilarityScore = MeasureSimilarityBySSIM(richness_MI,richness_Neigh_MI);
                    }
                        break;
                    case SST_Hu:
                    {
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, KeyPtr,richness_MI);
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, Neigh_Key,richness_Neigh_MI);
                        SimilarityScore=MeasureSimilarityByHu(richness_MI,richness_Neigh_MI,KeyPtr,Neigh_Key);
                    }
                        break;
                    case SST_FourierDescriptors:
                    {
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, KeyPtr,richness_MI);
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, Neigh_Key,richness_Neigh_MI);

                        SimilarityScore=MeasureSimilarityByFD(richness_MI,richness_Neigh_MI);
                    }
                        break;
                    case SimilarityByRichness:
                    {
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, KeyPtr,richness_MI);
                        m_ptrDepthSolver->GetRichnessScoreMI(strFrameName, Neigh_Key,richness_Neigh_MI);
                        SimilarityScore=MeasureSimilarityByRichness(richness_MI,richness_Neigh_MI);
                    }
                        break;
                    case SST_ShiftOverlap:
                    {
                        SimilarityScore=MeasureSimilarityByShift(m_Image_gray,Image_Neigh_gray,KeyPtr,Neigh_Key);
                    }
                        break;
                    /*case SST_Hausdorff:
                    {
                        MeasureSimilarityByHausdorff();
                    }
                        break;
                    case SST_ShapeContext:
                    {
                        MeasureSimilarityByShapeContext();
                    }
                        break;*/
                }
                Problem.m_NeighScoreMapForRefocus[Neigh_Key].m_Similarity=SimilarityScore;
                int32_t tile_X_neigh = Neigh_Key->GetTileX();
                int32_t tile_Y_neigh = Neigh_Key->GetTileY();
                std::cout<<"("<<tile_X<<"-"<<tile_Y<<"):"<< "("<<tile_X_neigh<<"-"<<tile_Y_neigh<<")--"<<SimilarityScore<<std::endl;
            }
        }
    }

        //计划在这里拿取邻域集合的key，用不同的函数 分别比较srcImage与neighborImage的相似程度（暂时使用自适应邻域图像选择方法）
       /* for (int i = 0; i < m_NeigKeyPtrVec.size(); i++)
        {
            QuadTreeTileKeyPtr key = m_NeigKeyPtrVec[i];
            QuadTreeProblemMap problem_map = itrP->second;
            QuadTreeProblemMap::iterator itr = problem_map.find(key);
            MLA_Problem& problem = itr->second;
            problem.m_NeigKeyPtrVec;
        }*/

    // 纹理图像切割函数
    void MISimilarityMeasure::Slice_RichnessMLAImage(QuadTreeProblemMapMap::iterator& itrFrame)
    {
        if (m_ptrDepthSolver == NULL)
            return;
        std::string strFrameName = itrFrame->first;
        std::map<std::string, cv::Mat>& richnessImageMap = m_ptrDepthSolver->GetRichnessImageMap();
        cv::Mat& richnessImage = richnessImageMap[strFrameName];
        QuadTreeProblemMap& problem_map = itrFrame->second;
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        int total_abandon = 0;
        for(QuadTreeTileInfoMap::iterator itr = MLA_info_map.begin(); itr != MLA_info_map.end(); itr++)
        {
            QuadTreeTileKeyPtr ptrKey = itr->first;
            MLA_InfoPtr ptrInfo = itr->second;
            if (ptrInfo->IsAbandonByArea())
            {
                total_abandon++;
                continue;
            }
            QuadTreeProblemMap::iterator itrP = problem_map.find(ptrKey);
            if (itrP != problem_map.end())
            {
                MLA_Problem& problem = itrP->second;
                // 切割
                m_Params = m_ptrDepthSolver->GetParams();
                cv::Rect rect(ptrInfo->GetLeftDownCorner().x, ptrInfo->GetLeftDownCorner().y,
                              m_Params.mi_width_for_match, m_Params.mi_height_for_match);
                richnessImage(rect).copyTo(problem.m_Image_Richness);

                //测试时使用的路径
                std::string strMLAPath = m_ptrDepthSolver->GetRootPath();
                std::string strSlice_MLAFullPath = m_ptrDepthSolver->GetSavePath() + strFrameName + LF_RAW_MLA_IMAGES_NAME + ptrKey->StrRemoveLOD()+ "_Richness" + ".png";
             //   std::string strSlice_MLAFullPath = strMLAPath +"/" + ptrKey->StrRemoveLOD()+"Richness"+ ".png";
                imwrite(strSlice_MLAFullPath, problem.m_Image_Richness);
                //路径截至到这里

            }
        }
    }



    //衡量结构相似性 SSIM ，结果值越接近1，代表图像约相似
    double MISimilarityMeasure::MeasureSimilarityBySSIM(cv::Mat srcImage,cv::Mat neighborImage )
    {
        cv::Mat i1 = srcImage.clone();
        cv::Mat i2 = neighborImage.clone();

        float c1 = 6.5025,c2 = 58.5225;
        cv::Mat I1 = cv::Mat::zeros(srcImage.rows,srcImage.cols,CV_32FC1);
        cv::Mat I2 = cv::Mat::zeros(srcImage.rows,srcImage.cols,CV_32FC1);
        cv::Mat I1_sq = I1.mul(I1);
        cv::Mat I2_sq = I2.mul(I2);
        cv::Mat I1_I2 = I1.mul(I2);

        cv::Mat mu1,mu2;
        cv::GaussianBlur(I1,mu1,cv::Size (11,11),1.5);
        cv::GaussianBlur(I2,mu2,cv::Size(11,11),1.5);

        cv::Mat mu1_sq = mu1.mul(mu1);
        cv::Mat mu2_sq = mu2.mul(mu2);
        cv::Mat mu1_mu2 = mu1.mul(mu2);

        cv::Mat sigmal1_sq,sigmal2_sq,sigmal_12;
        cv::GaussianBlur(I1_sq,sigmal1_sq,cv::Size(11,11),1.5);
        sigmal1_sq -= mu1_sq;
        cv::GaussianBlur(I2_sq,sigmal2_sq,cv::Size(11,11),1.5);
        sigmal2_sq -= mu2_sq;
        cv::GaussianBlur(I1_I2,sigmal_12,cv::Size(11,11),1.5);
        sigmal_12 -= mu1_mu2;

        cv::Mat t1 = 2*mu1_mu2+c1;
        cv::Mat t2 = 2*sigmal_12+c2;
        cv::Mat t3 = t1.mul(t2);
        t1 = mu1_sq+mu2_sq+c1;
        t2 = sigmal1_sq+sigmal2_sq+c2;
        t1 = t1.mul(t2);
        cv::Mat ssimMap;
        cv::divide(t1,t3,ssimMap);
        cv::Scalar ssim = cv::mean(ssimMap);     //灰度图只有一个通道，所以取值ssim[0]
        double similarScore = ssim[0];
        return similarScore;
    }

    //衡量两个物体轮廓的相似性 用opencv的接口 cv::matchShapes  ，值越小代表两个图像越相似，值为0代表是完全相同的图像
    double MISimilarityMeasure::MeasureSimilarityByHu(cv::Mat srcImage,cv::Mat neighborImage,QuadTreeTileKeyPtr KeyPtr,QuadTreeTileKeyPtr Neigh_Key)
    {
        cv::Mat BinSrcImage,BinNeighborImage;
        cv::Mat srcImage_8UC1;
        cv::Mat neighborImage_8UC1;
        srcImage.convertTo(srcImage_8UC1, CV_8UC1);
        neighborImage.convertTo(neighborImage_8UC1, CV_8UC1);
        cv::threshold(srcImage_8UC1,BinSrcImage,127,255,cv::THRESH_BINARY);
        cv::threshold(neighborImage_8UC1,BinNeighborImage,127,255,cv::THRESH_BINARY);

     //   std::cout<<"srcImage neighborImage type:"<<srcImage.type()<<neighborImage.type()<<std::endl;
     //   std::cout<<"over"<<std::endl;
        std::vector<std::vector<cv::Point>> srcContours,neighborContours;
    //    cv::Mat srcContours,neighborContours;
        cv::findContours(BinSrcImage,srcContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(BinNeighborImage,neighborContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        if (!srcContours.empty() && !neighborContours.empty())
        {
            cv::Mat neighborContoursImage = cv::Mat::zeros(67, 67, CV_8UC1);;
            cv::drawContours(neighborContoursImage, neighborContours, -1, 255, cv::FILLED);

            //测试用
            int32_t tile_X = KeyPtr->GetTileX();
            int32_t tile_Y = KeyPtr->GetTileY();
            if(tile_X==56 && tile_Y==9)
            {
                int32_t tile_X_neigh = Neigh_Key->GetTileX();
                int32_t tile_Y_neigh = Neigh_Key->GetTileY();
        //        cv::Mat neighborContoursImage = cv::Mat::zeros(67, 67, CV_8UC1);;
        //       cv::drawContours(neighborContoursImage, neighborContours, -1, 255, cv::FILLED);
                std::string path = "/home/wdy/work/test/xyytest/neighborContoursImageTest/";
                std::string ContoursImagePath =  path + std::to_string(tile_X_neigh) + "_" + std::to_string(tile_Y_neigh) + ".png";
                if (!neighborContoursImage.empty())
                    imwrite(ContoursImagePath, neighborContoursImage);
            }

            std::vector<cv::Point> srcCont = srcContours[0];
            std::vector<cv::Point> neighborCont = neighborContours[0];
            double similarity = cv::matchShapes(srcCont,neighborCont,2,0.0);
            return similarity;
        } else
        {
            std::cout<<"Contours is empty!"<<std::endl;
        }
    }

    //傅里叶描述子  值越接近0，代表相似度越高，0.1-0.5 大致相似，0.5-1.5 部分相似，>1.5基本不相似
    double MISimilarityMeasure::MeasureSimilarityByFD(cv::Mat srcImage,cv::Mat neighborImage)
    {
        CV_Assert(!srcImage.empty() && !neighborImage.empty());
        CV_Assert(srcImage.size() == neighborImage.size());
        //计算傅里叶描述子
        const int descriptor_number = 12; // 8-50之间的值
        cv::Mat srcImage_8UC1;
        cv::Mat neighborImage_8UC1;
        srcImage.convertTo(srcImage_8UC1, CV_8UC1);
        neighborImage.convertTo(neighborImage_8UC1, CV_8UC1);
        std::vector<double> srcDesc = calcFourierDescriptors(srcImage_8UC1, descriptor_number);
        std::vector<double> neighborDesc = calcFourierDescriptors(neighborImage_8UC1, descriptor_number);
        if(srcDesc.size() == neighborDesc.size() && !srcDesc.empty())
        {
            const size_t n = srcDesc.size();
            double mean1 = 0.0, mean2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                mean1 += srcDesc[i];
                mean2 += neighborDesc[i];
            }
            mean1 /= n;
            mean2 /= n;
            double cov = 0.0, var1 = 0.0, var2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double d1 = srcDesc[i] - mean1;
                double d2 = neighborDesc[i] - mean2;
                cov += d1 * d2;
                var1 += d1 * d1;
                var2 += d2 * d2;
            }
            // 归一化相关系数
            double similarity = 0.0;
            if (var1 > 0 && var2 > 0) {  // 避免除以零
                similarity = cov / std::sqrt(var1 * var2);
            }
            return similarity;
            // similarity ∈ [-1, 1]，1表示完全相似，-1表示完全相反
        } else
        {
            std::cout<<"srcDesc.size() != neighborDesc.size() || srcDesc.empty()"<<std::endl;
        }




        /*int number = 12;            //number一般取值 8-30，最大可取50+，值越大，计算复杂程度越高
        std::vector<double> srcDesc = calcFourierDescriptors(srcImage,number);
        std::vector<double> neighborDesc = calcFourierDescriptors(neighborImage,number);

        CV_Assert(srcDesc.size() == neighborDesc.size());

        double mean1 = std::accumulate(srcDesc.begin(),srcDesc.end(),0.0)/srcDesc.size();
        double mean2 = std::accumulate(neighborDesc.begin(),neighborDesc.end(),0.0)/neighborDesc.size();

        double var1=0,var2=0,cov=0;
        for(size_t i=0;i<srcDesc.size();i++)
        {
            double d1=srcDesc[i]-mean1;
            double d2=neighborDesc[i]-mean2;
            cov += d1*d2;
            var1 += d1*d2;
            var2 += d2*d2;
        }
        double description = cov/std::sqrt(var1*var2);*/
    }

    double MISimilarityMeasure::MeasureSimilarityByRichness(cv::Mat srcImage,cv::Mat neighborImage)
    {
        // 检查尺寸和类型是否一致
        CV_Assert(srcImage.size() == neighborImage.size());
        CV_Assert(srcImage.type() == CV_32F && neighborImage.type() == CV_32F);

        // 将图像展平为向量
        cv::Mat vec1 = srcImage.reshape(1, 1); // 单行
        cv::Mat vec2 = neighborImage.reshape(1, 1);

        // 计算欧氏距离
        cv::Mat diff = vec1 - vec2;
        double distance = std::sqrt(diff.dot(diff)); // dot 是向量内积
        return distance;
    }

    double MISimilarityMeasure::MeasureSimilarityByShift(cv::Mat srcImage,cv::Mat neighborImage,QuadTreeTileKeyPtr KeyPtr,QuadTreeTileKeyPtr Neigh_Key)
    {
        double SimilarityScore;
        srcImage.convertTo(srcImage, CV_8U);
        neighborImage.convertTo(neighborImage, CV_8U);
        cv::threshold(srcImage,srcImage,128,255,cv::THRESH_BINARY);
        cv::threshold(neighborImage,neighborImage,128,255,cv::THRESH_BINARY);
        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();
        QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(KeyPtr);
        MLA_InfoPtr ptrInfo = itrInfo->second;
        float center_X = ptrInfo->GetCenter().x;
        float center_Y = ptrInfo->GetCenter().y;
        QuadTreeTileInfoMap::iterator itrInfo_Neigh = MLA_info_map.find(Neigh_Key);
        MLA_InfoPtr ptrInfo_Neigh = itrInfo_Neigh->second;
        float Neigh_center_X = ptrInfo_Neigh->GetCenter().x;
        float Neigh_center_Y = ptrInfo_Neigh->GetCenter().y;
        float dx = Neigh_center_X - center_X;
        float dy = Neigh_center_Y - center_Y;
        float norm = std::sqrt(dx*dx + dy*dy);
        float stepSize = 2.0f;
        float dx_step = (dx/norm)*stepSize;
        float dy_step = (dy/norm)*stepSize;

        float shift_x = 0.0f;
        float shift_y = 0.0f;
        std::vector<int> match_counts;
        int img_width = srcImage.cols;
        int img_height = srcImage.rows;
        int max_steps = std::ceil(std::max(img_width/std::abs(dx_step),img_height/std::abs(dy_step)));
        int step = 0;
        while(step<max_steps)
        {
            if(std::abs(shift_x)>=img_width || std::abs(shift_y)>=img_height)
                break;

            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, shift_x, 0, 1, shift_y);
            cv::Mat shifted_neighborImage;
            cv::warpAffine(neighborImage, shifted_neighborImage, M, neighborImage.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

            //计算重叠区域掩码
            cv::Mat overlap_image = cv::Mat::zeros(srcImage.size(),CV_8UC1);
            float tx = shift_x;
            float ty = shift_y;
            int x1 = static_cast<int>(std::max(0.0f,tx));
            int y1 = static_cast<int>(std::max(0.0f,ty));
            int x2 = static_cast<int>(std::floor(std::min(static_cast<float>(img_width-1),tx+neighborImage.cols-1)));
            int y2 = static_cast<int>(std::floor(std::min(static_cast<float>(img_height-1),ty+neighborImage.rows-1)));
            if(x2>=x1&&y2>=y1)
            {
                cv::Rect overlab_rect(x1,y1,x2-x1+1,y2-y1+1);
                overlap_image(overlab_rect) =255;
            }

            cv::Mat mask_equal;
            cv::compare(srcImage, shifted_neighborImage, mask_equal, cv::CMP_EQ);
            cv::Mat final_mask;
            cv::bitwise_and(mask_equal,overlap_image,final_mask);
            int count = cv::countNonZero(final_mask);
            match_counts.push_back(count);
            shift_x += dx_step;
            shift_y += dy_step;
            step++;
        }
        if(match_counts.empty())
        {
            SimilarityScore = 0;
        }
        else
        {
            std::vector<int>::iterator max_match = std::max_element(match_counts.begin(),match_counts.end());
            int max_match_counts = *max_match;
            SimilarityScore = static_cast<double>(max_match_counts);
        }
        return SimilarityScore;
    }

    std::vector<double> MISimilarityMeasure::calcFourierDescriptors(cv::Mat image,int number)
    {
        cv::Mat binary;
        cv::threshold(image,binary,50,255,cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if(!contours.empty())
        {
            std::vector<cv::Point> contour = *std::max_element(contours.begin(),contours.end(),
             [](const std::vector<cv::Point>& a,const std::vector<cv::Point>& b)
              {
              return cv::contourArea(a) < cv::contourArea(b);
              });
        //    CV_Assert(contour.size()>(3*number));
            number = (number & -2);
            number = std::max(12,number);

            cv::Mat complexInput(contour.size(),1,CV_32FC2);
            for(int i=0;i<contour.size();i++)
            {
                complexInput.at<cv::Vec2f>(i)[0] = (float)contour[i].x;
                complexInput.at<cv::Vec2f>(i)[1] = (float)contour[i].y;
            }
            int m = cv::getOptimalDFTSize(complexInput.rows);
            cv::Mat padded;
            if (m%2==0)
            {
                cv::copyMakeBorder(complexInput,padded,0,m-complexInput.rows,0,0,cv::BORDER_CONSTANT,cv::Scalar::all(0));
                for(int i=0;i<padded.rows;i++)
                {
                    if(i%2 !=0)
                    {
                        padded.at<cv::Vec2f>(i)[0] *= -1.0f;
                        padded.at<cv::Vec2f>(i)[1] *= -1.0f;
                    }
                }
                //傅里叶变换
                cv::Mat fourier;
                cv::dft(padded,fourier,cv::DFT_COMPLEX_OUTPUT);

                //提取描述子
                int center = m/2;
                int half = number /2;
                std::vector<double> descriptors;
                double maxMag = 0.0;
                for(int i=center-half;i<=center+half;i++)
                {
                    if(i==center)
                        continue;
                    cv::Vec2f val = fourier.at<cv::Vec2f>(i);
                    double mag = std::sqrt(val[0]*val[0]+val[1]*val[1]);
                    descriptors.push_back(mag);
                    if(mag>maxMag) maxMag = mag;
                }
                if(descriptors.size()==number)
                {
                    for(double& v:descriptors)
                        v /= maxMag;
                    auto it = std::find(descriptors.begin(),descriptors.end(),1.0);
                    if(it != descriptors.end())
                        descriptors.erase(it);

                    return descriptors;
                }
                else
                {
                    std::cout<<"descriptors.size()!=number,error!"<<std::endl;
                }
            }
            else
            {
                std::cout<<"m is invalid!"<<std::endl;
            }
        }
        else
        {
            std::cout<<"contours is empty"<<std::endl;
        }






    }
}

