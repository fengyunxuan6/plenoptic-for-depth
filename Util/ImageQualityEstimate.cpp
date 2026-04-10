/********************************************************************
file base:      ImageQualityEstimate.cpp
author:         LZD
created:        2025/01/13
purpose:
*********************************************************************/
#include "ImageQualityEstimate.h"

#include "Common/QuadTree.h"
#include "../MVStereo/LFDepthInfo.h"
#include "DepthSolver.h"
#include "numeric"

#include "boost/filesystem.hpp"
#include "Util/MISimilarityMeasure.h"

namespace LFMVS
{
    ImageQualityEstimate::ImageQualityEstimate(DepthSolver* pDepthSolver)
    {
        m_ptrDepthSolver = pDepthSolver;
    }

    ImageQualityEstimate::~ImageQualityEstimate()
    {
        m_ptrDepthSolver = NULL;
    }

    void ImageQualityEstimate::SetBlurEstimateType(BlurScoreType type)
    {
        m_eBlurScoreType = type;
    }

    BlurScoreType ImageQualityEstimate::GetBlurEstimateType()
    {
        return m_eBlurScoreType;
    }

    void ImageQualityEstimate::SetRichnessEstimateType(RichnessScoreType type)
    {
        m_eRichnessScoreType = type;
    }

    RichnessScoreType ImageQualityEstimate::GetRichnessEstimateType()
    {
        return m_eRichnessScoreType;
    }

    void ImageQualityEstimate::QuantizeBlurLevelForMIC(QuadTreeProblemMapMap::iterator& itrP,
                                               bool bExcute, bool bWrite)
    {
        if (!bExcute)
            return;
        if (m_ptrDepthSolver == NULL)
            return;

        // 量化模糊程度
        std::string strNameLessExt = itrP->first;
        std::map<std::string, cv::Mat>& RawImageMap = m_ptrDepthSolver->GetRawImageMap();
        cv::Mat& raw_image = RawImageMap[strNameLessExt];
        cv::cvtColor(raw_image, m_MI_gray, cv::COLOR_BGR2GRAY);

        switch (m_eBlurScoreType)
        {
        case BST_SMD2:
            {
                QuantizeBlurBySMD2(itrP, bWrite);

                NormalizeAndCluster();
            }
            break;
        case BST_Gradient:
            {
                QuantizeBlurByGradient(itrP, bWrite);
            }
            break;
        case BST_Laplacian:
            {
                QuantizeBlurByLaplacian(itrP, bWrite);
            }
            break;
        case BST_FrequencyEnergy:
            {
                QuantizeBlurByFrequencyEnergy(itrP, bWrite);
            }
            break;
        case BST_MultiScalarGradient:
            {
                QuantizeBlurBySMD2(itrP, bWrite);
            }
            break;
            default:
                break;
        }

    }

    void ImageQualityEstimate::QuantizeRichnessLevelForMIC(QuadTreeProblemMapMap::iterator& itrP,
                                                            bool bExcute, bool bWrite)
    {
        if (!bExcute)
            return;
        if (m_ptrDepthSolver == NULL)
            return;

        // 量化纹理丰富度
        std::string strNameLessExt = itrP->first;
        std::map<std::string, cv::Mat>& RawImageMap = m_ptrDepthSolver->GetRawImageMap();

         switch (m_eRichnessScoreType)
        {
        case RST_GLCM:
            {
                QuantizeRichnessByGLCM(256,1,0, itrP, bWrite);
            }
            break;
        case RST_Tamura:
            {
                QuantizeRichnessByTamura(itrP, bWrite);
            }
            break;
        case RST_Gabor:
            {
                QuantizeRichnessByGabor(itrP, bWrite);
            }
            break;
//        case RST_GLDM:
//            {
//                QuantizeRichnessByGLDM();
//                if(bWrite)
//                {
//                    WriteQuantifiedBlureScoreImage(itrP,m_Image_RichnessScore_GLDM);
//                }
//            }
        case RST_LBP:
            {
                QuantizeRichnessByLBP(itrP, bWrite);
            }
            break;
            case RST_Wavelet:
            {
                QuantizeRichnessByWavelet(itrP, bWrite);
            }
            break;
            case RST_HOG:
            {
                QuantizeRichnessByHOG(itrP, bWrite);
            }
            break;
            default:
                break;
        }
    }

    void ImageQualityEstimate::QuantizeBlurBySMD2(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if (m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate, QBBS: Image emtpy."<<std::endl;
            return;
        }

        m_Image_blurScore_SMD2 = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);
        for (int col = 0; col < m_MI_gray.cols-1; col++)
        {
            for (int row = 0; row < m_MI_gray.rows-1; row++)
            {
                float v1 = fabs(m_MI_gray.at<uchar>(row,col) - m_MI_gray.at<uchar>(row+1, col) );
                float v2 = fabs(m_MI_gray.at<uchar>(row,col) - m_MI_gray.at<uchar>(row, col+1) );
                float va = v1 * v2;
                m_Image_blurScore_SMD2.at<float>(row, col) = va;
            }
        }

        // 存储模糊量化结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetBlurScoreImageMap(strNameLessExt, m_Image_blurScore_SMD2) ;
        // 存储模糊量化结果值

        if (bWrite)
        {
            WriteQuantifiedBlureScoreImage(itrP, m_Image_blurScore_SMD2);
        }
    }

    void ImageQualityEstimate::QuantizeBlurByGradient(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if (m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate, QBBS: Image emtpy."<<std::endl;
            return;
        }
        m_Image_blurScore_Gradient = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);
        cv::Mat_<float> gradient_dx = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);
        cv::Mat_<float> gradient_dy = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);
        cv::Sobel(m_MI_gray, gradient_dx, CV_32F, 1,0,3);
        cv::Sobel(m_MI_gray, gradient_dy, CV_32F, 0,1,3);
        cv::sqrt(gradient_dx.mul(gradient_dx)+gradient_dy.mul(gradient_dy), m_Image_blurScore_Gradient);
        //cv::mean(m_Image_blurScore_Gradient)[0];


        // 存储模糊量化结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetBlurScoreImageMap(strNameLessExt, m_Image_blurScore_Gradient) ;
        if (bWrite)
        {
            WriteQuantifiedBlureScoreImage(itrP, m_Image_blurScore_Gradient);
        }
    }

    void ImageQualityEstimate::QuantizeBlurByLaplacian(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if (m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate, QBBS: Image emtpy."<<std::endl;
            return;
        }
        m_Image_blurScore_Laplacian = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);
        cv::Laplacian(m_MI_gray, m_Image_blurScore_Laplacian, CV_32F);
        //cv::mean(abs(m_Image_blurScore_Laplacian))[0];

        // 存储模糊量化结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetBlurScoreImageMap(strNameLessExt, m_Image_blurScore_Laplacian) ;

        if (bWrite)
        {
            WriteQuantifiedBlureScoreImage(itrP, m_Image_blurScore_Laplacian);
        }
    }

    void ImageQualityEstimate::QuantizeBlurByFrequencyEnergy(QuadTreeProblemMapMap::iterator& itrP, bool bWrite,
        double radiusRatio)
    {
        if (m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate, QBBS: Image emtpy."<<std::endl;
            return;
        }

        m_Image_blurScore_FrequencyEnergy = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);

        cv::Mat planes[] = {cv::Mat_<double>(m_MI_gray), cv::Mat::zeros(m_MI_gray.size(), CV_32F)};
        cv::Mat complexImg;
        cv::merge(planes, 2, complexImg);
        cv::dft(complexImg, complexImg);

        // 频域移位
        int cx = complexImg.cols/2;
        int cy = complexImg.rows/2;
        cv::Mat q0 (complexImg, cv::Rect(0,0,cx,cy));
        cv::Mat q1 (complexImg, cv::Rect(cx,0,cx,cy));
        cv::Mat q2 (complexImg, cv::Rect(0,cy,cx,cy));
        cv::Mat q3 (complexImg, cv::Rect(cx,cy,cx,cy));
        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        // 创建掩码
        int radius = static_cast<int>(cv::min(cx, cy)*radiusRatio);
        cv::Mat mask = cv::Mat::zeros(complexImg.size(), CV_8U);
        cv::circle(mask, cv::Point(cx, cy), radius,  cv::Scalar(255), -1);

        // 计算能量
        cv::Mat lowFreq;
        complexImg.copyTo(lowFreq, mask);
        cv::Scalar totalEnergy = cv::sum(cv::abs(complexImg));
        cv::Scalar lowEnergy = cv::sum(cv::abs(lowFreq));
        m_Image_blurScore_FrequencyEnergy = (totalEnergy[0] - lowEnergy[0])/totalEnergy[0];

        // 存储模糊量化结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetBlurScoreImageMap(strNameLessExt, m_Image_blurScore_FrequencyEnergy) ;
        if (bWrite)
        {
            WriteQuantifiedBlureScoreImage(itrP, m_Image_blurScore_FrequencyEnergy);
        }
    }

    void ImageQualityEstimate::QuantizeBlurByMultiScale(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        std::vector<double> scales = {1.0, 0.5, 0.25}; // 多尺度参数

        double totalScore = 0.0;
        for (int i = 0; i < scales.size(); i++)
        {
            double scale = scales.at(i);
            cv::Mat reseized_img;
            cv::resize(m_MI_gray, reseized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
            totalScore += GradiantBlurScore(reseized_img);
        }
        totalScore/scales.size();
    }

    double ImageQualityEstimate::GradiantBlurScore(const cv::Mat& img)
    {
        cv::Mat dx, dy;
        cv::Sobel(img, dx, CV_64F, 1, 0, 3);
        cv::Sobel(img, dy, CV_64F, 0, 1, 3);
        cv::Mat magnitude;
        cv::sqrt(dx.mul(dx)+dy.mul(dy), magnitude);
        return cv::mean(magnitude)[0];
    }

    // 目前的GLCM函数，输入一张灰度图，输出的是整个图像的纹理特征值，后续可按实际需求修改成滑动窗口纹理值
    void ImageQualityEstimate::QuantizeRichnessByGLCM(int leves,int dx,int dy,
        QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if(m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate, QBBS: Image emtpy."<<std::endl;
            return;
        }
        //设置滑动窗口计算
        m_Image_RichnessScore_GLCM = cv::Mat::zeros(m_MI_gray.rows,m_MI_gray.cols,CV_32FC1);
        cv::Rect roi;
        roi.x = 0;
        roi.y = 0;
        roi.width = 8;
        roi.height = 8;
        for(int row=4;row<m_MI_gray.rows-4;row++)
        {
            for(int col=4;col<m_MI_gray.cols-4;col++)
            {
                roi.x = col-4;
                roi.y = row-4;
                cv::Mat roi_image = cv::Mat::zeros(8,8,CV_32FC1);
                roi_image = m_MI_gray(roi);
                float pv = calculate_matrix(roi_image);
                m_Image_RichnessScore_GLCM.at<float>(row,col) = pv;
  //              std::cout <<"glcm的值"<<glcm.at<float>(row,col)<<std::endl;
            }
        }
        cv::normalize(m_Image_RichnessScore_GLCM,m_Image_RichnessScore_GLCM,0,255,cv::NORM_MINMAX);

    //    m_Image_RichnessScore_GLCM = glcm;
     //   cv::imshow("m_Image_RichnessScore_GLCM",m_Image_RichnessScore_GLCM);



        //备用
//        RichnessByGLCMFeatures features;
//        cv::Mat glcm= cv::Mat::zeros(leves,leves,CV_32F);
//        m_Image_RichnessScore_LBP = cv::Mat(m_MI_gray.rows,m_MI_gray.cols,CV_8UC1);
//
//        cv::Mat quantized;
//        m_MI_gray.convertTo(quantized, CV_32F, static_cast<float>(leves) / 256.0f);
//        quantized.convertTo(quantized, CV_8U);
//
//        //计算GLCM方向
//        for(int y=0;y<m_MI_gray.rows-1;y++)
//        {
//            for(int x=0;x<m_MI_gray.cols-1;x++)
//            {
//                int i = quantized.at<uchar>(y,x);
//                int j = quantized.at<uchar>(y,x+1);
//                glcm.at<float>(i,j)++;
//                //std::cout<<glcm.at<float>(i,j)<<"glcm的值"<<std::endl;
//            }
//        }
//
//        //归一化
//        float sumGLCM = static_cast<float>(cv::sum(glcm)[0]);
//        if (sumGLCM != 0)
//            glcm /=sumGLCM;
//
//        //提取特征
//        for(int i=0;i<leves;i++)
//        {
//            for(int j=0;j<leves;j++)
//            {
//                float p = glcm.at<float>(i,j);
//                if(p==0)
//                    continue;
//                features.energy += p*p;
//                features.contrast +=(i-j)*(i-j)*p;
//                features.homogeneity += p/(1+std::abs(i-j));
//                features.IDM += p/((i-j)*(i-j)+1);     //+1是为了避免除以0
//                features.entropy -= p*std::log10(p);
//                features.mean += (i+j)*0.5f*p;
//            }
//        }
//        features.glcm = glcm.clone();  //保留原始GLCM 可视化用
//        m_Image_RichnessScore_GLCM = cv::Mat_<float>(m_MI_gray.size(), features.contrast);

        // 存储纹理丰富性指标结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetRichnessImageMap(strNameLessExt, m_Image_RichnessScore_GLCM) ;

        // test
        if(bWrite)
        {
            WriteQuantifiedRichnessImage(itrP, m_Image_RichnessScore_GLCM);
        }
    }

    void ImageQualityEstimate::QuantizeRichnessByTamura(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if (m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate,QBBS:Image emtpy."<<std::endl;
            return;
        }
        m_Image_RichnessScore_Tamura = cv::Mat::zeros(m_MI_gray.rows,m_MI_gray.cols,CV_32FC1);
        int windowSize = 15;
        cv::Mat gray=cv::Mat::zeros(m_MI_gray.rows,m_MI_gray.cols,CV_32FC1);
        gray = m_MI_gray.clone();
        CV_Assert(gray.channels() == 1);

        int r = windowSize / 2;
        cv::Mat padded;
        copyMakeBorder(gray, padded, r, r, r, r, cv::BORDER_REFLECT);

        cv::Mat contrastMap(gray.size(), CV_32F);
        cv::Mat roughnessMap(gray.size(), CV_32F);
        cv::Mat directionMap(gray.size(), CV_32F);

        for (int y = r; y < padded.rows - r; ++y)
        {
            for (int x = r; x < padded.cols - r; ++x)
            {
                cv::Rect roi(x - r, y - r, windowSize, windowSize);
                cv::Mat patch = padded(roi);
              //计算了三种特征，对比度、粗糙度、方向性
            //    contrastMap.at<float>(y - r, x - r) = localContrast(patch);
                roughnessMap.at<float>(y - r, x - r) = localDirectionality(patch);
             //   directionMap.at<float>(y - r, x - r) = localRoughness(patch);
            }
        }

        m_Image_RichnessScore_Tamura=roughnessMap.clone();
//        cv::Mat tamuraMap;
//        std::vector<cv::Mat> channels = {contrastMap, roughnessMap, directionMap};
//        merge(channels, tamuraMap);

        // 存储纹理丰富性指标结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetRichnessImageMap(strNameLessExt, m_Image_RichnessScore_Tamura);

        // test
        if(bWrite)
        {
            WriteQuantifiedRichnessImage(itrP, m_Image_RichnessScore_Tamura);
        }
    }

    void ImageQualityEstimate::QuantizeRichnessByGabor(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if(m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate,QBBS:Image emtpy."<<std::endl;
            return;
        }
      //  m_Image_RichnessScore_Gabor = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);
        m_Image_RichnessScore_Gabor = cv::Mat::zeros(m_MI_gray.rows, m_MI_gray.cols, CV_32FC1);

        int ks = 21;
        cv::Mat_<float> gaborKernel (ks,ks,CV_32FC1);
        GetGaborKernel(ks,5.0,45.0,10.0,90.0, gaborKernel);   //参数值可修改
        cv::Mat_<float> MI_gray_float;
        m_MI_gray.convertTo(MI_gray_float,CV_32FC1);
        cv::filter2D(MI_gray_float,m_Image_RichnessScore_Gabor,-1,gaborKernel);

        cv::imwrite("/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/kk.png", gaborKernel);

        cv::imwrite("/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/gg.png", MI_gray_float);
        cv::imwrite("/home/lzd/work/data/plenoptic_data/2600w_1128/scene1/scene/depth_intra/8/MVSResult/tt.png", m_Image_RichnessScore_Gabor);
        // 存储纹理丰富性指标结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetRichnessImageMap(strNameLessExt, m_Image_RichnessScore_Gabor) ;

        // test
        if(bWrite)
        {
            WriteQuantifiedRichnessImage(itrP, m_Image_RichnessScore_Gabor);
        }
    }

    void ImageQualityEstimate::QuantizeRichnessByLBP(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if(m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate,QBBS:Image emtpy."<<std::endl;
        }
        m_Image_RichnessScore_LBP = cv::Mat::zeros(m_MI_gray.rows,m_MI_gray.cols,CV_32FC1);
        m_Image_RichnessScore_LBP = circularLbp(m_MI_gray,2,16);


//        for(int i = 1;i<m_MI_gray.rows-1;i++)
//        {
//            for(int j=1;j<m_MI_gray.cols-1;j++)
//            {
//                uchar tpix = m_MI_gray.at<uchar>(i,j);
//                uchar color;
//                color |=(m_MI_gray.at<uchar>(i-1,j-1)>tpix)<<7;
//                color |=(m_MI_gray.at<uchar>(i-1,j)>tpix)<<6;
//                color |=(m_MI_gray.at<uchar>(i-1,j+1)>tpix)<<5;
//                color |=(m_MI_gray.at<uchar>(i,j+1)>tpix)<<4;
//                color |=(m_MI_gray.at<uchar>(i+1,j+1)>tpix)<<3;
//                color |=(m_MI_gray.at<uchar>(i+1,j)>tpix)<<2;
//                color |=(m_MI_gray.at<uchar>(i+1,j-1)>tpix)<<1;
//                color |=(m_MI_gray.at<uchar>(i,j-1)>tpix)<<0;
//                m_Image_RichnessScore_LBP.at<uchar>(i-1,j-1) = color;
//            }
//            // res["LBP"] = m_Image_RichnessScore_LBP;
//        }

        // 存储纹理丰富性指标结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetRichnessImageMap(strNameLessExt, m_Image_RichnessScore_LBP) ;

        // test
        if(bWrite)
        {
            WriteQuantifiedRichnessImage(itrP, m_Image_RichnessScore_LBP);
        }
    }

    void ImageQualityEstimate::QuantizeRichnessByWavelet(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if(m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate,QBBS:Image emtpy."<<std::endl;
        }
        m_Image_RichnessScore_Wavelet = cv::Mat::zeros(m_MI_gray.rows/2,m_MI_gray.cols/2,CV_32F);
        int rows = m_MI_gray.rows/2;
        int cols = m_MI_gray.cols/2;

        cv::Mat temp = m_MI_gray.clone();
        cv::Mat low = cv::Mat(rows,cols,CV_32F);
        cv::Mat high = cv::Mat(rows,cols,CV_32F);

        for(int y=0;y<rows;y++)
        {
            for(int x=0;x<cols;x++)
            {
                float a=temp.at<float>(y*2,x*2);
                float b=temp.at<float>(y*2,x*2+1);
                float c=temp.at<float>(y*2+1,x*2);
                float d=temp.at<float>(y*2+1,x*2+1);

                m_Image_RichnessScore_Wavelet.at<float>(y,x) = (a-b-c+d)/2.0;
            }
        }

        // 存储纹理丰富性指标结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetRichnessImageMap(strNameLessExt, m_Image_RichnessScore_Wavelet) ;

        // test
        if(bWrite)
        {
            WriteQuantifiedRichnessImage(itrP, m_Image_RichnessScore_Wavelet);
        }
    }

    void ImageQualityEstimate::QuantizeRichnessByHOG(QuadTreeProblemMapMap::iterator& itrP, bool bWrite)
    {
        if(m_MI_gray.empty())
        {
            std::cout<<"ImageQualityEstimate,QBBS:Image emtpy."<<std::endl;
        }

        QuadTreeProblemMap MIA_Problem_Map = itrP->second;

        //设置滑动窗口计算
        m_Image_RichnessScore_HOG = cv::Mat::zeros(m_MI_gray.rows,m_MI_gray.cols,CV_32FC1);
        // 设置 HOG 参数
        int windowSize = 32;
        cv::Size win_size(windowSize, windowSize);
        cv::Size block_size(16, 16);
        cv::Size block_stride(4, 4);
        cv::Size cell_size(4, 4);
        int nbins = 6;
        int stride = 4;
        cv::HOGDescriptor hog(win_size, block_size, block_stride, cell_size, nbins);

        for (int y = 0; y <= m_MI_gray.rows - windowSize; y += stride)
        {
            for (int x = 0; x <= m_MI_gray.cols - windowSize; x += stride)
            {
                cv::Rect roi(x, y, windowSize, windowSize);
                cv::Mat patch = m_MI_gray(roi);
                std::vector<float> descriptors;
                hog.compute(patch, descriptors);

                if (!descriptors.empty())
                {
//                    float score = accumulate(descriptors.begin(), descriptors.end(), 0.0f) / descriptors.size();    均值
//                    float richness = 1.0f / (score + 1e-6f);
                    float entropy = 0.0f;
                    for (float val : descriptors)
                    {
                        if (val > 1e-6f)
                        {
                            entropy -= val * log(val);
                        }
                    }
                    float richness = entropy;

                    for (int j = y; j < y + stride && j < m_MI_gray.rows; ++j)
                    {
                        for (int i = x; i < x + stride && i < m_MI_gray.cols; ++i)
                        {
                            m_Image_RichnessScore_HOG.at<float>(j, i) = richness;
                        }
                    }
                }
            }
        }
        cv::normalize(m_Image_RichnessScore_HOG,m_Image_RichnessScore_HOG,0,255,cv::NORM_MINMAX);



//        std::vector<float> hog_features;
//        std::vector<float> MIA_hog_features_Result;
//        m_Image_RichnessScore_HOG = cv::Mat::zeros(49,64,CV_32F);
//        //设置HOG参数
//        cv::Size win_size(67,67);
//        cv::Size block_size(16,16);
//        cv::Size block_stride(1,1);
//        cv::Size cell_size(8,8);
//        int nbins = 9;
//        for(QuadTreeProblemMap::iterator itr=MIA_Problem_Map.begin();itr!=MIA_Problem_Map.end();itr++)
//        {
//            MLA_Problem& problem = itr->second;
//            cv::Mat m_Image_gray = problem.m_Image_gray;
//
//            cv::HOGDescriptor hog(win_size,block_size,block_stride,cell_size,nbins);
//            cv::Mat a = cv::Mat::zeros(67,67,CV_8UC1);
//            hog.compute(m_Image_gray,hog_features);
//            float Hog_Richness_Score = accumulate(hog_features.begin(),hog_features.end(),0.0f)/hog_features.size();
//            Hog_Richness_Score = 1.0f /(Hog_Richness_Score + 1e-6);
//            m_Image_RichnessScore_HOG.at<float>(row,col) = Hog_Richness_Score;
//            MIA_hog_features_Result.push_back(Hog_Richness_Score);
//        }
//        for(int col=0;col<m_Image_RichnessScore_HOG.cols;col++)
//        {
//            for (int row = 0; row < m_Image_RichnessScore_HOG.rows; row++)
//            {
//                int index = col*m_Image_RichnessScore_HOG.rows + row;
//                m_Image_RichnessScore_HOG.at<float>(row,col) = MIA_hog_features_Result[index];
//            }
//        }
//        cv::normalize(m_Image_RichnessScore_HOG,m_Image_RichnessScore_HOG,0,255,cv::NORM_MINMAX);
//        std::cout<<MIA_hog_features_Result.size()<<std::endl;

        // 存储纹理丰富性指标结果图
        std::string strNameLessExt = itrP->first;
        m_ptrDepthSolver->SetRichnessImageMap(strNameLessExt, m_Image_RichnessScore_HOG) ;

        // test
        if(bWrite)
        {
            WriteQuantifiedRichnessImage(itrP, m_Image_RichnessScore_HOG);
        }
    }

    float ImageQualityEstimate::calculate_matrix(cv::Mat& m)
    {
    //    cv::Mat m = cv::Mat::zeros(8,8,CV_32FC1);
        cv::Mat glcm = cv::Mat::zeros(8,8,CV_32FC1);
  //      m = gray_image;
        for(int row=0;row<m.rows-1;row++)
        {
            for(int col=0;col<m.cols-1;col++)
            {
                int i = m.at<uchar>(row,col)/32;
                int j = m.at<uchar>(row+1,col+1)/32;
                glcm.at<float>(i,j)++;
            }
        }
        float contrast = 0;
        for(int i=0;i<8;i++)
        {
            for(int j=0;j<8;j++)
            {
                contrast += glcm.at<float>(i,j)*(i-j)*(i-j);
            }
        }
        return contrast;
    }

    // 生成一个Gabor卷积核，用这个核去处理图像，提取纹理
    void ImageQualityEstimate::GetGaborKernel(int ks,double sig,double th,double lm,double ps, cv::Mat_<float>& gaborKernel)
    {
        int hks = (ks-1)/2;
        double theta = th*CV_PI/180;
        double psi = ps*CV_PI/180;
        double  del = 2.0/(ks-1);
        double lmbd = lm;
        double sigma = sig/ks;
        double x_theta;
        double y_theta;
        for(int y=-hks;y<hks;y++)
        {
            for(int x=-hks;x<hks;x++)
            {
                x_theta = x*del* cos(theta)+y*del*sin(theta);
                y_theta = -x*del*sin(theta)+y*del*cos(theta);
                gaborKernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))*cos(2*CV_PI*x_theta/lmbd+psi);
            }
        }
    }

    //圆形LBP提取纹理特征
    cv::Mat ImageQualityEstimate::circularLbp(cv::Mat grayImage,int radius,int points)
    {
        int height = grayImage.rows;
        int width = grayImage.cols;
        cv::Mat dst = cv::Mat::zeros(grayImage.rows,grayImage.cols,CV_32FC1);
        dst = grayImage.clone();
        std::vector<float> neighbours(points);
        std::vector<int> lbpValue(points);
        for(int x=radius;x<width-radius-1;x++)
        {
            for(int y=radius;y<height-radius-1;y++)
            {
                float lbp = 0;
                for(int n=0;n<points;n++)
                {
                    float theta = 2*CV_PI*n/points;
                    float x_n = x+radius*std::cos(theta);
                    float y_n = y-radius*std::sin(theta);

                    int x1 = static_cast<int>(std::floor(x_n));
                    int y1 = static_cast<int>(std::floor(y_n));
                    int x2 = static_cast<int>(std::ceil(x_n));
                    int y2 = static_cast<int>(std::ceil(y_n));

                    float tx = abs(x_n-x1);
                    float ty = abs(y_n-y1);

                    float w1 = (1-tx)*(1-ty);
                    float w2 = tx*(1-ty);
                    float w3 = (1-tx)*ty;
                    float w4 = tx*ty;

                    float neighbour = grayImage.at<float>(y1,x1)*w1 + grayImage.at<float>(y2,x1)*w2 + grayImage.at<float>(y1,x2)*w3 + grayImage.at<float>(y2,x2)*w4;
                    neighbours[n] = neighbour;
                }
                float center = grayImage.at<float>(y,x);
                for(int n=0;n<points;n++)
                {
                    if (neighbours[n]>center)
                        lbpValue[n] = 1;
                    else
                        lbpValue[n] = 0;
                }
                for(int n=0;n<points;n++)
                {
                    lbp +=lbpValue[n]*std::pow(2,n);
                }
                dst.at<float>(y,x) = lbp/(std::pow(2,points)-1)*255.0f;
            }
        }
        std::cout<<dst.empty()<<"判断dst是否为空"<<std::endl;
        return dst;
    }

    float  ImageQualityEstimate::localContrast(cv::Mat patch)
    {
        cv::Mat gradX, gradY;
        Sobel(patch, gradX, CV_32F, 1, 0);
        Sobel(patch, gradY, CV_32F, 0, 1);
        cv::Mat magnitude;
        magnitude = abs(gradX) + abs(gradY);
        return mean(magnitude)[0];
    }
    float ImageQualityEstimate::localDirectionality(cv::Mat patch)
    {
        cv::Mat gradX, gradY;
        Sobel(patch, gradX, CV_32F, 1, 0);
        Sobel(patch, gradY, CV_32F, 0, 1);
        cv::Mat magnitude;
        magnitude = abs(gradX) + abs(gradY);
        return mean(magnitude)[0];
    }
    float ImageQualityEstimate::localRoughness(cv::Mat patch)
    {
        cv::Mat gx, gy;
        Sobel(patch, gx, CV_32F, 1, 0);
        Sobel(patch, gy, CV_32F, 0, 1);
        cv::Mat angle;
        phase(gx, gy, angle, true); // 角度图 0-360

        int bins = 16;
        float binWidth = 360.f / bins;
        std::vector<float> hist(bins, 0);

        for (int y = 0; y < angle.rows; ++y) {
            for (int x = 0; x < angle.cols; ++x) {
                int bin = static_cast<int>(angle.at<float>(y, x) / binWidth) % bins;
                hist[bin]++;
            }
        }

        float meanVal = 0, total = 0;
        for (int i = 0; i < bins; ++i) {
            meanVal += i * hist[i];
            total += hist[i];
        }
        if (total == 0) return 0;
        meanVal /= total;

        float variance = 0;
        for (int i = 0; i < bins; ++i) {
            variance += pow(i - meanVal, 2) * hist[i];
        }
        return variance / total;
    }

    void ImageQualityEstimate::WriteQuantifiedBlureScoreImage(QuadTreeProblemMapMap::iterator& itrP, cv::Mat_<float> blur_score_img)
    {
        cv::Mat_<float> blur_score_img_copy = blur_score_img.clone();

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();

        std::string strNameLessExt = itrP->first;
        QuadTreeProblemMap& problem_map = itrP->second;

        // 准备路径，并写出
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strNameLessExt + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
                }
            }
        }
        std::string strMI_BlurValue_path;
        switch (m_eBlurScoreType)
        {
        case BST_SMD2:
            {
               strMI_BlurValue_path = m_strSavePath + std::string("/blur_ssim");
            }
            break;
        case BST_Gradient:
            {
                strMI_BlurValue_path = m_strSavePath + std::string("/blur_gradient");
            }
            break;
        case BST_Laplacian:
            {
                strMI_BlurValue_path = m_strSavePath + std::string("/blur_lap");
            }
            break;
        case BST_FrequencyEnergy:
            {
                strMI_BlurValue_path = m_strSavePath + std::string("/blur_frequency");
            }
            break;
        case BST_MultiScalarGradient:
            {
                strMI_BlurValue_path = m_strSavePath + std::string("/blur_multiGradient");
            }
            break;
            default:
                break;
        }

//        std::string str_MIRichnessValue_path;
//        switch (m_eRichnessScoreType)
//        {
//        case RST_GLCM:
//            {
//                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_glcm");
//            }
//        case RST_Tamura:
//            {
//                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_tamura");
//            }
//        case RST_Gabor:
//            {
//                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_gabor");
//            }
//        case RST_LBP:
//            {
//                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_lbp");
//            }
//            break;
//            default:
//                break;
//        }

        std::string strMI_BlurValue_gray_path = strMI_BlurValue_path + "raw.png";
        imwrite(strMI_BlurValue_gray_path, blur_score_img_copy);

//        std::string str_MIRichnessValue_gray_path = str_MIRichnessValue_path + "raw.png";
//        imwrite(str_MIRichnessValue_gray_path, blur_score_img_copy);


        // 加微图像的外边缘和编号
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++)
        {
            MLA_Problem& problem = itr->second;

            QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(problem.m_ptrKey);
            if (itrInfo == MLA_info_map.end())
                continue;
            MLA_InfoPtr ptrInfo = itrInfo->second;
            if (ptrInfo->IsAbandonByArea())
                continue;

            // MLA_Tilekey: 微透镜编码
            // 设置字体和颜色
            int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
            double fontScale = 0.4;                  // 字体大小
            int thickness = 1;                        // 线条粗细
            // 文字内容
            std::string text = problem.m_ptrKey->StrRemoveLOD();

            // 文字位置，(x, y)为文字左下角的坐标
            LightFieldParams& lf_params = m_ptrDepthSolver->GetLightFieldParams();
            cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y-lf_params.mi_height_for_match*0.5);
            // 将文字写入图片
            cv::putText(blur_score_img_copy, text, textOrg, fontFace, fontScale, 255, thickness, cv::LINE_AA);

            // 绘制中心点
            cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
            cv::circle(blur_score_img_copy, center_p, 2, 255, -1);

            // 绘制圆环：根据标定的微透镜直径
            cv::circle(blur_score_img_copy, center_p, lf_Params.baseline*0.5, 255, 1);
        }
        std::string strMI_BlurValue_gray_full_path = strMI_BlurValue_path + "raw_key.png";
        imwrite(strMI_BlurValue_gray_full_path, blur_score_img_copy);
    }

   // void ImageQualityEstimate::WriteQuantifiedRichnessImage(QuadTreeProblemMapMap::iterator& itrP, cv::Mat_<float> richness_score_img)
    void ImageQualityEstimate::WriteQuantifiedRichnessImage(QuadTreeProblemMapMap::iterator& itrP, cv::Mat richness_score_img)
    {
        cv::Mat_<float> richness_score_img_copy = richness_score_img.clone();

        QuadTreeTileInfoMap& MLA_info_map = m_ptrDepthSolver->GetMLAInfoMap();

        LightFieldParams& lf_Params = m_ptrDepthSolver->GetLightFieldParams();

        std::string strNameLessExt = itrP->first;
        QuadTreeProblemMap& problem_map = itrP->second;

        // 准备路径，并写出
        m_strSavePath = m_ptrDepthSolver->GetSavePath() + strNameLessExt + LF_MVS_RESULT_DATA_NAME;
        {
            boost::filesystem::path dir_save_path(m_strSavePath);
            if (!boost::filesystem::exists(dir_save_path))
            {
                if (!boost::filesystem::create_directory(dir_save_path))
                {
                    std::cout << "dir failed to create: " << m_strSavePath << std::endl;
                }
            }
        }

        std::string str_MIRichnessValue_path;
        switch (m_eRichnessScoreType)
        {
            case RST_GLCM:
            {
                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_glcm");
            }
            break;
        case RST_Tamura:
            {
                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_tamura");
            }
            break;
        case RST_Gabor:
            {
                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_gabor");
            }
            break;
        case RST_LBP:
            {
                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_lbp");
            }
            break;
        case RST_HOG:
            {
                str_MIRichnessValue_path = m_strSavePath + std::string("/richness_HOG");
            }
            break;
        default:
            break;
        }

        std::string str_MIRichnessValue_gray_path = str_MIRichnessValue_path + "_raw.png";
        imwrite(str_MIRichnessValue_gray_path, richness_score_img_copy);

        // 加微图像的外边缘和编号
        for(QuadTreeProblemMap::iterator itr = problem_map.begin(); itr != problem_map.end(); itr++)
        {
            MLA_Problem& problem = itr->second;

            QuadTreeTileInfoMap::iterator itrInfo = MLA_info_map.find(problem.m_ptrKey);
            if (itrInfo == MLA_info_map.end())
                continue;
            MLA_InfoPtr ptrInfo = itrInfo->second;
            if (ptrInfo->IsAbandonByArea())
                continue;

            // MLA_Tilekey: 微透镜编码
            // 设置字体和颜色
            int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体
            double fontScale = 0.4;                  // 字体大小
            int thickness = 1;                        // 线条粗细
            // 文字内容
            std::string text = problem.m_ptrKey->StrRemoveLOD();

            // 文字位置，(x, y)为文字左下角的坐标
            LightFieldParams& lf_params = m_ptrDepthSolver->GetLightFieldParams();
            cv::Point textOrg(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y-lf_params.mi_height_for_match*0.5);
            // 将文字写入图片
            cv::putText(richness_score_img_copy, text, textOrg, fontFace, fontScale, 255, thickness, cv::LINE_AA);

            // 绘制中心点
            cv::Point center_p(ptrInfo->GetCenter().x, ptrInfo->GetCenter().y);
            cv::circle(richness_score_img_copy, center_p, 2, 255, -1);

            // 绘制圆环：根据标定的微透镜直径
            cv::circle(richness_score_img_copy, center_p, lf_Params.baseline*0.5, 255, 1);
        }
        std::string strMI_BlurValue_gray_full_path = str_MIRichnessValue_path + "_raw_key.png";
        imwrite(strMI_BlurValue_gray_full_path, richness_score_img_copy);

//        MISimilarityMeasure image_SM(this);
//        image_SM.Slice_RichnessMLAImage(itrP);
    }

    //测试用的临时函数
    QuadTreeProblemMapMap::iterator& ImageQualityEstimate::ReturnItrP(QuadTreeProblemMapMap::iterator& itrP)
    {
        QuadTreeProblemMapMap::iterator& itrp =itrP;
        return itrp;
    }


    void ImageQualityEstimate::NormalizeAndCluster(int kCluster)
    {
        std::string strPath_test = "/home/lzd/work/data/plenoptic_data/HR260_H2_Shilong/scene1/depth_intra/scene2/MVSResult";

        // 归一化到0-1之间
        cv::Mat normalized_img;
        cv::normalize(m_Image_blurScore_SMD2, normalized_img, 0, 1, cv::NORM_MINMAX);

        // K-means聚类
        cv::Mat data_img = normalized_img.reshape(1, normalized_img.rows*normalized_img.cols);
        data_img.convertTo(data_img, CV_32F);
        cv::Mat labels_img, centers_img;
        cv::kmeans(data_img, kCluster, labels_img,
                 cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,
                     10, 0.1),
            3, cv::KMEANS_PP_CENTERS, centers_img);


        // 生成聚类结果图
        m_Clustered_img = labels_img.reshape(0, normalized_img.rows);
        m_Clustered_img.convertTo(m_Clustered_img, CV_8U);

        cv::Mat tmp_img = m_Clustered_img.clone();
        cv::Mat color_map;
        cv::applyColorMap(tmp_img*80, color_map, cv::COLORMAP_JET);
        cv::imwrite(strPath_test+"/blur_ssim_cluster_color.png", color_map);

        cv::convertScaleAbs(m_Clustered_img, m_Clustered_img, 255);

        cv::imwrite(strPath_test+"/blur_ssim_cluster.png", m_Clustered_img);
    }

}
