/********************************************************************
file base:      ImageQualityEstimate.h
author:         LZD
created:        2025/01/13
purpose:        对微图像进行模糊程度的量化
*********************************************************************/
#ifndef LFMVS_BLURESTIMATE_H
#define LFMVS_BLURESTIMATE_H
#include <memory>
#include <sstream>

#include "DepthSolver.h"

namespace LFMVS
{
    enum BlurScoreType
    {
        BST_SMD2, //
        BST_Gradient, // 梯度幅度值
        BST_Laplacian, // 拉普拉斯响应
        BST_FrequencyEnergy, // 频域能量
        BST_MultiScalarGradient, // 多尺度梯度幅度值
        BST_All // 综合
    };

    enum RichnessScoreType
    {
        RST_GLCM, // 灰度共生矩阵
        RST_Tamura, // 基于人类视觉感知，提取对比度、粗糙度等特征，
        RST_Gabor, // Gabor滤波器，频域分析纹理。适合多尺度分析
        RST_All, // 综合
        RST_GLDM, // 灰度差方法
        RST_LBP, // 局部二值模式
        RST_Wavelet,  //小波变换
        RST_HOG,    //方向梯度直方图

    };

    //QuantizeRichnessByGLCM函数里提取出的多个特征结构体
    struct RichnessByGLCMFeatures
    {
        float energy=0.0;
        float contrast=0.0;
        float  homogeneity=0.0;
        float  IDM=0.0;
        float  entropy=0.0;
        float mean=0.0;
        cv::Mat glcm;  //可视化用，用不到可去掉
    };

    // 评价以及量化：微图像的模糊程度、纹理丰富度
    class ImageQualityEstimate
    {
        friend DepthSolver;

    public:
        ImageQualityEstimate(DepthSolver* pDepthSolver);

        ~ImageQualityEstimate();

    public:
        void SetBlurEstimateType(BlurScoreType type);
        BlurScoreType GetBlurEstimateType();

        void SetRichnessEstimateType(RichnessScoreType type);
        RichnessScoreType GetRichnessEstimateType();

        // 量化微图像的模糊程度
        void QuantizeBlurLevelForMIC(QuadTreeProblemMapMap::iterator& itrP,
                                    bool bExcute=false, bool bWrite=false);

        // 量化微图像的纹理丰富度
        void QuantizeRichnessLevelForMIC(QuadTreeProblemMapMap::iterator& itrP,
                                    bool bExcute=false, bool bWrite=false);

    private:
        //////////////////////////////////////////////////////////////////////////////
        // 模糊程度量化

        // SMD2算子：灰度方差乘积函数。对每一个像素邻域两个灰度差相乘后再逐个像素累加
        void QuantizeBlurBySMD2(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);

        // 计算梯度幅度值的模糊得分
        void QuantizeBlurByGradient(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);

        // 计算拉普拉斯响应的模糊得分
        void QuantizeBlurByLaplacian(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);

        // 计算频域能量的模糊得分
        void QuantizeBlurByFrequencyEnergy(QuadTreeProblemMapMap::iterator& itrP, bool bWrite, double radiusRatio = 0.1);

        // 多尺度（梯度幅度）的模糊得分
        void QuantizeBlurByMultiScale(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);
        double GradiantBlurScore(const cv::Mat& img);

        // 写出模糊量化的可视化图像
        void WriteQuantifiedBlureScoreImage(QuadTreeProblemMapMap::iterator& itrP, cv::Mat_<float> blur_score_img);

        // 像素模糊程度量化值的归一化与区域划分
        void NormalizeAndCluster(int kCluster = 8);

        //////////////////////////////////////////////////////////////////////////////
        // 纹理丰富性量化
        //灰度共生矩阵量化纹理：统计不同方向和距离下灰度对的联合概率，构建矩阵
        void QuantizeRichnessByGLCM(int leves,int dx,int dy,
            QuadTreeProblemMapMap::iterator& itrP, bool bWrite);
        float calculate_matrix(cv::Mat& m);

        //Tamura量化纹理：模拟人类视觉感知，提取对比度、粗糙度等特征
        void QuantizeRichnessByTamura(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);
        float localContrast(cv::Mat grayImage);
        float localDirectionality(cv::Mat grayImage);
        float localRoughness(cv::Mat grayImage);

        //Gabor滤波器量化纹理：对每个方向/频率响应计算
        void QuantizeRichnessByGabor(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);
        // kernelSize:核尺寸 sigma:高斯函数标准差 theta:滤波器方向（0：水平；90：垂直）lambda:控制条纹的间距（5～15像素）psi:相位偏移（0/90）
        void GetGaborKernel(int ks, double sig, double th, double lm, double ps, cv::Mat_<float>& gaborKernel );

        //灰度差量化纹理：计算图像中像素与其邻居间灰度差值的统计分布
     //   void QuantizeRichnessByGLDM();

        //局部二值量化纹理:将中心像素与其邻域像素比较
        void QuantizeRichnessByLBP(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);
        cv::Mat circularLbp(cv::Mat grayImage,int radius,int points);

        //小波变换提取纹理特征信息
        void QuantizeRichnessByWavelet(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);

        //方向梯度直方图
        void QuantizeRichnessByHOG(QuadTreeProblemMapMap::iterator& itrP, bool bWrite);

        // 写出纹理量化的可视化图像
     //   void WriteQuantifiedRichnessImage(QuadTreeProblemMapMap::iterator& itrP, cv::Mat_<float> richness_score_img);
        void WriteQuantifiedRichnessImage(QuadTreeProblemMapMap::iterator& itrP, cv::Mat richness_score_img);


        QuadTreeProblemMapMap::iterator& ReturnItrP(QuadTreeProblemMapMap::iterator& itrP);


    private:
        DepthSolver*                            m_ptrDepthSolver;
        std::string                             m_strSavePath;

        cv::Mat                                 m_MI_gray; // 微图像的灰度图

        // 模糊程度量化
        BlurScoreType                           m_eBlurScoreType;
        cv::Mat_<float>                         m_Image_blurScore_SMD2;
        cv::Mat_<float>                         m_Image_blurScore_Gradient;
        cv::Mat_<float>                         m_Image_blurScore_Laplacian;
        cv::Mat_<float>                         m_Image_blurScore_FrequencyEnergy;

        // 纹理丰富程度量化
        RichnessScoreType                       m_eRichnessScoreType;
        cv::Mat_<float>                         m_Image_RichnessScore_GLCM;
        cv::Mat_<float>                         m_Image_RichnessScore_Tamura;
        cv::Mat_<float>                         m_Image_RichnessScore_Gabor;
     //   cv::Mat_<float>                         m_Image_RichnessScore_GLDM;
        cv::Mat_<float>                         m_Image_RichnessScore_LBP;
        cv::Mat                                 m_Image_RichnessScore_Wavelet;
        cv::Mat                                 m_Image_RichnessScore_HOG;

        cv::Mat                                 m_Clustered_img;
    };
}
#endif //LFMVS_BLURESTIMATE_H
