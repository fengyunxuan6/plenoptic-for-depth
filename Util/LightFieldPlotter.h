/********************************************************************
file base:      LightFieldPlotter.h
author:         LZD
created:        2025/08/10
purpose:        此头文件声明了一个用于可视化伽利略型聚焦光场相机中视差 d、
                虚拟深度 v 和真实深度 a 之间关系的类。输入相机参数后，
                类可以计算这些关系并使用 OpenCV 绘制曲线图。注释均使用中文。
*********************************************************************/
#ifndef LIGHTFIELDPLOTTER_H
#define LIGHTFIELDPLOTTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace LFMVS
{
    // 用于保存相机参数的简单结构体，所有单位均为毫米
    struct CameraParams
    {
        double f;    // 主透镜焦距 f
        double bL0;  // 主透镜到微透镜阵列的距离 bL0
        double D;    // 微透镜阵列到传感器的距离 D
        double p;    // 微透镜中心间距 p
    };

    // LightFieldPlotter 类负责计算并绘制三种两两关系：
    // 1. 视差对真实深度的函数；
    // 2. 虚拟深度对真实深度的函数；
    // 3. 视差对虚拟深度的函数。
    class LightFieldPlotter
    {
    public:
        // 构造函数，接收一份相机参数。
        explicit LightFieldPlotter(const CameraParams& params, std::string& strRootPath);

        // 根据输入的景物距离范围（米）绘制三幅关系图。numSamples 为采样点数。
        void plotRelations(double minDepthMeters, double maxDepthMeters, int numSamples = 2000);

    private:
        // 计算虚像距 b：根据薄透镜成像公式 1/f = 1/a + 1/(bL0 + b) 解出 b
        static double computeVirtualImageDistance(double a, const CameraParams& params);

        // 根据虚像距 b 计算视差 d：d = (p * D) / b
        static double computeDisparity(double b, const CameraParams& params);

        // 根据虚像距 b 计算虚拟深度 v：v = b / D
        static double computeVirtualDepth(double b, const CameraParams& params);

        // 绘制单幅折线图。传入 x 和 y 数据以及标题和坐标轴标签，返回绘制好的图像。
        cv::Mat drawPlot(const std::vector<double> &xs,
                         const std::vector<double> &ys,
                         const std::string &title,
                         const std::string &xlabel,
                         const std::string &ylabel);

    private:
        CameraParams        params_;
        std::string         m_strSavePath;
    };
}
#endif //LIGHTFIELDPLOTTER_H
