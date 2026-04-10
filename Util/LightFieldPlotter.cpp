/********************************************************************
file base:      LightFieldPlotter.cpp
author:         LZD
created:        2025/08/10
purpose:        此头文件声明了一个用于可视化伽利略型聚焦光场相机中视差 d、
                虚拟深度 v 和真实深度 a 之间关系的类。输入相机参数后，
                类可以计算这些关系并使用 OpenCV 绘制曲线图。
// 主要功能包括：
//   1. 根据薄透镜模型和微透镜针孔模型计算虚像距、视差和虚拟深度；
//   2. 采样真实深度范围，生成三组数据用于绘图；
//   3. 使用 OpenCV 绘制折线图，直观展示不同关系的变化趋势。
*********************************************************************/
#include "LightFieldPlotter.h"

#include "Common/Common.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <boost/filesystem.hpp>

namespace LFMVS
{
    // 构造函数：保存相机参数
    LightFieldPlotter::LightFieldPlotter(const CameraParams& params, std::string& strRootPath)
        : params_(params)
    {
        m_strSavePath = strRootPath + LF_CAMERA_PLOTTER_NAME;
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
    }

    // 计算虚像距 b，根据薄透镜成像公式：
    // 1/f = 1/a + 1/(bL0 + b) => b = 1/(1/f - 1/a) - bL0
    double LightFieldPlotter::computeVirtualImageDistance(double a, const CameraParams& params)
        {
        double denom = 1.0 / params.f - 1.0 / a;
        if (denom <= 0.0)
        {
            // 如果分母小于等于零，说明景物在焦距以内或者退化，返回负值表示无效
            return -1.0;
        }
        double b = 1.0 / denom - params.bL0;
        return b;
    }

    // 根据虚像距 b 计算视差 d：d = (p * D) / b
    double LightFieldPlotter::computeDisparity(double b, const CameraParams& params) {
        if (b == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return (params.p * params.D) / b;
    }

    // 根据虚像距 b 计算虚拟深度 v：v = b / D
    double LightFieldPlotter::computeVirtualDepth(double b, const CameraParams& params) {
        return b / params.D;
    }

    // 私有函数：绘制一幅折线图。
    // xs 和 ys 为数据向量，title 为图标题，xlabel 和 ylabel 为坐标轴的文本。
    cv::Mat LightFieldPlotter::drawPlot(const std::vector<double> &xs,
                                        const std::vector<double> &ys,
                                        const std::string &title,
                                        const std::string &xlabel,
                                        const std::string &ylabel) {
        const int width = 800;
        const int height = 600;
        const int margin = 60;
        cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        // 计算数据范围
        double xMin = std::numeric_limits<double>::max();
        double xMax = std::numeric_limits<double>::lowest();
        double yMin = std::numeric_limits<double>::max();
        double yMax = std::numeric_limits<double>::lowest();
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i] < xMin) xMin = xs[i];
            if (xs[i] > xMax) xMax = xs[i];
            if (ys[i] < yMin) yMin = ys[i];
            if (ys[i] > yMax) yMax = ys[i];
        }
        // 确保范围不是零，避免除零错误
        if (std::abs(xMax - xMin) < 1e-12) xMax = xMin + 1.0;
        if (std::abs(yMax - yMin) < 1e-12) yMax = yMin + 1.0;

        // 绘制坐标轴
        cv::line(img, cv::Point(margin, height - margin),
                 cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2);
        cv::line(img, cv::Point(margin, margin),
                 cv::Point(margin, height - margin), cv::Scalar(0, 0, 0), 2);

        // 添加标题
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.8;
        int thickness = 1;
        cv::putText(img, title, cv::Point(margin, margin - 20), fontFace, fontScale,
                    cv::Scalar(0, 0, 0), thickness);
        // 添加 x 轴和 y 轴标签
        cv::putText(img, xlabel,
                    cv::Point((width - margin) / 2, height - 10), fontFace, 0.6,
                    cv::Scalar(0, 0, 0), 1);
        // y 轴标签需要旋转，这里通过在辅助图像上绘制然后旋转实现
        cv::Mat rotated = cv::Mat::zeros(height, width, CV_8UC3);
        cv::putText(rotated, ylabel, cv::Point(height / 2, margin / 2), fontFace,
                    0.6, cv::Scalar(0, 0, 0), 1);
        cv::Mat rotatedTransposed;
        cv::transpose(rotated, rotatedTransposed);
        cv::flip(rotatedTransposed, rotatedTransposed, 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cv::Vec3b pixel = rotatedTransposed.at<cv::Vec3b>(y, x);
                if (pixel != cv::Vec3b(0, 0, 0)) {
                    img.at<cv::Vec3b>(y, x) = pixel;
                }
            }
        }

        // 绘制刻度线和刻度值（默认 5 个刻度）
        int numTicks = 5;
        for (int i = 0; i <= numTicks; ++i) {
            double tx = xMin + (xMax - xMin) * i / numTicks;
            double ty = yMin + (yMax - yMin) * i / numTicks;
            // x 轴刻度
            int xPos = margin + static_cast<int>((tx - xMin) / (xMax - xMin) * (width - 2 * margin));
            cv::line(img, cv::Point(xPos, height - margin - 5),
                     cv::Point(xPos, height - margin + 5), cv::Scalar(0, 0, 0), 1);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.2f", tx);
            cv::putText(img, buf,
                        cv::Point(xPos - 30, height - margin + 20), fontFace, 0.4,
                        cv::Scalar(0, 0, 0), 1);
            // y 轴刻度
            int yPos = height - margin - static_cast<int>((ty - yMin) / (yMax - yMin) * (height - 2 * margin));
            cv::line(img, cv::Point(margin - 5, yPos),
                     cv::Point(margin + 5, yPos), cv::Scalar(0, 0, 0), 1);
            std::snprintf(buf, sizeof(buf), "%.2f", ty);
            cv::putText(img, buf, cv::Point(5, yPos + 5), fontFace, 0.4,
                        cv::Scalar(0, 0, 0), 1);
        }

        // 绘制折线
        cv::Point prevPoint;
        bool firstPoint = true;
        for (size_t i = 0; i < xs.size(); ++i) {
            if (!std::isfinite(xs[i]) || !std::isfinite(ys[i])) {
                continue; // 跳过无效值
            }
            int xPixel = margin + static_cast<int>((xs[i] - xMin) / (xMax - xMin) * (width - 2 * margin));
            int yPixel = height - margin - static_cast<int>((ys[i] - yMin) / (yMax - yMin) * (height - 2 * margin));
            cv::Point pt(xPixel, yPixel);
            if (!firstPoint) {
                cv::line(img, prevPoint, pt, cv::Scalar(255, 0, 0), 2);
            }
            prevPoint = pt;
            firstPoint = false;
        }
        return img;
    }

    // plotRelations：采样真实深度区间，生成三种关系的数据并绘制图像
    void LightFieldPlotter::plotRelations(double minDepthMeters, double maxDepthMeters, int numSamples)
    {
        if (minDepthMeters <= 0.0 || maxDepthMeters <= 0.0 || maxDepthMeters <= minDepthMeters) {
            std::cerr << "景物距离范围无效，请确保 0 < minDepth < maxDepth。" << std::endl;
            return;
        }
        // 将景物距离转换为毫米
        double minAmm = minDepthMeters * 1000.0;
        double maxAmm = maxDepthMeters * 1000.0;
        // 预留空间
        std::vector<double> realDepths;
        std::vector<double> disparities;
        std::vector<double> virtualDepths;
        realDepths.reserve(numSamples);
        disparities.reserve(numSamples);
        virtualDepths.reserve(numSamples);
        // 逐点计算
        for (int i = 0; i < numSamples; ++i) {
            double a = minAmm + (maxAmm - minAmm) * static_cast<double>(i) / (numSamples - 1);
            double b = computeVirtualImageDistance(a, params_);
            if (b <= 0.0) {
                // 无效情况用 NaN 填充，方便绘制时跳过
                realDepths.push_back(a / 1000.0);
                disparities.push_back(std::numeric_limits<double>::quiet_NaN());
                virtualDepths.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            double d = computeDisparity(b, params_);
            double v = computeVirtualDepth(b, params_);
            realDepths.push_back(a / 1000.0);
            disparities.push_back(d);
            virtualDepths.push_back(v);
        }
        // 准备三组数据
        // 1. 视差与真实深度
        std::vector<double> xs1 = realDepths;
        std::vector<double> ys1 = disparities;
        // 2. 虚拟深度与真实深度
        std::vector<double> xs2 = realDepths;
        std::vector<double> ys2 = virtualDepths;
        // 3. 视差与虚拟深度
        std::vector<double> xs3;
        std::vector<double> ys3;
        xs3.reserve(numSamples);
        ys3.reserve(numSamples);
        for (size_t i = 0; i < realDepths.size(); ++i) {
            if (std::isfinite(virtualDepths[i]) && std::isfinite(disparities[i])) {
                xs3.push_back(virtualDepths[i]);
                ys3.push_back(disparities[i]);
            }
        }
        // 绘制并显示
        cv::Mat plot1 = drawPlot(xs1, ys1, "Disparity-RealDepth", "RealDepth(m)", "Disparity(mm)");
        cv::Mat plot2 = drawPlot(xs2, ys2, "VirtualDepth-RealDepth", "RealDepth(m)", "VirtualDepth(v)");
        cv::Mat plot3 = drawPlot(xs3, ys3, "Disparity-VirtualDepth", "VirtualDepth(v)", "Disparity(mm)");
        std::string strDis_RealDepth_Name = m_strSavePath+"Dis_RealDepth.png";
        cv::imwrite(strDis_RealDepth_Name, plot1);
        std::string strVD_RealDepth_Name = m_strSavePath+"VD_RealDepth.png";
        cv::imwrite(strVD_RealDepth_Name, plot2);
        std::string strDis_VD_Name = m_strSavePath+"Dis_VD.png";
        cv::imwrite(strDis_VD_Name, plot3);
    }
}
