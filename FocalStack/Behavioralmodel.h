//
// Created by wdy on 25-10-17.
//

#ifndef ACMP_BEHAVIORALMODEL_H
#define ACMP_BEHAVIORALMODEL_H

namespace LFMVS
{
    /*class Behavioralmodel
    {*/
        bool ExtractDepthsFromImages(
                cv::Mat& real_img,                // 已经加载好的图像
                cv::Mat& virt_img,
                std::vector<cv::Point>& coords_real,
                std::vector<cv::Point>& coords_virtual,
                std::vector<float>& realdepth,
                std::vector<float>& virtualdepth,
                int& radius);

        cv::Mat toFloat3(const cv::Mat& img);

        float sampleMean3(const cv::Mat& img32f3, int x, int y);

        std::array<double,3> BehavioralModel( std::vector<float>& realdepth, std::vector<float>& virtualdepth);

        cv::Mat convertVirtualToRealDepth( cv::Mat& virtualDepth, std::array<double, 3>& coeffs);

        enum class DistanceType { Chamfer, Euclidean, Mean, Median};
        double imageDistance(cv::Mat rdImage_Bea, cv::Mat& refcasd_img,
                             DistanceType dt, double threshold /* mm */,
                             bool rejectLocalOutliers,
                             bool useLocalNearest);

       void imageDistanceSampling(cv::Mat rdImage_Bea, cv::Mat& refcasd_img,
                             double threshold /* mm */,
                             int searchRadius,
                             std::vector<cv::Point> coords_virtual,
                             std::vector<cv::Point>& sampledPoints,
                             int windowSize,int samplesPerPoint);

        cv::Mat buildOutlierMask(cv::Mat& img, int ksize, double k);

        bool loadPointsXML(std::string& path,
                           std::vector<cv::Point>& coords_casd,
                           std::vector<cv::Point>& coords_virtual);

        cv::Mat drawRandomColorCrosses(const cv::Mat& img, const std::vector<cv::Point>& points);

  //  }

}




#endif //ACMP_BEHAVIORALMODEL_H
