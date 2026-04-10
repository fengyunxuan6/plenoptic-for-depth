/********************************************************************
file base:      Common.h
author:         LZD
created:        2024/05/13
purpose:        全局变量、枚举、结构体等
*********************************************************************/
#ifndef ACMP_COMMON_H
#define ACMP_COMMON_H

#include "vector"
#include "string"

#include "opencv2/core/core.hpp"
#include <vector_types.h>
#include "Eigen/StdVector"

namespace LFMVS
{
    // MLA：Micro-Lens Array 微透镜阵列
    // MIA：Micro-Images Array 微图像集合

    /////////////////////////////////////////////////////////////////////////
    using Mat3 = Eigen::Matrix<double, 3, 3>;
    using Mat34 = Eigen::Matrix<double, 3, 4>;
    using Vec2i = Eigen::Vector2i;
    using Vec2d = Eigen::Vector2d;
    using Vec3f = Eigen::Vector3f;
    using Vec3d = Eigen::Vector3d;

    /////////////////////////////////////////////////////////////////////////
    #define MAX_IMAGES 256
    #define LF_LOGGER_FOLDER_NAME ("/logs")
    #define LF_RAW_DATASET_NAME ("/raw_dataset/")
    #define LF_CALIB_FOLDER_NAME ("/calibration/")
    #define LF_DEPTH_INTRA_NAME ("/depth_intra/")
    #define LF_DEPTH_INTER_NAME ("/depth_inter/")
    #define LF_UNDISTOR_NAME ("/Undistorion/")
    #define LF_RAW_MLA_IMAGES_NAME ("/MLA_images/")
    #define LF_MVS_RESULT_DATA_NAME ("/MVSResult/")
    #define LF_CAMERA_PLOTTER_NAME ("/CameraPlotter/")
    #define MLA_WHITE_CENTERS_INFO_NAME ("CalibInfo.xml") // 利用白图像自动计算的微透镜图像中心点坐标
    #define MLA_WHITE_CENTERS_CIRCLE_INFO_NAME ("CalibInfo_circle.png") // 利用白图像自动计算的微透镜图像中心点坐标
    #define MLA_WHITE_CENTERS_NAME ("Center.txt")
    #define LF_INTRINSICS_NAME ("intrinsics.xml") // 标定阶段解算的相机内参

    #define LF_RAW_IMAGE_NAME ("RawImage") // 原始光场影像的名称
    #define LF_WHITE_IMAGE_NAME ("mask") // 白图像的名称
    #define LF_MLA_DISPARITYMAPS_NAME ("/MLA_DMaps/") // 微透镜图像的深度图
    #define LF_MLA_DISPARITYMAPS_PLANNER_NAME ("/MLA_DMaps_Planner/") // 微透镜图像的平面先验深度图


    #define LF_DISPARITYHMAP_NAME ("Disparity") // 虚拟深度图名称
    #define LF_VIRTUALDEPTHMAP_NAME ("VitualDepthMap") // 虚拟深度图名称
    #define LF_VIRTUALDEPTH_CONFMAP_RAW_NAME ("Depth_ConfMap_Raw") // 虚拟深度空间中深度置信图名称
    #define LF_VIRTUALDEPTH_CONFMAP_NAME ("Depth_ConfMap") // 二值化：虚拟深度空间中深度置信图名称
    #define LF_ALLINFOUCSIMAGE_NAME ("/AIF_Color") // 全聚焦图像名称
    #define LF_FOUCSIMAGE_NAME ("fullfocus") // 全聚焦图像名称

    ////////////////////////////////////////////////////////////////
    // brief: 基础类型别名
    typedef int8_t          sint8; // 有符号8位整数
    typedef uint8_t         uint8; // 无符号8位整数
    typedef int16_t         sint16; // 有符号16位整数
    typedef uint16_t        uint16; // 无符号16位整数
    typedef int32_t         sint32; // 有符号32位整数
    typedef uint32_t        uint32; // 无符号32位整数
    typedef int64_t         sint64; // 有符号64位整数
    typedef uint64_t        uint64; // 无符号64位整数
    typedef float           float32; // 单精度浮点
    typedef double          float64; // 双精度浮点
    ///////////////////////////////////////////////////////////////////////////
    //全局变量
    extern std::vector<std::string> g_Common_image_formats; // 常见图像文件的格式
    typedef int  g_row_index;

    extern int8_t g_Debug_Save;
    extern int8_t g_Debug_Static;



    //////////////////////////////////////////////////////////////////////////////////
    // 枚举
    enum SimilarityScoreType
    {
        SST_SSIM,    //衡量结构相似性
        SST_Hu,      //衡量两个物体轮廓的相似性
        SST_FourierDescriptors,   //傅里叶描述子，较精细的匹配
        SST_Hausdorff,   //豪斯多夫距离，先提取轮廓点，再比较，对边缘清晰度有要求，不一定适用
        SST_ShapeContext,    //形状上下文，精确匹配，但计算量大
        SimilarityByRichness,
        SST_ShiftOverlap     //沿着极线方向滑动，计算重合度
    };

    enum eImageType
    {
        eIT_CustomMultiView = 0, // 传统多视影像
        eIT_LightFieldView // 光场微透镜图像
    };

    // 微透镜（图像）模糊程度
    enum eMLABlurType
    {
        eBT_Level_default = 0,
        eBT_Level0, // 最清晰的一类微透镜
        eBT_Level1, // 次清晰的一类微透镜
        eBT_Level2,
        eBT_Level3,
        eBT_Level4,
        eBT_Level5
    };

    enum eParseMLACentersType
    {
        ePMLACT_wts = 0, //
        ePMLACT_Auto, // 本方法中自研
        ePMLACT_ParseFromCalib // 从标定结果文件解析
    };

    // 邻居选择类型
    enum eSelectNeighborsType
    {
        eSNT_FixedPosition, // 固定位置
        eSNT_Features, // 特征点的匹配
        eSNT_Gradients, // 梯度灰度的块匹配
        eSNT_Other
    };

    // 匹配算法的类型
    enum eStereoType
    {
        eST_ACMH = 0, // 基本算法 MVS method with Adaptive Checkerboard sampling and Multi-Hypothesis joint view selection
        eST_PlannerPrior, // 平面先验
        eST_Horizontal_Propagation, // 同层级横向传播
        eST_BlurFeature, // 模糊特征:多尺度特征类扩散的估计方法（类似ACMM）
    };

    // 深度估计阶段
    enum eStereoStage
    {
        eSS_ACMH_Begin = 0, // ACMH
        eSS_ACMH_Finished,
        eSS_PlannerPrior_Begin, // 平面先验
        eSS_PlannerPrior_Finished,
        eSS_LRGeometric_Check, // 左右一致性检测
        eSS_BlurFeature_Coarse, // 模糊特征：粗略层级
        eSS_BlurFeature_Fine, // 模糊特征：精细层级
        eSS_BlurFeature_Finest // 模糊特征：最精细层级
    };

        ////////////////////////////////////////////////////////////////
    struct LightFieldParams
    {
        LightFieldParams();

        // 根据基线计算微透镜图像中，可用于深度估计的像素区域
        void ComputeMIA_Math_Info();

    public:
        // 微透镜阵列，即角度分辨率
        int             mla_u_size; // 角度分辨率（水平方向微透镜的数量）列号 x
        int             mla_v_size; // 角度分辨率（垂直方向微透镜的数量）行号 y

        // 可用于匹配的微图像的有效像素：目前，为白图像直径的根号2的一半
        int             mi_width_for_match;
        int             mi_height_for_match;

        float           baseline; // 微透镜之间的基线距离（单位：像素）邻居微图像中心点的长度
        float           mainlen_flocal_length; // 主透镜焦距（单位：毫米）

        // float           m_bl; // 主透镜到微透镜阵列的距离(单位毫米, xml中的D)
        // float           m_B; // 微透镜阵列到传感器距离(单位毫米, xml中的d)

        // 传感器参数
        float           sensor_pixel_size; // 传感器(像元)尺寸  像素转换毫米（mm/pixel 每个像素代表了多少毫米）
        Mat3            sensor_rotation;
        Vec3d           sensor_translate;

        // 主透镜畸变参数
        Vec3d           distor_depth; // 深度畸变
        Vec3d           distor_radial; // 径向畸变
        Vec2d           distor_tangential; // 切向畸变
    };
    // 主透镜到微透镜阵列的距离
    extern float g_bl0; // 单位毫米 (xml中的D)
    // 微透镜阵列到传感器距离
    extern float g_B; // 单位毫米 (xml中的d)

    extern float g_Invalid_image;
    extern float g_Invalid_Match_ratio;

    // 一般情况下，LightFieldParamsCUDA是LightFieldParams的子集
    struct LightFieldParamsCUDA
    {
        LightFieldParamsCUDA(LightFieldParams& params);

        // 可用于匹配的微图像的有效像素：目前，为白图像直径的根号2的一半
        int             mi_width_for_match;
        int             mia_height_for_match;

        float           baseline; // 微透镜之间的基线距离（单位：像素）邻居微图像中心点的长度
    };
    struct Camera
    {
        float       K[9];
        float       R[9];
        float       t[3];
        int         height;
        int         width;
        float       depth_min;
        float       depth_max;
    };

    struct Problem
    {
        int ref_image_id;
        std::vector<int> src_image_ids;
    };

    struct Triangle
    {
        Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3)
        : pt1(_pt1)
        , pt2(_pt2)
        , pt3(_pt3)
        {}

        cv::Point pt1, pt2, pt3;
    };

    struct PointList
    {
        float3              coord;
        float3              normal;
        float3              color;
    };

}
#endif // ACMP_COMMON_H
