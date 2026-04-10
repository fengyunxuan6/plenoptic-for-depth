/********************************************************************
file base:      Common.cpp
author:         LZD
created:        2024/05/13
purpose:        全局变量、枚举、结构体等
*********************************************************************/
#include "Common.h"

namespace LFMVS
{
    LightFieldParams::LightFieldParams()
    {
        mla_u_size = 0;
        mla_v_size = 0;

        mi_width_for_match = 0;
        mi_height_for_match = 0;

        baseline = 0.0;
        mainlen_flocal_length = 0.0;

        sensor_pixel_size = 0;
        sensor_rotation = Mat3::Identity();
        sensor_translate = Vec3d(0,0,0);

        // m_bl = 0.0;
        // m_B = 0.0;

        distor_depth = Vec3d(0,0,0);
        distor_radial = Vec3d(0,0,0);
        distor_tangential = Vec2d(0,0);
    }

    void LightFieldParams::ComputeMIA_Math_Info()
    {
        mi_width_for_match = baseline * 0.707106781186; // 目前按照内接正方形进行裁剪
        mi_height_for_match = mi_width_for_match;
    }

    LightFieldParamsCUDA::LightFieldParamsCUDA(LightFieldParams& params)
    {
        mia_height_for_match = params.mi_height_for_match;
        mi_width_for_match = params.mi_width_for_match;
        baseline = params.baseline;
    }


    // 主透镜到微透镜阵列的距离
    float g_bl0 = 0.0; // 单位毫米 (xml中的D)
    // 微透镜阵列到传感器距离
    float g_B = 0.0; // 单位毫米 (xml中的d)

    float g_Invalid_image = -9999.0;

    float g_Invalid_Match_ratio = 0.00; // 0.05

    int8_t g_Debug_Save = 0;
    int8_t g_Debug_Static = 0;
    std::vector<std::string> g_Common_image_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
}
