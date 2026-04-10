/********************************************************************
file base:      CommonUtil.cpp
author:         LZD
created:        2025/06/07
purpose:
*********************************************************************/
#include "CommonUtil.h"

#include "../../../../../../../../usr/local/cuda/targets/x86_64-linux/include/vector_functions.h"

namespace LFMVS
{
    MLA_Info::MLA_Info()
    {
        m_Center.x = -1.0;
        m_Center.y = -1.0;
        m_Col = -1;
        m_Row = -1;

        m_Area = 0.0;
        m_BAbandon_area = false;
        score = -1;
        m_LeftDownCorner = cv::Point2f(-1.0, -1.0);

        m_blurType = eBT_Level_default;
        m_blurRadius = 0.0;
    }

    MLA_Info::~MLA_Info()
    {
    }

    void MLA_Info::SetCenter(cv::Point2f& center)
    {
        m_Center = center;
    }

    cv::Point2f& MLA_Info::GetCenter()
    {
        return m_Center;
    }

    void MLA_Info::SetAbandonByArea(bool b)
    {
        m_BAbandon_area = b;
    }

    bool MLA_Info::IsAbandonByArea()
    {
        return m_BAbandon_area;
    }

    cv::Point2f& MLA_Info::GetLeftDownCorner()
    {
        return m_LeftDownCorner;
    }

    void MLA_Info::SetArea(float area)
    {
        m_Area = area;
    }

    float& MLA_Info::GetArea()
    {
        return m_Area;
    }

    void MLA_Info::SetBlurType(eMLABlurType type)
    {
        m_blurType = type;
    }

    const eMLABlurType MLA_Info::GetBlurType()
    {
        return m_blurType;
    }

    void MLA_Info::SetBlurRadius(float blurRadius)
    {
        m_blurRadius = blurRadius;
    }

    float MLA_Info::GetBlurRadius()
    {
        return m_blurRadius;
    }

    void MLA_Info::SetCol(int col)
    {
        m_Col = col;
    }
    int MLA_Info::GetCol()
    {
        return m_Col;
    }
    void MLA_Info::SetRow(int row)
    {
        m_Row = row;
    }
    int MLA_Info::GetRow()
    {
        return m_Row;
    }

}
