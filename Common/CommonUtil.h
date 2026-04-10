/********************************************************************
file base:      CommonUtil.h
author:         LZD
created:        2025/06/07
purpose:
*********************************************************************/
#ifndef COMMONUTIL_H
#define COMMONUTIL_H

#include "QuadTree.h"
#include "../../../../../../../../usr/local/cuda/targets/x86_64-linux/include/vector_functions.h"

namespace LFMVS
{
    struct Proxy_DisPlane
    {
        float4      plane[8];
        Proxy_DisPlane()
        {
            for(int i=0;i<8;i++)
            {
                plane[i]=make_float4(-9999,-9999,-9999,-9999);
            }
        }
    };

    class MLA_Info
    {
    public:
        MLA_Info();

        ~MLA_Info();

    public:
        void SetCenter(cv::Point2f& center);
        cv::Point2f& GetCenter();

        void SetCol(int col);
        int GetCol();
        void SetRow(int row);
        int GetRow();

        void SetAbandonByArea(bool b);
        bool IsAbandonByArea();

        cv::Point2f& GetLeftDownCorner();

        void SetArea(float area);
        float& GetArea();

        void SetBlurType(eMLABlurType type);
        const eMLABlurType GetBlurType();

        void SetBlurRadius(float blurRadius);
        float GetBlurRadius();

    private:
        cv::Point2f             m_Center;
        int                     m_Col;
        int                     m_Row;

        eMLABlurType            m_blurType;
        float                   m_blurRadius;

        float                   m_Area;
        bool                    m_BAbandon_area; // 因面积较少而删除:该值为true,文档中写为 1 表示面积过小而被剔除
        float                   score;

        cv::Point2f             m_LeftDownCorner; // 左下角坐标：用于切割矩形图像，匹配时使用
    };
    typedef std::shared_ptr<MLA_Info> MLA_InfoPtr;
    typedef std::vector<MLA_InfoPtr> MLA_InfoVec;
    typedef std::map<QuadTreeTileKeyPtr, MLA_InfoPtr, QuadTreeTileKeyMapCmpLess> QuadTreeTileInfoMap; // <tilekey, MLA_Info>
    typedef std::vector<QuadTreeTileInfoMap> QuadTreeTileInfoMapVec;


    struct NeighborCorrespondingInfo
    {
        NeighborCorrespondingInfo()
        {
            m_Point2d = Vec2i(-1, -1);
            m_Contribute = 1.0;
        }
        Vec2i           m_Point2d;
        float           m_Contribute; // 融合时，对参考图像的贡献
    };
    typedef std::map<QuadTreeTileKeyPtr, NeighborCorrespondingInfo, QuadTreeTileKeyMapCmpLess> NeighborCorrInfoMap;
    struct PointPacket
    {
        Vec3f                   m_Point3D; // 深度图中的（x,y,depth）
        NeighborCorrInfoMap     m_NeigInfoMap; // 邻居信息
    };
    typedef std::map<int64_t, PointPacket> PointPacketMap;

}
#endif //COMMONUTIL_H
