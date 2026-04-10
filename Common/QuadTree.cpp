/********************************************************************
file base:      QuadTree.cpp
author:         LZD
created:        2024/05/13
purpose:
*********************************************************************/
#include "QuadTree.h"
#include "iostream"

namespace LFMVS
{
    QuadTreeBoundingbox::QuadTreeBoundingbox(double minX, double minY, double minZ,
                                            double maxX, double maxY, double maxZ)
    {
        min_X = minX;
        min_Y = minY;
        min_Z = minZ;
        max_X = maxX;
        max_Y = maxY;
        max_Z = maxZ;
    }

    QuadTreeTileKey::QuadTreeTileKey(eTileKeyQuadrant eQuadrant, int32_t lod, int32_t tile_x,
                                    int32_t tile_y, QuadTreeBoundingbox box)
             : m_eQuadrant(eQuadrant)
             , m_LOD(lod)
             , m_TileX(tile_x)
             , m_TileY(tile_y)
             , m_BoundingBox(box)
    {
        std::stringstream stream;
        stream<<m_LOD<<"_"<<m_TileX<<"_"<<m_TileY;
        m_strKey = stream.str();
        std::stringstream stream2;
        stream2<<m_TileX<<"_"<<m_TileY;
        m_strKey_RemoveLOD = stream2.str();
    }

    QuadTreeBoundingbox::~QuadTreeBoundingbox()
    {
        Release();
    }

    void QuadTreeBoundingbox::Release()
    {
        m_pLeftNeighborVec.clear();
        m_pRightNeighborVec.clear();
        m_pDownNeighborVec.clear();
        m_pUpNeighborVec.clear();
    }

    void QuadTreeBoundingbox::Print()
    {
        std::cout<< "min_x: " << min_X << ", min_y: " << min_Y<< ", min_z: " << min_Z <<std::endl;
        std::cout<< "max_x: " << max_X << ", max_y: " << max_Y<< ", max_z: " << max_Z <<std::endl;
        std::cout<< "x_length: " << max_X - min_X << ", y_length: "
        << max_Y - min_Y << ", z_length: " << max_Z - min_Z <<std::endl;
    }

    bool QuadTreeTileKey::IsSame(const QuadTreeTileKey* pKey)
    {
        if (this == pKey)
            return true;
        return *this == *pKey;
    }


    std::shared_ptr<QuadTreeTileKey> QuadTreeTileKey::CreateChildKey(int32_t quadrant)
    {
        int32_t lod = m_LOD + 1;
        int32_t x = m_TileX << 1; // 位移
        int32_t y = m_TileY << 1; // 位移

        // 计算包围盒
        QuadTreeBoundingbox box;
        switch (quadrant)
        {
            case 0: // 左上角孩子
            {
                box.min_X = m_BoundingBox.min_X;
                box.max_X = m_BoundingBox.min_X + (m_BoundingBox.max_X - m_BoundingBox.min_X)*0.5;
                box.min_Y = m_BoundingBox.min_Y + (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*0.5;
                box.max_Y = m_BoundingBox.max_Y;
            }
                break;
            case 1: // 右上角孩子
            {
                x += 1;
                box.min_X = m_BoundingBox.min_X + (m_BoundingBox.max_X - m_BoundingBox.min_X)*0.5;
                box.max_X = m_BoundingBox.max_X;
                box.min_Y = m_BoundingBox.min_Y + (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*0.5;
                box.max_Y = m_BoundingBox.max_Y;
            }
                break;
            case 2: // 左下角孩子
            {
                y += 1;
                box.min_X = m_BoundingBox.min_X;
                box.max_X = m_BoundingBox.min_X + (m_BoundingBox.max_X - m_BoundingBox.min_X)*0.5;
                box.min_Y = m_BoundingBox.min_Y;
                box.max_Y = m_BoundingBox.min_Y + (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*0.5;
            }
                break;
            case 3: // 右下角孩子
            {
                x += 1;
                y += 1;
                box.min_X = m_BoundingBox.min_X + (m_BoundingBox.max_X - m_BoundingBox.min_X)*0.5;
                box.max_X = m_BoundingBox.max_X;
                box.min_Y = m_BoundingBox.min_Y;
                box.max_Y = m_BoundingBox.min_Y + (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*0.5;
            }
                break;
            default:
                assert(0);
                break;
        }
        box.min_Z = m_BoundingBox.min_Z;
        box.max_Z = m_BoundingBox.max_Z;

        QuadTreeTileKeyPtr ptrKey = std::make_shared<QuadTreeTileKey>((eTileKeyQuadrant)quadrant, lod, x, y, box);
        return ptrKey;
    }


    std::shared_ptr<QuadTreeTileKey> QuadTreeTileKey::CreateParentKey()
    {
        if (m_LOD == 0)
            return NULL;
        int32_t lod = m_LOD - 1;
        int32_t x = m_TileX >> 1;
        int32_t y = m_TileY >> 1;

        // 计算包围盒
        QuadTreeBoundingbox parentBox;
        switch (m_eQuadrant)
        {
            case TileKey_LeftUp:
            {
                parentBox.min_X = m_BoundingBox.min_X;
                parentBox.max_X = m_BoundingBox.min_X + (m_BoundingBox.max_X - m_BoundingBox.min_X)*2.0;
                parentBox.min_Y = m_BoundingBox.max_Y - (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*2.0;
                parentBox.max_Y = m_BoundingBox.max_Y;
            }
                break;
            case TileKey_RightUp:
            {
                parentBox.min_X = m_BoundingBox.max_X - (m_BoundingBox.max_X - m_BoundingBox.min_X)*2.0;
                parentBox.max_X = m_BoundingBox.max_X;
                parentBox.min_Y = m_BoundingBox.max_Y - (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*2.0;
                parentBox.max_Y = m_BoundingBox.max_Y;
            }
                break;
            case TileKey_LeftDown:
            {
                parentBox.min_X = m_BoundingBox.min_X;
                parentBox.max_X = m_BoundingBox.min_X + (m_BoundingBox.max_X - m_BoundingBox.min_X)*2.0;
                parentBox.min_Y = m_BoundingBox.min_Y;
                parentBox.max_Y = m_BoundingBox.min_Y + (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*2.0;
            }
                break;
            case TileKey_RightDown:
            {
                parentBox.min_X = m_BoundingBox.max_X - (m_BoundingBox.max_X - m_BoundingBox.min_X)*2.0;
                parentBox.max_X = m_BoundingBox.max_X;
                parentBox.min_Y = m_BoundingBox.min_Y;
                parentBox.max_Y = m_BoundingBox.min_Y + (m_BoundingBox.max_Y - m_BoundingBox.min_Y)*2.0;
            }
                break;
            default:
                break;
        }

        QuadTreeTileKeyPtr ptrKey = std::make_shared<QuadTreeTileKey>(TileKey_None, lod, x, y, parentBox);
        return ptrKey;
    }

    std::shared_ptr<QuadTreeTileKey> QuadTreeTileKey::CreateInstance(eTileKeyQuadrant eQuadrant, int32_t lod,
        int32_t tile_x, int32_t tile_y, QuadTreeBoundingbox box)
    {
        QuadTreeTileKeyPtr ptrKey = std::make_shared<QuadTreeTileKey>(TileKey_None, lod, tile_x, tile_y, box);
        return ptrKey;
    }

    QuadTreeTileKeyPtr QuadTreeTileKey::CreateNeighborKey(int x_offset, int y_offset)
    {
        //if (m_LOD == 0)
           // return NULL;

        // 计算编号
        int32_t lod = m_LOD;
        int32_t x = m_TileX + x_offset;
        int32_t y = m_TileY + y_offset;

        // 计算包围盒
        double x_length = m_BoundingBox.max_X - m_BoundingBox.min_X;
        double y_length = m_BoundingBox.max_Y - m_BoundingBox.min_Y;
        QuadTreeBoundingbox neighborBox;
        neighborBox.min_X = m_BoundingBox.min_X + x_offset*x_length;
        neighborBox.max_X = m_BoundingBox.max_X + x_offset*x_length;
        neighborBox.min_Y = m_BoundingBox.min_Y + y_offset*y_length;
        neighborBox.max_Y = m_BoundingBox.max_Y + y_offset*y_length;

        QuadTreeTileKeyPtr ptrKey = std::make_shared<QuadTreeTileKey>(TileKey_None, lod, x, y, neighborBox);
        return ptrKey;
    }
}

