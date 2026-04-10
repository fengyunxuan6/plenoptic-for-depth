/********************************************************************
file base:      QuadTree.h
author:         LZD
created:        2024/05/13
purpose:
*********************************************************************/
#ifndef ACMP_QUADTREE_H
#define ACMP_QUADTREE_H

#include "vector"
#include "map"
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "Common.h"

namespace LFMVS
{
    // 四叉树结构
    enum eTileKeyQuadrant  // 当前瓦片位于父亲瓦片的编码
    {
        TileKey_None = -1, // 根节点的相对空间关系
        TileKey_LeftUp = 0,
        TileKey_RightUp = 1,
        TileKey_LeftDown = 2,
        TileKey_RightDown = 3,
        TileKey_Error = 4,
    };

    class QuadTreeBoundingbox
    {
    public:
        QuadTreeBoundingbox()
        {
            min_X = 0;
            min_Y = 0;
            min_Z = 0;
            max_X = 0;
            max_Y = 0;
            max_Z = 0;
        }

        QuadTreeBoundingbox(double minX, double minY, double minZ, double maxX, double maxY, double maxZ);

        ~QuadTreeBoundingbox();
        void Release();

        void Print();

    public:
        double                  min_X;
        double                  min_Y;
        double                  min_Z;
        double                  max_X;
        double                  max_Y;
        double                  max_Z;
        // 四个方位的边邻居
        std::vector<QuadTreeBoundingbox*>    m_pLeftNeighborVec;
        std::vector<QuadTreeBoundingbox*>    m_pRightNeighborVec;
        std::vector<QuadTreeBoundingbox*>    m_pDownNeighborVec;
        std::vector<QuadTreeBoundingbox*>    m_pUpNeighborVec;
    };
    typedef std::shared_ptr<QuadTreeBoundingbox> QuadTreeBoundingboxPtr;

    class QuadTreeTileKey
    {
        // 四叉树编码规则：横向为X，从左向右依次递增；纵向为Y，从上到下依次递增。
        // 父子之间的空间位置关系：0号孩子为父亲的左上角，1号为右上角，2号为左下角，3号为右下角
    public:
        QuadTreeTileKey()
        {
        }

        QuadTreeTileKey(eTileKeyQuadrant eQuadrant, int32_t lod, int32_t tile_x,
                        int32_t tile_y, QuadTreeBoundingbox box = QuadTreeBoundingbox());

        // 创建Tilekey
        static  std::shared_ptr<QuadTreeTileKey> CreateInstance(eTileKeyQuadrant eQuadrant, int32_t lod,
                int32_t tile_x, int32_t tile_y, QuadTreeBoundingbox box = QuadTreeBoundingbox());

        virtual ~QuadTreeTileKey()
        {}

    public:
        int32_t GetLOD()
        {
            return m_LOD;
        }
        int32_t GetTileX()
        {
            return m_TileX;
        }
        int32_t GetTileY()
        {
            return m_TileY;
        }
        QuadTreeBoundingbox& GetBoundingBox()
        {
            return m_BoundingBox;
        }

        // 获取当前TileKey所属的父节点的子节点编号
        int32_t GetQuadrant() const
        {
            return m_eQuadrant;
        }

    public:
        // 创建同层级邻居的Tilekey
        std::shared_ptr<QuadTreeTileKey> CreateNeighborKey(int x_offset, int y_offset);
        // 创建指定的子节点TileKey
        std::shared_ptr<QuadTreeTileKey> CreateChildKey(int32_t quadrant);
        // 创建该瓦片的父级别的TileKey
        std::shared_ptr<QuadTreeTileKey> CreateParentKey();

        const std::string& Str()
        {
            return m_strKey;
        }
        const std::string& StrRemoveLOD()
        {
            return m_strKey_RemoveLOD;
        }
        bool IsSame(const QuadTreeTileKey* pKey);

    public:
        bool operator == (const QuadTreeTileKey& rhs) const
        {
            return m_LOD == rhs.m_LOD && m_TileX == rhs.m_TileX && m_TileY == rhs.m_TileY;
        }

        bool operator != (const QuadTreeTileKey& rhs) const
        {
            return !(*this == rhs);
        }

        bool operator < (const QuadTreeTileKey& rhs) const
        {
            if (m_LOD < rhs.m_LOD)
                return true;
            if (m_LOD > rhs.m_LOD)
                return false;
            if (m_TileY < rhs.m_TileY)
                return true;
            if (m_TileY > rhs.m_TileY)
                return false;
            return m_TileX < rhs.m_TileX;
        }

    private:
        eTileKeyQuadrant                 m_eQuadrant;
        std::string                      m_strKey;
        std::string                      m_strKey_RemoveLOD;
        int32_t                          m_LOD;
        int32_t                          m_TileX;
        int32_t                          m_TileY;
        QuadTreeBoundingbox              m_BoundingBox;
    };
    typedef std::shared_ptr<QuadTreeTileKey> QuadTreeTileKeyPtr;
    typedef std::vector<QuadTreeTileKeyPtr> QuadTreeTileKeyPtrVec;

    struct QuadTreeTileKeyMapCmpLess
    {
        bool operator () (const QuadTreeTileKeyPtr& pLhs, const QuadTreeTileKeyPtr& pRhs) const
        {
            return ((*(pLhs.get())) < (*(pRhs.get())));
        }
    };

    typedef std::map<int, QuadTreeTileKeyPtrVec> QuadTreeTileKeyPtrCircles; // <圈的索引，>
    typedef std::map<QuadTreeTileKeyPtr, QuadTreeTileKeyPtr, QuadTreeTileKeyMapCmpLess> QuadTreeTileKeysMapFast; // 快速查找

}
#endif //ACMP_QUADTREE_H
