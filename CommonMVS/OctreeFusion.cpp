#include "OctreeFusion.h"


//////////////////////////////////////////////////////////////////////////
namespace SEACAVE
{
    // max number of items in one cell
#define OCTREE_CELLITEMS SIZE

#ifdef _USE_OPENMP
    // minimum number of polygons for which we do multi-threading
#define OCTREE_MIN_ITEMS_MINTHREAD 1024*2
#endif

    // m_pointNum of a cell
#define OCTREE_CELLSIZE (TYPE(NOM)/TYPE(DENOM))
    // radius of a cell
#define OCTREE_CELLRADIUS (OCTREE_CELLSIZE/2)

    /////////////////////////////////////////////////////
    inline TOctreeCellFusion::TOctreeCellFusion(TOctreeCellFusion* children)
        : m_child(children)
        , m_pCoord(NULL)
    {
    }

    inline TOctreeCellFusion::~TOctreeCellFusion()
    {
        Release();
    }

    inline void TOctreeCellFusion::Release()
    {
        delete[] m_child;
        m_child = NULL;
    }

    void TOctreeCellFusion::SetCoord(POINT_TYPE* pCoord)
    {
        m_pCoord = pCoord;
    }

    SEACAVE::TOctreeCellFusion::POINT_TYPE* TOctreeCellFusion::GetCoord()
    {
        return m_pCoord;
    }

    void TOctreeCellFusion::ResetCoord()
    {
        m_pCoord = NULL;
    }

    void TOctreeCellFusion::DeleteCoord()
    {
        if (m_pCoord != NULL)
        {
            delete m_pCoord;
            m_pCoord = NULL;
        }
    }

    bool TOctreeCellFusion::IsCoordEmpty()
    {
        if (m_pCoord == NULL)
            return true;
        return false;
    }

    // compute item's index corresponding to the containing cell
    inline unsigned TOctreeCellFusion::ComputeChild(const POINT_TYPE& item , const POINT_TYPE& center) const
    {
        unsigned idx = 0;
        if (item[0] >= center[0])
            idx |= (1 << 0);
        if (3 > 1)
            if (item[1] >= center[1])
                idx |= (1 << 1);
        if (3 > 2)
            if (item[2] >= center[2])
                idx |= (1 << 2);
        return idx;
    } // ComputeChild

    void TOctreeCellFusion::ComputeCenter(POINT_TYPE centers[])
    {
        centers[0] << -1, -1, -1;
        centers[1] << 1, -1, -1;
        centers[2] << -1, 1, -1;
        centers[3] << 1, 1, -1;
        centers[4] << -1, -1, 1;
        centers[5] << 1, -1, 1;
        centers[6] << -1, 1, 1;
        centers[7] << 1, 1, 1;
    } // ComputeCenter


    bool TOctreeCellFusion::NoChildren() const
    {
        return (m_child == NULL);
    }

    // count the number of items contained by the given octree-cell
    //size_t TOctreeCellFusion::GetNumItemsHeld() const
    //{
    //    if (HasChildren())
    //        return GetNumItems();
    //    size_t numItems = 0;
    //    for (int i = 0; i < numChildren; ++i)
    //        numItems += GetChild(i).GetNumItemsHeld();
    //    return numItems;
    //}

    //////////////////////////////////////////////////////////////////////////
    inline void TOctreeFusion::GetTOctreeLeafs(std::vector<F_CELL_TYPE*>& toctreeleafs)
    {
        toctreeleafs.swap(m_TOctreeLeafs);
        //toctreeleafs.assign(m_TOctreeLeafs.begin(), m_TOctreeLeafs.end());
    }

    void TOctreeFusion::IniOctree(const F_AABB_TYPE& AB, int NUM, float SIZE)
    {
        m_VoxelSize = SIZE;
        m_VoxelNum = NUM;
        cellTotalNum = 0;

        minPoint = AB.GetCenter() - F_POINT_TYPE::Constant(GetRadius(AB));
        m_AB = AB;

        bool IsFinishe = false;
        TYPE _Radius = GetRadius(AB);
        while (!IsFinishe)
        {
            _Radius = _Radius / TYPE(2);
            if (_Radius <= m_VoxelSize)
            {
                TYPE UP_Radius = _Radius * TYPE(2);
                float ElementOne = UP_Radius - m_VoxelSize;
                float ElementTwo = m_VoxelSize - _Radius;
                if (ElementOne > ElementTwo)
                {
                    m_VoxelSize = _Radius;
                }
                else
                {
                    m_VoxelSize = UP_Radius;
                }
                IsFinishe = true;
            }
        }
        m_RootCell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];

        // LZD:�������
        //m_VoxelSize *= 2.0;

        VERBOSE("OCF, Voxel size for density optimize, Previous: %f, computed: %f", SIZE, m_VoxelSize);
    }

    bool TOctreeFusion::InsertForFusion(F_POINT_TYPE& pointCoord)
    {
        TYPE f_radius = GetRadius(m_AB);
        const F_POINT_TYPE f_center = m_AB.GetCenter();
        const unsigned idx = m_RootCell.ComputeChild(pointCoord, f_center);
        F_CELL_TYPE& cell = m_RootCell.m_child[idx];
        bool Judge = _InsertForFusion(idx, cell, pointCoord, f_center, f_radius);
        return Judge;
    }

    inline bool TOctreeFusion::_InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius)
    {
        if (F_Cell.NoChildren())
        {
            if (f_radius <= m_VoxelSize)
            {
                if (F_Cell.IsCoordEmpty())
                {
                    F_Cell.SetCoord(new F_POINT_TYPE(F_Point));
                    return true;
                }
                return false;
            }
            else
            {
                if (F_Cell.IsCoordEmpty())
                {
                    F_Cell.SetCoord(new F_POINT_TYPE(F_Point));
                    return true;
                }
                else
                {
                    // �ѷ�
                    bool Judge = InsertForFusion_(idx, F_Cell, f_center, f_radius, F_Point);
                    return Judge;
                }
            }
        }
        else // �ݹ�
        {
            TYPE childradius = f_radius / TYPE(2);
            F_POINT_TYPE childcenter(ComputeChildCenter(f_center, childradius, idx));

            const unsigned _idx = F_Cell.ComputeChild(F_Point, childcenter);
            F_CELL_TYPE& cell = F_Cell.m_child[_idx];
            bool Judge = _InsertForFusion(_idx, cell, F_Point, childcenter, childradius);
            return Judge;
        }
    }

    inline bool TOctreeFusion::InsertForFusion_(const unsigned idx, F_CELL_TYPE& F_Cell, const F_POINT_TYPE& Center, TYPE F_Radius, F_POINT_TYPE& F_Point)
    {
        // ��ʼ���ӽڵ�
        F_Cell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];

        TYPE childradius = F_Radius / TYPE(2);
        F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));

        F_POINT_TYPE* pCell_Point0 = F_Cell.GetCoord();
        const unsigned idx0 = F_Cell.ComputeChild(*pCell_Point0, childcenter);
        const unsigned idx1 = F_Cell.ComputeChild(F_Point, childcenter);

        if (idx0 != idx1)
        {
            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
            child0.SetCoord(pCell_Point0);
            F_Cell.ResetCoord();

            F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
            child1.SetCoord(new F_POINT_TYPE(F_Point));
            return true;
        }
        else
        {
            if (childradius <= m_VoxelSize)
            {
                F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
                child0.SetCoord(pCell_Point0);
                F_Cell.ResetCoord();
                return false;
            }
            else
            {
                F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
                child0.SetCoord(pCell_Point0);
                F_Cell.ResetCoord();

                bool Judge = InsertForFusion_(idx0, child0, childcenter, childradius, F_Point);
                return Judge;
            }
        }
    }

    /////////�����˲���///////////////////////
    inline void TOctreeFusion::TraverseOctree(int& leafNum, int& pointNum)
    {
        for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
        {
            F_CELL_TYPE& cell = m_RootCell.m_child[i];
            TraverseOctree_(cell, leafNum, pointNum);
        }
    }

    inline void TOctreeFusion::TraverseOctree_(F_CELL_TYPE& F_Cell, int& leafNum, int& pointNum)
    {
        if (F_Cell.NoChildren())
        {
            leafNum += 1;
            if (F_Cell.IsCoordEmpty())
            {
                pointNum += 1;
            }
        }
        else
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                F_CELL_TYPE& cell = F_Cell.m_child[i];
                TraverseOctree_(cell, leafNum, pointNum);
            }
        }
        return; // 添加显式返回以避免编译器警告
    }

    void TOctreeFusion::DeleteCellInfo()
    {
        if (!m_RootCell.NoChildren())
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                F_CELL_TYPE& cell = m_RootCell.m_child[i];
                DeleteCellInfo_(cell);
            }
        }
    }

    void TOctreeFusion::DeleteCellInfo_(F_CELL_TYPE& F_Cell)
    {
        if (F_Cell.NoChildren())
        {
            F_Cell.DeleteCoord();
        }
        else
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                F_CELL_TYPE& cell = F_Cell.m_child[i];
                DeleteCellInfo_(cell);
            }
        }
    }

    TOctreeFusion::~TOctreeFusion()
    {
        Release();
    }

    void TOctreeFusion::Release()
    {
        DeleteCellInfo();
    }

    SEACAVE::TOctreeFusion::TYPE TOctreeFusion::GetRadius(const F_AABB_TYPE& aabb)
    {
        const F_POINT_TYPE size(aabb.GetSize() / TYPE(2));
        TYPE radius = size[0];
        if (3 > 1 && radius < size[1])
            radius = size[1];
        if (3 > 2 && radius < size[2])
            radius = size[2];
        return radius;
    }

    SEACAVE::TOctreeFusion::F_POINT_TYPE TOctreeFusion::ComputeChildCenter(const F_POINT_TYPE& center, TYPE radius, unsigned idxChild)
    {
        struct CENTERARR_TYPE
        {
            F_POINT_TYPE        child[F_CELL_TYPE::numChildren];
            inline CENTERARR_TYPE() { F_CELL_TYPE::ComputeCenter(child); }
        };
        static const CENTERARR_TYPE centers;
        return center + centers.child[idxChild] * radius;
    }

    SEACAVE::VoxelKey TOctreeFusion::ComputeVoxelKey(const F_POINT_TYPE& center, TYPE radius, unsigned levelChild, unsigned idxChild)
    {
        VoxelKey m_Num;
        m_Num.m_LOD = levelChild;
        F_POINT_TYPE m_point;
        if (idxChild == 0)
        {
            F_POINT_TYPE point_(center[0] - radius, center[1] - radius, center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 1)
        {
            F_POINT_TYPE point_(center[0], center[1] - radius, center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 2)
        {
            F_POINT_TYPE point_(center[0] - radius, center[1], center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 3)
        {
            F_POINT_TYPE point_(center[0], center[1], center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 4)
        {
            F_POINT_TYPE point_(center[0] - radius, center[1] - radius, center[2]);
            m_point = point_;
        }
        else if (idxChild == 5)
        {
            F_POINT_TYPE point_(center[0], center[1] - radius, center[2]);
            m_point = point_;
        }
        else if (idxChild == 6)
        {
            F_POINT_TYPE point_(center[0] - radius, center[1], center[2]);
            m_point = point_;
        }
        else if (idxChild == 7)
        {
            F_POINT_TYPE point_(center[0], center[1], center[2]);
            m_point = point_;
        }

        m_Num.m_VoxelX = round((m_point[0] - minPoint[0]) / radius);
        m_Num.m_VoxelY = round((m_point[1] - minPoint[1]) / radius);
        m_Num.m_VoxelZ = round((m_point[2] - minPoint[2]) / radius);
        return m_Num;
    }
}
