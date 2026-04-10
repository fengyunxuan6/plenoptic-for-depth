////////////////////////////////////////////////////////////////////
#ifndef __SEACAVE_OCTREEBYFUSION_H__
#define __SEACAVE_OCTREEBYFUSION_H__

#include "Common.h"
//#include "AABB.h"
//#include "Ray.h"
//#include "List.h"

#include "Octree.h"

//////////////////////////////////////////////////////////////////////////
namespace SEACAVE
{
// octree cell class
class TOctreeCellFusion
{
public:
    typedef float TYPE;
    typedef uint32_t DATA_TYPE;

    typedef Eigen::Matrix<float, 3, 1> POINT_TYPE;
    typedef SEACAVE::TAABB<float, 3> AABB_TYPE;
    typedef uint32_t SIZE_TYPE;
    // enum { dataSize = (sizeof(NODE_TYPE) > sizeof(LEAF_TYPE) ? sizeof(NODE_TYPE) : sizeof(LEAF_TYPE)) };
    enum { numChildren = (2 << (3 - 1)) };

public:
    inline TOctreeCellFusion()
        : m_child(NULL)
        , m_pCoord(NULL)
    {}

    inline TOctreeCellFusion(TOctreeCellFusion*);
    inline ~TOctreeCellFusion();

    inline void Release();

public:
    void SetCoord(POINT_TYPE* pCoord);
    POINT_TYPE* GetCoord();
    void ResetCoord();
    void DeleteCoord();
    bool IsCoordEmpty();

    inline unsigned ComputeChild(const POINT_TYPE& item , const POINT_TYPE& center) const;
    static void ComputeCenter(POINT_TYPE[]);

    bool	NoChildren() const;

    inline const TOctreeCellFusion& GetChild(int i) const
    { ASSERT(!NoChildren() && i < numChildren); return m_child[i]; }

    SIZE_TYPE GetNumItems()
    {
        if (m_pCoord)
            return 1;
        return 0;
    }
    //size_t GetNumItemsHeld() const;

public:
    TOctreeCellFusion*                  m_child; // if not a leaf, 2^DIMS child objects
    POINT_TYPE*                         m_pCoord;
};

//////////////////////////////////////////////////////////////////////////
class TOctreeFusion
{
public:
    typedef float TYPE;
    typedef uint32_t DATA_TYPE;

    typedef TOctreeCellFusion F_CELL_TYPE;
    typedef Eigen::Matrix<TYPE, 3, 1> F_POINT_TYPE;
    typedef SEACAVE::TAABB<TYPE, 3> F_AABB_TYPE;

public:
    inline TOctreeFusion() {};
    ~TOctreeFusion();
    void Release();

public:
    inline std::map<VoxelKey, F_CELL_TYPE*>& GetLeafsMap()
    {
        return arrOctreeLeafMap;
    }

    TYPE GetRadius(const F_AABB_TYPE& aabb);

    F_POINT_TYPE ComputeChildCenter(const F_POINT_TYPE& center, TYPE radius, unsigned idxChild);

    VoxelKey ComputeVoxelKey(const F_POINT_TYPE& center, TYPE radius, unsigned levelChild, unsigned idxChild);

    inline F_POINT_TYPE GetMinPoint(const F_POINT_TYPE& center, TYPE radius)
    {
        return center - F_POINT_TYPE::Constant(radius);
    }

    void IniOctree(const F_AABB_TYPE&, int, float);

    bool InsertForFusion(F_POINT_TYPE& pointCoord);
    inline void GetTOctreeLeafs(std::vector<F_CELL_TYPE*>& toctreeleafs);
    inline void TraverseOctree(int& leafNum, int& pointNum);
    void DeleteCellInfo();

protected:
    inline bool _InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius);
    inline bool InsertForFusion_(const unsigned idx, F_CELL_TYPE& F_Cell, const F_POINT_TYPE& Center, TYPE F_Radius, F_POINT_TYPE& F_Point);

    inline void TraverseOctree_(F_CELL_TYPE& F_Cell, int& leafNum, int& pointNum);
    void DeleteCellInfo_(F_CELL_TYPE& F_Cell);

public:
    std::vector<VoxelKey>                   arrOctreeCellNum;
    std::map<VoxelKey, F_CELL_TYPE*>        arrOctreeCellMap; // ���нڵ�
    std::map<VoxelKey, F_CELL_TYPE*>        arrOctreeLeafMap; // Ҷ�ڵ�
    int                                     cellTotalNum;
    F_POINT_TYPE                            minPoint;

protected:
    float                                   m_VoxelSize;
    int                                     m_VoxelNum;
    F_CELL_TYPE                             m_RootCell;
    std::vector<F_CELL_TYPE*>               m_TOctreeLeafs;
    F_AABB_TYPE                             m_AB;
};
} // namespace SEACAVE
#endif // __SEACAVE_OCTREE_H__
