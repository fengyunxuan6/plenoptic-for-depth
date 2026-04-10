////////////////////////////////////////////////////////////////////
#ifndef __SEACAVE_OCTREEBYFUSIONVOXEL_H__
#define __SEACAVE_OCTREEBYFUSIONVOXEL_H__

#include "Common.h"
//#include "AABB.h"
//#include "Ray.h"
//#include "List.h"

#include "Octree.h"
#include <Eigen/Core>

//////////////////////////////////////////////////////////////////////////
namespace SEACAVE
{
// octree cell class
class TOctreeCellFusionVoxel
{
public:
    typedef float TYPE;
    typedef uint32_t DATA_TYPE;

    typedef Eigen::Matrix<float, 3, 1> POINT_TYPE;
    typedef SEACAVE::TAABB<float, 3> AABB_TYPE;
    typedef uint32_t SIZE_TYPE;
    // enum { dataSize = (sizeof(NODE_TYPE) > sizeof(LEAF_TYPE) ? sizeof(NODE_TYPE) : sizeof(LEAF_TYPE)) };
    enum { numChildren = (2 << (3 - 1)) };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    inline TOctreeCellFusionVoxel()
        : m_child(NULL)
        , m_pCoord(NULL)
        , m_nodeRadius(0)
    {}

    inline TOctreeCellFusionVoxel(TOctreeCellFusionVoxel*);
    inline ~TOctreeCellFusionVoxel();

    inline void Release();

public:
    void SetCoord(POINT_TYPE* pCoord);
    POINT_TYPE* GetCoord();
    void ResetCoord();
    void DeleteCoord();
    bool IsCoordValid();

    void SetCenter(const POINT_TYPE& nodeCenter);
    const POINT_TYPE& GetCenter() const;
    void SetRadius(TYPE radius);
    TYPE& GetRadius();

    inline unsigned ComputeChild(const POINT_TYPE& item) const;
    static void ComputeCenter(POINT_TYPE[]);

    bool	NoChildren() const;

    inline const TOctreeCellFusionVoxel& GetChild(int i) const
    { ASSERT(!NoChildren() && i < numChildren); return m_child[i]; }

    AABB_TYPE GetAabb(TYPE radius)
    {
        return AABB_TYPE(m_nodeCenter, radius);
    }

    SIZE_TYPE GetNumItems()
    {
        if (m_pCoord)
            return 1;
        return 0;
    }

    SIZE_TYPE GetNumItems_Bucket()
    {
        return static_cast<SIZE_TYPE>(m_points.size());
    }

    //size_t GetNumItemsHeld() const;

    inline std::vector<Point3f>& GetCornerPoints()
    {
        return m_CornerPoints;
    }

public:
    TOctreeCellFusionVoxel*             m_child; // if not a leaf, 2^DIMS child objects
    TOctreeCellFusionVoxel*             m_father; //父节点

    POINT_TYPE                          m_nodeCenter; // m_nodeCenter of the current cell
    TYPE                                m_nodeRadius;
    POINT_TYPE*                         m_pCoord;

    VoxelKey                            m_VoxelKey;
    std::vector<Point3f>                m_CornerPoints;

    std::vector<POINT_TYPE, Eigen::aligned_allocator<POINT_TYPE>> m_points;
};

//////////////////////////////////////////////////////////////////////////
class TOctreeFusionVoxel
{
public:
    typedef float TYPE;
    typedef uint32_t DATA_TYPE;

    typedef TOctreeCellFusionVoxel CELL_TYPE;
    typedef Eigen::Matrix<TYPE, 3, 1> F_POINT_TYPE;
    typedef SEACAVE::TAABB<TYPE, 3> F_AABB_TYPE;

    // by xyy
    struct Int3Key {
        int x, y, z;
        bool operator==(const Int3Key& o) const { return x==o.x && y==o.y && z==o.z; }
    };
    struct Int3KeyHash {
        size_t operator()(const Int3Key& k) const noexcept {
            // 简单 hash，足够用
            size_t h = 1469598103934665603ull;
            h ^= (size_t)k.x + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
            h ^= (size_t)k.y + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
            h ^= (size_t)k.z + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
            return h;
        }
    };

public:
    inline TOctreeFusionVoxel() {};
    ~TOctreeFusionVoxel();
    void Release();

public:
    inline std::map<VoxelKey, CELL_TYPE*>& GetLeafsMap()
    {
        return arrOctreeLeafMap;
    }

    float GetVoxelSize()
    {
        return VoxelSize;
    }

    TYPE GetRadius(const F_AABB_TYPE& aabb);

    F_POINT_TYPE ComputeChildCenter(const F_POINT_TYPE& center, TYPE radius, unsigned idxChild);

    VoxelKey ComputeVoxelKey(const F_POINT_TYPE& center, TYPE radius, unsigned levelChild, unsigned idxChild);

    inline F_POINT_TYPE GetMinPoint(const F_POINT_TYPE& center, TYPE radius)
    {
        return center - F_POINT_TYPE::Constant(radius);
    }

    void GetNeighbor(VoxelKey num, std::vector<VoxelKey>& neighbor);
    void Get_Neighbor(VoxelKey num, VoxelKey h_num,int handle,std::vector<VoxelKey>& returnNumArr);
    bool _GetNeighbor(CELL_TYPE& cell, VoxelKey& returnNum,int handle);

    void GetNeighbor_(CELL_TYPE& cell, std::vector<VoxelKey>& returnNumArr, int handle);

    void IniOctree(const F_AABB_TYPE&, int, float);

    bool InsertForFusion(const F_POINT_TYPE& pointCoord);
    bool InsertForFusion_Bucket(const F_POINT_TYPE& pointCoord);    // by xyy
    bool InsertForFusionOS(const F_POINT_TYPE& F_Point, uint32_t imageID);
    inline void GetTOctreeLeafs(std::vector<CELL_TYPE*>& toctreeleafs);
    void CollectLeafs(bool bBox = false); // 遍历八叉树，搜集叶节点
    inline void TraverseOctree(int& leafNum, int& pointNum);
    void DeleteCellInfo();

    // by xyy----
    bool ExportOccupiedLeafVoxelsObj(std::string& objPath,    // by xyy
                                     bool includeRoot,
                                     float voxelScale);
    bool ExportOccupiedLeafSurfaceObj(const std::string& objPath,
                                                          const std::vector<CELL_TYPE*>& occupiedLeaves,
                                                          bool includeRoot /*=false*/);
    inline int GetOrAddVertex(std::ofstream& os,
                              std::unordered_map<Int3Key, int, Int3KeyHash>& vtxMap,
                              const Eigen::Vector3f& minPoint,
                              float cellSize,
                              int vx, int vy, int vz,
                              int& vBase);
    bool ExportInitOctreeObj_RootPlus8Children(F_AABB_TYPE& ab,
                                               std::string objPath,
                                               bool includeRoot,
                                               float childScale);
    // by xyy-----

protected:
    inline bool _InsertForFusion(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius);
    inline bool InsertForFusion_(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& Center, TYPE F_Radius, const F_POINT_TYPE& F_Point);

    inline bool _InsertForFusion_Bucket(CELL_TYPE& F_Cell, const F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius);   // by xyy

    inline float GetRootRadiusFromAABB( F_AABB_TYPE& ab);   // by xyy
    inline void AppendBoxAsTriangles(std::ofstream& os,     // by xyy
                                     TOctreeCellFusionVoxel::POINT_TYPE& c,
                                     float r,
                                     int& vBase);


    inline bool _InsertForFusionOS(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius, uint32_t imageID);
    bool InsertForFusion_OS(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& Center, TYPE F_Radius, const F_POINT_TYPE& F_Point, uint32_t imageID);

    inline void TraverseOctree_(CELL_TYPE& F_Cell, bool bBox=false);
    inline void TraverseOctree_Bucket(CELL_TYPE& F_Cell, bool bBox=false);    // by xyy
    inline void AddLeaf(CELL_TYPE& F_Cell, bool bBox = false);

    inline void TraverseOctree_(CELL_TYPE& F_Cell, int& leafNum, int& pointNum);
    void DeleteCellInfo_(CELL_TYPE& F_Cell);

public:
    std::vector<VoxelKey>                   arrOctreeCellNum;
    std::map<VoxelKey, CELL_TYPE*>        arrOctreeCellMap; // 所有节点
    std::map<VoxelKey, CELL_TYPE*>        arrOctreeLeafMap; // 叶节点
    int                                     cellTotalNum;
    F_POINT_TYPE                            minPoint;

    // by xyy
    int occupiedLeafCount = 0;
    int occupiedPointCount = 0;
    std::vector<CELL_TYPE*> occupiedLeaves;


protected:
    float                                   VoxelSize;
    int                                     VoxelNum;
    CELL_TYPE                             F_Root;
    std::vector<CELL_TYPE*>               TOctreeLeafs;
};
} // namespace SEACAVE
#endif // __SEACAVE_OCTREE_H__