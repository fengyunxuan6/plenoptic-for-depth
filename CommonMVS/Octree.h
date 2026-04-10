////////////////////////////////////////////////////////////////////
// Octree.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __SEACAVE_OCTREE_H__
#define __SEACAVE_OCTREE_H__

#define _XADD 0
#define _XSUBTRACT 1
#define _YADD 2
#define _YSUBTRACT 3
#define _ZADD 4
#define _ZSUBTRACT 5

// I N C L U D E S /////////////////////////////////////////////////

//#include "AABB.h"
//#include "Ray.h"
//#include "List.h"
//#include "PointCloud.h"


// D E F I N E S ///////////////////////////////////////////////////

namespace SEACAVE
{
    typedef TPoint3<float> Point;
    typedef SEACAVE::cList<Point, const Point&, 2, 8192> PointArr;

    typedef uint32_t View;
    typedef SEACAVE::cList<View, const View, 0, 4, uint32_t> ViewArr;
    typedef SEACAVE::cList<ViewArr> PointViewArr;

    typedef uint32_t PointIdx;
    typedef SEACAVE::cList<PointIdx, const PointIdx, 0, 4, uint32_t> PointIdxArr;
    typedef std::map<View, PointIdxArr> ViewPointIdxMap;

    typedef float Weight;
    typedef SEACAVE::cList<Weight, const Weight, 0, 4, uint32_t> WeightArr;
    typedef SEACAVE::cList<WeightArr> PointWeightArr;

// S T R U C T S ///////////////////////////////////////////////////
    struct Point3D
    {
        float X;
        float Y;
        float Z;
        float Score;
    };

    struct VoxelKey
    {
        VoxelKey(int lod = 0, int x = 0, int y = 0, int z = 0)
        {
            m_LOD = lod;
            m_VoxelX = x;
            m_VoxelY = y;
            m_VoxelZ = z;
        }
        bool operator==(const VoxelKey &num)
        {
            if (num.m_LOD == m_LOD && num.m_VoxelX == m_VoxelX && num.m_VoxelY == m_VoxelY && num.m_VoxelZ == m_VoxelZ)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        int m_LOD;
        int m_VoxelX;
        int m_VoxelY;
        int m_VoxelZ;
    };
    
    inline bool operator <(const VoxelKey& Num1, const VoxelKey& Num2)
    {
        if (Num1.m_LOD != Num2.m_LOD)
            return Num1.m_LOD < Num2.m_LOD;
        if (Num1.m_VoxelX != Num2.m_VoxelX)
            return Num1.m_VoxelX < Num2.m_VoxelX;
        if (Num1.m_VoxelY != Num2.m_VoxelY)
            return Num1.m_VoxelY < Num2.m_VoxelY;
        return Num1.m_VoxelZ < Num2.m_VoxelZ;
    }

// raw array wrapper
template <typename TYPE>
class TItemArr
{
public:
	typedef TYPE Type;

	inline TItemArr() {}
	inline TItemArr(const TYPE* data, IDX size) : m_data(data), m_size(size) {}
	inline void Set(const TYPE* data, IDX size) { m_data = data; m_size = size; }

	inline const TYPE& operator[](IDX i) const { return m_data[i]; }
	inline const TYPE* Begin() const { return m_data; }
	inline const TYPE* GetData() const { return m_data; }
	inline IDX GetSize() const { return m_size; }

protected:
	const TYPE* m_data;
	IDX m_size;
}; // class TItemArr
/*----------------------------------------------------------------*/


// octree cell class
template <typename TYPE, int DIMS, typename DATA_TYPE>
class TOctreeCell
{
	STATIC_ASSERT(DIMS > 0 && DIMS <= 3);

public:
	typedef Eigen::Matrix<TYPE,DIMS,1> POINT_TYPE;
	typedef SEACAVE::TAABB<TYPE,DIMS> AABB_TYPE;
	typedef uint32_t SIZE_TYPE;

    typedef uint32_t View;
    typedef float Weight;
    typedef TPoint3<float> Normal;

	typedef struct
    {
		POINT_TYPE center; // center of the current cell
        TYPE F_Radius;
	} NODE_TYPE;

    typedef struct
    {
		IDX idxBegin; // index in the global array of the first item contained by this cell
		SIZE_TYPE size; // number of items contained by this cell in the global array
		DATA_TYPE data; // user data associated with this leaf
        float FusionPointScore;   //�洢һ��Ҷ�ڵ����ںϵ��Ƶĳɼ�
        POINT_TYPE FusionPoint;     //�洢һ��Ҷ�ڵ��е��ںϵ�
	}LEAF_TYPE;

	enum { dataSize = (sizeof(NODE_TYPE)>sizeof(LEAF_TYPE) ? sizeof(NODE_TYPE) : sizeof(LEAF_TYPE)) };
	enum { numChildren = (2<<(DIMS-1)) };

public:
	inline TOctreeCell();
	inline TOctreeCell(TOctreeCell*);
	inline ~TOctreeCell();

	inline void Release();
	inline void Swap(TOctreeCell&);

	inline unsigned ComputeChild(const POINT_TYPE& item) const;
	static void ComputeCenter(POINT_TYPE []);

	inline bool	IsLeaf() const { return (m_child==NULL); }
	inline const TOctreeCell& GetChild(int i) const { ASSERT(!IsLeaf() && i<numChildren); return m_child[i]; }
	inline const NODE_TYPE& Node() const { ASSERT(!IsLeaf()); return *((const NODE_TYPE*)m_data); }
	inline NODE_TYPE& Node() { ASSERT(!IsLeaf()); return *((NODE_TYPE*)m_data); }
	inline const LEAF_TYPE& Leaf() const { ASSERT(IsLeaf()); return *((const LEAF_TYPE*)m_data); }
	inline LEAF_TYPE& Leaf() { ASSERT(IsLeaf()); return *((LEAF_TYPE*)m_data); }
	inline const POINT_TYPE& GetCenter() const { return Node().center; }
	inline AABB_TYPE GetAabb(TYPE radius) const { return AABB_TYPE(Node().center, radius); }
	inline IDX GetLastItem() const { return Leaf().idxBegin + Leaf().size; }
	inline SIZE_TYPE GetNumItems() const { return Leaf().size; }
	inline const DATA_TYPE& GetUserData() const { return Leaf().data; }
	inline DATA_TYPE& GetUserData() { return Leaf().data; }
	size_t GetNumItemsHeld() const;

    inline std::vector<Point3f>& GetCornerPoints()
    {
        return m_CornerPoints;
    }

public:
	TOctreeCell*                m_child; // if not a leaf, 2^DIMS child objects
    TOctreeCell*                m_father; //���ڵ�
	uint8_t                     m_data[dataSize]; // a LEAF_TYPE or NODE_TYPE object, if it is a leaf or not respectively
    // all
    //std::vector<Point3D>        Points_3D;
    // best
    ViewArr*                     FusionPointViewArr;    // �洢һ��Ҷ�ڵ����ںϵ��Ƶ�ͼ��
    WeightArr*                   FusionPointWeightArr;    //�洢һ��Ҷ�ڵ����ںϵ��Ƶ�Ȩ��
    //POINT_TYPE                  FusionPointNormal;          //�洢һ��Ҷ�ڵ����ںϵ�ķ���

    VoxelKey                    m_VoxelKey;
    std::vector<Point3f>        m_CornerPoints;
    uint32_t                    imageIDCell;     //�����vector���ܵ���Ч�ʱ��
    bool                        addJudge=false;
    bool                        haveImage=false;
}; // class TOctreeCell
/*----------------------------------------------------------------*/


// basic octree class
// each item should define the operator const POINT_TYPE& returning its center
// SIZE is the minimum number of items contained by the cell so that this to be divided further
// NOM/DENOM is the size of the cell minimum allowed to be divided further
// both conditions represent exclusive limits and both should be true for the division to take place
template <typename ITEMARR_TYPE, typename TYPE, int DIMS, typename DATA_TYPE=uint32_t, int SIZE=16, int NOM=0, int DENOM=1>
class TOctree
{
public:
	typedef TYPE Type;
	typedef TOctreeCell<TYPE,DIMS,DATA_TYPE> CELL_TYPE;
	typedef typename ITEMARR_TYPE::Type ITEM_TYPE;
	typedef SEACAVE::cList<IDX,IDX,0,1024> IDXARR_TYPE;
	typedef SEACAVE::cList<CELL_TYPE*,CELL_TYPE*,0,256> CELLPTRARR_TYPE;
	typedef typename IDXARR_TYPE::Type IDX_TYPE;
	typedef typename CELL_TYPE::POINT_TYPE POINT_TYPE;
	typedef typename CELL_TYPE::AABB_TYPE AABB_TYPE;

	struct IndexInserter {
		IDXARR_TYPE& indices;
		IndexInserter(IDXARR_TYPE& _indices) : indices(_indices) {}
		void operator()(IDX_TYPE idx) { indices.Insert(idx); }
		void operator()(const IDX_TYPE* idices, size_t size) { indices.Join(idices, size); }
	};

	struct CellInserter {
		CELLPTRARR_TYPE& cells;
		CellInserter(CELLPTRARR_TYPE& _cells) : cells(_cells) {}
		void operator()(CELL_TYPE& cell) { cells.Insert(&cell); }
	};

public:
	inline TOctree() {}
	inline TOctree(const ITEMARR_TYPE&);
	inline TOctree(const ITEMARR_TYPE&, const AABB_TYPE&);

	inline void Release();
	inline void Swap(TOctree&);

	void Insert(const ITEMARR_TYPE&);
	void Insert(const ITEMARR_TYPE&, const AABB_TYPE&);

	template <typename INSERTER>
	inline void CollectCells(INSERTER&) const;
	template <typename PARSER>
	inline void ParseCells(PARSER&);

	template <typename INSERTER>
	inline void Collect(INSERTER& inserter, const AABB_TYPE& aabb) const;
	inline void Collect(IDXARR_TYPE& indices, const AABB_TYPE& aabb) const;
	inline void Collect(IDX_TYPE maxNeighbors, IDXARR_TYPE& indices, const AABB_TYPE& aabb) const;

	template <typename INSERTER>
	inline void Collect(INSERTER& inserter, const POINT_TYPE& center, TYPE radius) const;
	inline void Collect(IDXARR_TYPE& indices, const POINT_TYPE& center, TYPE radius) const;
	inline void Collect(IDX_TYPE maxNeighbors, IDXARR_TYPE& indices, const POINT_TYPE& center, TYPE radius) const;

	template <typename INSERTER, typename COLLECTOR>
	inline void Collect(INSERTER& inserter, const COLLECTOR& collector) const;
	template <typename COLLECTOR>
	inline void Collect(IDXARR_TYPE& indices, const COLLECTOR& collector) const;

	template <typename FTYPE, int FDIMS, typename INSERTER>
	inline void Traverse(const TFrustum<FTYPE,FDIMS>&, INSERTER&) const;
	template <typename FTYPE, int FDIMS>
	inline void Traverse(const TFrustum<FTYPE,FDIMS>&, IDXARR_TYPE&) const;
	template <typename FTYPE, int FDIMS, typename PARSER>
	inline void TraverseCells(const TFrustum<FTYPE,FDIMS>&, PARSER&);
	template <typename FTYPE, int FDIMS>
	inline void TraverseCells(const TFrustum<FTYPE,FDIMS>&, CELLPTRARR_TYPE&);

	inline AABB_TYPE GetAabb() const { return m_root.GetAabb(m_radius); }
	inline TYPE GetRadius(const AABB_TYPE& aabb) const {
		// radius of the root cell
		const POINT_TYPE size(aabb.GetSize() / TYPE(2));
		TYPE radius = size[0];
		if (DIMS > 1 && radius < size[1])
			radius = size[1];
		if (DIMS > 2 && radius < size[2])
			radius = size[2];
		return radius;
	}
	inline bool IsEmpty() const { return m_indices.IsEmpty(); }
	inline size_t GetNumItems() const { return m_indices.GetSize(); }
	inline const IDXARR_TYPE& GetIndexArr() const { return m_indices; }
	inline const ITEM_TYPE* GetItems() const { return m_items; }

protected:
	static inline POINT_TYPE ComputeChildCenter(const POINT_TYPE&, TYPE, unsigned);

	void _Insert(CELL_TYPE&, TYPE, IDXARR_TYPE []);
	void _Insert(CELL_TYPE&, const POINT_TYPE&, TYPE, IDXARR_TYPE&);

	template <typename INSERTER>
	void _CollectCells(const CELL_TYPE&, INSERTER&) const;
	template <typename PARSER>
	void _ParseCells(CELL_TYPE&, TYPE, PARSER&);

	template <typename INSERTER>
	void _Collect(const CELL_TYPE&, const AABB_TYPE&, INSERTER&) const;
	template <typename INSERTER, typename COLLECTOR>
	void _Collect(const CELL_TYPE&, TYPE, const COLLECTOR&, INSERTER&) const;

	template <typename FTYPE, int FDIMS, typename INSERTER>
	void _Traverse(const CELL_TYPE&, TYPE, const TFrustum<FTYPE,FDIMS>&, INSERTER&) const;
	template <typename FTYPE, int FDIMS, typename PARSER>
	void _TraverseCells(CELL_TYPE&, TYPE, const TFrustum<FTYPE,FDIMS>&, PARSER&);

protected:
	const ITEM_TYPE* m_items; // original input items (the only condition is that every item to resolve to a position)
	IDXARR_TYPE m_indices; // indices to input items re-arranged spatially (as dictated by the octree)
	CELL_TYPE m_root; // first cell of the tree (always of Node type)
	TYPE m_radius; // size of the sphere containing all cells

#ifndef _RELEASE
public:
	typedef struct DEBUGINFO_TYPE {
		size_t memSize;		// total memory used
		size_t memStruct;	// memory used for the tree structure
		size_t memItems;	// memory used for the contained items
		size_t numItems;	// number of contained items
		size_t numNodes;	// total nodes...
		size_t numLeaves;	// ... from which this number of leaves
		size_t minDepth;	// minimum tree depth
		size_t maxDepth;	// maximum tree depth
		float avgDepth;		// average tree depth
		void Init() { memset(this, 0, sizeof(DEBUGINFO_TYPE)); }
		void operator += (const DEBUGINFO_TYPE& r) {
			avgDepth = avgDepth*numNodes + r.avgDepth*r.numNodes;
			memSize += r.memSize; memStruct += r.memStruct; memItems += r.memItems;
			numItems += r.numItems; numNodes += r.numNodes; numLeaves += r.numLeaves;
			if (minDepth > r.minDepth) minDepth = r.minDepth;
			if (maxDepth < r.maxDepth) maxDepth = r.maxDepth;
			avgDepth /= numNodes;
		}
	} DEBUGINFO;

	void GetDebugInfo(DEBUGINFO* =NULL, bool bPrintStats=false) const;
	static void LogDebugInfo(const DEBUGINFO&);

protected:
	void _GetDebugInfo(const CELL_TYPE&, unsigned, DEBUGINFO&) const;
#endif
}; // class TOctree


//////////////////////////////////////////////////////////////////////////
// octree cell class
template <typename TYPE, int DIMS, typename DATA_TYPE>
class TOctreeCellByFusion
{
    STATIC_ASSERT(DIMS > 0 && DIMS <= 3);

public:
    typedef Eigen::Matrix<TYPE, DIMS, 1> POINT_TYPE;
    typedef SEACAVE::TAABB<TYPE, DIMS> AABB_TYPE;
    typedef uint32_t SIZE_TYPE;

    typedef uint32_t View;
    typedef float Weight;
    typedef TPoint3<float> Normal;

    typedef struct
    {
        POINT_TYPE center; // center of the current cell
        TYPE F_Radius;
    } NODE_TYPE;

    typedef struct
    {
        SIZE_TYPE size; // number of items contained by this cell in the global array
        POINT_TYPE FusionPoint;     //�洢һ��Ҷ�ڵ��е��ںϵ�
    }LEAF_TYPE;

    enum { dataSize = (sizeof(NODE_TYPE) > sizeof(LEAF_TYPE) ? sizeof(NODE_TYPE) : sizeof(LEAF_TYPE)) };
    enum { numChildren = (2 << (DIMS - 1)) };

public:
    inline TOctreeCellByFusion();
    inline TOctreeCellByFusion(TOctreeCellByFusion*);
    inline ~TOctreeCellByFusion();

    inline void Release();
    inline void Swap(TOctreeCellByFusion&);

    inline unsigned ComputeChild(const POINT_TYPE& item) const;
    static void ComputeCenter(POINT_TYPE[]);

    inline bool	IsLeaf() const { return (m_child == NULL); }
    inline const TOctreeCellByFusion& GetChild(int i) const { ASSERT(!IsLeaf() && i < numChildren); return m_child[i]; }
    inline const NODE_TYPE& Node() const { ASSERT(!IsLeaf()); return *((const NODE_TYPE*)m_data); }
    inline NODE_TYPE& Node() { ASSERT(!IsLeaf()); return *((NODE_TYPE*)m_data); }
    inline const LEAF_TYPE& Leaf() const { ASSERT(IsLeaf()); return *((const LEAF_TYPE*)m_data); }
    inline LEAF_TYPE& Leaf() { ASSERT(IsLeaf()); return *((LEAF_TYPE*)m_data); }
    inline const POINT_TYPE& GetCenter() const { return Node().center; }
    inline AABB_TYPE GetAabb(TYPE radius) const { return AABB_TYPE(Node().center, radius); }
    inline IDX GetLastItem() const { return Leaf().idxBegin + Leaf().size; }
    inline SIZE_TYPE GetNumItems() const { return Leaf().size; }
    inline const DATA_TYPE& GetUserData() const { return Leaf().data; }
    inline DATA_TYPE& GetUserData() { return Leaf().data; }
    size_t GetNumItemsHeld() const;

    inline std::vector<Point3f>& GetCornerPoints()
    {
        return m_CornerPoints;
    }

public:
    TOctreeCellByFusion*                m_child; // if not a leaf, 2^DIMS child objects
    TOctreeCellByFusion*                m_father; //���ڵ�
    uint8_t                     m_data[dataSize]; // a LEAF_TYPE or NODE_TYPE object, if it is a leaf or not respectively

    VoxelKey                    m_VoxelKey;
    std::vector<Point3f>        m_CornerPoints;
};

/*----------------------------------------------------------------*/
template <typename TYPE, int DIMS, typename DATA_TYPE = uint32_t>
class TOctreeByFusion
{
public:
    typedef TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE> F_CELL_TYPE;
    typedef Eigen::Matrix<TYPE, DIMS, 1> F_POINT_TYPE;
    typedef SEACAVE::TAABB<TYPE, DIMS> F_AABB_TYPE;

    //typedef TPoint3<float> Normal;
    std::vector<VoxelKey> arrOctreeCellNum;
    std::map<VoxelKey, F_CELL_TYPE*> arrOctreeCellMap; // ���нڵ�
    std::map<VoxelKey, F_CELL_TYPE*> arrOctreeLeafMap; // Ҷ�ڵ�
    int cellTotalNum;
    F_POINT_TYPE minPoint;
public:
    inline TOctreeByFusion() {};

    inline std::map<VoxelKey, F_CELL_TYPE*>& GetLeafsMap()
    {
        return arrOctreeLeafMap;
    }

    inline TYPE GetRadius(const F_AABB_TYPE& aabb){
        const F_POINT_TYPE size(aabb.GetSize() / TYPE(2));
        TYPE radius = size[0];
        if (DIMS > 1 && radius < size[1])
            radius = size[1];
        if (DIMS > 2 && radius < size[2])
            radius = size[2];
        return radius;
    }

    inline F_POINT_TYPE ComputeChildCenter(const F_POINT_TYPE& center, TYPE radius, unsigned idxChild){
        struct CENTERARR_TYPE {
            F_POINT_TYPE child[F_CELL_TYPE::numChildren];
            inline CENTERARR_TYPE() { F_CELL_TYPE::ComputeCenter(child); }
        };
        static const CENTERARR_TYPE centers;
        return center + centers.child[idxChild] * radius;
    }

    inline VoxelKey ComputeNum(const F_POINT_TYPE& center, TYPE radius, unsigned levelChild, unsigned idxChild) {
        VoxelKey m_Num;
        m_Num.m_LOD = levelChild;
        F_POINT_TYPE m_point;
        if (idxChild == 0) {
            F_POINT_TYPE point_(center[0] - radius, center[1] - radius, center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 1) {
            F_POINT_TYPE point_(center[0], center[1] - radius, center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 2) {
            F_POINT_TYPE point_(center[0] - radius, center[1], center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 3) {
            F_POINT_TYPE point_(center[0], center[1], center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 4) {
            F_POINT_TYPE point_(center[0] - radius, center[1] - radius, center[2]);
            m_point = point_;
        }
        else if (idxChild == 5) {
            F_POINT_TYPE point_(center[0], center[1] - radius, center[2]);
            m_point = point_;
        }
        else if (idxChild == 6) {
            F_POINT_TYPE point_(center[0] - radius, center[1], center[2]);
            m_point = point_;
        }
        else if (idxChild == 7) {
            F_POINT_TYPE point_(center[0], center[1], center[2]);
            m_point = point_;
        }
        m_Num.m_VoxelX = round((m_point[0] - minPoint[0]) / radius);
        m_Num.m_VoxelY = round((m_point[1] - minPoint[1]) / radius);
        m_Num.m_VoxelZ = round((m_point[2] - minPoint[2]) / radius);
        return m_Num;
    }

    inline F_POINT_TYPE GetMinPoint(const F_POINT_TYPE& center, TYPE radius)
    {
        F_POINT_TYPE minpoint_(center[0] - radius, center[1] - radius, center[2] - radius);
        return minpoint_;
    }

    inline void GetNeighbor(VoxelKey num, std::vector<VoxelKey>& neighbor)
    {
        //////////////////////////Z+//////////////////////////////////
        VoxelKey zAddNum(num.m_LOD, num.m_VoxelX, num.m_VoxelY, num.m_VoxelZ + 1);
        Get_Neighbor(num, zAddNum, _ZADD, neighbor);
        //////////////////////////Z-//////////////////////////////////
        VoxelKey zSubtractNum(num.m_LOD, num.m_VoxelX, num.m_VoxelY, num.m_VoxelZ - 1);
        Get_Neighbor(num, zSubtractNum, _ZSUBTRACT, neighbor);
        //////////////////////////X+//////////////////////////////////
        VoxelKey xAddNum(num.m_LOD, num.m_VoxelX+1, num.m_VoxelY, num.m_VoxelZ);
        Get_Neighbor(num, xAddNum, _XADD, neighbor);
        //////////////////////////X-//////////////////////////////////
        VoxelKey xSubtractNum(num.m_LOD, num.m_VoxelX - 1, num.m_VoxelY, num.m_VoxelZ);
        Get_Neighbor(num, xSubtractNum, _XSUBTRACT, neighbor);
        //////////////////////////Y+//////////////////////////////////
        VoxelKey yAddNum(num.m_LOD, num.m_VoxelX, num.m_VoxelY + 1, num.m_VoxelZ);
        Get_Neighbor(num, yAddNum, _YADD, neighbor);
        //////////////////////////Y-//////////////////////////////////
        VoxelKey ySubtractNum(num.m_LOD, num.m_VoxelX, num.m_VoxelY - 1, num.m_VoxelZ);
        Get_Neighbor(num, ySubtractNum, _YSUBTRACT, neighbor);
    }

    void Get_Neighbor(VoxelKey num, VoxelKey h_num,int handle,std::vector<VoxelKey>& returnNumArr)
    {
        typename std::map<VoxelKey, F_CELL_TYPE*>::iterator iter = arrOctreeCellMap.find(h_num);
        if (iter == arrOctreeCellMap.end())
        {
            F_CELL_TYPE& cell = *(arrOctreeCellMap[num]);
            VoxelKey returnNum(0,0,0,0);
            bool judge = _GetNeighbor(cell, returnNum, handle);
            if (judge)
            {
                returnNumArr.push_back(returnNum);
            }
        }
        else
        {
            F_CELL_TYPE& h_cell = *(arrOctreeCellMap[h_num]);
            if (h_cell.IsLeaf())
            {
                returnNumArr.push_back(h_num);
            }
            else
            {
                GetNeighbor_(h_cell, returnNumArr, handle);
            }
        }
    }

    inline bool _GetNeighbor(F_CELL_TYPE& cell, VoxelKey& returnNum,int handle)
    {
        if (cell.m_father!=NULL)
        {
            F_CELL_TYPE& cell_father = *(cell.m_father);
            VoxelKey num_father = cell_father.m_VoxelKey;
            if (handle==_XADD)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX + 1;
                returnNum.m_VoxelY = num_father.m_VoxelY;
                returnNum.m_VoxelZ = num_father.m_VoxelZ;
            }
            else if(handle == _XSUBTRACT)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX - 1;
                returnNum.m_VoxelY = num_father.m_VoxelY;
                returnNum.m_VoxelZ = num_father.m_VoxelZ;
            }
            else if (handle == _YADD)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX;
                returnNum.m_VoxelY = num_father.m_VoxelY + 1;
                returnNum.m_VoxelZ = num_father.m_VoxelZ;
            }
            else if (handle == _YSUBTRACT)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX;
                returnNum.m_VoxelY = num_father.m_VoxelY - 1;
                returnNum.m_VoxelZ = num_father.m_VoxelZ;
            }
            else if (handle == _ZADD)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX;
                returnNum.m_VoxelY = num_father.m_VoxelY;
                returnNum.m_VoxelZ = num_father.m_VoxelZ + 1;
            }
            else if (handle == _ZSUBTRACT)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX;
                returnNum.m_VoxelY = num_father.m_VoxelY;
                returnNum.m_VoxelZ = num_father.m_VoxelZ - 1;
            }
            typename std::map<VoxelKey, F_CELL_TYPE*>::iterator iter = arrOctreeCellMap.find(returnNum);
            if (iter == arrOctreeCellMap.end())
            {
                bool judge = _GetNeighbor(cell_father, returnNum, handle);
                return judge;
            }
            else
            {
                return true;
            }
        }
        else
        {
            return false;
        }
    }

    inline void GetNeighbor_(F_CELL_TYPE& cell, std::vector<VoxelKey>& returnNumArr, int handle)
    {
        if (handle==_XADD)
        {
            for (int i = 0; i<F_CELL_TYPE::numChildren; i++)
            {
                if (i==0 || i==2 || i==4 || i==6)
                {
                    F_CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.IsLeaf())
                    {
                        returnNumArr.push_back(cellChild.m_VoxelKey);
                    }
                    else
                    {
                        GetNeighbor_(cellChild, returnNumArr, handle);
                    }
                }
            }
        }
        if (handle == _XSUBTRACT)
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                if (i == 1 || i == 3 || i == 5 || i == 7)
                {
                    F_CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.IsLeaf())
                    {
                        returnNumArr.push_back(cellChild.m_VoxelKey);
                    }
                    else
                    {
                        GetNeighbor_(cellChild, returnNumArr, handle);
                    }
                }
            }
        }
        if (handle == _YADD)
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                if (i == 0 || i == 1 || i == 4 || i == 5)
                {
                    F_CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.IsLeaf())
                    {
                        returnNumArr.push_back(cellChild.m_VoxelKey);
                    }
                    else
                    {
                        GetNeighbor_(cellChild, returnNumArr, handle);
                    }
                }
            }
        }
        if (handle == _YSUBTRACT)
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                if (i == 2 || i == 3 || i == 6 || i == 7)
                {
                    F_CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.IsLeaf())
                    {
                        returnNumArr.push_back(cellChild.m_VoxelKey);
                    }
                    else
                    {
                        GetNeighbor_(cellChild, returnNumArr, handle);
                    }
                }
            }
        }
        if (handle == _ZADD)
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                if (i == 0 || i == 1 || i == 2 || i == 3)
                {
                    F_CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.IsLeaf())
                    {
                        returnNumArr.push_back(cellChild.m_VoxelKey);
                    }
                    else
                    {
                        GetNeighbor_(cellChild, returnNumArr, handle);
                    }
                }
            }
        }
        if (handle == _ZSUBTRACT)
        {
            for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
            {
                if (i == 4 || i == 5 || i == 6 || i == 7)
                {
                    F_CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.IsLeaf())
                    {
                        returnNumArr.push_back(cellChild.m_VoxelKey);
                    }
                    else
                    {
                        GetNeighbor_(cellChild, returnNumArr, handle);
                    }
                }
            }
        }
    }

    inline void IniOctree(const F_AABB_TYPE&, int, float);
    //inline void InsertForFusion(F_POINT_TYPE& F_Point,float Point_Score, ViewArr& viewArr, WeightArr& weightArr, Normal& normal);
    //inline void InsertForFusion(F_POINT_TYPE& F_Point, float Point_Score, ViewArr* viewArr, WeightArr* weightArr);
    inline bool InsertForFusion(F_POINT_TYPE& F_Point);
    TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE>* InsertForFusion(F_POINT_TYPE& F_Point, float Point_Score);
    inline bool InsertForFusionOS(F_POINT_TYPE& F_Point, uint32_t imageID);
    inline void GetTOctreeLeafs(std::vector<F_CELL_TYPE*>& toctreeleafs);
    inline void CollectLeafs(bool bBox = false); // �����˲������Ѽ�Ҷ�ڵ�
    inline void TraverseOctree(PointArr& , PointViewArr& , PointWeightArr& ); // �����˲���
    inline void TraverseOctree(int& leafNum, int& pointNum);

protected:
    //inline void _InsertForFusion(const unsigned, F_CELL_TYPE&, F_POINT_TYPE&, float ,ViewArr&, WeightArr&, Normal& ,F_POINT_TYPE& ,TYPE);
    //inline void InsertForFusion_(const unsigned, F_CELL_TYPE&, F_POINT_TYPE& , TYPE , float, F_POINT_TYPE&, ViewArr& , WeightArr& , Normal& );

    //inline void _InsertForFusion(const unsigned, F_CELL_TYPE&, F_POINT_TYPE&, float, ViewArr*, WeightArr*, F_POINT_TYPE&, TYPE);
    //inline void InsertForFusion_(const unsigned, F_CELL_TYPE&, F_POINT_TYPE&, TYPE, float, F_POINT_TYPE&, ViewArr*, WeightArr*);

    TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE>* _InsertForFusion(const unsigned, F_CELL_TYPE&, F_POINT_TYPE&, float, F_POINT_TYPE&, TYPE );
    TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE>* InsertForFusion_(const unsigned, F_CELL_TYPE&, F_POINT_TYPE&, TYPE, float, F_POINT_TYPE& );

    inline bool _InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, F_POINT_TYPE& f_center, TYPE f_radius);
    inline bool InsertForFusion_(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center, TYPE F_Radius, F_POINT_TYPE& F_Point);

    inline bool _InsertForFusionOS(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, F_POINT_TYPE& f_center, TYPE f_radius, uint32_t imageID);
    inline bool InsertForFusion_OS(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center, TYPE F_Radius, F_POINT_TYPE& F_Point, uint32_t imageID);

    inline void TraverseOctree_(F_CELL_TYPE& F_Cell, bool bBox=false);
    inline void AddLeaf(F_CELL_TYPE& F_Cell, bool bBox = false);

    inline void TraverseOctree_(F_CELL_TYPE& F_Cell,PointArr& pointArr, PointViewArr& pointViewArr, PointWeightArr& pointWeightArr);
    inline void TraverseOctree_(F_CELL_TYPE& F_Cell, int& leafNum, int& pointNum);
protected:
    float VoxelSize;
    int VoxelNum;
    F_CELL_TYPE F_Root;
    std::vector<F_CELL_TYPE*> TOctreeLeafs;
};

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::GetTOctreeLeafs(std::vector<F_CELL_TYPE*>& toctreeleafs)
{
    toctreeleafs.swap(TOctreeLeafs);
    //toctreeleafs.assign(TOctreeLeafs.begin(), TOctreeLeafs.end());
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::IniOctree(const F_AABB_TYPE& AB,int NUM,float SIZE)
{
    VoxelSize = SIZE;
    VoxelNum = NUM;
    cellTotalNum = 0;

    F_Root.Node().F_Radius= GetRadius(AB);
    F_Root.Node().center = AB.GetCenter();
    F_Root.m_father = NULL;
    minPoint = GetMinPoint(F_Root.Node().center, F_Root.Node().F_Radius);

    bool IsFinishe = false;
    TYPE _Radius = F_Root.Node().F_Radius;
    while (!IsFinishe)
    {
        _Radius = _Radius / TYPE(2);
        if (_Radius<= VoxelSize)
        {
            TYPE UP_Radius = _Radius * TYPE(2);
            float ElementOne = UP_Radius - VoxelSize;
            float ElementTwo = VoxelSize - _Radius;
            if (ElementOne>ElementTwo)
            {
                VoxelSize = _Radius;
            }
            else
            {
                VoxelSize = UP_Radius;
            }
            IsFinishe = true;
        }
    }

    F_Root.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];
    for (int i=0;i<F_CELL_TYPE::numChildren;i++)
    {
        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Root.m_child[i].Leaf();
        leaf.size = 0;
        F_Root.m_child[i].m_father = &F_Root;
        F_Root.m_child[i].m_VoxelKey= ComputeNum(F_Root.Node().center, F_Root.Node().F_Radius,1,i);
        arrOctreeCellMap[F_Root.m_child[i].m_VoxelKey] = &F_Root.m_child[i];
    }
}

/////////////�洢����(�ӵ÷�)/////////////////////////

//template<typename TYPE, int DIMS, typename DATA_TYPE>
//inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion(F_POINT_TYPE& F_Point, float Point_Score, ViewArr& viewArr, WeightArr& weightArr, Normal& normal)
//{
//    TYPE f_radius = F_Root.Node().F_Radius;
//    F_POINT_TYPE& f_center = F_Root.Node().center;
//    const unsigned idx = F_Root.ComputeChild(F_Point);
//    F_CELL_TYPE& cell = F_Root.m_child[idx];
//    _InsertForFusion(idx, cell, F_Point, Point_Score, viewArr, weightArr, normal, f_center,f_radius);
//}
//
//template<typename TYPE, int DIMS, typename DATA_TYPE>
//inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::_InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, float Point_Score, ViewArr& viewArr, WeightArr& weightArr, Normal& normal,F_POINT_TYPE& f_center,TYPE f_radius)
//{
//    if (F_Cell.IsLeaf())
//    {
//        if (f_radius<= VoxelSize)
//        {
//            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
//            if (leaf.size<VoxelNum)
//            {
//                leaf.FusionPointScore= Point_Score;
//                leaf.FusionPoint = F_Point;
//                //for (int i=0;i<viewArr.size();i++)
//                //{
//                //    leaf.FusionPointViewArr.push_back(viewArr[i]);
//                //    leaf.FusionPointWeightArr.push_back(weightArr[i]);
//                //}
//                //leaf.FusionPointNormal[0] = normal.x;
//                //leaf.FusionPointNormal[1] = normal.y;
//                //leaf.FusionPointNormal[2] = normal.z;
//                leaf.size = 1;
//                TOctreeLeafs.push_back(&F_Cell);
//            }
//            else
//            {
//                if (Point_Score>leaf.FusionPointScore)
//                {
//                    leaf.FusionPointScore = Point_Score;
//                    leaf.FusionPoint = F_Point;
//                    //for (int i = 0; i < viewArr.size(); i++)
//                    //{
//                    //    leaf.FusionPointViewArr.push_back(viewArr[i]);
//                    //    leaf.FusionPointWeightArr.push_back(weightArr[i]);
//                    //}
//                    //leaf.FusionPointNormal[0] = normal.x;
//                    //leaf.FusionPointNormal[1] = normal.y;
//                    //leaf.FusionPointNormal[2] = normal.z;
//                    leaf.size = 1;
//                }
//            }
//            if (Point_Score >= 60)
//            {
//                Point3D Point_3D;
//                Point_3D.X = F_Point[0];
//                Point_3D.Y = F_Point[1];
//                Point_3D.Z = F_Point[2];
//                Point_3D.Score = Point_Score;
//                F_Cell.Points_3D.push_back(Point_3D);
//            }
//        }
//        else
//        {
//            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
//            if (leaf.size<VoxelNum)
//            {
//                leaf.size = 1;
//                leaf.FusionPoint=F_Point;
//                leaf.FusionPointScore= Point_Score;
//                //for (int i = 0; i < viewArr.size(); i++)
//                //{
//                //    leaf.FusionPointViewArr.push_back(viewArr[i]);
//                //    leaf.FusionPointWeightArr.push_back(weightArr[i]);
//                //}
//                //leaf.FusionPointNormal[0] = normal.x;
//                //leaf.FusionPointNormal[1] = normal.y;
//                //leaf.FusionPointNormal[2] = normal.z;
//            }
//            else
//            {
//                InsertForFusion_(idx, F_Cell, f_center, f_radius, Point_Score, F_Point, viewArr, weightArr, normal);
//            }
//        }
//    }
//    else
//    {
//        //TYPE childradius = f_radius / TYPE(2);
//        //F_POINT_TYPE childcenter(ComputeChildCenter(f_center, childradius, idx));
//        //F_Cell.Node().center = childcenter;
//        //F_Cell.Node().F_Radius = childradius;
//        TYPE childradius = F_Cell.Node().F_Radius;
//        F_POINT_TYPE childcenter = F_Cell.Node().center;
//
//        const unsigned _idx = F_Cell.ComputeChild(F_Point);
//        F_CELL_TYPE& cell = F_Cell.m_child[_idx];
//        _InsertForFusion(_idx, cell, F_Point, Point_Score, viewArr, weightArr, normal, childcenter, childradius);
//    }
//}
//
//template<typename TYPE, int DIMS, typename DATA_TYPE>
//inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion_(const unsigned idx,F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center,TYPE F_Radius, float Point_Score, F_POINT_TYPE& F_Point, ViewArr& viewArr, WeightArr& weightArr, Normal& normal)
//{
//    F_Cell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];
//    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
//    {
//        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.m_child[i].Leaf();
//        leaf.size = 0;
//    }
//    TYPE childradius = F_Radius / TYPE(2);
//    F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
//    F_Cell.Node().center = childcenter;
//    F_Cell.Node().F_Radius = childradius;
//
//    typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
//
//    F_POINT_TYPE& F_Cell_Point0 = leaf.FusionPoint;
//    float F_Cell_Score0 = leaf.FusionPointScore;
//    //std::vector<View> F_Cell_ViewArr0;
//    //F_Cell_ViewArr0.assign(leaf.FusionPointViewArr.begin(), leaf.FusionPointViewArr.end());
//    //std::vector<Weight> F_Cell_WeightArr0;
//    //F_Cell_WeightArr0.assign(leaf.FusionPointWeightArr.begin(), leaf.FusionPointWeightArr.end());
//    //F_POINT_TYPE& F_Cell_Normal0 = leaf.FusionPointNormal;
//
//    const unsigned idx0 = F_Cell.ComputeChild(F_Cell_Point0);
//    const unsigned idx1 = F_Cell.ComputeChild(F_Point);
//
//    if (idx0!= idx1)
//    {
//        F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//        typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();
//
//        leaf0.FusionPoint=F_Cell_Point0;
//        leaf0.FusionPointScore= F_Cell_Score0;
//        //leaf0.FusionPointViewArr.assign(F_Cell_ViewArr0.begin(), F_Cell_ViewArr0.end());
//        //leaf0.FusionPointWeightArr.assign(F_Cell_WeightArr0.begin(), F_Cell_WeightArr0.end());
//        //leaf0.FusionPointNormal = F_Cell_Normal0;
//        leaf0.size = 1;
//
//        F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
//        typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();
//
//        leaf1.FusionPoint= F_Point;
//        leaf1.FusionPointScore= Point_Score;
//        //for (int i = 0; i < viewArr.size(); i++)
//        //{
//        //    leaf1.FusionPointViewArr.push_back(viewArr[i]);
//        //    leaf1.FusionPointWeightArr.push_back(weightArr[i]);
//        //}
//        //leaf1.FusionPointNormal[0] = normal.x;
//        //leaf1.FusionPointNormal[1] = normal.y;
//        //leaf1.FusionPointNormal[2] = normal.z;
//        leaf1.size = 1;
//
//        if (childradius<= VoxelSize)
//        {
//            TOctreeLeafs.push_back(&child0);
//            TOctreeLeafs.push_back(&child1);
//            if (F_Cell_Score0>=60)
//            {
//                Point3D Point_3D;
//                Point_3D.X = F_Cell_Point0[0];
//                Point_3D.Y = F_Cell_Point0[1];
//                Point_3D.Z = F_Cell_Point0[2];
//                Point_3D.Score = F_Cell_Score0;
//                child0.Points_3D.push_back(Point_3D);
//            }
//            if (Point_Score>=60)
//            {
//                Point3D Point_3D;
//                Point_3D.X = F_Point[0];
//                Point_3D.Y = F_Point[1];
//                Point_3D.Z = F_Point[2];
//                Point_3D.Score = Point_Score;
//                child1.Points_3D.push_back(Point_3D);
//            }
//        }
//    }
//    else
//    {
//        if (childradius<= VoxelSize)
//        {
//            if (F_Cell_Score0 >= Point_Score)
//            {
//                F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//                typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();
//
//                leaf0.FusionPoint = F_Cell_Point0;
//                leaf0.FusionPointScore = F_Cell_Score0;
//                //leaf0.FusionPointViewArr.assign(F_Cell_ViewArr0.begin(), F_Cell_ViewArr0.end());
//                //leaf0.FusionPointWeightArr.assign(F_Cell_WeightArr0.begin(), F_Cell_WeightArr0.end());
//                //leaf0.FusionPointNormal = F_Cell_Normal0;
//                leaf0.size = 1;
//                TOctreeLeafs.push_back(&child0);
//            }
//            else
//            {
//                F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
//                typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();
//
//                leaf1.FusionPoint = F_Point;
//                leaf1.FusionPointScore = Point_Score;
//                //for (int i = 0; i < viewArr.size(); i++)
//                //{
//                //    leaf1.FusionPointViewArr.push_back(viewArr[i]);
//                //    leaf1.FusionPointWeightArr.push_back(weightArr[i]);
//                //}
//                //leaf1.FusionPointNormal[0] = normal.x;
//                //leaf1.FusionPointNormal[1] = normal.y;
//                //leaf1.FusionPointNormal[2] = normal.z;
//                leaf1.size = 1;
//                TOctreeLeafs.push_back(&child1);
//            }
//            if (F_Cell_Score0 >= 60)
//            {
//                Point3D Point_3D;
//                Point_3D.X = F_Cell_Point0[0];
//                Point_3D.Y = F_Cell_Point0[1];
//                Point_3D.Z = F_Cell_Point0[2];
//                Point_3D.Score = F_Cell_Score0;
//                F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//                child0.Points_3D.push_back(Point_3D);
//            }
//            if (Point_Score >= 60)
//            {
//                Point3D Point_3D;
//                Point_3D.X = F_Point[0];
//                Point_3D.Y = F_Point[1];
//                Point_3D.Z = F_Point[2];
//                Point_3D.Score = Point_Score;
//                F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
//                child1.Points_3D.push_back(Point_3D);
//            }
//        }
//        else
//        {
//            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//            typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();
//
//            leaf0.FusionPoint=F_Cell_Point0;
//            leaf0.FusionPointScore = F_Cell_Score0;
//            //leaf0.FusionPointViewArr.assign(F_Cell_ViewArr0.begin(), F_Cell_ViewArr0.end());
//            //leaf0.FusionPointWeightArr.assign(F_Cell_WeightArr0.begin(), F_Cell_WeightArr0.end());
//            //leaf0.FusionPointNormal = F_Cell_Normal0;
//
//            InsertForFusion_(idx0, child0, childcenter, childradius, Point_Score, F_Point, viewArr, weightArr, normal);
//        }
//    }
//}

///////////���洢���ߣ��ӵ÷֣�/////////////////////////

//template<typename TYPE, int DIMS, typename DATA_TYPE>
//inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion(F_POINT_TYPE& F_Point, float Point_Score, ViewArr* viewArr, WeightArr* weightArr)
//{
//    TYPE f_radius = F_Root.Node().F_Radius;
//    F_POINT_TYPE& f_center = F_Root.Node().center;
//    const unsigned idx = F_Root.ComputeChild(F_Point);
//    F_CELL_TYPE& cell = F_Root.m_child[idx];
//    _InsertForFusion(idx, cell, F_Point, Point_Score, viewArr, weightArr, f_center, f_radius);
//}
//
//template<typename TYPE, int DIMS, typename DATA_TYPE>
//inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::_InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, float Point_Score,
//    ViewArr* viewArr, WeightArr* weightArr, F_POINT_TYPE& f_center, TYPE f_radius)
//{
//    if (F_Cell.IsLeaf())
//    {
//        if (f_radius <= VoxelSize)
//        {
//            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
//            if (leaf.size < VoxelNum)
//            {
//                leaf.FusionPointScore = Point_Score;
//                leaf.FusionPoint = F_Point;
//                F_Cell.FusionPointViewArr = viewArr;
//                F_Cell.FusionPointWeightArr = weightArr;
//                leaf.size = 1;
//                //TOctreeLeafs.push_back(&F_Cell);
//            }
//            else
//            {
//                if (Point_Score > leaf.FusionPointScore)
//                {
//                    leaf.FusionPointScore = Point_Score;
//                    leaf.FusionPoint = F_Point;
//                    F_Cell.FusionPointViewArr->Release();
//                    F_Cell.FusionPointViewArr->Release();
//                    delete F_Cell.FusionPointViewArr;
//                    delete F_Cell.FusionPointWeightArr;
//                    F_Cell.FusionPointViewArr = NULL;
//                    F_Cell.FusionPointWeightArr = NULL;
//                    F_Cell.FusionPointViewArr = viewArr;
//                    F_Cell.FusionPointWeightArr = weightArr;
//                    leaf.size = 1;
//                }
//            }
//            //if (Point_Score >= 60)
//            //{
//            //    Point3D Point_3D;
//            //    Point_3D.X = F_Point[0];
//            //    Point_3D.Y = F_Point[1];
//            //    Point_3D.Z = F_Point[2];
//            //    Point_3D.Score = Point_Score;
//            //    F_Cell.Points_3D.push_back(Point_3D);
//            //}
//        }
//        else
//        {
//            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
//            if (leaf.size < VoxelNum)
//            {
//                leaf.size = 1;
//                leaf.FusionPoint = F_Point;
//                leaf.FusionPointScore = Point_Score;
//                F_Cell.FusionPointViewArr = viewArr;
//                F_Cell.FusionPointWeightArr = weightArr;
//            }
//            else
//            {
//                InsertForFusion_(idx, F_Cell, f_center, f_radius, Point_Score, F_Point, viewArr, weightArr);
//            }
//        }
//    }
//    else
//    {
//        //TYPE childradius = f_radius / TYPE(2);
//        //F_POINT_TYPE childcenter(ComputeChildCenter(f_center, childradius, idx));
//        //F_Cell.Node().center = childcenter;
//        //F_Cell.Node().F_Radius = childradius;
//        TYPE childradius = F_Cell.Node().F_Radius;
//        F_POINT_TYPE childcenter = F_Cell.Node().center;
//
//        const unsigned _idx = F_Cell.ComputeChild(F_Point);
//        F_CELL_TYPE& cell = F_Cell.m_child[_idx];
//        _InsertForFusion(_idx, cell, F_Point, Point_Score, viewArr, weightArr, childcenter, childradius);
//    }
//}
//
//template<typename TYPE, int DIMS, typename DATA_TYPE>
//inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion_(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center, TYPE F_Radius, float Point_Score,
//    F_POINT_TYPE& F_Point, ViewArr* viewArr, WeightArr* weightArr)
//{
//    F_Cell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];
//    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
//    {
//        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.m_child[i].Leaf();
//        leaf.size = 0;
//    }
//    TYPE childradius = F_Radius / TYPE(2);
//    F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
//    F_Cell.Node().center = childcenter;
//    F_Cell.Node().F_Radius = childradius;
//
//    typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
//    const unsigned idx0 = F_Cell.ComputeChild(leaf.FusionPoint);
//    const unsigned idx1 = F_Cell.ComputeChild(F_Point);
//
//    if (idx0 != idx1)
//    {
//        F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//        typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();
//
//        leaf0.FusionPoint = leaf.FusionPoint;
//        leaf0.FusionPointScore = leaf.FusionPointScore;
//        child0.FusionPointViewArr = F_Cell.FusionPointViewArr;
//        child0.FusionPointWeightArr = F_Cell.FusionPointWeightArr;
//        leaf0.size = 1;
//
//        F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
//        typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();
//
//        leaf1.FusionPoint = F_Point;
//        leaf1.FusionPointScore = Point_Score;
//        child1.FusionPointViewArr = viewArr;
//        child1.FusionPointWeightArr = weightArr;
//        leaf1.size = 1;
//
//        //if (childradius <= VoxelSize)
//        //{
//        //    //TOctreeLeafs.push_back(&child0);
//        //    //TOctreeLeafs.push_back(&child1);
//        //    if (leaf.FusionPointScore >= 60)
//        //    {
//        //        Point3D Point_3D;
//        //        Point_3D.X = leaf.FusionPoint[0];
//        //        Point_3D.Y = leaf.FusionPoint[1];
//        //        Point_3D.Z = leaf.FusionPoint[2];
//        //        Point_3D.Score = leaf.FusionPointScore;
//        //        child0.Points_3D.push_back(Point_3D);
//        //    }
//        //    if (Point_Score >= 60)
//        //    {
//        //        Point3D Point_3D;
//        //        Point_3D.X = F_Point[0];
//        //        Point_3D.Y = F_Point[1];
//        //        Point_3D.Z = F_Point[2];
//        //        Point_3D.Score = Point_Score;
//        //        child1.Points_3D.push_back(Point_3D);
//        //    }
//        //}
//    }
//    else
//    {
//        if (childradius <= VoxelSize)
//        {
//            if (leaf.FusionPointScore >= Point_Score)
//            {
//                F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//                typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();
//
//                leaf0.FusionPoint = leaf.FusionPoint;
//                leaf0.FusionPointScore = leaf.FusionPointScore;
//                child0.FusionPointViewArr = F_Cell.FusionPointViewArr;
//                child0.FusionPointWeightArr = F_Cell.FusionPointWeightArr;
//                leaf0.size = 1;
//                //TOctreeLeafs.push_back(&child0);
//            }
//            else
//            {
//                F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
//                typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();
//
//                leaf1.FusionPoint = F_Point;
//                leaf1.FusionPointScore = Point_Score;
//                child1.FusionPointViewArr = viewArr;
//                child1.FusionPointWeightArr = weightArr;
//                leaf1.size = 1;
//                //TOctreeLeafs.push_back(&child1);
//            }
//            //if (leaf.FusionPointScore >= 60)
//            //{
//            //    Point3D Point_3D;
//            //    Point_3D.X = leaf.FusionPoint[0];
//            //    Point_3D.Y = leaf.FusionPoint[1];
//            //    Point_3D.Z = leaf.FusionPoint[2];
//            //    Point_3D.Score = leaf.FusionPointScore;
//            //    F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//            //    child0.Points_3D.push_back(Point_3D);
//            //}
//            //if (Point_Score >= 60)
//            //{
//            //    Point3D Point_3D;
//            //    Point_3D.X = F_Point[0];
//            //    Point_3D.Y = F_Point[1];
//            //    Point_3D.Z = F_Point[2];
//            //    Point_3D.Score = Point_Score;
//            //    F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
//            //    child1.Points_3D.push_back(Point_3D);
//            //}
//        }
//        else
//        {
//            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
//            typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();
//
//            leaf0.FusionPoint = leaf.FusionPoint;
//            leaf0.FusionPointScore = leaf.FusionPointScore;
//            child0.FusionPointViewArr = F_Cell.FusionPointViewArr;
//            child0.FusionPointWeightArr = F_Cell.FusionPointWeightArr;
//            InsertForFusion_(idx0, child0, childcenter, childradius, Point_Score, F_Point, viewArr, weightArr);
//        }
//    }
//}

////////////�������أ��е÷֣�/////////////////////////
template<typename TYPE, int DIMS, typename DATA_TYPE>
TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE>* TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion(F_POINT_TYPE& F_Point, float Point_Score)
{
    TYPE f_radius = F_Root.Node().F_Radius;
    F_POINT_TYPE& f_center = F_Root.Node().center;
    const unsigned idx = F_Root.ComputeChild(F_Point);
    F_CELL_TYPE& cell = F_Root.m_child[idx];
    F_CELL_TYPE* returnCell = _InsertForFusion(idx, cell, F_Point, Point_Score, f_center, f_radius);
    int a = 0;
    return returnCell;
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE>* TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::_InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, float Point_Score, F_POINT_TYPE& f_center, TYPE f_radius)
{
    if (F_Cell.IsLeaf())
    {
        if (f_radius <= VoxelSize)
        {
            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
            if (leaf.size < VoxelNum)
            {
                leaf.FusionPointScore = Point_Score;
                leaf.FusionPoint = F_Point;
                //leaf.size = 1;
                return &F_Cell;
            }
            else
            {
                if (Point_Score > leaf.FusionPointScore)
                {
                    leaf.FusionPointScore = Point_Score;
                    leaf.FusionPoint = F_Point;
                    //leaf.size = 1;
                    return &F_Cell;
                }
                else
                {
                    return NULL;
                }
            }
        }
        else
        {
            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
            if (leaf.size < VoxelNum)
            {
                //leaf.size = 1;
                leaf.FusionPoint = F_Point;
                leaf.FusionPointScore = Point_Score;
                return &F_Cell;
            }
            else
            {
                F_CELL_TYPE* returnCell=InsertForFusion_(idx, F_Cell, f_center, f_radius, Point_Score, F_Point);
                return returnCell;
            }
        }
    }
    else
    {
        TYPE childradius = F_Cell.Node().F_Radius;
        F_POINT_TYPE childcenter = F_Cell.Node().center;

        const unsigned _idx = F_Cell.ComputeChild(F_Point);
        F_CELL_TYPE& cell = F_Cell.m_child[_idx];
        F_CELL_TYPE* returnCell = _InsertForFusion(_idx, cell, F_Point, Point_Score, childcenter, childradius);
        return returnCell;
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
TOctreeCellByFusion<TYPE, DIMS, DATA_TYPE>* TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion_(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center, TYPE F_Radius, float Point_Score, F_POINT_TYPE& F_Point)
{
    F_Cell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];
    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
    {
        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.m_child[i].Leaf();
        leaf.size = 0;
    }
    TYPE childradius = F_Radius / TYPE(2);
    F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
    F_Cell.Node().center = childcenter;
    F_Cell.Node().F_Radius = childradius;

    typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
    const unsigned idx0 = F_Cell.ComputeChild(leaf.FusionPoint);
    const unsigned idx1 = F_Cell.ComputeChild(F_Point);

    if (idx0 != idx1)
    {
        F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
        typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

        leaf0.FusionPoint = leaf.FusionPoint;
        leaf0.FusionPointScore = leaf.FusionPointScore;
        child0.FusionPointViewArr = F_Cell.FusionPointViewArr;
        child0.FusionPointWeightArr = F_Cell.FusionPointWeightArr;
        leaf0.size = 1;

        F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
        typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();

        leaf1.FusionPoint = F_Point;
        leaf1.FusionPointScore = Point_Score;
        //leaf1.size = 1;
        return &child1;
    }
    else
    {
        if (childradius <= VoxelSize)
        {
            if (leaf.FusionPointScore >= Point_Score)
            {
                F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
                typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

                leaf0.FusionPoint = leaf.FusionPoint;
                leaf0.FusionPointScore = leaf.FusionPointScore;
                child0.FusionPointViewArr = F_Cell.FusionPointViewArr;
                child0.FusionPointWeightArr = F_Cell.FusionPointWeightArr;
                leaf0.size = 1;
            }
            else
            {
                F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
                typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();

                leaf1.FusionPoint = F_Point;
                leaf1.FusionPointScore = Point_Score;
                //leaf1.size = 1;
                return &child1;
            }
        }
        else
        {
            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
            typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

            leaf0.FusionPoint = leaf.FusionPoint;
            leaf0.FusionPointScore = leaf.FusionPointScore;
            child0.FusionPointViewArr = F_Cell.FusionPointViewArr;
            child0.FusionPointWeightArr = F_Cell.FusionPointWeightArr;
            F_CELL_TYPE* returnCell=InsertForFusion_(idx0, child0, childcenter, childradius, Point_Score, F_Point);
            return returnCell;
        }
    }
}
////////////���ӵ÷�///////////////////////////////////
template<typename TYPE, int DIMS, typename DATA_TYPE>
inline bool TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion(F_POINT_TYPE& F_Point)
{
    TYPE f_radius = F_Root.Node().F_Radius;
    F_POINT_TYPE& f_center = F_Root.Node().center;
    const unsigned idx = F_Root.ComputeChild(F_Point);
    F_CELL_TYPE& cell = F_Root.m_child[idx];
    bool Judge = _InsertForFusion(idx, cell, F_Point, f_center, f_radius);
    return Judge;
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline bool TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::_InsertForFusion(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, F_POINT_TYPE& f_center, TYPE f_radius)
{
    if (F_Cell.IsLeaf())
    {
        if (f_radius <= VoxelSize)
        {
            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
            if (leaf.size < VoxelNum)
            {
                leaf.FusionPoint = F_Point;
                leaf.size = 1;
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
            if (leaf.size < VoxelNum)
            {
                leaf.size = 1;
                leaf.FusionPoint = F_Point;
                return true;
            }
            else
            {
                bool Judge = InsertForFusion_(idx, F_Cell, f_center, f_radius, F_Point);
                return Judge;
            }
        }
    }
    else
    {
        TYPE childradius = f_radius / TYPE(2);
        F_POINT_TYPE childcenter(ComputeChildCenter(f_center, childradius, idx));
        F_Cell.Node().center = childcenter;
        F_Cell.Node().F_Radius = childradius;

        const unsigned _idx = F_Cell.ComputeChild(F_Point);
        F_CELL_TYPE& cell = F_Cell.m_child[_idx];
        bool Judge = _InsertForFusion(_idx, cell, F_Point, childcenter, childradius);
        return Judge;
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline bool TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion_(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center, TYPE F_Radius, F_POINT_TYPE& F_Point)
{
    F_Cell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];
    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
    {
        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.m_child[i].Leaf();
        leaf.size = 0;
    }
    TYPE childradius = F_Radius / TYPE(2);
    F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
    F_Cell.Node().center = childcenter;
    F_Cell.Node().F_Radius = childradius;

    typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();

    F_POINT_TYPE& F_Cell_Point0 = leaf.FusionPoint;
    const unsigned idx0 = F_Cell.ComputeChild(F_Cell_Point0);
    const unsigned idx1 = F_Cell.ComputeChild(F_Point);

    if (idx0 != idx1)
    {
        F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
        typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

        leaf0.FusionPoint = F_Cell_Point0;
        leaf0.size = 1;

        F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
        typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();

        leaf1.FusionPoint = F_Point;
        leaf1.size = 1;

        return true;
    }
    else
    {
        if (childradius <= VoxelSize)
        {
            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
            typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

            leaf0.FusionPoint = F_Cell_Point0;
            leaf0.size = 1;
            return false;
        }
        else
        {
            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
            typename F_CELL_TYPE::LEAF_TYPE& leaf = child0.Leaf();
            leaf.FusionPoint = F_Cell_Point0;

            bool Judge=InsertForFusion_(idx0, child0, childcenter, childradius, F_Point);
            return Judge;
        }
    }
}
//////////////�������س����Ĵ��幹��///////////////
template<typename TYPE, int DIMS, typename DATA_TYPE>
inline bool TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusionOS(F_POINT_TYPE& F_Point, uint32_t imageID)
{
    TYPE f_radius = F_Root.Node().F_Radius;
    F_POINT_TYPE& f_center = F_Root.Node().center;
    const unsigned idx = F_Root.ComputeChild(F_Point);
    F_CELL_TYPE& cell = F_Root.m_child[idx];
    bool Judge = _InsertForFusionOS(idx, cell, F_Point, f_center, f_radius, imageID);
    return Judge;
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline bool TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::_InsertForFusionOS(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& F_Point, F_POINT_TYPE& f_center, TYPE f_radius, uint32_t imageID)
{
    if (F_Cell.IsLeaf())
    {
        if (f_radius <= VoxelSize)
        {
            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
            if (leaf.size < VoxelNum)
            {
                leaf.FusionPoint = F_Point;
                leaf.size = 1;
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
            if (leaf.size < VoxelNum)
            {
                leaf.size = 1;
                leaf.FusionPoint = F_Point;
                return true;
            }
            else
            {
                bool Judge = InsertForFusion_OS(idx, F_Cell, f_center, f_radius, F_Point, imageID);
                return Judge;
            }
        }
    }
    else
    {
        TYPE childradius = F_Cell.Node().F_Radius;
        F_POINT_TYPE childcenter = F_Cell.Node().center;

        const unsigned _idx = F_Cell.ComputeChild(F_Point);
        F_CELL_TYPE& cell = F_Cell.m_child[_idx];
        bool Judge = _InsertForFusionOS(_idx, cell, F_Point, childcenter, childradius, imageID);
        return Judge;
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline bool TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::InsertForFusion_OS(const unsigned idx, F_CELL_TYPE& F_Cell, F_POINT_TYPE& Center, TYPE F_Radius, F_POINT_TYPE& F_Point, uint32_t imageID)
{
    TYPE childradius = F_Radius / TYPE(2);
    F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
    F_Cell.Node().center = childcenter;
    F_Cell.Node().F_Radius = childradius;
    F_Cell.m_child = new F_CELL_TYPE[F_CELL_TYPE::numChildren];
    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
    {
        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.m_child[i].Leaf();
        leaf.size = 0;
        F_Cell.m_child[i].m_father = &F_Cell;
        F_Cell.m_child[i].m_VoxelKey = ComputeNum(F_Cell.Node().center, F_Cell.Node().F_Radius, F_Cell.m_VoxelKey.m_LOD+1, i);
        arrOctreeCellMap[F_Cell.m_child[i].m_VoxelKey] = &F_Cell.m_child[i];
        F_Cell.m_child[i].Node().F_Radius = childradius / TYPE(2);
    }

    typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();

    F_POINT_TYPE& F_Cell_Point0 = leaf.FusionPoint;
    const unsigned idx0 = F_Cell.ComputeChild(F_Cell_Point0);
    const unsigned idx1 = F_Cell.ComputeChild(F_Point);

    if (idx0 != idx1)
    {
        F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
        typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

        leaf0.FusionPoint = F_Cell_Point0;
        leaf0.size = 1;

        F_CELL_TYPE& child1 = F_Cell.m_child[idx1];
        typename F_CELL_TYPE::LEAF_TYPE& leaf1 = child1.Leaf();

        leaf1.FusionPoint = F_Point;
        leaf1.size = 1;

        return true;
    }
    else
    {
        if (childradius <= VoxelSize)
        {
            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
            typename F_CELL_TYPE::LEAF_TYPE& leaf0 = child0.Leaf();

            leaf0.FusionPoint = F_Cell_Point0;
            leaf0.size = 1;
            return false;
        }
        else
        {
            F_CELL_TYPE& child0 = F_Cell.m_child[idx0];
            typename F_CELL_TYPE::LEAF_TYPE& leaf = child0.Leaf();
            leaf.FusionPoint = F_Cell_Point0;
            leaf.size = 1;

            bool Judge = InsertForFusion_OS(idx0, child0, childcenter, childradius, F_Point, imageID);
            return Judge;
        }
    }
}

/////////�����˲���///////////////////////
template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::CollectLeafs(bool bBox/* = false*/)
{
    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
    {
        F_CELL_TYPE& cell = F_Root.m_child[i];
        TraverseOctree_(cell, bBox);
    }
}
template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::TraverseOctree_(F_CELL_TYPE& F_Cell, bool bBox/* = false*/)
{
    if (F_Cell.IsLeaf())
    {
        cellTotalNum++;
        arrOctreeCellNum.push_back(F_Cell.m_VoxelKey);
        AddLeaf(F_Cell, bBox);
    }
    else
    {
        for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
        {
            F_CELL_TYPE& cell = F_Cell.m_child[i];
            TraverseOctree_(cell, bBox);
        }
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::AddLeaf(F_CELL_TYPE& F_Cell, bool bBox/* = false*/)
{
    arrOctreeLeafMap[F_Cell.m_VoxelKey] = &F_Cell;

    // ����˸��ǵ�����
    if (bBox)
    {
        float radius = F_Cell.Node().F_Radius * 2.0;

        Point3f corner1; // ��ǰ��
        corner1.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
        corner1.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
        corner1.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
        F_Cell.m_CornerPoints.push_back(corner1);

        Point3f corner2; // ��ǰ��
        corner2.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
        corner2.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
        corner2.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
        F_Cell.m_CornerPoints.push_back(corner2);

        Point3f corner3; // �����
        corner3.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
        corner3.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
        corner3.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
        F_Cell.m_CornerPoints.push_back(corner3);

        Point3f corner4; // �Һ���
        corner4.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
        corner4.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
        corner4.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
        F_Cell.m_CornerPoints.push_back(corner4);

        Point3f corner5; // ��ǰ��
        corner5.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
        corner5.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
        corner5.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
        F_Cell.m_CornerPoints.push_back(corner5);

        Point3f corner6; // ��ǰ��
        corner6.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
        corner6.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
        corner6.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
        F_Cell.m_CornerPoints.push_back(corner6);

        Point3f corner7; // �����
        corner7.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
        corner7.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
        corner7.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
        F_Cell.m_CornerPoints.push_back(corner7);

        Point3f corner8; // �Һ���
        corner8.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
        corner8.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
        corner8.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
        F_Cell.m_CornerPoints.push_back(corner8);
    }
}


template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::TraverseOctree(PointArr& pointArr, PointViewArr& pointViewArr, PointWeightArr& pointWeightArr)
{
    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
    {
        F_CELL_TYPE& cell = F_Root.m_child[i];
        TraverseOctree_(cell, pointArr, pointViewArr, pointWeightArr);
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::TraverseOctree_(F_CELL_TYPE& F_Cell, PointArr& pointArr, PointViewArr& pointViewArr, PointWeightArr& pointWeightArr)
{
    if (F_Cell.IsLeaf())
    {
        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
        if (leaf.size==1)
        {
            //TOctreeLeafs.push_back(&F_Cell);
            Point& point_mvs = pointArr.AddEmpty();
            point_mvs[0] = leaf.FusionPoint[0];
            point_mvs[1] = leaf.FusionPoint[1];
            point_mvs[2] = leaf.FusionPoint[2];

            ViewArr& views_mvs = pointViewArr.AddEmpty();
            WeightArr& weights_mvs = pointWeightArr.AddEmpty();
            views_mvs = *(F_Cell.FusionPointViewArr);
            weights_mvs = *(F_Cell.FusionPointWeightArr);
            F_Cell.FusionPointViewArr->Release();
            F_Cell.FusionPointWeightArr->Release();
            delete F_Cell.FusionPointViewArr;
            delete F_Cell.FusionPointWeightArr;
            F_Cell.FusionPointViewArr = NULL;
            F_Cell.FusionPointWeightArr = NULL;
        }
    }
    else
    {
        for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
        {
            F_CELL_TYPE& cell = F_Cell.m_child[i];
            TraverseOctree_(cell, pointArr, pointViewArr, pointWeightArr);
        }
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::TraverseOctree(int& leafNum, int& pointNum)
{
    for (int i = 0; i < F_CELL_TYPE::numChildren; i++)
    {
        F_CELL_TYPE& cell = F_Root.m_child[i];
        TraverseOctree_(cell,leafNum, pointNum);
    }
}

template<typename TYPE, int DIMS, typename DATA_TYPE>
inline void TOctreeByFusion<TYPE, DIMS, DATA_TYPE>::TraverseOctree_(F_CELL_TYPE& F_Cell, int& leafNum, int& pointNum)
{
    if (F_Cell.IsLeaf())
    {
        leafNum += 1;
        typename F_CELL_TYPE::LEAF_TYPE& leaf = F_Cell.Leaf();
        if (leaf.size == 1)
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
}
/*----------------------------------------------------------------*/
#include "Octree.inl"
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // __SEACAVE_OCTREE_H__
