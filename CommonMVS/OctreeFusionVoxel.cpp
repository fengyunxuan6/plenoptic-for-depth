#include "OctreeFusionVoxel.h"


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
    inline TOctreeCellFusionVoxel::TOctreeCellFusionVoxel(TOctreeCellFusionVoxel* children)
        : m_child(children)
        , m_pCoord(NULL)
        , m_nodeRadius(0)
    {
    }

    inline TOctreeCellFusionVoxel::~TOctreeCellFusionVoxel()
    {
        Release();
    }

    inline void TOctreeCellFusionVoxel::Release()
    {
        delete[] m_child;
        m_child = NULL;
    }

    void TOctreeCellFusionVoxel::SetCoord(POINT_TYPE* pCoord)
    {
        m_pCoord = pCoord;
    }

    SEACAVE::TOctreeCellFusionVoxel::POINT_TYPE* TOctreeCellFusionVoxel::GetCoord()
    {
        return m_pCoord;
    }

    void TOctreeCellFusionVoxel::ResetCoord()
    {
        m_pCoord = NULL;
    }


    void TOctreeCellFusionVoxel::DeleteCoord()
    {
        if (m_pCoord != NULL)
        {
            delete m_pCoord;
            m_pCoord = NULL;
        }
    }

    bool TOctreeCellFusionVoxel::IsCoordValid()
    {
        if (m_pCoord == NULL)
            return true;
        return false;
    }

    void TOctreeCellFusionVoxel::SetCenter(const POINT_TYPE& nodeCenter)
    {
        m_nodeCenter = nodeCenter;
    }

    const SEACAVE::TOctreeCellFusionVoxel::POINT_TYPE& TOctreeCellFusionVoxel::GetCenter() const
    {
        return m_nodeCenter;
    }

    void TOctreeCellFusionVoxel::SetRadius(TYPE radius)
    {
        m_nodeRadius = radius;
    }

    SEACAVE::TOctreeCellFusionVoxel::TYPE& TOctreeCellFusionVoxel::GetRadius()
    {
        return m_nodeRadius;
    }

    // compute item's index corresponding to the containing cell
    inline unsigned TOctreeCellFusionVoxel::ComputeChild(const POINT_TYPE& item) const
    {
        unsigned idx = 0;
        if (item[0] >= m_nodeCenter[0])
            idx |= (1 << 0);
        if (3 > 1)
            if (item[1] >= m_nodeCenter[1])
                idx |= (1 << 1);
        if (3 > 2)
            if (item[2] >= m_nodeCenter[2])
                idx |= (1 << 2);
        return idx;
    } // ComputeChild

    void TOctreeCellFusionVoxel::ComputeCenter(POINT_TYPE centers[])
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


    bool TOctreeCellFusionVoxel::NoChildren() const
    {
        return (m_child == NULL);
    }

    // count the number of items contained by the given octree-cell
    //size_t TOctreeCellFusionVoxel::GetNumItemsHeld() const
    //{
    //    if (HasChildren())
    //        return GetNumItems();
    //    size_t numItems = 0;
    //    for (int i = 0; i < numChildren; ++i)
    //        numItems += GetChild(i).GetNumItemsHeld();
    //    return numItems;
    //}

    //////////////////////////////////////////////////////////////////////////
    inline void TOctreeFusionVoxel::GetTOctreeLeafs(std::vector<CELL_TYPE*>& toctreeleafs)
    {
        toctreeleafs.swap(TOctreeLeafs);
        //toctreeleafs.assign(TOctreeLeafs.begin(), TOctreeLeafs.end());
    }

    // 以用户输入的用于预测像素区域的物方体素的尺寸（SIZE）为依据，优化计算出最终算法使用的物方体素的尺寸（VoxelSize）
    void TOctreeFusionVoxel::IniOctree(const F_AABB_TYPE& AB, int NUM, float SIZE)
    {
        VoxelSize = SIZE;
        VoxelNum = NUM;
        cellTotalNum = 0;

        F_Root.SetRadius(GetRadius(AB));
        F_Root.SetCenter(AB.GetCenter());
        F_Root.m_father = NULL;
        minPoint = F_Root.GetCenter() - F_POINT_TYPE::Constant(F_Root.GetRadius());

        bool IsFinishe = false;
        TYPE _Radius = F_Root.GetRadius();
        // by xyy
//        while (!IsFinishe)
//        {
//            _Radius = _Radius / TYPE(2);
//            if (_Radius <= VoxelSize)
//            {
//                TYPE UP_Radius = _Radius * TYPE(2);
//                float ElementOne = UP_Radius - VoxelSize;
//                float ElementTwo = VoxelSize - _Radius;
//                if (ElementOne > ElementTwo)
//                {
//                    VoxelSize = _Radius;
//                }
//                else
//                {
//                    VoxelSize = UP_Radius;
//                }
//                IsFinishe = true;
//            }
//        }

        F_Root.m_child = new CELL_TYPE[CELL_TYPE::numChildren];
        for (int i = 0; i < CELL_TYPE::numChildren; i++)
        {
            F_Root.m_child[i].m_father = &F_Root;
            F_Root.m_child[i].m_VoxelKey = ComputeVoxelKey(F_Root.GetCenter(), F_Root.GetRadius(), 1, i);
            F_Root.m_child[i].SetRadius(F_Root.m_nodeRadius / TYPE(2));
            arrOctreeCellMap[F_Root.m_child[i].m_VoxelKey] = &F_Root.m_child[i];
        }
        VERBOSE("OCFVoxel, Voxel size for predict, Previous: %f, computed: %f", SIZE, VoxelSize);
    }

    bool TOctreeFusionVoxel::InsertForFusion(const F_POINT_TYPE& pointCoord)
    {
        TYPE f_radius = F_Root.GetRadius();
        const F_POINT_TYPE f_center = F_Root.GetCenter();
        const unsigned idx = F_Root.ComputeChild(pointCoord);
        CELL_TYPE& cell = F_Root.m_child[idx];
        bool Judge = _InsertForFusion(idx, cell, pointCoord, f_center, f_radius);
        return Judge;
    }

    inline bool TOctreeFusionVoxel::_InsertForFusion(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius)
    {
        if (F_Cell.NoChildren())
        {
            if (f_radius <= VoxelSize)
            {
                if (F_Cell.IsCoordValid())
                {
                    F_Cell.SetCoord(new F_POINT_TYPE(F_Point));
                    return true;
                }
                else
                    return false;
            }
            else
            {
                if (F_Cell.IsCoordValid())
                {
                    F_Cell.SetCoord(new F_POINT_TYPE(F_Point));
                    return true;
                }
                else
                {
                    // 裂分
                    bool Judge = InsertForFusion_(idx, F_Cell, f_center, f_radius, F_Point);
                    return Judge;
                }
            }
        }
        else // 递归
        {
            TYPE childradius = f_radius / TYPE(2);
            F_POINT_TYPE childcenter(ComputeChildCenter(f_center, childradius, idx));
            F_Cell.SetCenter(childcenter);
            F_Cell.SetRadius(childradius);

            const unsigned _idx = F_Cell.ComputeChild(F_Point);
            CELL_TYPE& cell = F_Cell.m_child[_idx];
            bool Judge = _InsertForFusion(_idx, cell, F_Point, childcenter, childradius);
            return Judge;
        }
    }

    inline bool TOctreeFusionVoxel::InsertForFusion_(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& Center, TYPE F_Radius, const F_POINT_TYPE& F_Point)
    {
        // 初始化子节点
        F_Cell.m_child = new CELL_TYPE[CELL_TYPE::numChildren];
        TYPE childradius = F_Radius / TYPE(2);
        F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
        F_Cell.SetCenter(childcenter);
        F_Cell.SetRadius(childradius);
        F_POINT_TYPE* pCell_Point0 = F_Cell.GetCoord();
        const unsigned idx0 = F_Cell.ComputeChild(*pCell_Point0);
        const unsigned idx1 = F_Cell.ComputeChild(F_Point);

        if (idx0 != idx1)
        {
            CELL_TYPE& child0 = F_Cell.m_child[idx0];
            child0.SetCoord(pCell_Point0);
            F_Cell.ResetCoord();

            CELL_TYPE& child1 = F_Cell.m_child[idx1];
            child1.SetCoord(new F_POINT_TYPE(F_Point));
            return true;
        }
        else
        {
            if (childradius <= VoxelSize)
            {
                CELL_TYPE& child0 = F_Cell.m_child[idx0];
                child0.SetCoord(pCell_Point0);
                F_Cell.ResetCoord();
                return false;
            }
            else
            {
                CELL_TYPE& child0 = F_Cell.m_child[idx0];
                child0.SetCoord(pCell_Point0);
                F_Cell.ResetCoord();

                bool Judge = InsertForFusion_(idx0, child0, childcenter, childradius, F_Point);
                return Judge;
            }
        }
    }

    // 体素八叉数插入--一个体素允许存放多个点
    bool TOctreeFusionVoxel::InsertForFusion_Bucket(const F_POINT_TYPE& pointCoord)
    {
        TYPE f_radius = F_Root.GetRadius();
        const F_POINT_TYPE f_center = F_Root.GetCenter();
        return _InsertForFusion_Bucket(F_Root, pointCoord, f_center, f_radius);
    }

    inline bool TOctreeFusionVoxel::_InsertForFusion_Bucket(CELL_TYPE& F_Cell, const F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius)
    {
        F_Cell.SetCenter(f_center);
        F_Cell.SetRadius(f_radius);

        // 1) 到达最小体素：叶子体素收集多点，不丢弃
        if (f_radius <= VoxelSize)
        {
            F_Cell.m_points.push_back(F_Point);
            return true;
        }

        // 2) 未到最小体素：固定下沉 -> 必须保证 children 存在
        if (F_Cell.NoChildren())
        {
            F_Cell.m_child = new CELL_TYPE[CELL_TYPE::numChildren];
            for (int i = 0; i < CELL_TYPE::numChildren; ++i)
                F_Cell.m_child[i].m_father = &F_Cell;
        }

        const TYPE childRadius = f_radius / TYPE(2);
        const unsigned idx = F_Cell.ComputeChild(F_Point);  // 用 cell.m_nodeCenter 判象限
        const F_POINT_TYPE childCenter = ComputeChildCenter(f_center, childRadius, idx);

        return _InsertForFusion_Bucket(F_Cell.m_child[idx], F_Point, childCenter, childRadius);
    }

    // 由 AABB 计算 root 半径：取最大边长的一半（与 GetRadius(AB) 的常见实现一致）
    inline float TOctreeFusionVoxel::GetRootRadiusFromAABB(F_AABB_TYPE& ab)
    {
        TOctreeCellFusionVoxel::POINT_TYPE sz = ab.GetSize(); // (sx, sy, sz) 这里是边长，不是半边长
        const float maxSide = std::max({sz.x(), sz.y(), sz.z()});
        return 0.5f * maxSide;
    }

// 将一个盒子（center, radius）写成：
// 8 顶点 + 12 三角面 + 12 条边线（wireframe）
    inline void TOctreeFusionVoxel::AppendBoxAsTriangles(std::ofstream& os,
                                                         TOctreeCellFusionVoxel::POINT_TYPE& c,
                                                         float r,
                                                         int& vBase)   // 引用传递
    {
        const float xmin = c.x() - r, xmax = c.x() + r;
        const float ymin = c.y() - r, ymax = c.y() + r;
        const float zmin = c.z() - r, zmax = c.z() + r;

        // 8 vertices（顺序保持不变）
        const float V[8][3] = {
                {xmin, ymin, zmin}, // 1
                {xmax, ymin, zmin}, // 2
                {xmax, ymax, zmin}, // 3
                {xmin, ymax, zmin}, // 4
                {xmin, ymin, zmax}, // 5
                {xmax, ymin, zmax}, // 6
                {xmax, ymax, zmax}, // 7
                {xmin, ymax, zmax}, // 8
        };

        for (int i = 0; i < 8; ++i)
            os << "v " << V[i][0] << " " << V[i][1] << " " << V[i][2] << "\n";

        // 这里不用 lambda，保持“普通写法”
        auto writeFace = [&](int a, int b, int c_) { os << "f " << a << " " << b << " " << c_ << "\n"; };
        auto writeLine = [&](int a, int b) { os << "l " << a << " " << b << "\n"; };

        const int i0 = vBase; // OBJ 1-based

        // 12 triangles
        writeFace(i0+0, i0+1, i0+2);  writeFace(i0+0, i0+2, i0+3); // bottom
        writeFace(i0+4, i0+6, i0+5);  writeFace(i0+4, i0+7, i0+6); // top
        writeFace(i0+0, i0+5, i0+1);  writeFace(i0+0, i0+4, i0+5); // front
        writeFace(i0+3, i0+2, i0+6);  writeFace(i0+3, i0+6, i0+7); // back
        writeFace(i0+0, i0+3, i0+7);  writeFace(i0+0, i0+7, i0+4); // left
        writeFace(i0+1, i0+6, i0+2);  writeFace(i0+1, i0+5, i0+6); // right

        // 12 edges（wireframe lines）
        // bottom loop: 1-2-3-4-1
        writeLine(i0+0, i0+1);
        writeLine(i0+1, i0+2);
        writeLine(i0+2, i0+3);
        writeLine(i0+3, i0+0);

        // top loop: 5-6-7-8-5
        writeLine(i0+4, i0+5);
        writeLine(i0+5, i0+6);
        writeLine(i0+6, i0+7);
        writeLine(i0+7, i0+4);

        // vertical edges: 1-5, 2-6, 3-7, 4-8
        writeLine(i0+0, i0+4);
        writeLine(i0+1, i0+5);
        writeLine(i0+2, i0+6);
        writeLine(i0+3, i0+7);

        vBase += 8; // 关键：更新索引基址，给下一个盒子用
    }

    bool TOctreeFusionVoxel::ExportOccupiedLeafVoxelsObj(std::string& objPath,
                                                         bool includeRoot,
                                                         float voxelScale)
    {
        // 0) 收集 occupied leaf（确保 leafmap 是最新的）
//        cellTotalNum = 0;
//        arrOctreeCellNum.clear();
//        arrOctreeLeafMap.clear();
//        TOctreeLeafs.clear();

//        CollectLeafs(false);  // 这里不一定要 cornerPoints，因为我们直接用 center/radius画
        // 若你更想用 cornerPoints，也可以传 true

        std::ofstream os(objPath);
        if (!os.is_open())
            return false;

        os << "# Occupied leaf voxels (bucket mode)\n";
        int vBase = 1;

        // 1) 可选：画 root 框，便于参照
        if (includeRoot)
        {
            auto c = F_Root.GetCenter();
            float r = F_Root.GetRadius() * voxelScale;
            AppendBoxAsTriangles(os, c, r, vBase);
        }

        // 2) 画所有已填充叶子体素
        for (CELL_TYPE* leaf : occupiedLeaves)
        {
            if (!leaf) continue;
            auto c = leaf->GetCenter();
            float r = leaf->GetRadius() * voxelScale;
            AppendBoxAsTriangles(os, c, r, vBase);
        }
        return true;
    }

    // 输入：网格顶点整数坐标 (vx,vy,vz)
    // 输出：OBJ 顶点索引（1-based）
    inline int TOctreeFusionVoxel::GetOrAddVertex(std::ofstream& os,
                                     std::unordered_map<Int3Key, int, Int3KeyHash>& vtxMap,
                                     const Eigen::Vector3f& minPoint,
                                     float cellSize,
                                     int vx, int vy, int vz,
                                     int& vBase)
    {
        Int3Key key{vx, vy, vz};
        auto it = vtxMap.find(key);
        if (it != vtxMap.end())
            return it->second;

        // 由网格整数坐标还原为真实坐标
        const float x = minPoint.x() + vx * cellSize;
        const float y = minPoint.y() + vy * cellSize;
        const float z = minPoint.z() + vz * cellSize;

        os << "v " << x << " " << y << " " << z << "\n";
        const int idx = vBase++;
        vtxMap.emplace(key, idx);
        return idx;
    }

    bool TOctreeFusionVoxel::ExportOccupiedLeafSurfaceObj(const std::string& objPath,
                                                          const std::vector<CELL_TYPE*>& occupiedLeaves,
                                                          bool includeRoot /*=false*/)
    {
        if (occupiedLeaves.empty())
            return false;

        std::ofstream os(objPath);
        if (!os.is_open())
            return false;

        os << "# Occupied leaf surface mesh (boundary faces only)\n";

        // 叶子半径（默认所有叶子相同）
        const float r = occupiedLeaves.front()->GetRadius();
        const float cellSize = 2.0f * r;

        // 你类里已有 minPoint（rootMin = rootCenter - rootRadius）
        const Eigen::Vector3f rootMin(minPoint[0], minPoint[1], minPoint[2]);

        // 1) 建 occupied voxel 的索引集合：用 leaf 的 minCorner 映射到 (ix,iy,iz)
        std::unordered_set<Int3Key, Int3KeyHash> occSet;
        occSet.reserve(occupiedLeaves.size() * 2);

        auto ToIndex = [&](const CELL_TYPE* leaf)->Int3Key {
            Eigen::Vector3f c = leaf->GetCenter();
            Eigen::Vector3f minCorner = c - Eigen::Vector3f::Constant(r);
            Eigen::Vector3f t = (minCorner - rootMin) / cellSize;

            // 用 round 抵抗浮点误差（minCorner 理论应落在网格上）
            int ix = (int)std::llround(t.x());
            int iy = (int)std::llround(t.y());
            int iz = (int)std::llround(t.z());
            return Int3Key{ix,iy,iz};
        };

        for (const CELL_TYPE* leaf : occupiedLeaves) {
            if (!leaf) continue;
            if (leaf->m_points.empty()) continue; // 二次保险
            occSet.insert(ToIndex(leaf));
        }

        // 2) 顶点去重表
        std::unordered_map<Int3Key, int, Int3KeyHash> vtxMap;
        vtxMap.reserve(occSet.size() * 8);

        int vBase = 1;

        // 3) 可选：输出 root 立方体（只用于参照，不参与剔面）
        if (includeRoot) {
            // 注意：这里 root 不是网格化去重的一部分，直接写一份独立立方体也行
            // 如果你必须也参与去重，需要把它映射到同一网格，这里先不做。
            Eigen::Vector3f c(F_Root.GetCenter().x(), F_Root.GetCenter().y(), F_Root.GetCenter().z());
            float rr = F_Root.GetRadius();
            // 直接写 8顶点 + 12三角（不去重），省事
            // 如果你一定要 root 也去重，我可以再给你一版。
            auto xmin=c.x()-rr, xmax=c.x()+rr;
            auto ymin=c.y()-rr, ymax=c.y()+rr;
            auto zmin=c.z()-rr, zmax=c.z()+rr;

            int i0 = vBase;
            os << "v " << xmin << " " << ymin << " " << zmin << "\n";
            os << "v " << xmax << " " << ymin << " " << zmin << "\n";
            os << "v " << xmax << " " << ymax << " " << zmin << "\n";
            os << "v " << xmin << " " << ymax << " " << zmin << "\n";
            os << "v " << xmin << " " << ymin << " " << zmax << "\n";
            os << "v " << xmax << " " << ymin << " " << zmax << "\n";
            os << "v " << xmax << " " << ymax << " " << zmax << "\n";
            os << "v " << xmin << " " << ymax << " " << zmax << "\n";
            vBase += 8;

            // 12 triangles
            os << "f " << i0+0 << " " << i0+1 << " " << i0+2 << "\n";
            os << "f " << i0+0 << " " << i0+2 << " " << i0+3 << "\n";
            os << "f " << i0+4 << " " << i0+6 << " " << i0+5 << "\n";
            os << "f " << i0+4 << " " << i0+7 << " " << i0+6 << "\n";
            os << "f " << i0+0 << " " << i0+5 << " " << i0+1 << "\n";
            os << "f " << i0+0 << " " << i0+4 << " " << i0+5 << "\n";
            os << "f " << i0+3 << " " << i0+2 << " " << i0+6 << "\n";
            os << "f " << i0+3 << " " << i0+6 << " " << i0+7 << "\n";
            os << "f " << i0+0 << " " << i0+3 << " " << i0+7 << "\n";
            os << "f " << i0+0 << " " << i0+7 << " " << i0+4 << "\n";
            os << "f " << i0+1 << " " << i0+6 << " " << i0+2 << "\n";
            os << "f " << i0+1 << " " << i0+5 << " " << i0+6 << "\n";
        }

        // 4) 输出 boundary faces
        // 对每个 occupied voxel：检查 6 邻域缺失则输出该面
        auto HasOcc = [&](int ix,int iy,int iz)->bool {
            return occSet.find(Int3Key{ix,iy,iz}) != occSet.end();
        };

        auto EmitQuad = [&](int v0,int v1,int v2,int v3){
            // 两个三角：v0-v1-v2, v0-v2-v3
            os << "f " << v0 << " " << v1 << " " << v2 << "\n";
            os << "f " << v0 << " " << v2 << " " << v3 << "\n";
        };

        for (const auto& vox : occSet)
        {
            const int ix = vox.x, iy = vox.y, iz = vox.z;

            // 8 corner vertex grid coords
            // (ix,iy,iz) 是 minCorner 的格点索引，因此顶点在 [ix,ix+1] 等
            const int x0=ix, x1=ix+1;
            const int y0=iy, y1=iy+1;
            const int z0=iz, z1=iz+1;

            // 取 8 个角点（去重生成）
            int V000 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x0,y0,z0, vBase);
            int V100 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x1,y0,z0, vBase);
            int V110 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x1,y1,z0, vBase);
            int V010 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x0,y1,z0, vBase);
            int V001 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x0,y0,z1, vBase);
            int V101 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x1,y0,z1, vBase);
            int V111 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x1,y1,z1, vBase);
            int V011 = GetOrAddVertex(os, vtxMap, rootMin, cellSize, x0,y1,z1, vBase);

            // -X face (left): neighbor (ix-1,iy,iz) 缺失才输出
            if (!HasOcc(ix-1,iy,iz)) EmitQuad(V000, V010, V011, V001);
            // +X face (right)
            if (!HasOcc(ix+1,iy,iz)) EmitQuad(V100, V101, V111, V110);
            // -Y face (front)
            if (!HasOcc(ix,iy-1,iz)) EmitQuad(V000, V001, V101, V100);
            // +Y face (back)
            if (!HasOcc(ix,iy+1,iz)) EmitQuad(V010, V110, V111, V011);
            // -Z face (bottom)
            if (!HasOcc(ix,iy,iz-1)) EmitQuad(V000, V100, V110, V010);
            // +Z face (top)
            if (!HasOcc(ix,iy,iz+1)) EmitQuad(V001, V011, V111, V101);
        }

        os.close();
        return true;
    }

    // 导出初始化后的八叉树：root + 第一层 8 个子节点
    bool TOctreeFusionVoxel::ExportInitOctreeObj_RootPlus8Children(F_AABB_TYPE& ab,
                                                                   std::string objPath,
                                                                   bool includeRoot,
                                                                   float childScale)
    {
        std::ofstream os(objPath);
        if (!os.is_open())
            return false;

        os << "# Init octree visualization: root + 8 children (faces + wireframe)\n";

        // root center/radius
        TOctreeCellFusionVoxel::POINT_TYPE rootCenter = ab.GetCenter();
        float rootRadius = GetRootRadiusFromAABB(ab);
        // 子节点半径 = rootRadius / 2（可选缩放用于可视化）
        float childRadius = (rootRadius * 0.5f) * childScale;
        // 8 个方向符号表
        TOctreeCellFusionVoxel::POINT_TYPE signs[8];
        SEACAVE::TOctreeCellFusionVoxel::ComputeCenter(signs);
        int vBase = 1;
        // root box
        if (includeRoot)
        {
            AppendBoxAsTriangles(os, rootCenter, rootRadius, vBase);
        }
        // 8 child boxes
        float childOffset = rootRadius * 0.5f;
        for (int idx = 0; idx < 8; ++idx)
        {
            TOctreeCellFusionVoxel::POINT_TYPE childCenter = rootCenter + signs[idx] * childOffset;
            AppendBoxAsTriangles(os, childCenter, childRadius, vBase);
        }
        os.close();
        return true;
    }

    //////////////用于体素场景的大体构建///////////////
    bool TOctreeFusionVoxel::InsertForFusionOS(const F_POINT_TYPE& F_Point, uint32_t imageID)
    {
        TYPE f_radius = F_Root.GetRadius();
        const F_POINT_TYPE& f_center = F_Root.GetCenter();
        const unsigned idx = F_Root.ComputeChild(F_Point);
        CELL_TYPE& cell = F_Root.m_child[idx];
        bool Judge = _InsertForFusionOS(idx, cell, F_Point, f_center, f_radius, imageID);
        return Judge;
    }

    inline bool TOctreeFusionVoxel::_InsertForFusionOS(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& F_Point, const F_POINT_TYPE& f_center, TYPE f_radius, uint32_t imageID)
    {
        if (F_Cell.NoChildren())
        {
            if (f_radius <= VoxelSize)
            {
                if (F_Cell.IsCoordValid())
                {
                    F_Cell.SetCoord(new F_POINT_TYPE(F_Point));
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if (F_Cell.IsCoordValid())
                {
                    F_Cell.SetCoord(new F_POINT_TYPE(F_Point));
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
            TYPE childradius = F_Cell.GetRadius();
            F_POINT_TYPE childcenter = F_Cell.GetCenter();

            const unsigned _idx = F_Cell.ComputeChild(F_Point);
            CELL_TYPE& cell = F_Cell.m_child[_idx];
            bool Judge = _InsertForFusionOS(_idx, cell, F_Point, childcenter, childradius, imageID);
            return Judge;
        }
    }

    bool TOctreeFusionVoxel::InsertForFusion_OS(const unsigned idx, CELL_TYPE& F_Cell, const F_POINT_TYPE& Center, TYPE F_Radius, const F_POINT_TYPE& F_Point, uint32_t imageID)
    {
        TYPE childradius = F_Radius / TYPE(2);
        F_POINT_TYPE childcenter(ComputeChildCenter(Center, childradius, idx));
        F_Cell.SetCenter(childcenter);
        F_Cell.SetRadius(childradius);
        F_Cell.m_child = new CELL_TYPE[CELL_TYPE::numChildren];
        for (int i = 0; i < CELL_TYPE::numChildren; i++)
        {
            F_Cell.m_child[i].m_father = &F_Cell;
            F_Cell.m_child[i].m_VoxelKey = ComputeVoxelKey(F_Cell.GetCenter(), F_Cell.GetRadius(), F_Cell.m_VoxelKey.m_LOD + 1, i);
            arrOctreeCellMap[F_Cell.m_child[i].m_VoxelKey] = &F_Cell.m_child[i];
            F_Cell.m_child[i].SetRadius(childradius / TYPE(2));
        }

        F_POINT_TYPE* pCell_Point0 = F_Cell.GetCoord();
        const unsigned idx0 = F_Cell.ComputeChild(*pCell_Point0);
        const unsigned idx1 = F_Cell.ComputeChild(F_Point);

        if (idx0 != idx1)
        {
            CELL_TYPE& child0 = F_Cell.m_child[idx0];
            child0.SetCoord(pCell_Point0);
            F_Cell.ResetCoord();

            CELL_TYPE& child1 = F_Cell.m_child[idx1];
            child1.SetCoord(new F_POINT_TYPE(F_Point));
            return true;
        }
        else
        {
            if (childradius <= VoxelSize)
            {
                CELL_TYPE& child0 = F_Cell.m_child[idx0];
                child0.SetCoord(pCell_Point0);
                F_Cell.ResetCoord();

                return false;
            }
            else
            {
                CELL_TYPE& child0 = F_Cell.m_child[idx0];
                child0.SetCoord(pCell_Point0);
                F_Cell.ResetCoord();

                bool Judge = InsertForFusion_OS(idx0, child0, childcenter, childradius, F_Point, imageID);
                return Judge;
            }
        }
    }

    /////////遍历八叉树///////////////////////
    void TOctreeFusionVoxel::CollectLeafs(bool bBox/* = false*/)
    {
        for (int i = 0; i < CELL_TYPE::numChildren; i++)
        {
            CELL_TYPE& cell = F_Root.m_child[i];
//            TraverseOctree_(cell, bBox);
            TraverseOctree_Bucket(cell, bBox);
        }

        printf("[Bucket] occupied leaf voxels = %zu, total points = %zu",
               occupiedLeafCount, occupiedPointCount);
    }

    inline void TOctreeFusionVoxel::TraverseOctree_(CELL_TYPE& F_Cell, bool bBox/* = false*/)
    {
        if (F_Cell.NoChildren())
        {
            cellTotalNum++;
            arrOctreeCellNum.push_back(F_Cell.m_VoxelKey);
            AddLeaf(F_Cell, bBox);
        }
        else
        {
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                CELL_TYPE& cell = F_Cell.m_child[i];
                TraverseOctree_(cell, bBox);
            }
        }
    }

    inline void TOctreeFusionVoxel::TraverseOctree_Bucket(CELL_TYPE& F_Cell, bool bBox)     // by xyy
    {
        if (F_Cell.NoChildren())
        {
            if (F_Cell.m_points.empty())
                return;

            occupiedLeaves.push_back(&F_Cell);
            occupiedLeafCount += 1;
            occupiedPointCount += F_Cell.m_points.size();

            cellTotalNum++;
            arrOctreeCellNum.push_back(F_Cell.m_VoxelKey);
            AddLeaf(F_Cell, bBox);
        }
        else
        {
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                CELL_TYPE& cell = F_Cell.m_child[i];
                TraverseOctree_Bucket(cell, bBox);
            }
        }
    }


    inline void TOctreeFusionVoxel::AddLeaf(CELL_TYPE& F_Cell, bool bBox/* = false*/)
    {
        arrOctreeLeafMap[F_Cell.m_VoxelKey] = &F_Cell;

        // 计算八个角点坐标
        if (bBox)
        {
            F_Cell.m_CornerPoints.clear();  // by xyy
            float radius = F_Cell.GetRadius() * 2.0;
            TOctreeFusionVoxel::F_POINT_TYPE minPoint = this->GetMinPoint(this->F_Root.GetCenter(), this->F_Root.GetRadius());

            Point3f corner1; // 左前下
            corner1.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
            corner1.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
            corner1.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
            F_Cell.m_CornerPoints.push_back(corner1);

            Point3f corner2; // 右前下
            corner2.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
            corner2.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
            corner2.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
            F_Cell.m_CornerPoints.push_back(corner2);

            Point3f corner3; // 左后下
            corner3.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
            corner3.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
            corner3.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
            F_Cell.m_CornerPoints.push_back(corner3);

            Point3f corner4; // 右后下
            corner4.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
            corner4.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
            corner4.z = minPoint[2] + F_Cell.m_VoxelKey.m_VoxelZ * radius;
            F_Cell.m_CornerPoints.push_back(corner4);

            Point3f corner5; // 左前上
            corner5.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
            corner5.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
            corner5.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
            F_Cell.m_CornerPoints.push_back(corner5);

            Point3f corner6; // 右前上
            corner6.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
            corner6.y = minPoint[1] + F_Cell.m_VoxelKey.m_VoxelY * radius;
            corner6.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
            F_Cell.m_CornerPoints.push_back(corner6);

            Point3f corner7; // 左后上
            corner7.x = minPoint[0] + F_Cell.m_VoxelKey.m_VoxelX * radius;
            corner7.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
            corner7.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
            F_Cell.m_CornerPoints.push_back(corner7);

            Point3f corner8; // 右后上
            corner8.x = minPoint[0] + (F_Cell.m_VoxelKey.m_VoxelX + 1) * radius;
            corner8.y = minPoint[1] + (F_Cell.m_VoxelKey.m_VoxelY + 1) * radius;
            corner8.z = minPoint[2] + (F_Cell.m_VoxelKey.m_VoxelZ + 1) * radius;
            F_Cell.m_CornerPoints.push_back(corner8);
        }
    }


    inline void TOctreeFusionVoxel::TraverseOctree(int& leafNum, int& pointNum)
    {
        for (int i = 0; i < CELL_TYPE::numChildren; i++)
        {
            CELL_TYPE& cell = F_Root.m_child[i];
            TraverseOctree_(cell, leafNum, pointNum);
        }
    }

    inline void TOctreeFusionVoxel::TraverseOctree_(CELL_TYPE& F_Cell, int& leafNum, int& pointNum)
    {
        if (F_Cell.NoChildren())
        {
            leafNum += 1;
            if (F_Cell.IsCoordValid())
            {
                pointNum += 1;
            }
        }
        else
        {
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                CELL_TYPE& cell = F_Cell.m_child[i];
                TraverseOctree_(cell, leafNum, pointNum);
            }
        }
        return; // 添加显式返回以避免编译器警告
    }

    void TOctreeFusionVoxel::DeleteCellInfo()
    {
        if (!F_Root.NoChildren())
        {
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                CELL_TYPE& cell = F_Root.m_child[i];
                DeleteCellInfo_(cell);
            }
        }
    }

    void TOctreeFusionVoxel::DeleteCellInfo_(CELL_TYPE& F_Cell)
    {
        if (F_Cell.NoChildren())
        {
            F_Cell.DeleteCoord();
        }
        else
        {
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                CELL_TYPE& cell = F_Cell.m_child[i];
                DeleteCellInfo_(cell);
            }
        }
        return; // 添加显式返回以避免编译器警告
    }

    TOctreeFusionVoxel::~TOctreeFusionVoxel()
    {
        Release();
    }

    void TOctreeFusionVoxel::Release()
    {
        DeleteCellInfo();
    }

    SEACAVE::TOctreeFusionVoxel::TYPE TOctreeFusionVoxel::GetRadius(const F_AABB_TYPE& aabb)
    {
        const TOctreeFusionVoxel::F_POINT_TYPE size(aabb.GetSize() / TYPE(2));
        TYPE radius = size[0];
        if (3 > 1 && radius < size[1])
            radius = size[1];
        if (3 > 2 && radius < size[2])
            radius = size[2];
        return radius;
    }

    SEACAVE::TOctreeFusionVoxel::F_POINT_TYPE TOctreeFusionVoxel::ComputeChildCenter(const TOctreeFusionVoxel::F_POINT_TYPE& center, TYPE radius, unsigned idxChild)
    {
        struct CENTERARR_TYPE
        {
            F_POINT_TYPE        child[CELL_TYPE::numChildren];
            inline CENTERARR_TYPE() { 
                F_POINT_TYPE centers_array[CELL_TYPE::numChildren];
                CELL_TYPE::ComputeCenter(centers_array);
                for (int i = 0; i < CELL_TYPE::numChildren; ++i)
                    child[i] = centers_array[i];
            }
        };
        static const CENTERARR_TYPE centers;
        return center + centers.child[idxChild] * radius;
    }

    SEACAVE::VoxelKey TOctreeFusionVoxel::ComputeVoxelKey(const TOctreeFusionVoxel::F_POINT_TYPE& center, TYPE radius, unsigned levelChild, unsigned idxChild)
    {
        VoxelKey m_Num;
        m_Num.m_LOD = levelChild;
        TOctreeFusionVoxel::F_POINT_TYPE m_point = TOctreeFusionVoxel::F_POINT_TYPE::Zero();
        if (idxChild == 0)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0] - radius, center[1] - radius, center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 1)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0], center[1] - radius, center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 2)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0] - radius, center[1], center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 3)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0], center[1], center[2] - radius);
            m_point = point_;
        }
        else if (idxChild == 4)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0] - radius, center[1] - radius, center[2]);
            m_point = point_;
        }
        else if (idxChild == 5)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0], center[1] - radius, center[2]);
            m_point = point_;
        }
        else if (idxChild == 6)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0] - radius, center[1], center[2]);
            m_point = point_;
        }
        else if (idxChild == 7)
        {
            TOctreeFusionVoxel::F_POINT_TYPE point_(center[0], center[1], center[2]);
            m_point = point_;
        }

        TOctreeFusionVoxel::F_POINT_TYPE minPoint = this->GetMinPoint(this->F_Root.GetCenter(), this->F_Root.GetRadius());
        m_Num.m_VoxelX = round((m_point[0] - minPoint[0]) / radius);
        m_Num.m_VoxelY = round((m_point[1] - minPoint[1]) / radius);
        m_Num.m_VoxelZ = round((m_point[2] - minPoint[2]) / radius);
        return m_Num;
    }

    void TOctreeFusionVoxel::GetNeighbor(VoxelKey num, std::vector<VoxelKey>& neighbor)
    {
        //////////////////////////Z+//////////////////////////////////
        VoxelKey zAddNum(num.m_LOD, num.m_VoxelX, num.m_VoxelY, num.m_VoxelZ + 1);
        Get_Neighbor(num, zAddNum, _ZADD, neighbor);
        //////////////////////////Z-//////////////////////////////////
        VoxelKey zSubtractNum(num.m_LOD, num.m_VoxelX, num.m_VoxelY, num.m_VoxelZ - 1);
        Get_Neighbor(num, zSubtractNum, _ZSUBTRACT, neighbor);
        //////////////////////////X+//////////////////////////////////
        VoxelKey xAddNum(num.m_LOD, num.m_VoxelX + 1, num.m_VoxelY, num.m_VoxelZ);
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

    void TOctreeFusionVoxel::Get_Neighbor(VoxelKey num, VoxelKey h_num, int handle, std::vector<VoxelKey>& returnNumArr)
    {
        std::map<VoxelKey, CELL_TYPE*>::iterator iter = arrOctreeCellMap.find(h_num);
        if (iter == arrOctreeCellMap.end())
        {
            CELL_TYPE& cell = *(arrOctreeCellMap[num]);
            VoxelKey returnNum(0, 0, 0, 0);
            bool judge = _GetNeighbor(cell, returnNum, handle);
            if (judge)
            {
                returnNumArr.push_back(returnNum);
            }
        }
        else
        {
            CELL_TYPE& h_cell = *(arrOctreeCellMap[h_num]);
            if (h_cell.NoChildren())
            {
                returnNumArr.push_back(h_num);
            }
            else
            {
                GetNeighbor_(h_cell, returnNumArr, handle);
            }
        }
    }

    bool TOctreeFusionVoxel::_GetNeighbor(CELL_TYPE& cell, VoxelKey& returnNum, int handle)
    {
        if (cell.m_father != NULL)
        {
            CELL_TYPE& cell_father = *(cell.m_father);
            VoxelKey num_father = cell_father.m_VoxelKey;
            if (handle == _XADD)
            {
                returnNum.m_LOD = num_father.m_LOD;
                returnNum.m_VoxelX = num_father.m_VoxelX + 1;
                returnNum.m_VoxelY = num_father.m_VoxelY;
                returnNum.m_VoxelZ = num_father.m_VoxelZ;
            }
            else if (handle == _XSUBTRACT)
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
            std::map<VoxelKey, CELL_TYPE*>::iterator iter = arrOctreeCellMap.find(returnNum);
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
        return false; // 添加默认返回值
    }

    void TOctreeFusionVoxel::GetNeighbor_(CELL_TYPE& cell, std::vector<VoxelKey>& returnNumArr, int handle)
    {
        if (handle == _XADD)
        {
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                if (i == 0 || i == 2 || i == 4 || i == 6)
                {
                    CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.NoChildren())
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
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                if (i == 1 || i == 3 || i == 5 || i == 7)
                {
                    CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.NoChildren())
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
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                if (i == 0 || i == 1 || i == 4 || i == 5)
                {
                    CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.NoChildren())
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
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                if (i == 2 || i == 3 || i == 6 || i == 7)
                {
                    CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.NoChildren())
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
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                if (i == 0 || i == 1 || i == 2 || i == 3)
                {
                    CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.NoChildren())
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
            for (int i = 0; i < CELL_TYPE::numChildren; i++)
            {
                if (i == 4 || i == 5 || i == 6 || i == 7)
                {
                    CELL_TYPE& cellChild = cell.m_child[i];
                    if (cellChild.NoChildren())
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
}