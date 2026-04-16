/********************************************************************
file base:      AdaptMIPMPFPGR.h
author:         LZD
created:        2025/07/12
purpose:        微图像的视差计算
*********************************************************************/
#ifndef ADAPTMIPMPFPGR_H
#define ADAPTMIPMPFPGR_H

#include "AdaptMIPM.h"

namespace LFMVS
{
    class AdaptMIPMPFPGR: public AdaptMIPM
    {
    public:
        AdaptMIPMPFPGR(LightFieldParams& params);
        ~AdaptMIPMPFPGR();

    public: // 接口
        // 边缘像素感知的微图像密集匹配算法：补全 patch+propagation
        void RunPatchMatchCUDAForMI_PFPGR_Collect();

        void RunPatchMatchCUDAForMI_PFPGR_Repair();

        void Initialize(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                    QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                    std::vector<float>& costVec);
        void Initialize_SecondStage(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                    QuadTreeProblemMap& problem_map, std::vector<float4>& planeVec,
                    std::vector<float>& costVec,
                    Proxy_DisPlane* proxy_Plane, StereoResultInfo* pStereoResult);

        int3 GetNeighborPGR(const int index);

        void InitializeGPU_collect();
        void InitializeGPU_new();

        // test
        void TestWritePF_PGRInfo();
        void TestWriteNeighbour();
        void TestWriteNeighbour_color(MLA_Problem& problem, QuadTreeProblemMap& problem_map);

    private:
        // 第一次GPU使用
        int3*                    neighbor_PGR_host; // (邻域微图像id, 对应点像素坐标 x,y )
        int3*                    neighbor_PGR_cuda; // (邻域微图像id, 对应点像素坐标 x,y )

        // 第二次GPU使用
        Proxy_DisPlane*          proxy_plane_host; // (邻域微图像id, 对应点像素坐标 x,y )
        Proxy_DisPlane*          proxy_plane_cuda; // (邻域微图像id, 对应点像素坐标 x,y )
    };
}
#endif //ADAPTMIPMPFPGR_H
