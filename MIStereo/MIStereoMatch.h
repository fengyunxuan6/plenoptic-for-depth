/********************************************************************
file base:      MIStereoMatch.h
author:         LZD
created:        2025/06/04
purpose:        微图像的视差计算
*********************************************************************/
#ifndef MISTEREOMATCH_H
#define MISTEREOMATCH_H

#include "DepthSolver.h"
#include "AdaptMIPM.h"
#include "AdaptMIPMPFPGR.h"

namespace LFMVS
{
    class MIStereoMatch
    {
        friend DepthSolver;

    public:
        MIStereoMatch(DepthSolver* pDepthSolver);

        ~MIStereoMatch();

    public:
        void StereoMatchingForMIA(QuadTreeProblemMapMap::iterator& itrFrame);

        // 匹配块补充（替换）
        void StereoMatchingForMIA_SoftProxyRepair(QuadTreeProblemMapMap::iterator& itrFrame);

        void StereoMatchingForMIA_FrameCrossViews(QuadTreeProblemMapMap::iterator& itrFrame);

        // 匹配块补充（替换）+ 传播路径补充: softProxy and Propagation Graph Repair(边缘区域不完整Patch和传播路径的修复)
        void StereoMatchingForMIA_SoftProxyPGRRepair(QuadTreeProblemMapMap::iterator& itrFrame);

    private:
        void TestWriteDisparityImage(std::string& strFrameName, QuadTreeTileKeyPtr ptrKey, AdaptMIPM& adapt_MIPM);
        void TestWriteDisparityImage_PRG(std::string& strFrameName, QuadTreeTileKeyPtr ptrKey, AdaptMIPM& adapt_MIPM);

    private:
        DepthSolver*             m_ptrDepthSolver;

        std::string              m_strLogFileFullName; // 日志文件（.txt）的全路径
    };
}

#endif //MISTEREOMATCH_H
