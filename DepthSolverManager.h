/********************************************************************
file base:      DepthSolver.h
author:         LZD
created:        2024/05/13
purpose:
*********************************************************************/
#ifndef LF_DEPTHSOLVERMANAGER_H
#define LF_DEPTHSOLVERMANAGER_H

#include "DepthSolver.h"

namespace LFMVS
{
    class DepthSolverManager
    {
    public:
        DepthSolverManager(std::string& strRootPath, std::string& strSavePath);

        ~DepthSolverManager();

    public: // 接口
        // 顺序流程进行深度估计
        int DepthSolverPipelineBySequence(LFMVS::eStereoType eStereoType);

        // 按批次（多帧）进行深度估计
        int DepthSolverPipelineByBatch();

    private:
        std::shared_ptr<DepthSolver>        m_ptrDepthSolver;
    };
}
#endif //LF_DEPTHSOLVERMANAGER_H