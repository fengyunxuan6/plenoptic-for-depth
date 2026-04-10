/********************************************************************
file base:      LFDenseMatcher.cpp
author:         LZD
created:        2024/05/13
purpose:
*********************************************************************/
#include "DepthSolverManager.h"

#include "MINeighbor/SelectNeighbors.h"
#include "Util/ImageQualityEstimate.h"
#include "Util/Logger.h"

// 添加线程相关的头文件
#include <exception>
#include <thread>

namespace LFMVS
{
    DepthSolverManager::DepthSolverManager(std::string& strRootPath, std::string& strSavePath)
    {
        // 初始化日志
        Logger::instance().setLogDir(strRootPath+LF_LOGGER_FOLDER_NAME);
        Logger::instance().setLevel(LOG_WARN);

        m_ptrDepthSolver = std::make_shared<DepthSolver>(strRootPath, strSavePath);
    }

    DepthSolverManager::~DepthSolverManager()
    {
        // 正确释放DepthSolver资源
        if (m_ptrDepthSolver) {
            try {
                m_ptrDepthSolver->Reset();
            } catch (...) {
                // 忽略重置过程中可能出现的任何异常
            }
            m_ptrDepthSolver.reset();
        }
    }

    int DepthSolverManager::DepthSolverPipelineBySequence(LFMVS::eStereoType eStereoType)
    {
        if (!m_ptrDepthSolver)
            return -1;

        // 读取光场原始影像
        if (!m_ptrDepthSolver->ReadRawImagesAndCreateProblemMaps()) // 创建了problem_map
            return -1;

        /********************* 微透镜相关 *************************/
        // Step 1: 解析微透镜中心点、主透镜焦距、微透镜焦距等相机参数信息
        eParseMLACentersType parse_type = LFMVS::ePMLACT_Auto;
        bool bGetMLACenters = m_ptrDepthSolver->GetOrComputeMLAInfo(parse_type);
        if (!bGetMLACenters)
            return -1;
        // 影像畸变校正
        m_ptrDepthSolver->Undistortion();

        /********** 微图像相关：逐帧处理帧内微图像的深度估计 *******************/
        // 根据标定时抛弃的微透镜，创建每帧中包含的微图像的深度估计对象

        // Step 2: 开始深度估计
        QuadTreeProblemMapMap& problems_map_map = m_ptrDepthSolver->GetMIAProblemsMapMap();
        for (QuadTreeProblemMapMap::iterator itr = problems_map_map.begin(); itr != problems_map_map.end(); itr++)
        {
            std::string str_frame = itr->first;
            QuadTreeProblemMap& problem_map = itr->second;
            LOG_ERROR("Depth Estimate, Frame: ", str_frame.c_str());

            // 创建微图像、disparity map 和 noraml map
            m_ptrDepthSolver->CreateProblemsAndDisNormals_Frame(str_frame);

            bool bRaw_key = false; // true
            if (bRaw_key)
                m_ptrDepthSolver->TestRawImageTilekeySequence(bRaw_key, str_frame, problem_map);

            // 获取或切割微图像
            if (!m_ptrDepthSolver->GetMLAImagesSequence(str_frame, problem_map))
                continue;

            // 深度估计
            m_ptrDepthSolver->SetStereoType(eStereoType);
            bool geom_consistency = false;
            m_ptrDepthSolver->ProcessProblemsImpSequence(itr, geom_consistency);
            // 在每个微图像处理完成后立即释放资源，避免内存和显存累积
            m_ptrDepthSolver->ReleaseResources(str_frame);
        }
        m_ptrDepthSolver->Reset();
        LOG_ERROR("Depth solver pipeline completed successfully");
        return 0;
    }

    int DepthSolverManager::DepthSolverPipelineByBatch()
    {
        if (!m_ptrDepthSolver)
            return -1;

        if (!m_ptrDepthSolver->ReadRawImagesAndCreateProblemMaps())
            return -1;

        // Step 1: 获取微透镜中心点
        LFMVS::eParseMLACentersType parse_type = LFMVS::ePMLACT_Auto;
        bool bGetMLACenters = m_ptrDepthSolver->GetOrComputeMLAInfo(parse_type);
        if (!bGetMLACenters)
            return -1;

        // Step 2: 获取微透镜图像
        // 根据CalibInfo信息统计被抛弃的微透镜图像
        m_ptrDepthSolver->CreateMIAofProblemDisNormals();

        // 对原始的帧内微图像加上行列号，并写出，供测试使用
        bool bRaw_key = true;
        m_ptrDepthSolver->TestRawImageTilekey(bRaw_key);

        bool bGetMLAImages = m_ptrDepthSolver->GetMLAImages();
        if (!bGetMLAImages)
            return -1;

        // Step 3: 微透镜图像的邻居选择
        m_ptrDepthSolver->SelectNeighborsForProblems();

        // Step 4: 深度初始估计;转虚拟深度图和真实深度图
        bool geom_consistency = false;
        LFMVS::eStereoType stereo_type = LFMVS::eST_PlannerPrior; // 深度估计算法
        m_ptrDepthSolver->SetStereoType(stereo_type);
        m_ptrDepthSolver->ProcessProblemsImp(geom_consistency);

        // Step 5: 转虚拟深度图和真实深度图
        m_ptrDepthSolver->WriteDisMap_TileKey_new_Accu();
        m_ptrDepthSolver->Virtual_depth_map_TileKey();
        //m_ptrDepthSolver->Virtual_depth_map_TileKey_new();

        // 销毁内存
        m_ptrDepthSolver->Reset();
        return 0;
    }

}