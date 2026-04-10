/********************************************************************
file base:      LFACMP.h
author:         LZD
created:        2024/06/12
purpose:
*********************************************************************/
#ifndef LFACMP_H
#define LFACMP_H

#include "LFDepthInfo.h"

namespace LFMVS
{
    class LF_ACMP
    {
    public:
        LF_ACMP(LightFieldParams& params);
        ~LF_ACMP();

    public:
        void SetTileKey(int tile_x, int tile_y)
        {
            m_tile_x = tile_x;
            m_tile_y = tile_y;
        }

        void InuputInitializationLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,const MLA_Problem & mlaProblem,float &Base);
        void InuputInitialization_LF_TileKey(QuadTreeTileInfoMap& MLA_info_map, MLA_Problem& problem,
                                            QuadTreeProblemMap& problem_map,
                                            std::vector<float4>& planeVec,
                                            std::vector<float>& costVec);
        void InuputInitialization_planarLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,const MLA_Problem & mlaProblem,const DisparityAndNormal & DN,float &Base);
        void InuputInitialization_planarLF_TileKey(QuadTreeTileInfoMap& MLA_info_map,
                                                   MLA_Problem& problem,
                                                   QuadTreeProblemMap& problem_map,
                                                   QuadTreeDisNormalMap& MLA_DisNormalMap);

        void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
        void CudaSpaceInitializationLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,
                                       const MLA_Problem & mlaProblem,std::vector<float4> Plane,std::vector<float> Cost);
        void CudaSpaceInitialization_LF_TileKey();
        void CudaSpaceInitialization_planarLF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,const MLA_Problem & mlaProblem,const DisparityAndNormal & DN);
        void CudaSpaceInitialization_planarLF_TileKey();

    public:
        void RunPatchMatchLF();
        void RunPatchMatchLF_plane();

        void SetGeomConsistencyParams(bool multi_geometry);
        void SetPlanarPriorParams();

        int GetReferenceImageWidth();
        int GetReferenceImageHeight();

        cv::Mat GetReferenceImage();

        float4 GetPlaneHypothesis(const int index);
        unsigned int GetSelected_viewIndexs(const int pt_col, const int pt_row);
        void SelectedViewIndexConvert(MLA_Problem& problem, unsigned int neig_viewBit); // 邻居索引号的转换二进制转换为十进制

        float GetCost(const int index);

        void GetSupportPoints(std::vector<cv::Point>& support2DPoints);

        std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);

        float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);

        float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
        float GetDepthFromPlaneParam_LF_Tilekey(const float4 plane_hypothesis, const int x, const int y);

        float GetMinDepth();
        float GetMaxDepth();

        float3 Get3DPointonRefCamLF(const int x, const int y, const float depth);

        void CudaPlanarPriorInitialization_LF_Tilekey(const std::vector<float4>& PlaneParams, const cv::Mat_<float>& masks);

        void CudaPlanarPriorInitialization(const std::vector<float4>& PlaneParams, const cv::Mat_<float>& masks);

    public:
        void Delete_pc();

        void ReleaseMemory();

    private:
        LightFieldParamsCUDA                        m_ParamsCUDA;

        int                                         num_images;
        std::vector<cv::Mat>                        images;
        float2*                                     cp; // center_points
        std::vector<cv::Mat>                        depths;
        std::vector<Camera>                         cameras;
        cudaTextureObjects                          texture_objects_host;
        cudaTextureObjects                          texture_depths_host;
        float4*                                     plane_hypotheses_host;
        float*                                      costs_host;
        float4*                                     prior_planes_host;
        unsigned int*                               plane_masks_host;

        // CUDA中使用的变量
        PatchMatchParamsLF                          params;
        Camera*                                     cameras_cuda;
        cudaArray*                                  cuArray[MAX_IMAGES];
        cudaArray*                                  cuDepthArray[MAX_IMAGES];
        cudaTextureObjects*                         texture_objects_cuda;
        cudaTextureObjects*                         texture_depths_cuda;
        float2*                                     centers_cuda;

        float4*                                     plane_hypotheses_cuda;
        curandState*                                rand_states_cuda;
        float*                                      costs_cuda;

        unsigned int*                               selected_views_cuda;
        unsigned int*                               selected_views_host;

        float*                                      depths_cuda;
        float4*                                     prior_planes_cuda;
        unsigned int*                               plane_masks_cuda;
        int*                                        COL1;
        int*                                        ROW1;
        int*                                        WIDTH1;
        int*                                        HIGHT1;

        int                                         m_tile_x;
        int                                         m_tile_y;
    };
}
#endif //LFACMP_H
