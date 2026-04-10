/********************************************************************
file base:      LFDepth.h
author:         LZD
created:        2024/05/13
purpose:
*********************************************************************/
#ifndef ACMP_LFDEPTH_H
#define ACMP_LFDEPTH_H

#include "Common/CommonCUDA.h"
#include "Common/CommonUtil.h"

namespace LFMVS
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)
    const int MAX_PATCH_SIZE = 9;
    struct PatchMatchParamsLF
    {
        int max_iterations = 3;
        int patch_size = 5;
        int patch_Bound_size = 5; // 是否为边界像素
        int propagation_Graph_size = 5; // 是否为边界像素
        int num_images = 18;
        int max_image_size = 32000;
        int radius_increment = 1;
        float sigma_spatial = 5.0f;
        float sigma_color = 3.0f;
        int top_k = 5; // 2  5
        float baseline = 0.54f;
        float depth_min = 5.0f;
        float depth_max = 40.0f;
        float disparity_min = 5.0f;
        float disparity_max = 40.0f;
        bool geom_consistency = false;
        bool multi_geometry = false;
        bool planar_prior = false;
        float Base = 0; // 相邻两微透镜的像素距离
        int MLA_Mask_Width_Cuda = 0;
        int MLA_Mask_Height_Cuda= 0;

        float base_height_ratio = 0.2;
        float base_height_sigma = 0.05;
        bool  b_test = false;
    };

    /////////////////////////////////////////////////
    struct DisparityRange
    {
        DisparityRange()
        {
            m_dis_min = DBL_MAX;
            m_dis_max = -DBL_MAX;
            m_vd_min = DBL_MAX;
            m_vd_max = -DBL_MAX;
        }

    public:
        // dispartity
        double          m_dis_min;
        double          m_dis_max;

        // virtual depth
        double          m_vd_min;
        double          m_vd_max;
    };
    typedef std::map<std::string, DisparityRange> DisparityRangeMap;

    class MLA_img
    {
    public:
        MLA_img(cv::Mat image, cv::Point2f point1)
        {
            this->img = image;
            this->Lcenter_point = point1;
        }

        MLA_img()
        {
        }

    public:
        cv::Mat                             img; // 图像数据
        cv::Point2f                         Lcenter_point; // 微透镜中心点
        std::vector<cv::Point2f>            mine_point; // 匹配的自己图像的特征点
        std::vector<cv::Point2f>            other_point; // 匹配的中心图像的特征点
        double                              Base_line; // 基线长度
        //vector<double> PSNR_grade;
        std::vector<float32>                SAD_grade;
        std::vector<uint8>                  img_rgb; // 像素值&影像数据
        std::vector<float32>                cost; // 代价
        std::vector<std::vector<float32>>   cost_All; // 所有代价
        std::vector<float32>                disp; // 视差
        std::vector<std::vector<float32>>   disp_All; // 所有视差
    };

    struct Res_img
    {
        Res_img(int n, float baseline)
        {
            num = n;
            Base_line = baseline;
        }

    public:
        int                         num; // 微透镜图像的索引
        float                       Base_line; // 基线
        std::vector<cv::Point2f>    m_p;
        std::vector<cv::Point2f>    r_p;
        std::vector<float>          Ncc_grade;
    };

    class Res_image_Key
    {
    public:
        Res_image_Key(LFMVS::QuadTreeTileKeyPtr ptrKey, float baseline);
        ~Res_image_Key()
        {
            m_ptrKey.reset();
        }

        void ReSet();
        // 计算有效NCC点的数量和得分
        void ComputeValidNCCInfo();

    public:
        LFMVS::QuadTreeTileKeyPtr   m_ptrKey;
        int                         res_number; //在Res_img_TileKey中的索引
        float                       Base_line; // 基线

        int                         m_iLevel; // 从1开始计数：1，2，3

        // 匹配结果
        std::vector<cv::Point2f>    m_p; // 作为邻居图像的同名点坐标
        std::vector<cv::Point2f>    r_p; // 当前微透镜图像的同名点坐标
        std::vector<float>          Ncc_grade;

        int                         NCC_pointNum_valid;
        float                       NCC_Average_Score_valid;
        float                       score;
    };
    typedef std::vector<Res_image_Key>      Res_image_KeyVec;

    struct Neigh_Score
    {
        Neigh_Score()
        {
            m_SortIndex = -1;
            m_Score = 0.0;
            m_BlureValue = 0.0;
            m_Similarity = 0.0;
            m_Baseline = 0.0;
            m_Baseline_normalization = 0.0;

            m_Circle_index = -1;
            fPhotographicTerm = 0.0;
        }

        int     m_SortIndex; // 作为邻居，按照score的排序序号

        float   m_Score; // 该值越大，则作为邻居图像就越好

        float   m_BlureValue; // 邻域MI的模糊程度
        float   m_Similarity; // 当前MI与邻域MI的相似度

        float   m_Baseline; // 当前MI与邻域MI的基线长
        float   m_Baseline_normalization; // 归一化的基线

        int     m_Circle_index;
        float   fPhotographicTerm;   // 相似度值
    };
    typedef std::map<QuadTreeTileKeyPtr, Neigh_Score, QuadTreeTileKeyMapCmpLess> NeighScoreMap;

    class MLA_Problem
    {
    public:
        MLA_Problem();
        ~MLA_Problem();

        void Release();

    public:
        // neiKey_Vec转为res_img_tilekey
        void CreateResTileKeysFromNeiKeyVec(cv::Point2f& current_center_coord, QuadTreeTileInfoMap& MLA_info_map);

        // 计算当前微透镜图像与所以邻居图像在NCC指标方面的均值和标准差
        void Compute_avg_std_key(int iLevel);
        // 根据均值和标准差计算Score
        void ComputeScoreByNCC(int iLevel);

        static bool Compare_Res(Res_image_Key& rhs1, Res_image_Key& rhs2)
        {
            return rhs1.score > rhs2.score;
        }

        void SetM_NeigKeyPtrVec(QuadTreeTileKeyPtrVec& m_NeigKeyPtrVec);

        void ComputeBlurenessValue();
        void ComputeRichnessValue();

        void SortNeighScoreForRefocus();
        QuadTreeTileKeyPtrVec& GetSortedNeighScoreForRefocus();
        NeighScoreMap& GetNeighScoreMapForRefocus();

        void SortNeighScoreForMatch();
        QuadTreeTileKeyPtrVec& GetSortedNeighScoreForMatch();
        NeighScoreMap& GetNeighScoreMapForMatch();

        void WriteNeighbosInfoForRefocus();
        void WriteNeighbosInfoForMatch();
        void WriteNeighbosInfo_old();

        void RansacNeighborKeyForMatch(LightFieldParams& lf_Params);
        bool OutlierImpByRANSAC(LightFieldParams& lf_Params, float baseline, float similarity);

        float GetPhotographicValueRangeforMatch();
        float ComputePhotoRatio(float value);
        float ComputeBaselineRatio(float value);

        void ItemsNormalization();

    private:
        static bool CompareR_Tilekey(Res_image_Key & neibours_length, Res_image_Key & neibours_length1)
        {
            return neibours_length.Base_line < neibours_length1.Base_line;
        }

    public:
        int                                 main_img;
        std::vector<Res_img>                res_img; // 以基线长短对邻居进行排序
        std::vector<int>                    number; // 邻居图像的索引
        std::vector<float>                  result_vec;
        bool                                m_bGarbage; // 不需要深度估计

        QuadTreeTileKeyPtr                  m_ptrKey;
        cv::Mat                             m_Image_gray; // 微图像---灰度图
        cv::Mat                             m_Image_rgb; // 微图像---rgb图

        cv::Mat_<uint8>                     m_Image_Blureness_Bianry;  // 模糊程度图 (二值化)
        cv::Mat_<uint8>                     m_Image_Blureness;  // 模糊程度图
        cv::Mat_<uint8>                     m_Image_Richness;  // 纹理丰富性图 (二值化)
        unsigned long                       m_BlurenessValue;
        unsigned long                       m_RichnessValue;
        NeighScoreMap                       m_NeighScoreMapForRefocus;  // 融合用
        QuadTreeTileKeyPtrVec               m_NeighsSortVecForRefocus;  // 融合用

        NeighScoreMap                       m_NeighScoreMapForMatch;  // 视差匹配用
        QuadTreeTileKeyPtrVec               m_NeighsSortVecForMatch;  // 视差匹配用

        bool                                m_bComputRANSAC;
        float2                              m_ransac_ab_forMatch; // ransac函数的常数

        float2                              m_NeigDistance_range_forMatch; // 邻居中最大基线距离
        float2                              m_PhotographicValue_range_forMatch; // ransac函数的常数

        bool                                m_bNeedMatch;

        Res_image_KeyVec                    m_Res_Image_KeyVec; // 以基线长短对邻居进行排序
        QuadTreeTileKeyPtrVec               m_NeigKeyPtrVec;

        float                               m_Variance; // 方差
        float                               m_Standard_deviation; // 标准差

        //std::vector<cv::Point2f> mine_point;//匹配的自己图像的特征点
        //std::vector<cv::Point2f> other_point;//匹配的中心图像的特征点
    };
    typedef std::map<QuadTreeTileKeyPtr, MLA_Problem, QuadTreeTileKeyMapCmpLess> QuadTreeProblemMap; // <tilekey, match problem>
    typedef std::map<std::string, QuadTreeProblemMap> QuadTreeProblemMapMap;
    typedef std::map<QuadTreeTileKeyPtr, cv::Mat, QuadTreeTileKeyMapCmpLess> QuadTreeImageMap;

    /////////////////////////////////////////////////////
    class DisparityAndNormal
    {
    public:
        DisparityAndNormal(LightFieldParams& params);

        ~DisparityAndNormal();

        void Release();


    public: // 接口
        void CollectPropagationGraphLackedPixels(const int2 center, const int mi_width, const int mi_height
            , MLA_Problem& problem, QuadTreeTileInfoMap& mla_info_map,
            std::map<QuadTreeTileKeyPtr, std::shared_ptr<DisparityAndNormal>, QuadTreeTileKeyMapCmpLess>& disNormals_map,
            Proxy_DisPlane* proxy_dis_plane);

        bool IsBroken(const int2 p, const int mi_height, const int mi_width, const int propagation_Graph_size);

    public:
        LightFieldParams                m_Params;
        QuadTreeTileKeyPtr              m_ptrKey;

        std::vector<float>              dis;
        std::vector<cv::Vec3f>          nor;
        cv::Mat_<float>                 depths;
        cv::Mat_<cv::Vec3f>             normals;
        cv::Mat_<float>                 costs;

        float4*                         ph_cuda; // normal+w(disparity)
        float4*                         disp_v_cuda; // (标准视差d， 实际斜方视差d_real，baseline, 虚拟深度)
        float*                          c_cuda; // cost 匹配代价
        float*                          d_cuda; // disparity 视差值
        unsigned int*                   selected_views;

        int3*                           neighbor_Patch_info;//id+x+y
        int3*                           neighbor_PGR_info;//id+x+y

        eStereoStage                    m_StereoStage;

        /////////////////////////////////////////////////////////
        // 微透镜图像的行列号
        int                             row;
        int                             col;
        int                             num;
        /////////////////////////////////////////////////////////
        // test
        int                         m_iDelete_count;
    };
    typedef std::shared_ptr<DisparityAndNormal> DisparityAndNormalPtr;
    typedef std::map<QuadTreeTileKeyPtr, DisparityAndNormalPtr, QuadTreeTileKeyMapCmpLess> QuadTreeDisNormalMap; // < >
    typedef std::map<std::string, QuadTreeDisNormalMap> QuadTreeDisNormalMapMap;

}
#endif //ACMP_LFDEPTH_H