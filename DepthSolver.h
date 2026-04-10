/********************************************************************
file base:      DepthSolver.h
author:         LZD
created:        2025/03/12
purpose:        帧内的匹配、视差图、虚拟深度图及全聚焦图像生成
*********************************************************************/
#ifndef DEPTHSOLVER_H
#define DEPTHSOLVER_H

#include "pleno/geometry/distortions.h"

#include "Common/QuadTree.h"
#include "MVStereo/LFACMP.h"

namespace LFMVS
{
    class LF_ACMP;

    //////////////////////////////////////////////////////////////////////////////////////
    // 统计
    class KDE
    {
    public:
        KDE(const std::vector<float>& data, float bandwidth)
            : data_(data), h_(bandwidth)
        {
        }

        // 估计
        float operator()(float x)
        {
            float inv_sqrt_2pi = 1.0 / std::sqrt(2*3.1415926);
            float sum = 0.0;
            for (float xi:data_)
            {
                float u=(x-xi)/h_;
                sum += std::exp(-0.5*u*u);
            }
            return (sum/(data_.size()*h_))*inv_sqrt_2pi;
        }

    private:
        std::vector<float> data_;
        float h_;
    };

    struct kde_point
    {
        float x;
        float density;
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // 微透镜图像深度估计
    class DepthSolver
    {
    public:
        DepthSolver(std::string& strRootPath, std::string& strSavePath);
        ~DepthSolver();

    public:
        LightFieldParams& GetLightFieldParams();

        // 影像畸变校正
        void Undistortion();
        // 主透镜的畸变校正
        bool UndistortionWithMainLens();
        // 微透镜的畸变校正
        bool UndistortionWithMicroLens(cv::Mat& ori_image, cv::Mat& undis_image);

        /**
         * 释放当前处理问题所占用的资源
         * 在处理大量微图像时，每个微图像处理完成后应及时释放资源，避免内存和显存累积
         */
        void ReleaseResources(std::string& strFrame);

        // 内存和显存信息打印函数
        void PrintMemoryInfo(const std::string& tag);

        // 选择邻居
        void SelectNeighborsForProblems();
        void SelectNeighborsForProblemsBySequence(std::string& strName, QuadTreeProblemMap& problem_map);

        // 深度估计
        void ProcessProblemsImp(bool geom_consistency); // 全局式
        void ProcessProblemsImpSequence(QuadTreeProblemMapMap::iterator itr,
                                        bool geom_consistency); // 增量式

        std::vector<cv::Vec3b>& GetColorList()
        {
            return m_Colors;
        }

    public: // 统计
        /** 一步得到全局内点（MAD 版）：输入所有视差 -> 输出全局 inlier 和阈带 */
        void BuildGlobalDispInlierMAD(const std::vector<float> &disp_all_raw,
                                    std::vector<float> &disp_inlier_global, float &lo_out,
                                    float &hi_out, float k = 3.0f, float min_band = 0.02f);

    private:
        void PrintCudaMemoryInfo(const std::string& tag);
        void PrintSystemMemoryInfo(const std::string& tag);

    public:
        void Init();
        void Reset();

        void SetPlannar(bool b);
        bool GetPlannar();
        void SetLRCheck(bool b);
        bool GetLRCheck();

        // 设置深度估计算法的类型
        void SetStereoType(eStereoType type);
        eStereoType GetStereoType();

        // 获取微透镜的信息
        QuadTreeTileInfoMap& GetMLAInfoMap();
        // 获取原始微图像
        std::map<std::string, cv::Mat>& GetRawImageMap();
        // 获取微图像对应的深度估计问题集合
        QuadTreeProblemMapMap& GetMIAProblemsMapMap();
        QuadTreeProblemMap& GetMIAProblemsMap(std::string& strName);

        // 获取微透镜图像
        bool GetMLAImages();
        bool GetMLAImagesSequence(std::string strNameLessExt, QuadTreeProblemMap& problem_map);

        cv::Mat& GetWhiteImage();

        std::string& GetSavePath();
        std::string& GetRootPath();

        // 微图像模糊程度
        std::map<std::string, cv::Mat>& GetBlurScoreImageMap();
        void SetBlurScoreImageMap(std::string& strFrameName, cv::Mat& blurScoreImage);

        void GetRichnessScoreMI(std::string& strFrameName, QuadTreeTileKeyPtr ptrKey, cv::Mat& target);

        void GetBlurScoreMI(std::string& strFrameName, QuadTreeTileKeyPtr ptrKey, cv::Mat& target);

        // 纹理丰富性
        std::map<std::string, cv::Mat>& GetRichnessImageMap();
        void SetRichnessImageMap(std::string& strFrameName, cv::Mat& richnessImage);

        const int GetTopGPUDevice()
        {
            return m_top_device;
        }

    public:
        void SetRawImageWidth(int width);
        const int GetRawImageWidth();

        void SetRawImageHeight(int height);
        const int GetRawImageHeight();

        LightFieldParams GetParams();

    public:
        bool AddMLAInfo(QuadTreeTileKeyPtr ptrKey, MLA_InfoPtr ptrInfo);

        void CreateProblem_frame(QuadTreeTileKeyPtr ptrKey, std::string strFrame);
        void CreateDisNormal_frame(QuadTreeTileKeyPtr ptrKey, std::string strFrame);

        void CreateProblems(QuadTreeTileKeyPtr ptrKey);
        void CreateDisNormals(QuadTreeTileKeyPtr ptrKey);
        QuadTreeDisNormalMapMap& GetMLADisNormalMapMap();

        // 从白图像中获取微透镜图像的中心点坐标
        bool GetOrComputeMLAInfo(eParseMLACentersType parse_type);

        bool ReadIntrinsicsFromXML();
        // 读取原始的帧图像
        bool ReadRawImagesAndCreateProblemMaps();
        // 读取白图像
        bool ReadWhiteImage();

        // 计算并搜集有效的微透镜图像，创建problem、disnormal
        void CreateMIAofProblemDisNormals();
        void CreateProblemsAndDisNormals_Frame(std::string str_frame);

    private:
        // 移除problem、disnormal
        void RemoveProblemsAndDisNormals_Frame(std::string str_frame);

        void ProcessProblemsByACMH_LF_Tilekey();
        void ProcessProblemsByACMH_LF_TilekeySequence(std::string& strName, QuadTreeProblemMap& problem_map);
        void DepthEstimateByACMH_LF_Tilekey(std::string& strName, QuadTreeProblemMap& problem_map);

        void ProcessProblemsByPlannerPrior_LF_Tilekey(bool geom_consistency);
        void ProcessProblemsByPlannerPrior_LF_TilekeySequence(std::string& strName, QuadTreeProblemMap& problem_map, bool geom_consistency);
        void DepthEstimateByPlannerPrior_LF_Tilekey(std::string& strName, QuadTreeProblemMap& problem_map, bool geom_consistency);

        void ProcessProblemsByHP_LF_Tilekey(bool geom_consistency);
        void ProcessProblemsByHP_LF_TilekeySequence(std::string& strName, QuadTreeProblemMap& problem_map, bool geom_consistency);
        void DepthEstimateByHP_LF_Tilekey(std::string& strName, QuadTreeProblemMap& problem_map, bool geom_consistency);

        void ProcessProblemsByBlurFeature_LF_Tilekey();
        void ProcessDepthInfo_BlurFeature_LFFrame(QuadTreeProblemMapMap::iterator& itrFrame);

        void ProcessProblems_planarImp(QuadTreeProblemMap& problem_map, QuadTreeDisNormalMap& dis_normal_map,
                                      bool geom_consistency, bool planar_prior, bool multi_geometry, std::string& strName);
        void LRCheckImp_Tilekey(QuadTreeProblemMap& problem_map, QuadTreeDisNormalMap& dis_normal_map);

        // 写出标定结果
        bool WriteCalibrationParamsXML(std::string& strCalibFullPath);

        // 读取标定结果
        bool ReadCalibrationParamsFromXML();

        void VisualizeDisparityWithMaskOverlay(const cv::Mat& dis_gray, cv::Mat& disp_color);

    public:
        struct NCC_Info
        {
            int     point_number;
            float   avg_ncc_value;
        };

    private:
        bool ComputeRawImageFullPath();

        bool ComputeWhiteImageFullPath();

    private:

        // 方案A：MAD 全局稳健区间
        /** @brief 由全局视差样本计算 MAD 稳健区间 [lo, hi]（约等价 3σ 剔除）
        * @param disp_all   全局收集的视差（建议只放有效值）
        * @param k          截断系数（默认 3.0）
        * @param min_band   最小带宽防夹扁（默认 0.02，按你的视差量纲调）
        * @param lo_out     输出：下界
        * @param hi_out     输出：上界 */
        void ComputeGlobalBandByMAD(const std::vector<float> &disp_all,
                                    float k, float min_band,
                                    float &lo_out, float &hi_out);
        void Filter_by_band(const std::vector<float> &in, float lo, float hi, std::vector<float> &out);


        std::vector<std::string>& GetRawImagePathVec();
        std::string& GetWhiteImagePath();

        // 白图像的二值化
        int Otsu(cv::Mat& image);
        void SortCenter(std::vector<cv::Point2f>& Center, std::vector<std::vector<cv::Point2f>>& center_1);

        // 提取白图像的中心点
        bool ComputeMLACentersFromWhiteImage();
        bool ComputeCenterFromWhiteImage_Special();

        // 从原始影像中切割微透镜图像
        bool Slice_RawMLAImage(std::string& strName, std::string& strMLAPath,
            QuadTreeProblemMap& problem_map, bool bWriteMLAImages);

        // 量化微图像的模糊程度
        void QuantizeBlurLevelForMI(cv::Mat& gray_MI, std::string& strMI_BlurValue_path);

        bool ReadMLAImages(std::string strMLAPath, int MLA_valid_image_count, QuadTreeProblemMap& problem_map);

        // 计算特征匹配对需要满足的阈值
        std::vector<double> slope(cv::Point2f point1, cv::Point2f point2);

        std::vector<std::string> splitStrings(const std::string &str, char delimiter);

    private: // 邻域图像选择  QuadTreeProblemMap

        //////////////////////////////////////////////////////////////////////////////////////////////
        // 旧代码
        void Generate_MlaList(std::vector<cv::Mat>& img,std::vector<cv::Point2f> &center,
                              std::vector<MLA_Problem> &MLA_problems, cv::Mat& Test_Image);
        void sift_match(MLA_img & image, std::vector<MLA_img> & images);
        void sift_match1(std::vector<cv::Mat> & img, std::vector<cv::Point2f> & center, MLA_Problem & problem);

        void compute_NCC(MLA_img & image, std::vector<MLA_img> & images);
        void compute_NCC1(std::vector<cv::Mat> & img,  MLA_Problem & problem);

        void Get_len_img1(std::vector<cv::Mat> & img, std::vector<cv::Point2f> & center, MLA_Problem & problem,int i, int j);

        void Sort_len_img(MLA_img & main_img, std::vector<MLA_img> & MLa_img);
        void Sort_len_img1(std::vector<cv::Point2f>& center, MLA_Problem& problem);

        void SortMLANeigImagesByLength(MLA_Problem& problem);

        void Select_img2(MLA_img & main_img, std::vector<MLA_img> & MLa_img, std::vector<MLA_img> & res_img);
        void Select_img1(std::vector<cv::Mat> & img, std::vector<cv::Point2f> & center, MLA_Problem & problem);
        void Select_img(std::vector<cv::Mat> & img, std::vector<cv::Point2f> & center, MLA_Problem & problem,int i, int j);
        //////////////////////////////////////////////////////////////////////////////////////////////
        // 新代码
        // 固定位置选择邻居
        void SelectNeighborsFromFixedPosition(QuadTreeTileKeyPtr ptrKey);
        void CollectNeigFromFixedPosition(MLA_Problem& problem);
        void CollectNeighborKey(MLA_Problem& problem, QuadTreeTileKeyPtr ptrKey);

        void SelectNeighborsFromFixedPositionSequence(QuadTreeTileKeyPtr ptrKey, std::string& strName, QuadTreeProblemMap& problem_map);

        // 特征点的自适应选择邻居
        void SelectNeighborsFromFeatures(QuadTreeTileKeyPtr ptrKey);

        void SelectNeighborsFromFeaturesSequence(QuadTreeTileKeyPtr ptrKey, std::string& strName, QuadTreeProblemMap& problem_map);

        void CollectMLANeigImagesByPOSE(MLA_Problem& problem);
        void Sift_MatchFromTileKey(MLA_Problem& problem, QuadTreeProblemMap& problem_map);
        void Compute_NCCFromTileKey(MLA_Problem& problem);
        void Select_NeighborsByNCC(MLA_Problem& problem);

        std::vector<float>   Compute_avg_std_key(std::vector<Res_image_Key>& res_img_tilekey);

    public: // 微图像：深度估计
        void ProcessProblem_LF(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center, const MLA_Problem & mlaProblem,std::vector<DisparityAndNormal> &DNS,float &Base,int i,int j);
        void ProcessProblem_LF_TileKey(MLA_Problem& problem, QuadTreeProblemMap& problem_map,
            QuadTreeDisNormalMap& dis_normal_map, std::string& strName);

        void ProcessProblem_planner_LF_TileKey(MLA_Problem& problem, QuadTreeProblemMap& problem_map,
                                            QuadTreeDisNormalMap& dis_normal_map, std::string& strFrameName,
                                            bool geom_consistency);
        void ProcessProblem_HP_LF_TileKey(MLA_Problem& problem, QuadTreeProblemMap& problem_map,
                                    QuadTreeDisNormalMap& dis_normal_map, std::string& strName,
                                    bool geom_consistency);

        void ProcessProblem_planar(const std::vector<cv::Mat> &img, const std::vector<cv::Point2f> &center,const MLA_Problem & mlaProblem,DisparityAndNormal & DN,float &Base,bool geom_consistency, bool planar_prior, bool multi_geometrty);

        void ProcessProblem_planar_TileKey(MLA_Problem& problem, QuadTreeProblemMap& problem_map,
            QuadTreeDisNormalMap& dis_normal_map, std::string& strName,
            bool geom_consistency, bool planar_prior, bool multi_geometrty);

        // 确定当前的problem是否需要深度估计
        void ConfirmProblemForEstimation(MLA_Problem& problem);

        void Focus_AIF_VD(QuadTreeProblemMapMap::iterator& itrFrame);
        void Virtual_depth_map_TileKey();

    private: // 微图像: 深度值的优化
        float Sample(std::vector<float>  dis, float x0, float y0);

        void LRCheck(std::vector<DisparityAndNormal> &DNS, std::vector<cv::Point2f> & center);
        void LRCheck_TileKey(MLA_Problem& problem, QuadTreeDisNormalMap& dis_normal_map);
        void LRCheck2(std::vector<DisparityAndNormal> &DNS, std::vector<DisparityAndNormal> &DNS1);

        void Virtual_depth(std::vector<DisparityAndNormal> &DNS1, cv::Mat &VD_res,float &Base);

        void ShowDMap(float * dis,  cv::Mat &img);
        void ShowDisMap(std::vector<DisparityAndNormal> &DNS1, cv::Mat &img, std::string &path, int i);
        void WriteDisMap_TileKey_new();
        void WriteDisMap_TileKey_new_AccuSequence(std::string& strName, QuadTreeProblemMap& problem_map); // 精准坐标

        void WriteDisMapForMIA(std::string& strName, QuadTreeProblemMap& problem_map);

        void Reshape_img(cv::Mat & img, std::vector <std::vector<cv::Point2f>> & center_1);

        void TestSelectNeighbors(std::string& strName, QuadTreeProblemMap& problems_map);

        void checkNeiborhoods(std::vector<cv::Point2f> &center, std::vector<MLA_Problem> &MLA_problems,cv::Mat &BackGroundImage);

    private: // 虚拟深度图、全聚焦图像等的生成
        void Virtual_depth_map(cv::Mat &img, cv::Mat &result,
                                std::vector<std::vector<cv::Point2f>> &center_1,
                                std::string &path, float d);
        void Virtual_depth_map_TileKey_new();

        void Virtual_depth_map_TileKeySequence(std::string& strName, QuadTreeProblemMap& problem_map);

    private:
        // GPU设备信息
        void SelectGPUDevice();
        void GetDeviceInfo();


        void SliceAndIndicatlizeForMI(QuadTreeProblemMapMap::iterator& itrFrame);

    public:
        void TestRawImageTilekey(bool bTest_tilekey);
        void TestRawImageTilekeyWithCircleLine(bool bTest_tilekey, QuadTreeTileKeyPtr ptrCenterKey,
            QuadTreeTileKeyPtrCircles& circleKeyMap);
        void TestRawImageTilekeyWithSortNeighForRefocus(bool bTest_tilekey, QuadTreeTileKeyPtr ptrCenterKey,
                                                        MLA_Problem& curr_problem);

        void TestRawImageTilekeyWithSortNeighForMatch(bool bTest_tilekey, QuadTreeTileKeyPtr ptrCenterKey,
                                                        MLA_Problem& curr_problem);

        void TestRawImageTilekeySequence(bool bTest_tilekey, std::string& strName, QuadTreeProblemMap& problem_map);

        void WriteDisMap_TileKey_new_Accu(); // 精准坐标

    private:
        std::string                         m_strRootPath;
        std::string                         m_strSavePath;

        LightFieldParams                    m_Params;

        std::vector<std::string>            m_strRawImagePathVec;
        std::map<std::string, cv::Mat>      m_RawImageMap;
        std::map<std::string, cv::Mat>      m_BlurscoreImageMap; // 模糊程度指标结果图
        std::map<std::string, cv::Mat>      m_RichnessImageMap; // 纹理丰富性指标结果图
        int                                 m_RawImage_Width;
        int                                 m_RawImage_Height;

        std::string                         m_strWhiteImagePath;
        cv::Mat                             m_WhiteImage;

        QuadTreeTileInfoMap                 m_MLA_info_map; // 起始文件
        int                                 m_MLA_valid_image_count; // 帧内可有效进行深度估计的微图像数量
        int                                 m_iGarbageRows; // 微图像外围的丢弃的行列数
        QuadTreeProblemMapMap               m_MIA_problem_map_map; // 中间文件
        QuadTreeDisNormalMapMap             m_MIA_dispNormal_map_map; // 结果文件

        int                                 m_top_device; // 最优的GPU设备索引号

        bool                                m_bPlannar;
        bool                                m_bLRCheck;
        eStereoType                         m_eStereoType;

        // test
        int                                 m_iCount;
        int                                 m_iCount_2;
        DisparityRangeMap                   m_disparityRangeMap;
        cv::Mat                             m_raw_image_key_gray; // 带有key的灰度图
        std::string                         m_strLogFileFullName; // 日志文件（.txt）的全路径

        std::vector<cv::Vec3b>              m_Colors;
    };
}
#endif //DEPTHSOLVER_H
