/********************************************************************
file base:      PTotEstimator.cpp
author:         LZD
created:        2025/09/04
purpose:        pTotal-delate估计器
*********************************************************************/
#ifndef PTOTESTIMATOR_H
#define PTOTESTIMATOR_H
// 供 Host 端使用的 p_Δ^{tot} 估计与表构建工具（C++11）

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>              // 为了 float2/int2 定义
#include "MIStereo/AdaptMIPMUtil.cuh"  // 为了 lfhex::hex_ring_from_points_with_radius_kind

namespace LFMVS
{
	struct PTotOptions {
		float theta_max_deg = 12.0f; // 与 b̂_Δ 的夹角门限（度）
		int   min_obs       = 6;     // Direct-Δ 观测最少个数（不足则回退）
	};

	class PTotEstimator {
	public:
		PTotEstimator(int width, int height, float base_px, PTotOptions opt = {})
			: W_(width), H_(height), Base_(base_px), opt_(opt) {}

		// 线性下标工具（与 Device 端保持一致：y*W + x）
		inline int lin(const int2& tk) const { return tk.y * W_ + tk.x; }

		// 计算六边形步数 k_Δ（与 Device 保持一致）
		inline int hexSteps(const float2& c_ref, const float2& c_nei) const {
			int k = lfhex::hex_ring_from_points_with_radius_kind<double>(
						c_nei.x, c_nei.y, c_ref.x, c_ref.y, Base_*0.5, 1);
			return std::max(1, k);
		}

		// 直接从“本参考的 problem slice（参考 + 所有邻域中心）”估计 p_Δ^{tot}（像素）
		float computePTotDirectDelta(const std::vector<float2>& slice_centers,
									 const float2& c_ref,
									 const float2& c_nei) const;

		// 缓存版（评测常用）：以 (ref_lin, nei_lin) 为键缓存 pTot
		float getPTotCached(int ref_lin, int nei_lin,
							const std::vector<float2>& slice_centers,
							const float2& c_ref, const float2& c_nei);

		void clearCache() { cache_.clear(); }

		// 基于“参考 + 邻域 tilekeys + 对应中心（slice 顺序）”直接构建三张小表（供 CUDA 下发）
		// 约定：slice_centers[0] = c_ref； slice_centers[i+1] 对应 neighbors[i]
		void buildPerCallTablesFromSlice(const int2& tilekey_ref,
										 const std::vector<int2>& neighbors,
										 const std::vector<float2>& slice_centers,
										 std::vector<int>&   nei_lin,
										 std::vector<int>&   kSteps,
										 std::vector<float>& pTot) const;

	private:
		static inline float2 f2(float x,float y){ float2 a; a.x=x; a.y=y; return a; }
		static inline float  dot(const float2& a,const float2& b){ return a.x*b.x + a.y*b.y; }
		static inline float  nrm(const float2& a){ return std::sqrt(std::max(0.f, dot(a,a))); }
		static inline float2 unit(const float2& a){ float n=nrm(a); return (n>0.f? f2(a.x/n,a.y/n): f2(1.f,0.f)); }
		static inline float  clampf(float v,float lo,float hi){ return std::max(lo,std::min(hi,v)); }

		static float median(std::vector<float>& v){
			if(v.empty()) return 0.f;
			size_t m=v.size()/2;
			std::nth_element(v.begin(), v.begin()+m, v.end());
			float med=v[m];
			if((v.size()&1)==0){
				std::nth_element(v.begin(), v.begin()+m-1, v.end());
				med = 0.5f*(med + v[m-1]);
			}
			return med;
		}

		inline unsigned long long key64(int ref_lin, int nei_lin) const {
			return ( (unsigned long long)(unsigned int)ref_lin << 32 ) | (unsigned int)nei_lin;
		}

	private:
		int   W_ = 0, H_ = 0;
		float Base_ = 0.f;
		PTotOptions opt_;
		std::unordered_map<unsigned long long, float> cache_;
	};
}
#endif