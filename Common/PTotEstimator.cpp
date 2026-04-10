/********************************************************************
file base:      PTotEstimator.cpp
author:         LZD
created:        2025/09/04
purpose:        pTotal-delate估计器
*********************************************************************/
#include "PTotEstimator.h"

namespace LFMVS
	{
	float PTotEstimator::computePTotDirectDelta(const std::vector<float2>& slice_centers,
												const float2& c_ref,
												const float2& c_nei) const
	{
		const float2 b  = f2(c_ref.x - c_nei.x, c_ref.y - c_nei.y);
		const float  bn = nrm(b);
		const float2 bh = unit(b);

		// 与 Device 一致的 k_Δ
		const int k_delate = hexSteps(c_ref, c_nei);

		std::vector<float> obs; obs.reserve(32);
		const int M = static_cast<int>(slice_centers.size());

		for(int u=0; u<M; ++u){
			const float2 cu = slice_centers[u];
			for(int w=0; w<M; ++w){
				if(w==u) continue;
				const float2 cw = slice_centers[w];
				const float2 uw = f2(cw.x - cu.x, cw.y - cu.y);
				const float  uwN = nrm(uw);
				if (uwN < 1e-6f) continue;

				// 与 b̂_Δ 的夹角门限
				float cosang = clampf( (uw.x*bh.x + uw.y*bh.y) / uwN, -1.f, 1.f );
				float angdeg = std::acos(cosang) * 57.29578f;
				if (angdeg > opt_.theta_max_deg) continue;

				// 六边形步数一致
				int k_uw = hexSteps(cu, cw);
				if (k_uw != k_delate) continue;

				obs.push_back( std::fabs(uw.x*bh.x + uw.y*bh.y) ); // 投影长度（像素）
			}
		}

		if (static_cast<int>(obs.size()) >= opt_.min_obs)
			return median(obs);

		// 兜底：k_Δ * Base（防样本不足；推荐尽量让 min_obs 满足）
		return std::max(1e-6f, k_delate * Base_);
	}

	float PTotEstimator::getPTotCached(int ref_lin, int nei_lin,
									   const std::vector<float2>& slice_centers,
									   const float2& c_ref, const float2& c_nei)
	{
		const auto key = key64(ref_lin, nei_lin);
		auto it = cache_.find(key);
		if (it != cache_.end()) return it->second;

		float pTot = computePTotDirectDelta(slice_centers, c_ref, c_nei);
		cache_.emplace(key, pTot);
		return pTot;
	}

	void PTotEstimator::buildPerCallTablesFromSlice(const int2& tilekey_ref,
													const std::vector<int2>& neighbors,
													const std::vector<float2>& slice_centers,
													std::vector<int>&   nei_lin,
													std::vector<int>&   kSteps,
													std::vector<float>& pTot) const
	{
		// 约定：slice_centers[0] = c_ref； slice_centers[i+1] 对应 neighbors[i]
		const float2 c_ref = slice_centers.at(0);

		const size_t M = neighbors.size();
		nei_lin.resize(M);
		kSteps.resize(M);
		pTot.resize(M);

		for (size_t i=0; i<M; ++i){
			const float2 c_nei = slice_centers.at(1 + i);
			nei_lin[i] = lin(neighbors[i]);
			kSteps[i]  = hexSteps(c_ref, c_nei);
			pTot[i]    = computePTotDirectDelta(slice_centers, c_ref, c_nei);
		}
	}
}