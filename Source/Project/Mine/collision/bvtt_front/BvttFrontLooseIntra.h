/**	\file	BvttFrontLooseIntra.h
*	\brief	Bvtt-Intra-Front
*/

#ifndef __BVTT_FRONT_LOOSE_INTRA_H_
#define __BVTT_FRONT_LOOSE_INTRA_H_

#include <thrust/device_vector.h>

#include "setting\BvttFrontSettings.h"
#include "base\MultiArray.h"
#include "collision\auxiliary\FlOrderLog.h"
#include "collision\lbvh\LBvh.h"

namespace mn {

	class CudaDevice;

	/// for now, maintenance work is specified by this input alone
	enum class BvttFrontLooseIntraMaintenance {
		PURE_BVH_CD, GENERATE, UPDATE, REORDER, KEEP, RESTRUCTURE
	};

	/*
	* \brief	BvttFrontLooseIntra
	* \note		RAII
	*/
	class BvttFrontLooseIntra {
	public:
		BvttFrontLooseIntra() = delete;
		BvttFrontLooseIntra(BvttFrontIntraBuildConfig<LBvh<ModelType::FixedDeformableType>> config);
		~BvttFrontLooseIntra();
		void	maintain(BvttFrontLooseIntraMaintenance scheme);

		void	inspectResults();
		void	applyCpResults(uint* _idx, uint2* _front);
	private:
		void	proximityQuery();	///< narrow phase CD
		void	reorderFronts();	///< ordering
		void	separateFronts();	///< prepare for restructure
		/// quality inspection
		void	calcSnapshot();			// used for BVH quality evaluation
		void	checkQuality();			// used for BVH quality evaluation
		/// front-based CD pipeline
		void	generate();
		void	pruneSprout();
		void	balance();
		void	keep();
		void	restructure();
		/// BVH-based CD
		void	pureBvhCd();

		CudaDevice*					TheCudaDevice;
		LBvh<ModelType::FixedDeformableType>*	_pBvh;

		MultiArray<int2, 4>			_fronts;
		FlOrderLog					_log;

		bool						_restructured{ false };

		int							_numValidFrontNodes[2];	// 0: intfront 1: extfront
		int							_numInvalidFrontNodes[2];	// 0: intrestrfront 1: extrestrfront

		thrust::device_vector<float>	d_snapshot;	/// quality inspection

		/// coherent bvh-based cd
		thrust::device_vector<uint>	d_cpCntLog;
		thrust::device_vector<uint>	d_cpPosLog;
		thrust::device_vector<int>	d_cdpairOffsets;
		/// broad-phase results
		thrust::device_vector<int>	d_ExtFtIndices;
		thrust::device_vector<int>	d_IntFtIndices;

		int							*d_cpNum, _cpNum, *d_actualCpNum, _actualCpNum;
		uint                         *d_extFtNodeCnt, _extFtNodeCnt, *d_intFtNodeCnt, _intFtNodeCnt;
		thrust::device_vector<int2>	d_cpRes;
		thrust::device_vector<int2>	d_orderedCdpairs;
	};
	
}

#endif