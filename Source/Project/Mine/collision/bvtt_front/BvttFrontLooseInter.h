/**	\file	BvttFrontLooseInter.h
*	\brief	Bvtt-Inter-Front
*/

#ifndef __BVTT_FRONT_LOOSE_INTER_H_
#define __BVTT_FRONT_LOOSE_INTER_H_

#include <thrust/device_vector.h>

#include "setting\BvttFrontSettings.h"
#include "base\MultiArray.h"
#include "collision\auxiliary\FlOrderLog.h"
#include "collision\lbvh\LBvh.h"

namespace mn {

	class CudaDevice;

	/// for now, maintenance work is specified by this input alone
	enum class BvttFrontLooseInterMaintenance {
		PURE_BVH_CD, GENERATE, UPDATE, REORDER, KEEP, RESTRUCTURE
	};

	/*
	* \brief	BvttFrontLooseInter
	* \note		RAII
	*/
	class BvttFrontLooseInter {
	public:
		BvttFrontLooseInter() = delete;
		BvttFrontLooseInter(BvttFrontInterBuildConfig<LBvh<ModelType::FixedDeformableType>, LBvh<ModelType::RigidType>> config);
		~BvttFrontLooseInter();
		void	maintain(BvttFrontLooseInterMaintenance scheme);

		void	inspectResults();
		void	applyCpResults(uint* _idx, uint2* _front);
	private:
		void	reorderFronts();	///< ordering
		void	separateFronts();	///< prepare for restructure
		/// front-based CD pipeline
		void	generate();
		void	pruneSprout();
		void	balance();
		void	keep();
		void	restructure();
		/// BVH-based CD
		void	pureBvhCd();

		CudaDevice*					TheCudaDevice;
		LBvh<ModelType::FixedDeformableType>*	_pFixedDeformableBvh;
		LBvh<ModelType::RigidType>*	_pRigidBvh;

		MultiArray<int2, 4>			_fronts;
		FlOrderLog					_log;

		bool						_restructured{ false };

		int							_numValidFrontNodes[2];	// 0: intfront 1: extfront
		int							_numInvalidFrontNodes[2];	// 0: intrestrfront 1: extrestrfront

		/// coherent bvh-based cd
		thrust::device_vector<uint>	d_cpCntLog;
		thrust::device_vector<uint>	d_cpPosLog;
		thrust::device_vector<int>	d_cdpairOffsets;
		/// broad-phase results
		thrust::device_vector<int>	d_ExtFtIndices;
		thrust::device_vector<int>	d_IntFtIndices;

		int							*d_cpNum, _cpNum, *d_actualCpNum, _actualCpNum;
		thrust::device_vector<int2>	d_cpRes;
		thrust::device_vector<int2>	d_orderedCdpairs;
	};

}

#endif