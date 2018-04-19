/**	\file	BVTTFront.h
*	\brief	Front type selector & front kernel register
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVTT_FRONT_H_
#define __BVTT_FRONT_H_

#include "setting\BvttFrontSettings.h"
#include "BvttFrontLooseInter.h"
#include "BvttFrontLooseIntra.h"
#include "BvttFrontLooseKernels.cuh"
#include "collision\narrow_phase\narrow_phase.cuh"
#include "system\CudaDevice\CudaDevice.h"

namespace mn {

	template<BvttFrontType bvttFrontType>
	struct BvttFrontSelector;

	template<>
	struct BvttFrontSelector<BvttFrontType::LooseInterType> {
		using type = BvttFrontLooseInter;
	};
	template<>
	struct BvttFrontSelector<BvttFrontType::LooseIntraType> {
		using type = BvttFrontLooseIntra;
	};

	template<BvttFrontType bvttFrontType>
	using BvttFront = typename BvttFrontSelector<bvttFrontType>::type;

	class BvttFrontKernelRegister {
	public:
		BvttFrontKernelRegister() {
			/// three factors considered: 1\blocksize 2\cachePreference 3\tailEffect
			CudaDevice* device = CudaDevice::getInstance();
			device->registerKernel("FilterCnts", filterCnts, cudaFuncCachePreferL1, false);	
			device->registerKernel("FilterIntFrontCnts", filterIntFrontCnts, cudaFuncCachePreferL1, false);
			device->registerKernel("FilterExtFrontCnts", filterExtFrontCnts, cudaFuncCachePreferL1, false);

			/// coherent BVH based CD
			device->registerKernel("SimpleNarrowPhase", simpleNarrowPhase, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderCdPairs", reorderCdPairs, cudaFuncCachePreferL1, false);

			/// pure BVH based CD
			device->registerKernel("PureBvhSelfCD", pureBvhSelfCD, cudaFuncCachePreferL1, false);
			device->registerKernel("PureBvhInterCD", pureBvhInterCD, cudaFuncCachePreferL1, false);

			/// front ordering
			device->registerKernel("PureReorderLooseFrontsWithLog", pureReorderLooseFrontsWithLog, cudaFuncCachePreferL1, false);

			device->registerKernel("SeparateIntLooseIntraFrontWithLog", separateIntLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SeparateIntLooseInterFrontWithLog", separateIntLooseInterFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SeparateExtLooseIntraFrontWithLog", separateExtLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SeparateExtLooseInterFrontWithLog", separateExtLooseInterFrontWithLog, cudaFuncCachePreferL1, false);

			/// BVH quality inspection
			device->registerKernel("FrontSnapshot", frontSnapshot, cudaFuncCachePreferL1, false);
			device->registerKernel("CheckFrontQuality", checkFrontQuality, cudaFuncCachePreferL1, false);
			device->registerKernel("CountRestrFrontNodes", countRestrFrontNodes, cudaFuncCachePreferL1, false);

			///  front based CD
			device->registerKernel("GenLooseIntraFrontsWithLog", genLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);	///< wave definitely better
			device->registerKernel("GenLooseInterFrontsWithLog", genLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);	///< wave definitely better
			device->registerKernel("SproutIntLooseIntraFrontsWithLog", sproutIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SproutIntLooseInterFrontsWithLog", sproutIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneIntLooseIntraFrontsWithLog", pruneIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneIntLooseInterFrontsWithLog", pruneIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainIntLooseIntraFrontsWithLog", maintainIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainIntLooseInterFrontsWithLog", maintainIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SproutExtLooseIntraFrontsWithLog", sproutExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SproutExtLooseInterFrontsWithLog", sproutExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneExtLooseIntraFrontsWithLog", pruneExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneExtLooseInterFrontsWithLog", pruneExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainExtLooseIntraFrontsWithLog", maintainExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainExtLooseInterFrontsWithLog", maintainExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderIntLooseIntraFrontsWithLog", reorderIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderIntLooseInterFrontsWithLog", reorderIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderExtLooseIntraFrontsWithLog", reorderExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderExtLooseInterFrontsWithLog", reorderExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepIntLooseIntraFronts", keepIntLooseIntraFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepIntLooseInterFronts", keepIntLooseInterFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepExtLooseIntraFronts", keepExtLooseIntraFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepExtLooseInterFronts", keepExtLooseInterFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureIntLooseIntraFrontWithLog", restructureIntLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureIntLooseInterFrontWithLog", restructureIntLooseInterFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureExtLooseIntraFrontWithLog", restructureExtLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureExtLooseInterFrontWithLog", restructureExtLooseInterFrontWithLog, cudaFuncCachePreferL1, false);

		}
		~BvttFrontKernelRegister() {}
	};
}

#endif