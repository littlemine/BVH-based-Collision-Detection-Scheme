/**	\file	BVTTFront.h
*	\brief	Front type selector & front kernel register
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVTT_FRONT_H_
#define __BVTT_FRONT_H_

#include "setting/BvttFrontSettings.h"
#include "BvttFrontLooseInter.h"
#include "BvttFrontLooseIntra.h"
#include "BvttFrontLooseKernels.cuh"
#include "collision/narrow_phase/narrow_phase.cuh"
#include "system/CudaDevice/CudaDevice.h"

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
			device->registerKernel("FilterCnts", (void *)filterCnts, cudaFuncCachePreferL1, false);	
			device->registerKernel("FilterIntFrontCnts", (void *)filterIntFrontCnts, cudaFuncCachePreferL1, false);
			device->registerKernel("FilterExtFrontCnts", (void *)filterExtFrontCnts, cudaFuncCachePreferL1, false);

			/// coherent BVH based CD
			device->registerKernel("SimpleNarrowPhase", (void *)simpleNarrowPhase, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderCdPairs", (void *)reorderCdPairs, cudaFuncCachePreferL1, false);

			/// pure BVH based CD
			device->registerKernel("PureBvhSelfCD", (void *)pureBvhSelfCD, cudaFuncCachePreferL1, false);
			device->registerKernel("PureBvhInterCD", (void *)pureBvhInterCD, cudaFuncCachePreferL1, false);

			/// front ordering
			device->registerKernel("PureReorderLooseFrontsWithLog", (void *)pureReorderLooseFrontsWithLog, cudaFuncCachePreferL1, false);

			device->registerKernel("SeparateIntLooseIntraFrontWithLog", (void *)separateIntLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SeparateIntLooseInterFrontWithLog", (void *)separateIntLooseInterFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SeparateExtLooseIntraFrontWithLog", (void *)separateExtLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SeparateExtLooseInterFrontWithLog", (void *)separateExtLooseInterFrontWithLog, cudaFuncCachePreferL1, false);

			/// BVH quality inspection
			device->registerKernel("FrontSnapshot", (void *)frontSnapshot, cudaFuncCachePreferL1, false);
			device->registerKernel("CheckFrontQuality", (void *)checkFrontQuality, cudaFuncCachePreferL1, false);
			device->registerKernel("CountRestrFrontNodes", (void *)countRestrFrontNodes, cudaFuncCachePreferL1, false);

			///  front based CD
			device->registerKernel("GenLooseIntraFrontsWithLog", (void *)genLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);	///< wave definitely better
			device->registerKernel("GenLooseInterFrontsWithLog", (void *)genLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);	///< wave definitely better
			device->registerKernel("SproutIntLooseIntraFrontsWithLog", (void *)sproutIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SproutIntLooseInterFrontsWithLog", (void *)sproutIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneIntLooseIntraFrontsWithLog", (void *)pruneIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneIntLooseInterFrontsWithLog", (void *)pruneIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainIntLooseIntraFrontsWithLog", (void *)maintainIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainIntLooseInterFrontsWithLog", (void *)maintainIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SproutExtLooseIntraFrontsWithLog", (void *)sproutExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("SproutExtLooseInterFrontsWithLog", (void *)sproutExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneExtLooseIntraFrontsWithLog", (void *)pruneExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("PruneExtLooseInterFrontsWithLog", (void *)pruneExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainExtLooseIntraFrontsWithLog", (void *)maintainExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("MaintainExtLooseInterFrontsWithLog", (void *)maintainExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderIntLooseIntraFrontsWithLog", (void *)reorderIntLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderIntLooseInterFrontsWithLog", (void *)reorderIntLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderExtLooseIntraFrontsWithLog", (void *)reorderExtLooseIntraFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderExtLooseInterFrontsWithLog", (void *)reorderExtLooseInterFrontsWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepIntLooseIntraFronts", (void *)keepIntLooseIntraFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepIntLooseInterFronts", (void *)keepIntLooseInterFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepExtLooseIntraFronts", (void *)keepExtLooseIntraFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("KeepExtLooseInterFronts", (void *)keepExtLooseInterFronts, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureIntLooseIntraFrontWithLog", (void *)restructureIntLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureIntLooseInterFrontWithLog", (void *)restructureIntLooseInterFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureExtLooseIntraFrontWithLog", (void *)restructureExtLooseIntraFrontWithLog, cudaFuncCachePreferL1, false);
			device->registerKernel("RestructureExtLooseInterFrontWithLog", (void *)restructureExtLooseInterFrontWithLog, cudaFuncCachePreferL1, false);

		}
		~BvttFrontKernelRegister() {}
	};
}

#endif