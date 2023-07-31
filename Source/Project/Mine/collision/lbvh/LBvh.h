/**	\file	LBvh.h
*	\brief	LBvh selector & bvh kernel register
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __LBVH_H_
#define __LBVH_H_

#include "setting/BvhSettings.h"
#include "LBvhRigid.h"
#include "LBvhFixedDeformable.h"
#include "system/CudaDevice/CudaDevice.h"
#include "utility/CudaKernelUtils.cuh"

namespace mn {
	
	template<ModelType modelType>
	struct LBvhSelector;

	template<>
	struct LBvhSelector<ModelType::FixedDeformableType> {
		using type = LBvhFixedDeformable;
	};
	template<>
	struct LBvhSelector<ModelType::RigidType> {
		using type = LBvhRigid;
	};

	template<ModelType modelType>
	using LBvh = typename LBvhSelector<modelType>::type;

	class LBvhKernelRegister {
	public:
		LBvhKernelRegister() {
			CudaDevice* device = CudaDevice::getInstance();
			/// components
			device->registerKernel("GatherBVs", (void *)gatherBVs, cudaFuncCachePreferL1, false);
			device->registerKernel("ScatterBVs", (void *)scatterBVs, cudaFuncCachePreferL1, false);
			device->registerKernel("GatherPrims", (void *)gatherPrims, cudaFuncCachePreferL1, false);
			device->registerKernel("ScatterPrims", (void *)scatterPrims, cudaFuncCachePreferL1, false);
			device->registerKernel("MarkPrimSplitPos", (void *)markPrimSplitPos, cudaFuncCachePreferL1, false);
			device->registerKernel("CollapsePrimitives", (void *)collapsePrimitives, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcExtNodeSplitMetrics", (void *)calcExtNodeSplitMetrics, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcExtNodeRestrSplitMetrics", (void *)calcExtNodeRestrSplitMetrics, cudaFuncCachePreferL1, false);
			device->registerKernel("ScatterIntNodes", (void *)scatterIntNodes, cudaFuncCachePreferL1, false);

#if MACRO_VERSION
			device->registerKernel("CalcBVARCSim", (void *)calcMaxBVARCSim, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcMCsARCSim", (void *)calcMCsARCSim, cudaFuncCachePreferL1, false);
			device->registerKernel("BuildPrimsARCSim", (void *)buildPrimitivesARCSim, cudaFuncCachePreferL1, false);
			device->registerKernel("RefitExtNodeARCSim", (void *)refitExtNodeARCSim, cudaFuncCachePreferL1, false);
#endif
			/// morton codes
			device->registerKernel("CalcBV", (void *)calcMaxBV, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcMCs", (void *)calcMCs, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcMC64s", (void *)calcMC64s, cudaFuncCachePreferL1, false);
			/// build
			device->registerKernel("CalcPrimMap", (void *)calcInverseMapping, cudaFuncCachePreferL1, false);
			device->registerKernel("BuildPrims", (void *)buildPrimitives, cudaFuncCachePreferL1, false);
			device->registerKernel("BuildIntNodes", (void *)buildIntNodes, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcIntNodeOrders", (void *)calcIntNodeOrders, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdateBvhExtNodeLinks", (void *)updateBvhExtNodeLinks, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderIntNode", (void *)reorderIntNode, cudaFuncCachePreferL1, false);
			/// refit
			device->registerKernel("RefitExtNode", (void *)refitExtNode, cudaFuncCachePreferL1, false);
			device->registerKernel("RefitIntNode", (void *)refitIntNode, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdateIntNode", (void *)updateIntNode, cudaFuncCachePreferL1, true);
			/// restructure
			device->registerKernel("CalibrateLeafRangeMarks", (void *)calibrateLeafRangeMarks, cudaFuncCachePreferL1, false);
			device->registerKernel("CalibrateRestrRoots", (void *)calibrateRestrRoots, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcRestrMCs", (void *)calcRestrMCs, cudaFuncCachePreferL1, false);
			device->registerKernel("SelectPrimitives", (void *)selectPrimitives, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdatePrimMap", (void *)updatePrimMap, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdatePrimAndExtNode", (void *)updatePrimAndExtNode, cudaFuncCachePreferL1, false);
			device->registerKernel("RestrIntNodes", (void *)restrIntNodes, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcRestrIntNodeOrders", (void *)calcRestrIntNodeOrders, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderRestrIntNodes", (void *)reorderRestrIntNodes, cudaFuncCachePreferL1, false);
		}
		~LBvhKernelRegister() {}
	};
}

#endif