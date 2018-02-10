/**	\file	LBvh.h
*	\brief	LBvh selector & bvh kernel register
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __LBVH_H_
#define __LBVH_H_

#include "setting\BvhSettings.h"
#include "LBvhRigid.h"
#include "LBvhFixedDeformable.h"
#include "system\CudaDevice\CudaDevice.h"
#include "utility\CudaKernelUtils.cuh"

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
			device->registerKernel("GatherBVs", gatherBVs, cudaFuncCachePreferL1, false);
			device->registerKernel("ScatterBVs", scatterBVs, cudaFuncCachePreferL1, false);
			device->registerKernel("GatherPrims", gatherPrims, cudaFuncCachePreferL1, false);
			device->registerKernel("ScatterPrims", scatterPrims, cudaFuncCachePreferL1, false);
			device->registerKernel("MarkPrimSplitPos", markPrimSplitPos, cudaFuncCachePreferL1, false);
			device->registerKernel("CollapsePrimitives", collapsePrimitives, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcExtNodeSplitMetrics", calcExtNodeSplitMetrics, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcExtNodeRestrSplitMetrics", calcExtNodeRestrSplitMetrics, cudaFuncCachePreferL1, false);
			device->registerKernel("ScatterIntNodes", scatterIntNodes, cudaFuncCachePreferL1, false);

#if MACRO_VERSION
			device->registerKernel("CalcBVARCSim", calcMaxBVARCSim, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcMCsARCSim", calcMCsARCSim, cudaFuncCachePreferL1, false);
			device->registerKernel("BuildPrimsARCSim", buildPrimitivesARCSim, cudaFuncCachePreferL1, false);
			device->registerKernel("RefitExtNodeARCSim", refitExtNodeARCSim, cudaFuncCachePreferL1, false);
#endif
			/// morton codes
			device->registerKernel("CalcBV", calcMaxBV, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcMCs", calcMCs, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcMC64s", calcMC64s, cudaFuncCachePreferL1, false);
			/// build
			device->registerKernel("CalcPrimMap", calcInverseMapping, cudaFuncCachePreferL1, false);
			device->registerKernel("BuildPrims", buildPrimitives, cudaFuncCachePreferL1, false);
			device->registerKernel("BuildIntNodes", buildIntNodes, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcIntNodeOrders", calcIntNodeOrders, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdateBvhExtNodeLinks", updateBvhExtNodeLinks, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderIntNode", reorderIntNode, cudaFuncCachePreferL1, false);
			/// refit
			device->registerKernel("RefitExtNode", refitExtNode, cudaFuncCachePreferL1, false);
			device->registerKernel("RefitIntNode", refitIntNode, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdateIntNode", updateIntNode, cudaFuncCachePreferL1, true);
			/// restructure
			device->registerKernel("CalibrateLeafRangeMarks", calibrateLeafRangeMarks, cudaFuncCachePreferL1, false);
			device->registerKernel("CalibrateRestrRoots", calibrateRestrRoots, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcRestrMCs", calcRestrMCs, cudaFuncCachePreferL1, false);
			device->registerKernel("SelectPrimitives", selectPrimitives, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdatePrimMap", updatePrimMap, cudaFuncCachePreferL1, false);
			device->registerKernel("UpdatePrimAndExtNode", updatePrimAndExtNode, cudaFuncCachePreferL1, false);
			device->registerKernel("RestrIntNodes", restrIntNodes, cudaFuncCachePreferL1, false);
			device->registerKernel("CalcRestrIntNodeOrders", calcRestrIntNodeOrders, cudaFuncCachePreferL1, false);
			device->registerKernel("ReorderRestrIntNodes", reorderRestrIntNodes, cudaFuncCachePreferL1, false);
		}
		~LBvhKernelRegister() {}
	};
}

#endif