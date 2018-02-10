#ifndef __LBVH_KERNELS_H_
#define __LBVH_KERNELS_H_

#include <cuda_runtime.h>
#include "utility\Meta.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\auxiliary\BvhRestrLog.h"

namespace mn {

	/// Building Kernels
#if MACRO_VERSION
	__global__ void calcMaxBVARCSim(int size, g_box *_bxs, BOX* _bv);
	__global__ void calcMCsARCSim(int size, g_box *_bxs, BOX scene, uint* codes);
	__global__ void buildPrimitivesARCSim(int size, BvhPrimitiveCompletePort _prims, int *_primMap, uint3 *_faces, g_box *_bxs);
	__global__ void refitExtNodeARCSim(int primsize, BvhExtNodeCompletePort _lvs, int* _primMap, uint3* _faces, g_box* _bxs);
#endif
	
	__global__ void calcMaxBV(int size, const int3 *_faces, const PointType *_vertices, BOX* _bv);
	__global__ void calcMCs(int size, int3 *_faces, PointType *_vertices, BOX scene, uint* codes);
	__global__ void calcMC64s(int size, int3* _faces, PointType* _vertices, BOX* scene, uint64* codes);
	__global__ void copyBackCodes(int size, uint64* _primcodes, uint* _codes);	///< deprecated
	__global__ void buildPrimitives(int size, BvhPrimitiveCompletePort _prims, int *_primMap, int3 *_faces, PointType *_vertices);
	__global__ void buildIntNodes(int size, uint *_depths, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks);
	__global__ void calcIntNodeOrders(int size, BvhIntNodeCompletePort _tks, int *_lcas, uint *_depths, uint *_offsets, int *_tkMap);
	__global__ void updateBvhExtNodeLinks(int size, const int *_mapTable,  int* _lcas, int* _pars);
	__global__ void reorderIntNode(int intSize, const int *_tkMap, BvhIntNodeCompletePort _unorderedTks, BvhIntNodeCompletePort _tks);

	/// Updating Kernels
	__global__ void refitExtNode(int primsize, BvhExtNodeCompletePort _lvs, int* _primMap, int3* _faces, PointType* _vertices);
	__global__ void refitIntNode(int size, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks);
	__global__ void updateIntNode(int size, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, //FlOrderCompletePort _log,
		int* _lvMarks, int* _tkMarks, int* _restrLcas);

	/// Restructure Kernels
	__global__ void calibrateLeafRangeMarks(int size, BvhIntNodeCompletePort _tks, const int* _leafRestrRoots, const int* _intRestrMarks, int* _leafRangeMarks);	// mark relative leaves according to restr roots
	__global__ void calibrateRestrRoots(int size, BvhIntNodeCompletePort _tks, const int* _leafRestrRoots, 
		const int* _intRestrMarks, int* _leafRestrRootMarks, int* _numSubtree, uint* _subtreeSizes, int* _subtrees, int* _numRtIntNode);	// uniform inclusive_scanned restr marks
	__global__ void calcRestrMCs(int size, const int3* _faces, const PointType* _vertices, BOX scene,
		const int* _primRestrMarks, const int* _primmap, uint* codes);
	__global__ void selectPrimitives(int primsize, const int* _leafRestrRoots, const int* _gatherMap, const MCSize* _mtcodes, 
		uint64* _finalKeys, int* _scatterMap);
	__global__ void updatePrimMap(int restrPrimNum, int* _indices, int* _values, int* _primToIdx, int* _primmap);
	__global__ void updatePrimAndExtNode(int primsize, const int *_primRestrMarks, const int* _primMap, const int3* _faces, const PointType * _vertices, const BOX* scene, BvhExtNodeCompletePort _lvs);
	__global__ void restrIntNodes(int extSize, int numRtExtNode, const int *_restrExtNodes, const uint *_prevTkMarks,
		const int *_leafRestrRoots, uint *_depths, int *_localLcas, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks);
	__global__ void calcRestrIntNodeOrders(int numRtExtNode, const int *_restrExtNodes, const uint *_depths, const uint *_offsets, const int *_subtreeRoots,
		const int* _prevLbds, const uint *_prevTkMarks, const int* _localLcas, int *_lcas, int *_pars, BvhIntNodeCompletePort _unorderedTks, int *_tkMap, int *_sequence);
	__global__ void reorderRestrIntNodes(int numRtIntNode, const int *_restrIntNodes, const int *_tkMap, 
		BvhIntNodeCompletePort _unorderedTks, BvhIntNodeCompletePort _tks);

	/// Debug
	__global__ void checkRestrTrunkMap(int numRtIntNode, const int* _restrQueue, const int* _tkMap, const int * _restrIntMark, int *_rtIntCount);
	__global__ void checkPrimmap(int size, int* _primmap, int* _cnt);
	__global__ void checkBVHIntegrity(int size, BvhExtNodeCompletePort _leaves, BvhIntNodeCompletePort _trunks, int *tag);

}

#endif