#ifndef __BVTT_FRONT_LOOSE_KERNELS_CUH_
#define __BVTT_FRONT_LOOSE_KERNELS_CUH_

#include <cuda_runtime.h>
#include "utility\Meta.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\auxiliary\FlOrderLog.h"
#include "collision\auxiliary\BvhRestrLog.h"

namespace mn {

	__global__ void reorderCdPairs(uint cnt, uint* _segpos, uint* _cnt, int2* _pairs, int2* _orderedPairs);
	__global__ void pureReorderLooseFrontsWithLog(uint2 ftSize, int2 **_ftLists, int2 **_slideFtLists, FlOrderCompletePort _log);	///< key(cp.y), used on compacted fronts

	/// intra front
	__global__ void separateIntLooseIntraFrontWithLog(uint ftSize, uint2 numValid, const int2 *_intFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _intMarks, const int* _leafRestrRoots,
		const int* _prevLbds, const int* _lcas, FlOrderCompletePort _ftLog);
	__global__ void separateExtLooseIntraFrontWithLog(uint ftSize, uint2 numValid, const int2 *_extFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _extMarks, const int* _leafRestrRoots,
		const int* _lcas, FlOrderCompletePort _ftLog);

	__global__ void pureBvhSelfCD(uint primsize, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, int* _cpNum, int2* _cpRes);

	__global__ void genLooseIntraFrontsWithLog(uint primsize, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint *_ftSizes, int2 **_ftLists,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes);

	__global__ void sproutIntLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);
	__global__ void pruneIntLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists);
	__global__ void maintainIntLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);
	__global__ void sproutExtLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes);
	__global__ void pruneExtLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists);
	__global__ void maintainExtLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);

	__global__ void reorderIntLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, int2 *_intFront,
		FlOrderCompletePort _log, int2 *_slideIntFront, int *_cpNum, int2 *_cpRes);
	__global__ void reorderExtLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, uint ftSize, int2 *_extFront,
		FlOrderCompletePort _log, int2 *_slideExtFront, int *_cpNum, int2 *_cpRes);

	__global__ void keepIntLooseIntraFronts(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_ftList/*, uint* _primCpCnts*/, int *_cpNum, int2 *_cpRes);
	__global__ void keepExtLooseIntraFronts(const BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_ftList/*, uint* _primCpCnts*/, int *_cpNum, int2 *_cpRes);

	__global__ void restructureIntLooseIntraFrontWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_intRestrFront, FlOrderCompletePort _log,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);
	__global__ void restructureExtLooseIntraFrontWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_extRestrFront, FlOrderCompletePort _log, const int *_leafRestrRoots,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);

	/// inter front
	__global__ void separateIntLooseInterFrontWithLog(uint ftSize, uint2 numValid, const int2 *_intFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _intMarks, const int* _leafRestrRoots,
		const int* _prevLbds, const int* _lcas, FlOrderCompletePort _ftLog);
	__global__ void separateExtLooseInterFrontWithLog(uint ftSize, uint2 numValid, const int2 *_extFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _extMarks, const int* _leafRestrRoots,
		const int* _lcas, FlOrderCompletePort _ftLog);

	// trav is the obstacle one
	__global__ void pureBvhInterCD(uint travPrimSize, const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, int* _cpNum, int2* _cpRes);

	__global__ void genLooseInterFrontsWithLog(uint travPrimSize, const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint *_ftSizes, int2 **_ftLists,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes);

	__global__ void sproutIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);
	__global__ void pruneIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists);
	__global__ void maintainIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);
	__global__ void sproutExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes);
	__global__ void pruneExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists);
	__global__ void maintainExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);

	__global__ void reorderIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, int2 *_intFront,
		FlOrderCompletePort _log, int2 *_slideIntFront, int *_cpNum, int2 *_cpRes);
	__global__ void reorderExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, uint ftSize, int2 *_extFront,
		FlOrderCompletePort _log, int2 *_slideExtFront, int *_cpNum, int2 *_cpRes);

	__global__ void keepIntLooseInterFronts(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_ftList, int *_cpNum, int2 *_cpRes);
	__global__ void keepExtLooseInterFronts(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_ftList/*, uint* _primCpCnts*/, int *_cpNum, int2 *_cpRes);

	__global__ void restructureIntLooseInterFrontWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_intRestrFront, FlOrderCompletePort _log,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);
	__global__ void restructureExtLooseInterFrontWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_extRestrFront, FlOrderCompletePort _log, const int *_leafRestrRoots,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes);

	/// quality inspection
	__global__ void frontSnapshot(uint intSize, BvhIntNodeCompletePort _tks, FlOrderCompletePort _frontLog, float* _snapshot);
	__global__ void checkFrontQuality(uint intSize, BvhIntNodeCompletePort _tks, FlOrderCompletePort _frontLog, float* _snapshot, BvhRestrCompletePort _restrLog);
	__global__ void countRestrFrontNodes(uint2 bvhSizes, BvhRestrCompletePort _restrLog, FlOrderCompletePort _frontLog, uint* _intCount, uint* _extCount);

}

#endif