#include "BvttFrontLooseKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\auxiliary\FlOrderLog.h"

namespace mn {

	/// 
	__global__ void pureReorderLooseFrontsWithLog(uint2 ftSize, int2 **_ftLists, int2 **_slideFtLists, FlOrderCompletePort _log) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < ftSize.x) {	///< int front
			int2	cp = _ftLists[0][idx];
			_slideFtLists[0][_log.intbeg(cp.y) + atomicAdd(&_log.intcnt(cp.y), 1)] = cp;
			return;
		}
		if (idx < ftSize.x + ftSize.y) {	///< ext front
			idx -= ftSize.x;
			int2	cp = _ftLists[1][idx];
			_slideFtLists[1][_log.extbeg(cp.y) + atomicAdd(&_log.extcnt(cp.y), 1)] = cp;
		}
	}

	/// Balance Intra Front
	__global__ void reorderIntLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, int2 *_intFront,
		FlOrderCompletePort _log, int2 *_slideIntFront, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv{ _prims.getBV(cp.x) };
		if (!_tks.overlaps(st, bv)) {
			_slideIntFront[_log.intbeg(st) + atomicAdd(&_log.intcnt(st), 1)] = cp;
			return;
		}
#if MACRO_VERSION
		const int3 ids = _prims.getVids(cp.x);
#endif
		_slideIntFront[_log.intbeg(st + 1) - 1 - atomicAdd(&_log.intbackcnt(st), 1)] = cp;
		cp.y = _lvs.getlca(_tks.getrangey(st) + 1);
		st = (_tks.getlc(st) << 1) | (_tks.getmark(st) & 1);
		while (st != cp.y) {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st > t) {
				if (
#if MACRO_VERSION
					!covertex(ids, _prims.getVids(idx)) && 
#endif
					_lvs.overlaps(idx, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(idx));
				}
				st = _lvs.getlca(idx + 1);
			}
			else
				st = _lvs.getlca(_tks.getrangey(st) + 1);
		}
	}
	__global__ void reorderExtLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, uint ftSize, int2 *_extFront,
		FlOrderCompletePort _log, int2 *_slideExtFront, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		const int2 cp = _extFront[idx];
		idx = cp.y;
		if (
#if MACRO_VERSION
			!covertex(_prims.getVids(cp.x), _prims.getVids(idx)) && 
#endif
			_lvs.overlaps(idx, cp.x)) {
			_slideExtFront[_log.extbeg(idx + 1) - 1 - atomicAdd(&_log.extbackcnt(idx), 1)] = cp;
			_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(idx));
			return;
		}
		_slideExtFront[_log.extbeg(idx) + atomicAdd(&_log.extcnt(idx), 1)] = cp;
	}

	/// Balance Inter Front
	__global__ void reorderIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, int2 *_intFront,
		FlOrderCompletePort _log, int2 *_slideIntFront, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv{ _travPrims.getBV(cp.x) };
		if (!_tks.overlaps(st, bv)) {
			_slideIntFront[_log.intbeg(st) + atomicAdd(&_log.intcnt(st), 1)] = cp;
			return;
		}
		_slideIntFront[_log.intbeg(st + 1) - 1 - atomicAdd(&_log.intbackcnt(st), 1)] = cp;
		cp.y = _lvs.getlca(_tks.getrangey(st) + 1);
		st = (_tks.getlc(st) << 1) | (_tks.getmark(st) & 1);
		while (st != cp.y) {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st > t) {
				if (_lvs.overlaps(idx, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(idx), _travPrims.getidx(cp.x));
				}
				st = _lvs.getlca(idx + 1);
			}
			else
				st = _lvs.getlca(_tks.getrangey(st) + 1);
		}
	}

	__global__ void reorderExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, uint ftSize, int2 *_extFront,
		FlOrderCompletePort _log, int2 *_slideExtFront, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		const int2 cp = _extFront[idx];
		idx = cp.y;
		const BOX bv{ _travPrims.getBV(cp.x) };
		if (_lvs.overlaps(idx, bv)) {
			_slideExtFront[_log.extbeg(idx + 1) - 1 - atomicAdd(&_log.extbackcnt(idx), 1)] = cp;
			_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(idx), _travPrims.getidx(cp.x));
			return;
		}
		_slideExtFront[_log.extbeg(idx) + atomicAdd(&_log.extcnt(idx), 1)] = cp;
	}

	/// Restructure Intra Front
	__global__ void separateIntLooseIntraFrontWithLog(uint ftSize, uint2 numIntFrontNodes, const int2 *_intFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _intMarks, const int* _leafRestrRoots, 
		const int* _prevLbds, const int* _lcas, FlOrderCompletePort _ftLog) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const int2 cp = _intFront[idx];
		idx = _intMarks[cp.y];
		if (idx == 0)	///< valid front nodes
			_slideFronts[0][_ftLog.intbeg(cp.y) + atomicAdd(&_ftLog.intcnt(cp.y), 1)] = cp;
		else if (idx == -1) {	///< invalid front nodes that should be modified and preserved
			idx = _leafRestrRoots[_prevLbds[cp.y]];
			_slideFronts[0][numIntFrontNodes.x + _ftLog.intbegbak(idx) + atomicAdd(&_ftLog.intbackcnt(idx), 1)] = make_int2(cp.x, idx);
		}
		else if (cp.x + 1 == _prevLbds[cp.y]) {	///< also preserved particularly in intra-fronts
			_slideFronts[1][atomicAdd(_slideFrontSizes + 1, 1)] = make_int2(cp.x, cp.x + 1);
			//_slideFronts[idx & 1][atomicAdd(_slideFrontSizes + (idx & 1), 1)] = make_int2(cp.x, idx >> 1);
		}
	}

	__global__ void separateExtLooseIntraFrontWithLog(uint ftSize, uint2 numIntFrontNodes, const int2 *_extFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _extMarks, const int* _leafRestrRoots, 
		const int* _lcas, FlOrderCompletePort _ftLog) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const int2 cp = _extFront[idx];
		idx = _extMarks[cp.y];
		if (idx == 0)
			_slideFronts[1][_ftLog.extbeg(cp.y) + atomicAdd(&_ftLog.extcnt(cp.y), 1)] = cp;
		else if (idx == -1) {
			idx = _leafRestrRoots[cp.y];
			_slideFronts[0][numIntFrontNodes.x + _ftLog.intbegbak(idx) + atomicAdd(&_ftLog.intbackcnt(idx), 1)] = make_int2(cp.x, idx);
		}
		else if (cp.x + 1 == cp.y) {
			_slideFronts[1][atomicAdd(_slideFrontSizes + 1, 1)] = make_int2(cp.x, cp.y);
		}
	}

	/// Restructure Inter Front 
	__global__ void separateIntLooseInterFrontWithLog(uint ftSize, uint2 numIntFrontNodes, const int2 *_intFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _intMarks, const int* _leafRestrRoots,
		const int* _prevLbds, const int* _lcas, FlOrderCompletePort _ftLog) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const int2 cp = _intFront[idx];
		idx = _intMarks[cp.y];
		if (idx == 0)	///< valid front nodes
			_slideFronts[0][_ftLog.intbeg(cp.y) + atomicAdd(&_ftLog.intcnt(cp.y), 1)] = cp;
		else if (idx == -1) {	///< invalid front nodes that should be modified and preserved
			idx = _leafRestrRoots[_prevLbds[cp.y]];
			_slideFronts[0][numIntFrontNodes.x + _ftLog.intbegbak(idx) + atomicAdd(&_ftLog.intbackcnt(idx), 1)] = make_int2(cp.x, idx);
		}
	}

	__global__ void separateExtLooseInterFrontWithLog(uint ftSize, uint2 numIntFrontNodes, const int2 *_extFront,
		uint *_slideFrontSizes, int2 **_slideFronts, const int* _extMarks, const int* _leafRestrRoots,
		const int* _lcas, FlOrderCompletePort _ftLog) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const int2 cp = _extFront[idx];
		idx = _extMarks[cp.y];
		if (idx == 0)
			_slideFronts[1][_ftLog.extbeg(cp.y) + atomicAdd(&_ftLog.extcnt(cp.y), 1)] = cp;
		else if (idx == -1) {
			idx = _leafRestrRoots[cp.y];
			_slideFronts[0][numIntFrontNodes.x + _ftLog.intbegbak(idx) + atomicAdd(&_ftLog.intbackcnt(idx), 1)] = make_int2(cp.x, idx);
		}
	}
}