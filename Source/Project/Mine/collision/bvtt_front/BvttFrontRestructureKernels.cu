#include "BvttFrontLooseKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\auxiliary\FlOrderLog.h"

namespace mn {

	/// Restructure Intra Front
	__global__ void restructureIntLooseIntraFrontWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_intRestrFront, FlOrderCompletePort _log, 
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2	cp = _intRestrFront[idx];
		int		st = cp.y << 1;
		const BOX bv = _prims.getBV(cp.x);
#if MACRO_VERSION
		const int3 ids = _prims.getVids(cp.x);
#endif

		cp.y = _lvs.getlca(_tks.getrangey(cp.y) + 1);
		do {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st > t) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = make_int2(cp.x, idx);
				atomicAdd(&_log.extcnt(idx), 1);
				if (
#if MACRO_VERSION
					!covertex(ids, _prims.getVids(idx)) &&
#endif
					_lvs.overlaps(idx, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(idx));
				}
				st = _lvs.getlca(idx + 1);
			}
			else {
				_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st);
				atomicAdd(&_log.intcnt(st), 1);
				st = _lvs.getlca(_tks.getrangey(st) + 1);
			}
		} while (st != cp.y);
	}

	__global__ void restructureExtLooseIntraFrontWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_extRestrFront, FlOrderCompletePort _log, const int *_leafRestrRoots,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2	cp = _extRestrFront[idx];
		int		st = cp.y << 1 | 1;
#if MACRO_VERSION
		const int3 ids = _prims.getVids(cp.x);
#endif

		const BOX bv = _prims.getBV(cp.x);

		cp.y = _lvs.getlca(_tks.getrangey(_leafRestrRoots[cp.y]) + 1);

		do {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st > t) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = make_int2(cp.x, idx);
				atomicAdd(&_log.extcnt(idx), 1);
				if (
#if MACRO_VERSION
					!covertex(ids, _prims.getVids(idx)) && 
#endif
					_lvs.overlaps(idx, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(idx));
				}
				st = _lvs.getlca(idx + 1);
			}
			else {
				_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st);
				atomicAdd(&_log.intcnt(st), 1);
				st = _lvs.getlca(_tks.getrangey(st) + 1);
			}
		} while (st != cp.y);
	}

	/// Restructure Inter Front
	__global__ void restructureIntLooseInterFrontWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_intRestrFront, FlOrderCompletePort _log,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2	cp = _intRestrFront[idx];
		int		st = cp.y << 1;
		const BOX bv = _travPrims.getBV(cp.x);

		cp.y = _lvs.getlca(_tks.getrangey(cp.y) + 1);
		do {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st > t) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = make_int2(cp.x, idx);
				atomicAdd(&_log.extcnt(idx), 1);
				if (_lvs.overlaps(idx, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(idx), _travPrims.getidx(cp.x));
				}
				st = _lvs.getlca(idx + 1);
			}
			else {
				_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st);
				atomicAdd(&_log.intcnt(st), 1);
				st = _lvs.getlca(_tks.getrangey(st) + 1);
			}
		} while (st != cp.y);
	}

	__global__ void restructureExtLooseInterFrontWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_extRestrFront, FlOrderCompletePort _log, const int *_leafRestrRoots,
		uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2	cp = _extRestrFront[idx];
		int		st = cp.y << 1 | 1;

		const BOX bv = _travPrims.getBV(cp.x);

		cp.y = _lvs.getlca(_tks.getrangey(_leafRestrRoots[cp.y]) + 1);

		do {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st > t) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = make_int2(cp.x, idx);
				atomicAdd(&_log.extcnt(idx), 1);
				if (_lvs.overlaps(idx, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(idx), _travPrims.getidx(cp.x));
				}
				st = _lvs.getlca(idx + 1);
			}
			else {
				_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st);
				atomicAdd(&_log.intcnt(st), 1);
				st = _lvs.getlca(_tks.getrangey(st) + 1);
			}
		} while (st != cp.y);
	}
}