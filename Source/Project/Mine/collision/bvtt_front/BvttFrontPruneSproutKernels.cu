#include "BvttFrontLooseKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\auxiliary\FlOrderLog.h"

namespace mn {

	/// maintain intra fronts
	__global__ void maintainIntLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv = _prims.getBV(cp.x);
		int		t;
#if MACRO_VERSION
		const int3 ids = _prims.getVids(cp.x);
#endif
		if (!_tks.overlaps(st, bv)) {	///< prune
			for (t = _lvs.getlca(_tks.getrangex(st)) >> 1, st--; st >= t && !_tks.overlaps(st, bv); st--);
			if (st < t && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
				return;
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
			atomicAdd(&_log.intcnt(st + 1), 1);
			return;
		}
		cp.y = _lvs.getlca(_tks.getrangey(st) + 1);
		st = (_tks.getlc(st) << 1) | (_tks.getmark(st) & 1);
		do {							///< sprout
			t = st & 1;
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

	/// ext fronts
	__global__ void maintainExtLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		const int2 cp = _extFront[idx];
		int st = cp.y, gfa;
		const BOX bv = _prims.getBV(cp.x);

		if (!_lvs.overlaps(st, bv)) {
			gfa = _lvs.getpar(st);
			if (_tks.overlaps(gfa, bv)) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
				atomicAdd(&_log.extcnt(st), 1);
				return;
			}
			if ((_lvs.getmark(idx = st) & 4) == 4) 	///< or _lca[st] & 1
				return;
			for (st = gfa - 1, gfa = _lvs.getlca(idx) >> 1; st >= gfa && !_tks.overlaps(st, bv); st--);
			if (st < gfa && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
				return;
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
			atomicAdd(&_log.intcnt(st + 1), 1);
		}
		else {
			_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
			atomicAdd(&_log.extcnt(st), 1);

#if MACRO_VERSION
			if (!covertex(_prims.getVids(cp.x), _prims.getVids(st))) 
#endif
				_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(st));
		}
	}

	__global__ void sproutIntLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv = _prims.getBV(cp.x);
		if (!_tks.overlaps(st, bv)) {
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st);
			atomicAdd(&_log.intcnt(st), 1);
			return;
		}
		cp.y = _lvs.getlca(_tks.getrangey(st) + 1);
		st = (_tks.getlc(st) << 1) | (_tks.getmark(st) & 1);
#if MACRO_VERSION
		const int3 ids = _prims.getVids(cp.x);
#endif
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

	__global__ void sproutExtLooseIntraFrontsWithLog(BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2 cp = _extFront[idx];
		atomicAdd(&_log.extcnt(cp.y), 1);
		if (
#if MACRO_VERSION
			!covertex(_prims.getVids(cp.x), _prims.getVids(cp.y)) && 
#endif
			_lvs.overlaps(cp.x, cp.y)) {
			_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(cp.y));
		}
	}

	__global__ void pruneIntLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv = _prims.getBV(cp.x);
		int		t;
		/// assume not colliding
		for (t = _lvs.getlca(_tks.getrangex(st)) >> 1, st--; st >= t && !_tks.overlaps(st, bv); st--);
		if (st < t && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
			return;
		_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
		atomicAdd(&_log.intcnt(st + 1), 1);
		return;
	}

	__global__ void pruneExtLooseIntraFrontsWithLog(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		const int2 cp = _extFront[idx];
		int st = cp.y, gfa;
		const BOX bv = _prims.getBV(cp.x);

		if (!_lvs.overlaps(st, bv)) {
			gfa = _lvs.getpar(st);
			if (_tks.overlaps(gfa, bv)) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
				atomicAdd(&_log.extcnt(st), 1);
				return;
			}
			if ((_lvs.getmark(idx = st) & 4) == 4) 	///< or _lca[st] & 1
				return;
			for (st = gfa - 1, gfa = _lvs.getlca(idx) >> 1; st >= gfa && !_tks.overlaps(st, bv); st--);
			if (st < gfa && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
				return;
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
			atomicAdd(&_log.intcnt(st + 1), 1);
		}
		else {
			_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
			atomicAdd(&_log.extcnt(st), 1);
		}
	}

	/// maintain inter fronts
	__global__ void maintainIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv = _travPrims.getBV(cp.x);
		int		t;
		if (!_tks.overlaps(st, bv)) {
			for (t = _lvs.getlca(_tks.getrangex(st)) >> 1, st--; st >= t && !_tks.overlaps(st, bv); st--);
			if (st < t && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
				return;
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
			atomicAdd(&_log.intcnt(st + 1), 1);
			return;
		}
		cp.y = _lvs.getlca(_tks.getrangey(st) + 1);
		st = (_tks.getlc(st) << 1) | (_tks.getmark(st) & 1);
		do {
			t = st & 1;
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

	/// ext fronts
	__global__ void maintainExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		const int2 cp = _extFront[idx];
		int st = cp.y, gfa;
		const BOX bv = _travPrims.getBV(cp.x);
		if (!_lvs.overlaps(st, bv)) {
			gfa = _lvs.getpar(st);
			if (_tks.overlaps(gfa, bv)) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
				atomicAdd(&_log.extcnt(st), 1);
				return;
			}
			if ((_lvs.getmark(idx = st) & 4) == 4) 	///< or _lca[idx = st] & 1
				return;
			for (st = gfa - 1, gfa = _lvs.getlca(idx) >> 1; st >= gfa && !_tks.overlaps(st, bv); st--);
			if (st < gfa && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
				return;
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
			atomicAdd(&_log.intcnt(st + 1), 1);
		}
		else {
			_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
			atomicAdd(&_log.extcnt(st), 1);

			_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(st), _travPrims.getidx(cp.x));
		}
	}

	__global__ void sproutIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv = _travPrims.getBV(cp.x);
		if (!_tks.overlaps(st, bv)) {
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st);
			atomicAdd(&_log.intcnt(st), 1);
			return;
		}
		cp.y = _lvs.getlca(_tks.getrangey(st) + 1);
		st = (_tks.getlc(st) << 1) | (_tks.getmark(st) & 1);
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

	__global__ void sproutExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto& _prims = _lvs.primPort();
		int2 cp = _extFront[idx];
		atomicAdd(&_log.extcnt(cp.y), 1);
		if (_lvs.overlaps(cp.x, cp.y)) {
			_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.y), _travPrims.getidx(cp.x));
		}
	}

	__global__ void pruneIntLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_intFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		int2	cp = _intFront[idx];
		int		st = cp.y;
		const BOX bv = _travPrims.getBV(cp.x);
		int		t;
		/// certainly not colliding
		for (t = _lvs.getlca(_tks.getrangex(st)) >> 1, st--; st >= t && !_tks.overlaps(st, bv); st--);
		if (st < t && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
			return;
		_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
		atomicAdd(&_log.intcnt(st + 1), 1);
		return;
	}

	/// ext fronts
	__global__ void pruneExtLooseInterFrontsWithLog(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint ftSize, const int2 *_extFront,
		FlOrderCompletePort _log, uint *_ftSlideSizes, int2 **_slideFtLists) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto _prims = _lvs.getPrimPort();
		const int2 cp = _extFront[idx];
		int st = cp.y, gfa;
		const BOX bv = _travPrims.getBV(cp.x);
		if (!_lvs.overlaps(st, bv)) {
			gfa = _lvs.getpar(st);
			if (_tks.overlaps(gfa, bv)) {
				_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
				atomicAdd(&_log.extcnt(st), 1);
				return;
			}
			if ((_lvs.getmark(idx = st) & 4) == 4) 	///< or _lca[idx = st] & 1
				return;
			for (st = gfa - 1, gfa = _lvs.getlca(idx) >> 1; st >= gfa && !_tks.overlaps(st, bv); st--);
			if (st < gfa && (st + 1 > 0 && !_tks.overlaps(_tks.getpar(st + 1), bv)))
				return;
			_slideFtLists[0][atomicAdd(_ftSlideSizes, 1)] = make_int2(cp.x, st + 1);
			atomicAdd(&_log.intcnt(st + 1), 1);
		}
		else {
			_slideFtLists[1][atomicAdd(_ftSlideSizes + 1, 1)] = cp;
			atomicAdd(&_log.extcnt(st), 1);
		}
	}
}
