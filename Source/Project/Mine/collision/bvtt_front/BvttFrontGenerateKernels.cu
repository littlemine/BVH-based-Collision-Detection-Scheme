#include "BvttFrontLooseKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\auxiliary\FlOrderLog.h"
#include "collision\auxiliary\BvhRestrLog.h"

namespace mn {

	__global__ void countRestrFrontNodes(uint2 bvhSizes, BvhRestrCompletePort _restrLog, FlOrderCompletePort _frontLog, uint* _intCount, uint* _extCount) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= bvhSizes.x + bvhSizes.y) return;
		if (idx >= bvhSizes.x) {	///< ext branch
			idx -= bvhSizes.x;
			if (_restrLog.extrt(idx))
				atomicAdd(_extCount, _frontLog.extcnt(idx));
		}
		else {		///< int branch
			if (_restrLog.intrt(idx))
				atomicAdd(_intCount, _frontLog.intcnt(idx));
		}
	}

	__global__ void frontSnapshot(uint intSize, BvhIntNodeCompletePort _tks, FlOrderCompletePort _frontLog, float* _snapshot) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= intSize) return;
		int2 range{ _tks.rangex(idx), _tks.rangey(idx) };
		/// more precise evaluation should count the actual collision pairs
		_snapshot[idx] = 1.0 * (_frontLog.intbeg(idx + range.y - range.x) - _frontLog.intbeg(idx)) / (_frontLog.extbeg(range.y + 1) - _frontLog.extbeg(range.x) + 1);
	}

	__global__ void checkFrontQuality(uint intSize, BvhIntNodeCompletePort _tks, FlOrderCompletePort _frontLog, float* _snapshot, BvhRestrCompletePort _restrLog) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= intSize) return;
		int2 range{ _tks.rangex(idx), _tks.rangey(idx) };
		int	intCnt, extCnt;

		/// various quality threshold can be test here!
		if (1.0 * (intCnt = _frontLog.intbeg(idx + range.y - range.x) - _frontLog.intbeg(idx)) / (extCnt = _frontLog.extbeg(range.y + 1) - _frontLog.extbeg(range.x))
			//> _snapshot[idx] + 1 - 0.7 * (range.y - range.x) / intSize) {
			> _snapshot[idx] + 1 - 0.7 * intCnt / _frontLog.intbeg(intSize)) {
		
			atomicAdd(&_restrLog.intrange(idx), 1);
			atomicAdd(&_restrLog.intrange(idx + range.y - range.x), -1);
			atomicMin(&_restrLog.rtroot(range.x), idx);
		}
	}

	__global__ void genLooseIntraFrontsWithLog(uint primsize, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint *_ftSizes, int2 **_ftLists,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= primsize - 1) return;
		//int idx = primsize - 1 - (blockIdx.x * blockDim.x + threadIdx.x);
		//for (; idx >= 0; idx -= gridDim.x * blockDim.x) {
			int	t, lbd;
			const auto& _prims = _lvs.primPort();
			int st = _lvs.getlca(idx + 1);		///< _prims.getextno(idx)

			const BOX bv{ _prims.getBV(idx) };
#if MACRO_VERSION
			const int3 ids = _prims.getVids(idx);
#endif
			do {
				t = st & 1;
				st >>= 1;

				if (!t)	for (t = _lvs.getpar(lbd = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
				else	t = st - 1, lbd = st;
				if (st == t + 1) {
					if (
#if MACRO_VERSION
						!covertex(ids, _prims.getVids(lbd)) &&
#endif
						_lvs.overlaps(lbd, bv)) {
						_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(idx), _prims.getidx(lbd));		///< prim no, ext node no
					}
					_ftLists[1][atomicAdd(_ftSizes + 1, 1)] = make_int2(idx, lbd);
					atomicAdd(&_log.extcnt(lbd), 1);
					st = _lvs.getlca(lbd + 1);
				}
				else {
					_ftLists[0][atomicAdd(_ftSizes, 1)] = make_int2(idx, st);
					atomicAdd(&_log.intcnt(st), 1);
					st = _lvs.getlca(_tks.getrangey(st) + 1);
				}
			} while (st != -1);
		//}
	}

	__global__ void genLooseInterFrontsWithLog(uint travPrimSize, const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, uint *_ftSizes, int2 **_ftLists,
		FlOrderCompletePort _log, int *_cpNum, int2 *_cpRes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= travPrimSize - 1) return;
		//int idx = primsize - 1 - (blockIdx.x * blockDim.x + threadIdx.x);
		//for (; idx >= 0; idx -= gridDim.x * blockDim.x) {	///< size of primitives
		int	t, lbd;
		const auto& _prims = _lvs.primPort();
		int st = 0;

		const BOX bv{ _travPrims.getBV(idx) };
		do {
			t = st & 1;
			st >>= 1;

			if (!t)	for (t = _lvs.getpar(lbd = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, lbd = st;
			if (st == t + 1) {
				if (_lvs.overlaps(lbd, bv)) {
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(lbd), _travPrims.getidx(idx));
				}
				_ftLists[1][atomicAdd(_ftSizes + 1, 1)] = make_int2(idx, lbd);
				atomicAdd(&_log.extcnt(lbd), 1);
				st = _lvs.getlca(lbd + 1);
			}
			else {
				_ftLists[0][atomicAdd(_ftSizes, 1)] = make_int2(idx, st);
				atomicAdd(&_log.intcnt(st), 1);
				st = _lvs.getlca(_tks.getrangey(st) + 1);
			}
		} while (st != -1);
		//}
	}
}
