#include "BvttFrontLooseKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\lbvh\BvhIntNode.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\auxiliary\FlOrderLog.h"

namespace mn {

	__global__ void keepIntLooseIntraFronts(const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, 
		uint ftSize, const int2 *_ftList, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		const auto &_prims = _lvs.primPort();
		if (idx >= ftSize) return;
		int2	cp = _ftList[idx];
		const BOX bv{ _prims.getBV(cp.x) };
		if (!_tks.overlaps(cp.y, bv)) 
			return;

		int st = (_tks.getlc(cp.y) << 1) | (_tks.getmark(cp.y) & 1);
		cp.y = _lvs.getlca(_tks.getrangey(cp.y) + 1);
#if MACRO_VERSION
		const int3 ids = _prims.getVids(cp.x);
#endif
		while (st != cp.y) {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st == t + 1) {
				if (
#if MACRO_VERSION
					!covertex(ids, _prims.getVids(idx)) &&
#endif
					_lvs.overlaps(idx, bv)) 
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(cp.x), _prims.getidx(idx));
				st = _lvs.getlca(idx + 1);
			}
			else
				st = _lvs.getlca(_tks.getrangey(st) + 1);
		}
	}

	__global__ void keepExtLooseIntraFronts(const BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_ftList, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto &_prims = _lvs.primPort();
		int2 cp = _ftList[idx];

		if (
#if MACRO_VERSION
			!covertex(_prims.getVids(cp.x), _prims.getVids(cp.y)) &&
#endif
			_lvs.overlaps(cp.x, cp.y)) {
			_cpRes[atomicAggInc(_cpNum)] = make_int2(_prims.getidx(cp.x), _prims.getidx(cp.y));
		}
	}

	__global__ void keepIntLooseInterFronts(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks,
		uint ftSize, const int2 *_ftList, int *_cpNum, int2 *_cpRes) {
		uint	idx = blockIdx.x * blockDim.x + threadIdx.x;
		const auto &_prims = _lvs.primPort();
		if (idx >= ftSize) return;
		int2	cp = _ftList[idx];
		const BOX bv{ _travPrims.getBV(cp.x) };
		if (!_tks.overlaps(cp.y, bv))
			return;

		int st = (_tks.getlc(cp.y) << 1) | (_tks.getmark(cp.y) & 1);
		cp.y = _lvs.getlca(_tks.getrangey(cp.y) + 1);
		while (st != cp.y) {
			int t = st & 1;
			st >>= 1;
			if (!t)	for (t = _lvs.getpar(idx = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, idx = st;
			if (st == t + 1) {
				if (_lvs.overlaps(idx, bv))
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(_prims.getidx(idx), _travPrims.getidx(cp.x));
				st = _lvs.getlca(idx + 1);
			}
			else
				st = _lvs.getlca(_tks.getrangey(st) + 1);
		}
	}

	__global__ void keepExtLooseInterFronts(const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, uint ftSize, const int2 *_ftList, int *_cpNum, int2 *_cpRes) {
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ftSize) return;
		const auto &_prims = _lvs.primPort();
		int2 cp = _ftList[idx];

		if (_lvs.overlaps(cp.y, _travPrims.getBV(cp.x))) {
			_cpRes[atomicAggInc(_cpNum)] = make_int2(_prims.getidx(cp.y), _travPrims.getidx(cp.x));
		}
	}

}