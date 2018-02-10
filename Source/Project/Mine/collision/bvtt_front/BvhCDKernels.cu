#include "BvttFrontLooseKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\lbvh\BvhExtNode.h"
#include "collision\lbvh\BvhIntNode.h"

namespace mn {

	__global__ void reorderCdPairs(uint cnt, uint* _segpos, uint* _cnt, int2* _pairs, int2* _orderedPairs) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= cnt) return;
		int2 cp = _pairs[idx];
		_orderedPairs[_segpos[cp.y] + atomicAdd(_cnt + cp.y, 1)] = cp;
	}

	__global__ void pureBvhSelfCD(uint primsize, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, int* _cpNum, int2* _cpRes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= primsize - 1) return;
		const auto& prims = _lvs.primPort();
		//int idx = primsize - 1 - (blockIdx.x * blockDim.x + threadIdx.x);
		//for (; idx >= 0; idx -= gridDim.x * blockDim.x) {	///< size of primitives
			int	lbd;
			int st = _lvs.getlca(idx + 1);		///< prims.getextno(idx)
			const BOX bv{ prims.getBV(idx) };
#if MACRO_VERSION
			const int3 ids = prims.getVids(idx);
#endif
			do {
				int t = st & 1;
				st >>= 1;

				if (!t)	for (t = _lvs.getpar(lbd = _tks.getrangex(st));st <= t && _tks.overlaps(st, bv); st++);
				else	t = st - 1, lbd = st;
				if (st > t) {
					if (
#if MACRO_VERSION
						!covertex(ids, prims.getVids(lbd)) && 
#endif
						_lvs.overlaps(lbd, bv)) {
						_cpRes[atomicAdd(_cpNum, 1)] = make_int2(prims.getidx(idx), prims.getidx(lbd));
					}
					st = _lvs.getlca(lbd + 1);
				}
				else 
					st = _lvs.getlca(_tks.getrangey(st) + 1);
			} while (st != -1);
		//}
	}

	__global__ void pureBvhInterCD(uint travPrimSize, const BvhPrimitiveCompletePort _travPrims, const BvhExtNodeCompletePort _lvs, const BvhIntNodeCompletePort _tks, int* _cpNum, int2* _cpRes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= travPrimSize - 1) return;
		const auto& prims = _lvs.primPort();
		//int idx = primsize - 1 - (blockIdx.x * blockDim.x + threadIdx.x);
		//for (; idx >= 0; idx -= gridDim.x * blockDim.x) {	///< size of primitives
		int	lbd;
		int st = 0;
		const BOX bv{ _travPrims.getBV(idx) };
		do {
			int t = st & 1;
			st >>= 1;

			if (!t)	for (t = _lvs.getpar(lbd = _tks.getrangex(st)); st <= t && _tks.overlaps(st, bv); st++);
			else	t = st - 1, lbd = st;
			if (st > t) {
				if (_lvs.overlaps(lbd, bv))
					_cpRes[atomicAdd(_cpNum, 1)] = make_int2(prims.getidx(lbd), _travPrims.getidx(idx));
				st = _lvs.getlca(lbd + 1);
			}
			else {
				st = _lvs.getlca(_tks.getrangey(st) + 1);
			}
		} while (st != -1);
		//}
	}

}
