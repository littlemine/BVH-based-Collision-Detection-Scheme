#include "narrow_phase.cuh"
#include "collision\LBVH\BvhBV.h"
#include "utility\CudaDeviceUtils.h"
#include <gProximity\cuda_intersect_tritri.h>

namespace mn {

	__global__ void simpleNarrowPhase(uint numpair, int2* _cps, const int3* _indices, const PointType* _vertices, int* _actualCpNum) {
		int	idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numpair) return;
		const int2 cp = _cps[idx];
		const PointType P0 = _vertices[_indices[cp.x].x];
		const PointType P1 = _vertices[_indices[cp.x].y];
		const PointType P2 = _vertices[_indices[cp.x].z];
		const PointType Q0 = _vertices[_indices[cp.y].x];
		const PointType Q1 = _vertices[_indices[cp.y].y];
		const PointType Q2 = _vertices[_indices[cp.y].z];
		if (!triangleIntersection2(make_float3(P0.x, P0.y, P0.z), make_float3(P1.x, P1.y, P1.z), make_float3(P2.x, P2.y, P2.z),
			make_float3(Q0.x, Q0.y, Q0.z), make_float3(Q1.x, Q1.y, Q1.z), make_float3(Q2.x, Q2.y, Q2.z))) {
			_cps[idx] = { -1, -1 };
		}
		else
			atomicAggInc(_actualCpNum);
	}
}
