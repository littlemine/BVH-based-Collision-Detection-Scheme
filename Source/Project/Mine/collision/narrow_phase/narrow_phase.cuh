#ifndef __NARROW_PHASE_CUH_
#define __NARROW_PHASE_CUH_

#include "collision/lbvh/BvhPrimitiveNode.h"

namespace mn {

	using uint = unsigned int;

	__global__ void simpleNarrowPhase(uint numpair, int2* _cps, const int3* _indices, const PointType* _vertices, int* _actualCpNum);
}

#endif