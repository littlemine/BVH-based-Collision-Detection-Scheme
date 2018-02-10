#ifndef __CUDA_DEVICE_UTILS_H_
#define __CUDA_DEVICE_UTILS_H_

#include "Meta.h"
#include <device_launch_parameters.h>
#include "base\Bv.h"
#include "CudaHostUtils.h"

namespace mn {

	using uint = unsigned int;
	using uint64 = unsigned long long int;

#define PRINT_BOX_INFO(idx, box) printf("%d: %.3f, %.3f, %.3f ~ %.3f, %.3f, %.3f\n", idx, (box)._min.x, (box)._min.y, (box)._min.z, (box)._max.x, (box)._max.y, (box)._max.z); 
#define WARP_SIZE 32

	inline __device__ double dabs(double a) { return a < 0 ? -a : a; }
	inline __device__ double dmin(double a, double b) { return a < b ? a : b; }
	inline __device__ double dmax(double a, double b) { return a > b ? a : b; }
	/// CUDA atomic operations customized 
	__device__ bool atomicMinf(float* address, float val);
	__device__ bool atomicMaxf(float* address, float val);
	__device__ bool atomicMinD(double* address, double val);
	__device__ bool atomicMaxD(double* address, double val);

	template<typename T>
	inline __device__ bool atomicMinCustom(T* addr, T val);
	template<>
	inline __device__ bool atomicMinCustom<float>(float* addr, float val) { return atomicMinf(addr, val); }
	template<>
	inline __device__ bool atomicMinCustom<double>(double* addr, double val) { return atomicMinD(addr, val); }
	template<typename T>
	inline __device__ bool atomicMaxCustom(T* addr, T val);
	template<>
	inline __device__ bool atomicMaxCustom<float>(float* addr, float val) { return atomicMaxf(addr, val); }
	template<>
	inline __device__ bool atomicMaxCustom<double>(double* addr, double val) { return atomicMaxD(addr, val); }

	__device__ uint atomicAggInc(uint *ctr);
	__device__ int atomicAggInc(int *ctr);

	__device__ uint expandBits(uint v);
	__device__ uint64 expandBits64(uint v);
	__device__ uint compactBits(uint v);
	__device__ uint compactBits64(uint64 v);

	__device__ uint morton3D(float x, float y, float z);
	__device__ uint morton3Dbounds(float x, float y, float z,
		int dx, int dy, int dz);
	__device__ uint64 morton3D64(double x, double y, double z);
	__device__ uint3 morton3D64_d(uint64 c);

	__device__ bool covertex(int3 a, int3 b);

	// compute geometry
	inline __device__ char sgn(double a) {
		//return a < -DBL_EPSILON ? -1 : a > DBL_EPSILON;
		return a < -FLT_EPSILON ? -1 : a > FLT_EPSILON;
	}
	inline __device__ ExtentType dot(const PointType &a, const PointType &b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	inline __device__ PointType cross(const PointType& a, const PointType& b) {
		return MakePoint<ExtentType>::p(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}
	inline __device__ ExtentType proj(const PointType &v, const PointType &p0, const PointType &p1) {
		return dot(v, cross(p0, p1));
	}
	inline __device__ bool diff(const PointType &v0, const PointType &v1) {
		// what if both 0?
		return sgn(v0.x) != sgn(v1.x) && sgn(v0.y) != sgn(v1.y) && sgn(v0.z) != sgn(v1.z);
	}
	inline __device__ bool edgeBvOverlap(const PointType a0, const PointType a1,
		const PointType b0, const PointType b1) {
		return diff(a0 - b0, a0 - b1) || 
			diff(a1 - b0, a1 - b1);
	}
	__device__ PointType normalize(const PointType v);
	__device__ ExtentType signed_vf_distance(const PointType &x, const PointType &y0,
		const PointType &y1, const PointType &y2, PointType& n, ExtentType* const &w);
	__device__ ExtentType signed_ee_distance(const PointType& x0, const PointType& x1,
		const PointType& y0, const PointType& y1, PointType& n, ExtentType* const &w);
}

#endif
