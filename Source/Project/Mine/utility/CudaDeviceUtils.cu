#include "CudaDeviceUtils.h"

namespace mn {

	__device__ bool atomicMinf(float* address, float val) {
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		if (*address <= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fminf(val, __int_as_float(assumed))));
		} while (assumed != old);
		//return __int_as_float(old);
		return true;
	}

	__device__ bool atomicMaxf(float* address, float val) {
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		if (*address >= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
		} while (assumed != old);
		//return __int_as_float(old);
		return true;
	}

	__device__ bool atomicMinD(double* address, double val) {
		unsigned long long int* address_as_ull = (unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		if (*address <= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_ull, assumed,
				__double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
		} while (assumed != old);
		return true;
	}

	__device__ bool atomicMaxD(double* address, double val) {
		unsigned long long int* address_as_ull = (unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		if (*address >= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_ull, assumed,
				__double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
		} while (assumed != old);
		return true;
	}

	__device__ inline int lane_id(void) { return threadIdx.x % WARP_SIZE; }
	__device__ inline int warp_bcast(int v, int leader) { return __shfl(v, leader); }
	__device__ uint atomicAggInc(uint *ctr) {
		uint mask = __ballot(1), leader, res;
		// select the leader
		leader = __ffs(mask) - 1;
		// leader does the update
		if (lane_id() == leader)
			res = atomicAdd(ctr, __popc(mask));
		// broadcast result
		res = warp_bcast(res, leader);
		// each thread computes its own value
		return res + __popc(mask & ((1 << lane_id()) - 1));
	}
	__device__ int atomicAggInc(int *ctr) {
		int mask = __ballot(1), leader, res;
		// select the leader
		leader = __ffs(mask) - 1;
		// leader does the update
		if (lane_id() == leader)
			res = atomicAdd(ctr, __popc(mask));
		// broadcast result
		res = warp_bcast(res, leader);
		// each thread computes its own value
		return res + __popc(mask & ((1 << lane_id()) - 1));
	}
	
	__device__ uint expandBits(uint v) {					///< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}

	__device__ uint64 expandBits64(uint v) {
		// 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249
		uint64 x = v & 0x1fffff;
		x = (x | x << 32) & 0x1f00000000ffff;
		x = (x | x << 16) & 0x1f0000ff0000ff;
		x = (x | x << 8)  & 0x100f00f00f00f00f;
		x = (x | x << 4)  & 0x10c30c30c30c30c3;
		x = (x | x << 2)  & 0x1249249249249249;
		return x;
	}

	__device__ uint compactBits(uint v) {
		// 0, 0x000003ff, 0x30000ff, 0x0300f00f, 0x30c30c3, 0x9249249
		v &= 0x9249249;
		v = (v ^ (v >> 2)) & 0x30c30c3;
		v = (v ^ (v >> 4)) & 0x0300f00f;
		v = (v ^ (v >> 8)) & 0x30000ff;
		v = (v ^ (v >> 16)) & 0x000003ff;
		return v;
	}

	__device__ uint compactBits64(uint64 v) {
		// 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249
		v &= 0x1249249249249249;
		v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3;
		v = (v ^ (v >> 4)) & 0x100f00f00f00f00f;
		v = (v ^ (v >> 8)) & 0x1f0000ff0000ff;
		v = (v ^ (v >> 16)) & 0x1f00000000ffff;
		v = (v ^ (v >> 32)) & 0x1fffff;
		return static_cast<uint>(v);
	}

	__device__ uint morton3D(float x, float y, float z) {	///< Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
		x = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
		y = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
		z = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
		uint xx = expandBits((uint)x);
		uint yy = expandBits((uint)y);
		uint zz = expandBits((uint)z);
		return (xx * 4 + yy * 2 + zz);
	}
	__device__ uint morton3Dbounds(float x, float y, float z, int dx, int dy, int dz) {	///< Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
		x = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
		y = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
		z = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
		x += dx, y += dy, z += dz;	///< related with primitive size
		x = ::fmin(x, 1023.0f), y = ::fmin(y, 1023.0f), z = ::fmin(z, 1023.0f);
		uint xx = expandBits((uint)x);
		uint yy = expandBits((uint)y);
		uint zz = expandBits((uint)z);
		return (xx * 4 + yy * 2 + zz);
	}

	__device__ uint64 morton3D64(double x, double y, double z) {
		x = dmin(dmax(x * 2097152.0f, 0.0f), 2097151.0f);
		y = dmin(dmax(y * 2097152.0f, 0.0f), 2097151.0f);
		z = dmin(dmax(z * 2097152.0f, 0.0f), 2097151.0f);
		return expandBits64((uint)x) | expandBits64((uint)y) << 1 | expandBits64((uint)z) << 2;
	}
	__device__ uint3 morton3D64_d(uint64 c) {
		return make_uint3(compactBits64(c), compactBits64(c >> 1), compactBits64(c >> 2));
	}

	__device__ bool covertex(int3 a, int3 b) {
		return a.x == b.x || a.x == b.y || a.x == b.z || 
			a.y == b.x || a.y == b.y || a.y == b.z || 
			a.z == b.x || a.z == b.y || a.z == b.z;
	}
	__device__ PointType normalize(const PointType v) {
		ExtentType invLen = rsqrt(dot(v, v));
		return v * invLen;
	}

	__device__ ExtentType signed_vf_distance(const PointType& x, const PointType& y0,
		const PointType& y1, const PointType& y2, PointType& n, ExtentType* const &w) {
		n = cross(normalize(y1 - y0), normalize(y2 - y0));
		if (dot(n, n) < DBL_EPSILON) return DBL_MAX;
		n = normalize(n);
		ExtentType b0 = proj(y1 - x, y2 - x, n),
			b1 = proj(y2 - x, y0 - x, n), 
			b2 = proj(y0 - x, y1 - x, n);
		w[0] = 1;
		w[1] = -b0 / (b0 + b1 + b2);
		w[2] = -b1 / (b0 + b1 + b2);
		w[3] = -b2 / (b0 + b1 + b2);
		return dot(x - y0, n);
	}

	__device__ ExtentType signed_ee_distance(const PointType& x0, const PointType& x1,
		const PointType& y0, const PointType& y1, PointType& n, ExtentType* const &w) {
		n = cross(normalize(x1 - x0), normalize(y1 - y0));
		//if (dot(n, n) < DBL_EPSILON) return DBL_MAX;
		if (dot(n, n) < FLT_EPSILON) return FLT_MAX;
		n = normalize(n);
		ExtentType a0 = proj(y1 - x1, y0 - x1, n),
			a1 = proj(y0 - x0, y1 - x0, n),
			b0 = proj(x0 - y1, x1 - y1, n),
			b1 = proj(x1 - y0, x0 - y0, n);
		w[0] = a0 / (a0 + a1);
		w[1] = a1 / (a0 + a1);
		w[2] = -b0 / (b0 + b1);
		w[3] = -b1 / (b0 + b1);
		return dot(x0 - y0, n);
	}
}
