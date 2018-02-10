#ifndef __BV_H_
#define __BV_H_

#include "utility\Meta.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace mn {

	class AABB {
	public:
		PointType	_min, _max;

		__host__ __device__ AABB() { empty<ExtentType>(); }
		__host__ __device__ AABB(const AABB& b)  { _min = b._min; _max = b._max; }
		__host__ __device__ AABB(AABB&& b)  { _min = b._min; _max = b._max; }
		__host__ __device__ AABB(const ExtentType &minx, const ExtentType &miny, const ExtentType &minz,
			const ExtentType &maxx, const ExtentType &maxy, const ExtentType &maxz) {
			_min = MakePoint<ExtentType>::p(minx, miny, minz);
			_max = MakePoint<ExtentType>::p(maxx, maxy, maxz);
		}
		__host__ __device__ AABB(const PointType &v) { _min = _max = v; }
		__host__ __device__ AABB(const PointType &v1, const PointType &v2) {
			_min = MakePoint<ExtentType>::p(::fmin(v1.x, v2.x), ::fmin(v1.y, v2.y), ::fmin(v1.z, v2.z));
			_max = MakePoint<ExtentType>::p(::fmax(v1.x, v2.x), ::fmax(v1.y, v2.y), ::fmax(v1.z, v2.z));
		}
		__host__ __device__  void combines(const PointType &b) {
			_min = MakePoint<ExtentType>::p(::fmin(_min.x, b.x), ::fmin(_min.y, b.y), ::fmin(_min.z, b.z));
			_max = MakePoint<ExtentType>::p(::fmax(_max.x, b.x), ::fmax(_max.y, b.y), ::fmax(_max.z, b.z));
		}
		__host__ __device__  void combines(const ExtentType x, const ExtentType y, const ExtentType z) {
			_min = MakePoint<ExtentType>::p(::fmin(_min.x, x), ::fmin(_min.y, y), ::fmin(_min.z, z));
			_max = MakePoint<ExtentType>::p(::fmax(_max.x, x), ::fmax(_max.y, y), ::fmax(_max.z, z));
		}
		__host__ __device__  void combines(const AABB &b) {
			_min = MakePoint<ExtentType>::p(::fmin(_min.x, b._min.x), ::fmin(_min.y, b._min.y), ::fmin(_min.z, b._min.z));
			_max = MakePoint<ExtentType>::p(::fmax(_max.x, b._max.x), ::fmax(_max.y, b._max.y), ::fmax(_max.z, b._max.z));
		}

		__host__ __device__  bool overlaps(const AABB &b) const  {
			if (b._min.x > _max.x || b._max.x < _min.x) return false;
			if (b._min.y > _max.y || b._max.y < _min.y) return false;
			if (b._min.z > _max.z || b._max.z < _min.z) return false;
			return true;
		}
		__host__ __device__ bool contains(const PointType &v) const  {
			return v.x <= _max.x && v.x >= _min.x &&
				v.y <= _max.y && v.y >= _min.y &&
				v.z <= _max.z && v.z >= _min.z;
		}
		__host__ __device__ ExtentType merges(const AABB &a, const AABB &b, ExtentType *qualityMetric) {
			_min = MakePoint<ExtentType>::p(::fmin(a._min.x, b._min.x), ::fmin(a._min.y, b._min.y), ::fmin(a._min.z, b._min.z));
			_max = MakePoint<ExtentType>::p(::fmax(a._max.x, b._max.x), ::fmax(a._max.y, b._max.y), ::fmax(a._max.z, b._max.z));
			*qualityMetric = (a.volume() + b.volume()) / volume();
			return *qualityMetric;
		}
		__host__ __device__ void merges(const AABB &a, const AABB &b) {
			_min = MakePoint<ExtentType>::p(::fmin(a._min.x, b._min.x), ::fmin(a._min.y, b._min.y), ::fmin(a._min.z, b._min.z));
			_max = MakePoint<ExtentType>::p(::fmax(a._max.x, b._max.x), ::fmax(a._max.y, b._max.y), ::fmax(a._max.z, b._max.z));
		}

		__host__ __device__  ExtentType width()  const  { return _max.x - _min.x; }
		__host__ __device__  ExtentType height() const  { return _max.y - _min.y; }
		__host__ __device__  ExtentType depth()  const  { return _max.z - _min.z; }
		__host__ __device__ auto center() const  -> decltype(MakePoint<ExtentType>::p(0, 0, 0)){
			return MakePoint<ExtentType>::p((_min.x + _max.x)*0.5, (_min.y + _max.y) *0.5, (_min.z + _max.z)*0.5);
		}
		__host__ __device__  ExtentType volume() const  { return width()*height()*depth(); }

		template<typename T>
		__host__ __device__  void empty();
		template<>
		__host__ __device__  void empty<float>() {
			_max = MakePoint<float>::p(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			_min = MakePoint<float>::p(FLT_MAX, FLT_MAX, FLT_MAX);
		}
		//template<>
		//__host__ __device__  void empty<double>() {
		//	_max = MakePoint<double>(-DBL_MAX, -DBL_MAX, -DBL_MAX);
		//	_min = MakePoint<double>(DBL_MAX, DBL_MAX, DBL_MAX);
		//}
	};

	using BOX = AABB;
}

#endif
