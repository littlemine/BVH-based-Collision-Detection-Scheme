#ifndef __META_H_
#define __META_H_

#include <cuda_runtime.h>

namespace mn {
	using uint = unsigned int;
	using uint64 = unsigned long long int;
	using ExtentType = float;

	template<typename T>
	struct CorrespondingPoint;
	template<>
	struct CorrespondingPoint<float> {
		using Type = float3;
	};
	template<>
	struct CorrespondingPoint<double> {
		using Type = double3;
	};
	template<typename ExtentType>
	using PointTypeSelector = typename CorrespondingPoint<ExtentType>::Type;

	using PointType = PointTypeSelector<float>;

	template<typename T> struct MakePoint;

	template<>
	struct MakePoint<float> {
		MakePoint() = delete;
		static __host__ __device__ float3 p(float&& x, float&& y, float&& z) {
			return make_float3(x, y, z);
		}
		static __host__ __device__ float3 p(const float& x, const float& y, const float& z) {
			return make_float3(x, y, z);
		}
	};
	template<>
	struct MakePoint<double> {
		MakePoint() = delete;
		static __host__ __device__ double3 p(double&& x, double&& y, double&& z) {
			return make_double3(x, y, z);
		}
		static __host__ __device__ double3 p(const double& x, const double& y, const double& z) {
			return make_double3(x, y, z);
		}
	};

	inline __host__ __device__ PointType operator - (PointType a, PointType b) {
		return MakePoint<ExtentType>::p(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	inline __host__ __device__ PointType operator + (PointType a, PointType b) {
		return MakePoint<ExtentType>::p(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	inline __host__ __device__  PointType operator * (PointType a, ExtentType k) {
		return MakePoint<ExtentType>::p(a.x * k, a.y * k, a.z * k);
	}

}

#endif
