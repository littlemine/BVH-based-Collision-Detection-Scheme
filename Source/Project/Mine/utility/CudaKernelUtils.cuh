#ifndef __CUDA_KERNEL_UTILS_CUH_
#define __CUDA_KERNEL_UTILS_CUH_

#include <device_launch_parameters.h>

namespace mn {

	using uint = unsigned int;

	__host__ __device__ struct NotZero {
		__host__ __device__
			bool operator()(const int &x) {
			return (x != 0 ? 1 : 0);
		}
	};
	__host__ __device__ struct NotOne {
		__host__ __device__
			bool operator()(const int &x) {
			return (x != 1 ? 1 : 0);
		}
	};
	__host__ __device__ struct GreaterOne {
		__host__ __device__
			bool operator()(const int &x) {
			return (x > 1 ? 1 : 0);
		}
	};
	__global__ void checkArray(int size, uint* arr);
	__global__ void checkSame(int size, int* A, int* B);
	__global__ void checkLink(int size, int* A, int* B);

	__global__ void calcInverseMapping(int size, int* map, int* invMap);
	__global__ void calcFurtherMap(int size, int* A, int* B, int* map);
	__global__ void updateMapping(int size, int* map, int* scatterMap);
	__global__ void extractFrontKeys(int size, int2* _front, int* _frontKeys);

}

#endif