#ifndef __CUDA_HOST_UTILS_H_
#define __CUDA_HOST_UTILS_H_

#include "Meta.h"
#include <array>
#include <cuda_runtime.h>

namespace mn {

	inline void reportMemory() {
		size_t free_byte;
		size_t total_byte;
		cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
		if (cudaSuccess != cuda_status) {
			printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
			exit(1);
		}
		double free_db = (double)free_byte;
		double total_db = (double)total_byte;
		double used_db = total_db - free_db;
		printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
			used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
	}

	inline __host__ __device__ float diagonalLength(float3 ext) {
		return sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);
	}
	inline __host__ __device__ double diagonalLength(double3 ext) {
		return sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);
	}

}

#endif
