#include "CudaKernelUtils.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace mn {
	
	__global__ void checkArray(int size, uint* arr) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		printf("%d: %x\n", idx, arr[idx]);
	}

	__global__ void checkSame(int size, int* A, int* B) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		if (A[idx] != B[idx])
			printf("A[%d](%d) - B[%d](%d) not same!\n", idx, A[idx], idx, B[idx]);
	}

	__global__ void checkLink(int size, int* A, int* B) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		if (B[A[idx]] != idx)
			printf("%d-th A-B link failed!\n", idx);
	}

	/// incoherent access, thus poor performance
	__global__ void calcInverseMapping(int size, int* map, int* invMap) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		invMap[map[idx]] = idx;
	}

	__global__ void calcFurtherMap(int size, int* A, int* B, int* map) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		map[A[idx]] = B[idx];
	}
	
	__global__ void updateMapping(int size, int* map, int* scatterMap) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size || scatterMap[map[idx]] == -1) return;
		map[idx] = scatterMap[map[idx]];
	}
	
	__global__ void extractFrontKeys(int size, int2* _front, int* _frontKeys) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		_frontKeys[idx] = _front[idx].y;
	}
	/*
	/// DEBUG!
	__global__ void checkPrims(int size, BvhPrimitiveCompletePort prims) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		if (idx < 10)
			printf("%d-th prim: code(%x) tri-id(%d: %d, %d, %d) \n", idx, prims.mtcode(idx),
				prims.idx(idx), prims.vida(idx), prims.vidb(idx), prims.vidc(idx));
	}
	__global__ void checkBV(BOX* bv) {
		PRINT_BOX_INFO(0, *bv);
	}
	__global__ void checkRawInput(int size, glm::ivec3* d_faces, glm::vec3* d_vertices) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		if (idx < 10)
			printf("%d-th input: tri (%d: %.3f, %.3f, %.3f) (%d: %.3f, %.3f, %.3f) (%d: %.3f, %.3f, %.3f)\n", idx,
				d_faces[idx].x, d_vertices[d_faces[idx].x].x, d_vertices[d_faces[idx].x].y, d_vertices[d_faces[idx].x].z,
				d_faces[idx].y, d_vertices[d_faces[idx].y].x, d_vertices[d_faces[idx].y].y, d_vertices[d_faces[idx].y].z,
				d_faces[idx].z, d_vertices[d_faces[idx].z].x, d_vertices[d_faces[idx].z].y, d_vertices[d_faces[idx].z].z);
	}
	*/

}