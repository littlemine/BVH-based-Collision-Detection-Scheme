#include "BvhPrimitiveNode.h"
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "system\CudaDevice\CudaKernelLauncher.cu"

namespace mn {

	BvhPrimitiveArray::BvhPrimitiveArray() {}
	BvhPrimitiveArray::~BvhPrimitiveArray() {}

	void BvhPrimitiveArray::setup(uint primSize) {
		_bvArray.setup(primSize);
		_primSize = primSize;
		/// build attribs
		checkCudaErrors(cudaMalloc((void**)&_attribs[MTCODE], sizeof(MCSize)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[IDX], sizeof(int)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[VIDA], sizeof(int)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[VIDB], sizeof(int)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[VIDC], sizeof(int)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[TYPE], sizeof(uint)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_MARK], sizeof(uint)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_NO], sizeof(int)*primSize));
		/// build ports
		portptr(COMPLETE) = new BvhPrimitiveCompletePort(_bvArray.portobj<0>());
		/// link ports
		port<COMPLETE>()->link(_attribs, MTCODE);
	}
	void BvhPrimitiveArray::cleanup() {
		/// clean attribs
		for (int i = 0; i < NUM_ATTRIBS; i++)
			checkCudaErrors(cudaFree(_attribs[i]));
		/// clean ports
		delete port<COMPLETE>();
		///
		_bvArray.cleanup();
	}

	void BvhPrimitiveArray::gather(int size, const int* gatherPos, BvhPrimitiveArray& to) {
		_bvArray.gather(size, gatherPos, to._bvArray);
		recordLaunch("GatherPrims", (size + 255) / 256, 256, 0, gatherPrims,
			size, gatherPos, portobj<0>(), to.portobj<0>());
	}
	void BvhPrimitiveArray::scatter(int size, const int* scatterPos, BvhPrimitiveArray& to) {
		_bvArray.scatter(size, scatterPos, to._bvArray);
		recordLaunch("ScatterPrims", (size + 255) / 256, 256, 0, scatterPrims,
			size, scatterPos, portobj<0>(), to.portobj<0>());
	}

	void*& BvhPrimitiveArray::portptr(EnumBvhPrimPorts no) {
		assert(no >= COMPLETE && no < NUM_PORTS);
		return _ports[no];
	}

	__global__ void gatherPrims(int size, const int* gatherPos, BvhPrimitiveCompletePort from, BvhPrimitiveCompletePort to) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		int ori = gatherPos[idx];
		to.mtcode(idx) = from.mtcode(ori);
		to.idx(idx) = from.idx(ori);
		to.vida(idx) = from.vida(ori);
		to.vidb(idx) = from.vidb(ori);
		to.vidc(idx) = from.vidc(ori);
		to.type(idx) = from.type(ori);
	}
	__global__ void scatterPrims(int size, const int* scatterPos, BvhPrimitiveCompletePort from, BvhPrimitiveCompletePort to) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		int tar = scatterPos[idx];
		to.mtcode(tar) = from.mtcode(idx);
		to.idx(tar) = from.idx(idx);
		to.vida(tar) = from.vida(idx);
		to.vidb(tar) = from.vidb(idx);
		to.vidc(tar) = from.vidc(idx);
		to.type(tar) = from.type(idx);
	}

}
