#include "BvhIntNode.h"
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/fill.h>
#include "system\CudaDevice\CudaKernelLauncher.cu"
#include "utility\CudaThrustUtils.hpp"

namespace mn {

	BvhIntNodeArray::BvhIntNodeArray() {}
	BvhIntNodeArray::~BvhIntNodeArray() {}

	void BvhIntNodeArray::setup(uint intSize) {
		_bvArray.setup(intSize);
		_intSize = intSize;
		/// build attribs
		checkCudaErrors(cudaMalloc((void**)&_attribs[LC], sizeof(int)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[RC], sizeof(int)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[PAR], sizeof(int)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[RCD], sizeof(int)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MARK], sizeof(uint)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[RANGEX], sizeof(int)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[RANGEY], sizeof(int)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[FLAG], sizeof(uint)*intSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[QUALITY_METRIC], sizeof(ExtentType)*intSize));
		/// build ports
		portptr(COMPLETE) = new BvhIntNodeCompletePort(_bvArray.portobj<0>());
		/// link ports
		port<COMPLETE>()->link(_attribs, LC);
	}
	void BvhIntNodeArray::cleanup() {
		/// clean attribs
		for (int i = 0; i < NUM_ATTRIBS; i++)
			checkCudaErrors(cudaFree(_attribs[i]));
		/// clean ports
		delete port<COMPLETE>();
		_bvArray.cleanup();
	}

	void BvhIntNodeArray::scatter(int size, const int* scatterPos, BvhIntNodeArray& to) {
		_bvArray.scatter(size, scatterPos, to._bvArray);
		//recordLaunch("ScatterIntNodes", (size + 255) / 256, 256, 0, scatterIntNodes,
		configuredLaunch( {"ScatterIntNodes", size }, scatterIntNodes,
			size, scatterPos, portobj<0>(), to.portobj<0>());
	}

	void BvhIntNodeArray::clear(int size) {
		for (int i = LC; i <= PAR; i++)
			checkCudaErrors(cudaMemset(_attribs[i], 0xff, sizeof(int)*size));
		checkCudaErrors(cudaMemset(_attribs[RCD], 0, sizeof(int)*size));
		checkThrustErrors(thrust::fill(thrust::device_ptr<uint>((uint*)_attribs[MARK]), thrust::device_ptr<uint>((uint*)_attribs[MARK]) + size, 7));
		for (int i = RANGEX; i <= RANGEY; i++)
			checkCudaErrors(cudaMemset(_attribs[i], 0xff, sizeof(int)*size));
		checkCudaErrors(cudaMemset(_attribs[FLAG], 0, sizeof(uint)*size));
		_bvArray.clear(size);
	}

	void BvhIntNodeArray::clearIntNodes(int size) {
		checkCudaErrors(cudaMemset(_attribs[FLAG], 0, sizeof(uint)*size));
		_bvArray.clear(size);
	}

	void BvhIntNodeArray::clearFlags(int size) {
		checkCudaErrors(cudaMemset(_attribs[FLAG], 0, sizeof(uint)*size));
	}

	void*& BvhIntNodeArray::portptr(EnumBvhIntNodePorts no) {
		assert(no >= COMPLETE && no < NUM_PORTS);
		return _ports[no];
	}

	__global__ void scatterIntNodes(int size, const int* scatterPos, BvhIntNodeCompletePort from, BvhIntNodeCompletePort to) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		int tar = scatterPos[idx];
		//LC, RC, PAR, MARK, RANGEX, RANGEY, FLAG, QUALITY_METRIC,
		to.lc(tar) = from.getlc(idx);
		to.rc(tar) = from.getrc(idx);
		to.par(tar) = from.getpar(idx);
		to.rcd(tar) = from.getrcd(idx);
		to.mark(tar) = from.getmark(idx);
		to.rangex(tar) = from.getrangex(idx);
		to.rangey(tar) = from.getrangey(idx);
		to.metric(tar) = from.getmetric(idx);
	}

}
