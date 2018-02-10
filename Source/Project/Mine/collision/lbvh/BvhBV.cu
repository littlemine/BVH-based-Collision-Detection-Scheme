#include "BvhBV.h"
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/fill.h>
#include <system\CudaDevice\CudaKernelLauncher.cu>
#include "utility\CudaThrustUtils.hpp"

namespace mn {

	BvhBvArray::BvhBvArray() {}
	BvhBvArray::~BvhBvArray() {}

	void BvhBvArray::setup(uint count) {
		_count = count;
		/// build attribs
		checkCudaErrors(cudaMalloc((void**)&_attribs[MINX], sizeof(ExtentType)*count));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MINY], sizeof(ExtentType)*count));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MINZ], sizeof(ExtentType)*count));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MAXX], sizeof(ExtentType)*count));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MAXY], sizeof(ExtentType)*count));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MAXZ], sizeof(ExtentType)*count));
		/// build ports
		portptr(COMPLETE) = new BvhBvCompletePort;
		/// link ports
		port<COMPLETE>()->link(_attribs, MINX);
	}
	void BvhBvArray::cleanup() {
		/// clean attribs
		for (int i = 0; i < NUM_ATTRIBS; i++)
			checkCudaErrors(cudaFree(_attribs[i]));
		/// clean ports
		delete port<COMPLETE>();
	}

	void BvhBvArray::gather(int size, const int* gatherPos, BvhBvArray& to) {
		//recordLaunch("GatherBVs", (size + 255) / 256, 256, 0, gatherBVs,
		configuredLaunch( {"GatherBVs", size }, gatherBVs,
			size, gatherPos, portobj<0>(), to.portobj<0>());
	}
	void BvhBvArray::scatter(int size, const int* scatterPos, BvhBvArray& to) {
		//recordLaunch("ScatterBVs", (size + 255) / 256, 256, 0, scatterBVs,
		configuredLaunch( {"ScatterBVs", size }, scatterBVs,
			size, scatterPos, portobj<0>(), to.portobj<0>());
	}

	void BvhBvArray::clear(int size) {
		uchar i;
		for (i = MINX; i <= MINZ; i++)
			checkThrustErrors(thrust::fill(thrust::device_ptr<ExtentType>((ExtentType*)_attribs[i]), thrust::device_ptr<ExtentType>((ExtentType*)_attribs[i]) + size, DBL_MAX));
		for (; i <= MAXZ; i++)
			checkThrustErrors(thrust::fill(thrust::device_ptr<ExtentType>((ExtentType*)_attribs[i]), thrust::device_ptr<ExtentType>((ExtentType*)_attribs[i]) + size, -DBL_MAX));
	}

	void*& BvhBvArray::portptr(EnumBvhBvPorts no) {
		assert(no >= COMPLETE && no < NUM_PORTS);
		return _ports[no];
	}

	__global__ void gatherBVs(int size, const int* gatherPos, BvhBvCompletePort from, BvhBvCompletePort to) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		int ori = gatherPos[idx];
		to.minx(idx) = from.minx(ori);
		to.miny(idx) = from.miny(ori);
		to.minz(idx) = from.minz(ori);
		to.maxx(idx) = from.maxx(ori);
		to.maxy(idx) = from.maxy(ori);
		to.maxz(idx) = from.maxz(ori);
	}
	__global__ void scatterBVs(int size, const int* scatterPos, BvhBvCompletePort from, BvhBvCompletePort to) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		int tar = scatterPos[idx];
		to.minx(tar) = from.minx(idx);
		to.miny(tar) = from.miny(idx);
		to.minz(tar) = from.minz(idx);
		to.maxx(tar) = from.maxx(idx);
		to.maxy(tar) = from.maxy(idx);
		to.maxz(tar) = from.maxz(idx);
	}

}
