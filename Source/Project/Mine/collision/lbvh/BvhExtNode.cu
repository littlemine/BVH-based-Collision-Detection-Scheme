#include "BvhExtNode.h"
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "system\CudaDevice\CudaKernelLauncher.cu"
#include "utility\CudaThrustUtils.hpp"
#include "utility\CudaDeviceUtils.h"

namespace mn {

	BvhExtNodeArray::BvhExtNodeArray() {}
	BvhExtNodeArray::~BvhExtNodeArray() {}

	void BvhExtNodeArray::setup(uint primSize, uint extSize) {
		assert(extSize <= primSize);
		_primArray.setup(primSize);
		_bvArray.setup(extSize);
		_extSize = extSize;
		/// build attribs
		checkCudaErrors(cudaMalloc((void**)&_attribs[PAR], sizeof(int)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[MARK], sizeof(uint)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[LCA], sizeof(int)*(extSize + 1)));
		checkCudaErrors(cudaMalloc((void**)&_attribs[RCL], sizeof(int)*(extSize + 1)));
		checkCudaErrors(cudaMalloc((void**)&_attribs[STIDX], sizeof(int)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[SEGLEN], sizeof(uint)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[SPLIT_METRIC], sizeof(int)*extSize));
		/// build ports
		portptr(COMPLETE) = new BvhExtNodeCompletePort(_bvArray.portobj<0>(), _primArray.portobj<0>());
		/// link ports
		port<COMPLETE>()->link(_attribs, PAR);
	}
	void BvhExtNodeArray::cleanup() {
		/// clean attribs
		for (int i = 0; i < NUM_ATTRIBS; i++)
			checkCudaErrors(cudaFree(_attribs[i]));
		/// clean ports
		delete port<COMPLETE>();
		_bvArray.cleanup();
		_primArray.cleanup();
	}

	void BvhExtNodeArray::clearExtNodes(int size) {
		checkCudaErrors(cudaMemset(_attribs[PAR], 0xff, sizeof(int)*size));
		checkThrustErrors(thrust::fill(thrust::device_ptr<uint>((uint*)_attribs[MARK]), thrust::device_ptr<uint>((uint*)_attribs[MARK]) + size, 7));
		checkCudaErrors(cudaMemset(_attribs[SEGLEN], 0, sizeof(uint)*size));
		checkCudaErrors(cudaMemset(_attribs[LCA], 0xff, sizeof(int)*(size + 1)));
		checkCudaErrors(cudaMemset(_attribs[RCL], 0xff, sizeof(int)*(size + 1)));
		_bvArray.clear(size);
	}

	void BvhExtNodeArray::clearExtBvs(int size) {
		_bvArray.clear(size);
	}

	int BvhExtNodeArray::buildExtNodes(int primsize) {
		uint*	primMarks = _primArray.getMarks();
		int*	extIds = _primArray.getExtIds();
		int		extSize;
		/// should use strategy, delegate to 
		//recordLaunch("MarkSplitPostions", (primsize + 255) / 256, 256, 0, markPrimSplitPos,
		//	primsize, _primArray.portobj<0>(), primMarks);
		//checkThrustErrors(thrust::inclusive_scan(getDevicePtr(primMarks), getDevicePtr(primMarks) + primsize, getDevicePtr(extIds)));

		/// no primitive collapsing for now
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::fill(getDevicePtr(primMarks), getDevicePtr(primMarks) + primsize, 1));
		checkThrustErrors(thrust::sequence(getDevicePtr(extIds), getDevicePtr(extIds) + primsize, 1));
		checkCudaErrors(cudaMemcpy(&extSize, extIds + primsize - 1, sizeof(int), cudaMemcpyDeviceToHost));
		Logger::tock<TimerType::GPU>("PrepareCollapsing");

		clearExtNodes(extSize);
		//recordLaunch("CollapsePrimitives", (primsize + 255) / 256, 256, 0, collapsePrimitives,
		configuredLaunch({ "CollapsePrimitives", primsize }, collapsePrimitives,
			primsize, portobj<0>(), extIds);
		//printf("Collapsing %d primitives into %d leaves\n", primsize, extSize);
		return extSize;
	}

	void BvhExtNodeArray::calcSplitMetrics(int extsize) {
		//recordLaunch("CalcExtNodeSplitMetrics", (extsize + 255) / 256, 256, 0, calcExtNodeSplitMetrics,
		configuredLaunch({ "CalcExtNodeSplitMetrics", extsize }, calcExtNodeSplitMetrics,
			extsize, (const MCSize*)getMtCodes(), getMetrics());
	}

	void BvhExtNodeArray::calcRestrSplitMetrics(int extsize, const int * _leafRestrRoots) {
		//recordLaunch("CalcExtNodeRestrSplitMetrics", (extsize + 255) / 256, 256, 0, calcExtNodeRestrSplitMetrics,
		configuredLaunch({ "CalcExtNodeRestrSplitMetrics", extsize }, calcExtNodeRestrSplitMetrics,
			extsize, _leafRestrRoots, (const MCSize*)getMtCodes(), getMetrics());
	}

	void*& BvhExtNodeArray::portptr(EnumBvhExtNodePorts no) {
		assert(no >= COMPLETE && no < NUM_PORTS);
		return _ports[no];
	}

	__global__ void markPrimSplitPos(int size, BvhPrimitiveCompletePort _prims, uint* _mark) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size) return;
		_mark[idx] = 1;	///< should depend on collapsing policy

		//_mark[idx] = idx ? __clz(_prims.mtcode(idx) ^ _prims.mtcode(idx - 1)) <= MARK_TAG : 1;
	}

	__global__ void collapsePrimitives(int primsize, BvhExtNodeCompletePort _lvs, int* _extIds) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= primsize) return;
		int extId = _extIds[idx] - 1;
		//BvhBvCompletePort primBvs = _lvs.getPrimBvs(), extBvs = _lvs.getExtBvs();
		const BvhBvCompletePort &primBvs = _lvs.primBvs();
		auto &extBvs = _lvs.refExtBvs();
		atomicAdd(&_lvs.seglen(extId), 1);
		extBvs.setBV(extId, primBvs, idx);
		if (_lvs.getPrimMark(idx))
			_lvs.stidx(extId) = idx;
		/*
		atomicMinD(&extBvs.minx(extId), primBvs.getminx(idx));
		atomicMinD(&extBvs.miny(extId), primBvs.getminy(idx));
		atomicMinD(&extBvs.minz(extId), primBvs.getminz(idx));
		atomicMaxD(&extBvs.maxx(extId), primBvs.getmaxx(idx));
		atomicMaxD(&extBvs.maxy(extId), primBvs.getmaxy(idx));
		atomicMaxD(&extBvs.maxz(extId), primBvs.getmaxz(idx));
		*/
	}

	__global__ void calcExtNodeSplitMetrics(int extsize, const MCSize *_codes, int *_metrics) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= extsize) return;
		_metrics[idx] = idx != extsize - 1 ? 32 - __clz(_codes[idx] ^ _codes[idx + 1]) : 33;
		//if (idx < 10)
		//	printf("%d-ext node: split metric %d\n", idx, _lvs.metric(idx));
	}

	__global__ void calcExtNodeRestrSplitMetrics(int extsize, const int *_leafRestrRoots, const MCSize *_codes, int *_metrics) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		int subrt = _leafRestrRoots[idx];
		if (idx >= extsize || !subrt) return;
		//_lvs.metric(idx) = idx != extsize - 1 ? (_leafRestrRoots[idx + 1] == subrt ? 64 - __clzll(_lvs.getmtcode(idx) ^ _lvs.getmtcode(idx + 1)) : 65) : 65;
		_metrics[idx] = _leafRestrRoots[idx + 1] == subrt ? 32 - __clz(_codes[idx] ^ _codes[idx + 1]) : 33;
		//_metrics[idx] = idx != extsize - 1 ? (_leafRestrRoots[idx + 1] == subrt ? 32 - __clz(_codes[idx] ^ _codes[idx + 1]) : 33) : 33;
	}

}
