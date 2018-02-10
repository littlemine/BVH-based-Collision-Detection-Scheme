#include "BvhRestrLog.h"
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace mn {

	BvhRestrLog::BvhRestrLog() {}
	BvhRestrLog::~BvhRestrLog() {}
	void BvhRestrLog::setup(uint extSize, uint primSize) {
		_extSize = extSize;
		_primSize = primSize;
		/// build attribs
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_RANGE], sizeof(int)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[INT_RANGE], sizeof(int)*(extSize - 1)));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_RESTR], sizeof(int)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[INT_RESTR], sizeof(int)*(extSize - 1)));
		checkCudaErrors(cudaMalloc((void**)&_attribs[PRIM_RESTR], sizeof(int)*primSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[SUBBVH_ROOT_IDS], sizeof(int)*extSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[SUBBVH_OVERSIZE], sizeof(int)*(extSize - 1)));
		/// build ports
		portptr(COMPLETE) = new BvhRestrCompletePort;
		/// link ports
		port<COMPLETE>()->link(_attribs, EXT_RANGE);
	}
	void BvhRestrLog::cleanup() {
		/// clean attribs
		for (int i = 0; i < NUM_ATTRIBS; i++)
			checkCudaErrors(cudaFree(_attribs[i]));
		/// clean ports
		delete port<COMPLETE>();
	}

	void*& BvhRestrLog::portptr(EnumBvhRestrPorts no) {
		assert(no >= COMPLETE && no < NUM_PORTS);
		return _ports[no];
	}

}
