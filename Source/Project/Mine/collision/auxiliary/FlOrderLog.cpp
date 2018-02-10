#include "FlOrderLog.h"
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace mn {

	FlOrderLog::FlOrderLog() {}
	FlOrderLog::~FlOrderLog() {}
	void FlOrderLog::setup(uint intSize, uint extSize, uint intFtSize, uint extFtSize) {
		_intSize = intSize, _extSize = extSize;
		_intFtSize = intFtSize, _extFtSize = extFtSize;
		/// build attribs
		checkCudaErrors(cudaMalloc((void**)&_attribs[INT_FL_NODE_NUM], sizeof(uint)*intSize*2 + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_FL_NODE_NUM], sizeof(uint)*extSize*2 + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[INT_FL_BEG], sizeof(uint)*intSize*2 + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_FL_BEG], sizeof(uint)*extSize*2 + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[INT_FL_BACK_NUM], sizeof(uint)*intSize + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_FL_BACK_NUM], sizeof(uint)*extSize + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[INT_FL_BEG_BAK], sizeof(uint) * intSize + 1));
		checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_FL_BEG_BAK], sizeof(uint) * extSize + 1));
		//checkCudaErrors(cudaMalloc((void**)&_attribs[INT_OPT_MARK], sizeof(char) * intFtSize));
		//checkCudaErrors(cudaMalloc((void**)&_attribs[EXT_OPT_MARK], sizeof(char) * extFtSize));
		checkCudaErrors(cudaMalloc((void**)&_attribs[PRIM_CP_NUM], sizeof(uint) * extFtSize));	///< should be primsize
		checkCudaErrors(cudaMalloc((void**)&_attribs[PRIM_CP_OFFSET], sizeof(uint) * extFtSize));	///< should be primsize
		/// build ports
		portptr(COMPLETE) = new FlOrderCompletePort;
		/// link ports
		port<COMPLETE>()->link(_attribs, INT_FL_NODE_NUM);
	}
	void FlOrderLog::cleanup() {
		/// clean attribs
		for (int i = 0; i < NUM_ATTRIBS; i++)
			checkCudaErrors(cudaFree(_attribs[i]));
		/// clean ports
		delete port<COMPLETE>();
	}

	void FlOrderLog::clear(int size) {
		//checkCudaErrors(cudaMemset(_attribs[INT_FL_NODE_NUM], 0, sizeof(uint)*(size - 1)));
		//checkCudaErrors(cudaMemset(_attribs[EXT_FL_NODE_NUM], 0, sizeof(uint)*size));
		checkCudaErrors(cudaMemset(_attribs[INT_FL_NODE_NUM], 0, sizeof(uint)*size));
		checkCudaErrors(cudaMemset(_attribs[EXT_FL_NODE_NUM], 0, sizeof(uint)*(size + 1)));
		checkCudaErrors(cudaMemset(_attribs[INT_FL_BACK_NUM], 0, sizeof(uint)*size));
		checkCudaErrors(cudaMemset(_attribs[EXT_FL_BACK_NUM], 0, sizeof(uint)*(size + 1)));
	}

	void FlOrderLog::clearCpCnts(int size) {
		checkCudaErrors(cudaMemset(_attribs[PRIM_CP_NUM], 0, sizeof(uint)*size));
	}

	void*& FlOrderLog::portptr(EnumFlOrderPorts no) {
		assert(no >= COMPLETE && no < NUM_PORTS);
		return _ports[no];
	}

}
