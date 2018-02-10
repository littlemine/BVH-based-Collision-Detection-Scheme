#include "FlOrderLog.h"
#include "thrust\scan.h"
#include "utility\CudaThrustUtils.hpp"

namespace mn {

	void FlOrderLog::prepare(int size) {
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr<uint>((uint*)_attribs[INT_FL_NODE_NUM]),
			getDevicePtr<uint>((uint*)_attribs[INT_FL_NODE_NUM]) + size, getDevicePtr<uint>((uint*)_attribs[INT_FL_BEG])));
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr<uint>((uint*)_attribs[EXT_FL_NODE_NUM]),
			getDevicePtr<uint>((uint*)_attribs[EXT_FL_NODE_NUM]) + size + 1, getDevicePtr<uint>((uint*)_attribs[EXT_FL_BEG])));
	}

	void FlOrderLog::prepareBak(int size) {
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr<uint>((uint*)_attribs[INT_FL_BACK_NUM]),
			getDevicePtr<uint>((uint*)_attribs[INT_FL_BACK_NUM]) + size, getDevicePtr<uint>((uint*)_attribs[INT_FL_BEG_BAK])));
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr<uint>((uint*)_attribs[EXT_FL_BACK_NUM]),
			getDevicePtr<uint>((uint*)_attribs[EXT_FL_BACK_NUM]) + size + 1, getDevicePtr<uint>((uint*)_attribs[EXT_FL_BEG_BAK])));
	}

	void FlOrderLog::preserveCnts(int size) {
		checkThrustErrors(thrust::copy(getDevicePtr<uint>((uint*)_attribs[INT_FL_NODE_NUM]),
			getDevicePtr<uint>((uint*)_attribs[INT_FL_NODE_NUM]) + size, getDevicePtr<uint>((uint*)_attribs[INT_FL_BACK_NUM])));
		checkThrustErrors(thrust::copy(getDevicePtr<uint>((uint*)_attribs[EXT_FL_NODE_NUM]),
			getDevicePtr<uint>((uint*)_attribs[EXT_FL_NODE_NUM]) + size + 1, getDevicePtr<uint>((uint*)_attribs[EXT_FL_BACK_NUM])));
	}

	/// used to preserve back cnts (restr part)
	__global__ void filterCnts(uint size, int *_masks, uint *_validCnts, uint *_invalidCnts) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		if (_masks[idx])	_validCnts[idx] = 0;
		else				_invalidCnts[idx] = 0;
	}

	__global__ void filterIntFrontCnts(uint size, int *_masks, const int* _prevLbds, const int* _leafRestrRoots, 
		uint *_validIntCnts, uint *_invalidIntCnts) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		bool	tag = _masks[idx];
		int		lbd = _prevLbds[idx];
		int		rt = _leafRestrRoots[lbd];
		if (tag) {
			_validIntCnts[idx] = 0;
			if (lbd != _prevLbds[rt]) {
				_invalidIntCnts[idx] = 0;
				return;
			}
			if (idx != rt) {
				atomicAdd(_invalidIntCnts + rt, _invalidIntCnts[idx]);
				_invalidIntCnts[idx] = 0;
			}
			_masks[idx] = -1;	///< -1 marked restructured index preserved
		} else
			_invalidIntCnts[idx] = 0;
	}

	__global__ void filterExtFrontCnts(uint size, int *_masks, const int* _prevLbds, const int* _leafRestrRoots, 
		uint *_validExtCnts, uint *_invalidExtCnts, uint *_invalidIntCnts) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		int		rt = _leafRestrRoots[idx];
		if (_masks[idx]) {
			_validExtCnts[idx] = 0;
			if (idx != _prevLbds[rt]) {
				_invalidExtCnts[idx] = 0;
				return;
			}
			atomicAdd(_invalidIntCnts + rt, _invalidExtCnts[idx]);
			_masks[idx] = -1;
		}
		_invalidExtCnts[idx] = 0;
	}
}
