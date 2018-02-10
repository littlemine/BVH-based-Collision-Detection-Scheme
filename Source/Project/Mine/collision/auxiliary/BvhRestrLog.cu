#include "BvhRestrLog.h"
#include <thrust/scan.h>
#include "utility\CudaThrustUtils.hpp"
#include "system\CudaDevice\CudaKernelLauncher.cu"

namespace mn {

	void BvhRestrLog::clear(int extSize) {
		checkThrustErrors(thrust::fill(getDevicePtr(getRestrBvhRoot()), getDevicePtr(getRestrBvhRoot()) + extSize, INT_MAX));
		checkThrustErrors(thrust::fill(getDevicePtr(getIntRange()), getDevicePtr(getIntRange()) + extSize - 1, 0));
		//checkThrustErrors(thrust::fill(getDevicePtr(getExtRange()), getDevicePtr(getExtRange()) + extSize, 0));
	}

}
