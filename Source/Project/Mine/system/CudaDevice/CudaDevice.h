#ifndef __SYSTEM_CUDA_DEVICE_H_
#define __SYSTEM_CUDA_DEVICE_H_

#include "base\Singleton.h"
#include "utility\CudaExecutionPolicy.h"
#include <unordered_map>
#include <string>
#include <driver_types.h>

namespace mn {

	class CudaDevice : public ManagedSingleton<CudaDevice> {
	public:
		CudaDevice();
		~CudaDevice();

		using KernelFunc = const void*;
		
		struct KernelConfig {		///< static kernel attrib, could contain run-time debugger setting(error checking/ time recording etc...)
			KernelFunc		func;
			cudaFuncAttributes	attribs;
			cudaFuncCache	cachePreference;
			bool			waveFashion;		///< general fashion or loop fashion
			int				maxOccBlockSize;	///< condition: use no shared memory
			explicit CudaDevice::KernelConfig(KernelFunc f = nullptr, cudaFuncCache cacheConfig = cudaFuncCachePreferNone, bool isWave = false);
		};

		static void registerKernel(const std::string& tag, KernelFunc f, cudaFuncCache cacheConfig = cudaFuncCachePreferL1, bool waveFashion = true);
		static const KernelConfig& findKernel(const std::string& name);

		int generalGridSize(int& threadNum, int& blockSize) const;
		int waveGridSize(int& threadNum, int& blockSize) const;
		static int evalOptimalBlockSize(cudaFuncAttributes attribs, cudaFuncCache cachePreference, size_t smemBytes = 0);
		ExecutionPolicy launchConfig(const std::string& kernelName, int threadNum, bool sync = false, size_t smemSize = 0, cudaStream_t sid = nullptr) const;

		/// Launching Kernels
	private:
		static cudaDeviceProp*	_akDeviceProps;
		static int				_iDevID;	///< selected cuda device
		static std::unordered_map<std::string, KernelConfig>
								_kFuncTable;
	};

}

#endif