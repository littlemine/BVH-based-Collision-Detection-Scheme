#ifndef __CUDA_THRUST_UTILS_HPP_
#define __CUDA_THRUST_UTILS_HPP_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace mn {

#define checkThrustErrors(func) \
	try {func;}					\
	catch (thrust::system_error &e) { std::cout << std::string(__FILE__) << ":" << __LINE__ << " " << e.what() << std::endl; }

	template<class T>
	__inline__ __host__ T* getRawPtr(thrust::device_vector<T> &V) {
		return thrust::raw_pointer_cast(V.data());
	}
	template<class T>
	__inline__ __host__ thrust::device_ptr<T> getDevicePtr(thrust::device_vector<T> &V) {
		return thrust::device_ptr<T>(thrust::raw_pointer_cast(V.data()));
	}
	template<class T>
	__inline__ __host__ thrust::device_ptr<T> getDevicePtr(T* V) {
		return thrust::device_ptr<T>(V);
	}

}

#endif