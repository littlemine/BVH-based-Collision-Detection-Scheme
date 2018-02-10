/**	\file	MultiArray.h
*	\brief	Device Multi Array
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __MULTI_ARRAY_H_
#define __MULTI_ARRAY_H_

#include "MultiObject.h"
#include <array>

namespace mn {

	typedef unsigned int uint;
	/**
	 * \brief	MultiArray consists of _total number of arrays allocated on device
	 * \tparam	T is the element type
	 */
	template<typename T, uchar _total>
	class MultiArray : public MultiObjectD<T*, _total> {	///< another scheme is to use thrust::device_vector
	public:
		MultiArray();
		~MultiArray();

		///	host access
		void	retrieveSizes();
		uint	size(uchar absID);
		uint	cs(uchar relID = 0);	///< elem number of current object
		uint	ns(uchar relID = 0);	///< elem number of next object	
		/// device access
		T*		cbuf(uchar relID);
		T*		nbuf(uchar relID);
		T**		cbufs();
		T**		nbufs();
		uint*	dsize(uchar absID);
		uint*	csize(uchar relID);
		uint*	nsize(uchar relID);
		uint*	csizes();
		uint*	nsizes();
		/// 
		void	initBufs(uint* elemNums, uchar span = 1);
		void	destroyBufs();
		void	resetNextSizes();
	private:
		void	init(int) = delete;

		/// array size limit
		std::array<T*, _total>	_devAddrs;
		uint	*_bufSizes;
		uint	*h_lens, *d_lens;
	};

	template <typename T, uchar _total>
	MultiArray<T, _total>::MultiArray() :
		_bufSizes(nullptr), h_lens(nullptr), d_lens(nullptr) {}

	template <typename T, uchar _total>
	MultiArray<T, _total>::~MultiArray() {}

	/// host access
	template <typename T, uchar _total>
	void MultiArray<T, _total>::retrieveSizes() {
		checkCudaErrors(cudaMemcpy(h_lens, d_lens, sizeof(uint)*_total, cudaMemcpyDeviceToHost));
	}
	template <typename T, uchar _total>
	uint MultiArray<T, _total>::size(uchar absID) {
		assert(absID < _total);
		return h_lens[absID];
	}
	template <typename T, uchar _total>
	uint MultiArray<T, _total>::cs(uchar relID) {
		assert(relID < _span);
		return size(_offset + relID);
	}
	template <typename T, uchar _total>
	uint MultiArray<T, _total>::ns(uchar relID) {
		assert(relID < _span);
		return size((_offset + _span) % _total + relID);
	}

	/// device access
	template <typename T, uchar _total>
	T** MultiArray<T, _total>::cbufs() { return cdp(0); }
	template <typename T, uchar _total>
	T** MultiArray<T, _total>::nbufs() { return ndp(0); }
	template <typename T, uchar _total>
	T* MultiArray<T, _total>::cbuf(uchar relID) { return _devAddrs[_offset + relID]; }
	template <typename T, uchar _total>
	T* MultiArray<T, _total>::nbuf(uchar relID) { return _devAddrs[(_offset + _span) % _total + relID]; }
	

	template <typename T, uchar _total>
	uint* MultiArray<T, _total>::dsize(uchar absID) {
		assert(absID < _total);
		return d_lens + absID;
	}
	template <typename T, uchar _total>
	uint* MultiArray<T, _total>::csize(uchar relID) {
		assert(relID < _span);
		return dsize(_offset + relID);
	}
	template <typename T, uchar _total>
	uint* MultiArray<T, _total>::nsize(uchar relID) {
		assert(relID < _span);
		return dsize((_offset + _span) % _total + relID);
	}
	template <typename T, uchar _total>
	uint* MultiArray<T, _total>::csizes() { return dsize(_offset); }
	template <typename T, uchar _total>
	uint* MultiArray<T, _total>::nsizes() { return dsize((_offset + _span) % _total); }

	/**
	* \brief
	* \param elemNums	Buffer size of each array in a span
	* \param span		Number of arrays used currently
	* \param total		Total number of arrays
	*/
	template <typename T, uchar _total>
	void MultiArray<T, _total>::initBufs(uint* elemNums, uchar span) {
		_span = span;
		h_lens = new uint[_total];
		_bufSizes = new uint[_span];
		memcpy_s(_bufSizes, sizeof(uint) * _span, elemNums, sizeof(uint) * _span);

		checkCudaErrors(cudaMalloc((void**)&d_lens, sizeof(uint)*_total));
		for (int i = 0; i < _total; i++)
			checkCudaErrors(cudaMalloc((void**)&_devAddrs[i], sizeof(T)* elemNums[i % _span]));
		checkCudaErrors(cudaMemcpy(d_objs, _devAddrs.data(), sizeof(T*) * _total, cudaMemcpyHostToDevice));
	}

	template <typename T, uchar _total>
	void MultiArray<T, _total>::destroyBufs() {
		for (int i = 0; i < _total; i++) {
			checkCudaErrors(cudaFree(_devAddrs[i]));
			_devAddrs[i] = nullptr;
		}
		if (d_lens) { checkCudaErrors(cudaFree(d_lens)); d_lens = nullptr; }
		if (_bufSizes) { delete[] _bufSizes; _bufSizes = nullptr; }
		if (h_lens) { delete[] h_lens; h_lens = nullptr; }
	}

	/**
	* \brief	Only operate on the next span of 'd_lens'
	*/
	template <typename T, uchar _total>
	void MultiArray<T, _total>::resetNextSizes() {
		if (_offset + _span != _total)
			checkCudaErrors(cudaMemset(d_lens + _offset + _span, 0, sizeof(uint) * _span));
		else
			checkCudaErrors(cudaMemset(d_lens, 0, sizeof(uint) * _span));
	}
}

#endif