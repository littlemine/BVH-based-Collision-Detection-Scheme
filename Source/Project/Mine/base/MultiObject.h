/**	\file	MultiObject.h
*	\brief	Base structure
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __MULTI_OBJECT_H_
#define __MULTI_OBJECT_H_

#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace mn {

	typedef unsigned char uchar;

	template<typename T, uchar _total>
	class MultiObjectH {
	public:
		MultiObjectH() : _offset(0), _span(1) {
			assert(_total % _span == 0);
		}
		virtual ~MultiObjectH() {}

		///	host access
		T&		hobj(uchar absID) {
			assert(absID < _total);
			return h_objs[absID];
		}
		T&		cho(uchar relID = 0) {	///< current host object
			assert(relID < _span);
			return h_objs[_offset + relID];
		}
		T&		nho(uchar relID = 0) {	///< next host object
			assert(relID < _span);
			return h_objs[(_offset + _span) % _total + relID];
		}
		///
		void	init(uchar span = 1) { _span = span; }
		void	slide() { _offset = (_offset + _span) % _total; }

	protected:
		/// scroll
		uchar 	_offset, _span;

	private:
		/// objects
		std::array<T, _total>	h_objs;
	};

	template<typename T, uchar _total>
	class MultiObjectD {
	public:
		MultiObjectD() : _offset(0), _span(1) {
			assert(_total % _span == 0);
			checkCudaErrors(cudaMalloc((void**)&d_objs, sizeof(T) * _total));
		}
		virtual ~MultiObjectD() {
			checkCudaErrors(cudaFree(d_objs));
		}

		///	host access
		T*		dptr(uchar absID) {
			assert(absID < _total);
			return d_objs + absID;
		}
		T*		cdp(uchar relID) {	///< current host object
			assert(relID < _span);
			return d_objs + _offset + relID;
		}
		T*		ndp(uchar relID) {	///< next host object
			assert(relID < _span);
			return d_objs + ((_offset + _span) % _total + relID);
		}
									///
		void	init(uchar span = 1) { _span = span; }
		void	slide() { _offset = (_offset + _span) % _total; }

	protected:
		/// scroll
		uchar 	_offset, _span;
		/// objects
		T*		d_objs;
		
	};

}

#endif