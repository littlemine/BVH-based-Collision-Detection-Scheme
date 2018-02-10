/**	\file	BvhPrimitiveNode.h
*	\brief	Bridge layer between leaf layer and scene(model) data
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVH_PRIMITIVE_NODE_H_
#define __BVH_PRIMITIVE_NODE_H_

#include "base\AggregatedAttribs.h"
#include "BvhBV.h"

namespace mn {

	/**
	* \brief	Port
	*/
	class BvhPrimitiveCompletePort : public AttribPort<8> {
	public:
		__host__ __device__ BvhPrimitiveCompletePort() {}
		__host__ BvhPrimitiveCompletePort(BvhBvCompletePort bvPort) { _bvPort = bvPort; }
		__host__ __device__ ~BvhPrimitiveCompletePort() {}

		__device__ MCSize& mtcode(int i) { return ((MCSize*)_ddAttrs[MTCODE])[i]; }
		__device__ int& idx(int i) { return ((int*)_ddAttrs[IDX])[i]; }
		__device__ int& vid(int i, int j) { return ((int*)_ddAttrs[VIDA + j])[i]; }
		__device__ int& vida(int i) { return ((int*)_ddAttrs[VIDA])[i]; }
		__device__ int& vidb(int i) { return ((int*)_ddAttrs[VIDB])[i]; }
		__device__ int& vidc(int i) { return ((int*)_ddAttrs[VIDC])[i]; }
		__device__ uint& type(int i) { return ((uint*)_ddAttrs[TYPE])[i]; }
		__device__ uint& extmark(int i) { return ((uint*)_ddAttrs[EXT_MARK])[i]; }
		__device__ int& extno(int i) { return ((int*)_ddAttrs[EXT_NO])[i]; }

		__device__ MCSize getmtcode(int i) const { return ((MCSize*)_ddAttrs[MTCODE])[i]; }
		__device__ int getidx(int i) const { return ((int*)_ddAttrs[IDX])[i]; }
		__device__ int getvida(int i) const { return ((int*)_ddAttrs[VIDA])[i]; }
		__device__ int getvidb(int i) const { return ((int*)_ddAttrs[VIDB])[i]; }
		__device__ int getvidc(int i) const { return ((int*)_ddAttrs[VIDC])[i]; }
		__device__ int getextno(int i) const { return ((int*)_ddAttrs[EXT_NO])[i]; }

		__device__ void setBV(int i, const BOX& bv) { _bvPort.setBV(i, bv); }
		__device__ BOX	getBV(int i) const { return _bvPort.getBV(i); }
		__device__ BvhBvCompletePort getPrimBvs() const { return _bvPort; }
		__device__ BvhBvCompletePort &refPrimBvs() { return _bvPort; }
		__device__ const BvhBvCompletePort &primBvs() const { return _bvPort; }
		__device__ int3 getVids(int i) const { return make_int3(getvida(i), getvidb(i), getvidc(i)); }
	private:
		enum { MTCODE, IDX, VIDA, VIDB, VIDC, TYPE, EXT_MARK, EXT_NO };

		BvhBvCompletePort	_bvPort;
	};

	template<int N>
	struct BvhPrimitivePortType;
	template<>
	struct BvhPrimitivePortType<0> { typedef BvhPrimitiveCompletePort PortType; };
	/**
	* \brief	Connector, two-phase constructor
	* \note	Not CRTP here!
	*/
	class BvhPrimitiveArray : public AttribConnector<8, 1> {
	public:
		BvhPrimitiveArray();
		~BvhPrimitiveArray();
		void	setup(uint primSize);
		void	cleanup();

		template<int N>
		typename BvhPrimitivePortType<N>::PortType portobj() { return *port<N>(); }

		void		gather(int size, const int* gatherPos, BvhPrimitiveArray& to);
		void		scatter(int size, const int* scatterPos, BvhPrimitiveArray& to);

		MCSize*		getMtCodes() { return static_cast<MCSize*>(_attribs[MTCODE]); }
		uint*		getMarks() { return static_cast<uint*>(_attribs[EXT_MARK]); }
		int*		getExtIds() { return static_cast<int*>(_attribs[EXT_NO]); }
		int*		getIdx() { return static_cast<int*>(_attribs[IDX]); }
		BvhBvArray&	getBvArray() { return _bvArray; }
	private:
		typedef enum { MTCODE, IDX, VIDA, VIDB, VIDC, TYPE, EXT_MARK, EXT_NO, NUM_ATTRIBS } EnumBvhPrimAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumBvhPrimPorts;

		void*	&portptr(EnumBvhPrimPorts no);
		template<int N>
		typename BvhPrimitivePortType<N>::PortType* port() { return static_cast<typename BvhPrimitivePortType<N>::PortType*>(_ports[N]); }

		BvhBvArray	_bvArray;
		uint		_primSize;
	};

	__global__ void gatherPrims(int size, const int* gatherPos, BvhPrimitiveCompletePort from, BvhPrimitiveCompletePort to);
	__global__ void scatterPrims(int size, const int* scatterPos, BvhPrimitiveCompletePort from, BvhPrimitiveCompletePort to);
}

#endif