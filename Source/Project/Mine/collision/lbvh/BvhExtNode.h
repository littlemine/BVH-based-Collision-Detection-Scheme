/**	\file	BvhExtNode.h
*	\brief	BVH leaf layer
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVH_EXT_NODE_H_
#define __BVH_EXT_NODE_H_

#include "base\AggregatedAttribs.h"
#include "BvhPrimitiveNode.h"

namespace mn {

	/**
	* \brief	Port
	*/
	class BvhExtNodeCompletePort : public AttribPort<7> {
	public:
		__host__ __device__ BvhExtNodeCompletePort() {}
		__host__ BvhExtNodeCompletePort(BvhBvCompletePort bvPort, BvhPrimitiveCompletePort primPort) :
			_bvPort(bvPort), _primPort(primPort) {}
		__host__ __device__ ~BvhExtNodeCompletePort() {}

		__device__ int& par(int i) { return ((int*)_ddAttrs[PAR])[i]; }
		__device__ uint& mark(int i) { return ((uint*)_ddAttrs[MARK])[i]; }
		__device__ int& lca(int i) { return ((int*)_ddAttrs[LCA])[i]; }
		__device__ int& rcl(int i) { return ((int*)_ddAttrs[RCL])[i]; }
		__device__ int& stidx(int i) { return ((int*)_ddAttrs[STIDX])[i]; }
		__device__ uint& seglen(int i) { return ((uint*)_ddAttrs[SEGLEN])[i]; }
		__device__ int& metric(int i) { return ((int*)_ddAttrs[SPLIT_METRIC])[i]; }

		__device__ int getstidx(int i) const { return ((int*)_ddAttrs[STIDX])[i]; }
		__device__ MCSize getmtcode(int i) const { return _primPort.getmtcode(getstidx(i)); }
		__device__ int getpar(int i) const { return ((int*)_ddAttrs[PAR])[i]; }
		__device__ uint getmark(int i) const { return ((uint*)_ddAttrs[MARK])[i]; }
		__device__ int getlca(int i) const { return ((int*)_ddAttrs[LCA])[i]; }
		__device__ int getrcl(int i) const { return ((int*)_ddAttrs[RCL])[i]; }
		__device__ uint& getPrimMark(int i) { return _primPort.extmark(i); }
		__device__ int getmetric(int i) const { return ((int*)_ddAttrs[SPLIT_METRIC])[i]; }
		
		__device__ BvhBvCompletePort getExtBvs() const { return _bvPort; }
		__device__ BvhBvCompletePort &refExtBvs() { return _bvPort; }
		__device__ const BvhBvCompletePort &extBvs() const { return _bvPort; }
		__device__ BvhPrimitiveCompletePort getPrimBvs() const { return _primPort.getPrimBvs(); }
		__device__ BvhBvCompletePort &refPrimBvs() { return _primPort.refPrimBvs(); }
		__device__ const BvhBvCompletePort &primBvs() const { return _primPort.primBvs(); }
		__device__ BvhPrimitiveCompletePort getPrimPort() const { return _primPort; }
		__device__ BvhPrimitiveCompletePort &refPrimPort() { return _primPort; }
		__device__ const BvhPrimitiveCompletePort &primPort() const { return _primPort; }

		__device__ BOX getBV(int i) const { return _bvPort.getBV(i); }
		__device__ void setBV(int i, const BOX& bv) { _bvPort.setBV(i, bv); }
		__device__ bool overlaps(int i, const BOX&b) const { return _bvPort.overlaps(i, b); }
		__device__ bool overlaps(int i, int j) const { return _bvPort.overlaps(i, j); }
		__device__ unsigned int examine_overlap(int i, const BOX&b) const { return _bvPort.examine_overlap(i, b); }
	private:
		enum { PAR, MARK, LCA, RCL, STIDX, SEGLEN, SPLIT_METRIC };	///< order & metric

		BvhBvCompletePort			_bvPort;
		BvhPrimitiveCompletePort	_primPort;
	};


	template<int N>
	struct BvhExtNodePortType;
	template<>
	struct BvhExtNodePortType<0> { typedef BvhExtNodeCompletePort PortType; };
	/**
	* \brief	Connector, two-phase constructor
	* \note	Not CRTP here!
	*/
	class BvhExtNodeArray : public AttribConnector<7, 1> {
	public:
		BvhExtNodeArray();
		~BvhExtNodeArray();
		void	setup(uint primSize, uint extSize);
		void	cleanup();

		template<int N>
		typename BvhExtNodePortType<N>::PortType portobj() { return *port<N>(); }
		BvhBvArray&			getBvArray() { return _bvArray; }
		BvhPrimitiveArray&	getPrimitiveArray() { return _primArray; }
		int*				getLcas() { return static_cast<int*>(_attribs[LCA]); }
		int*				getPars() { return static_cast<int*>(_attribs[PAR]); }
		int*				getRcls() { return static_cast<int*>(_attribs[RCL]); }
		int*				getMetrics() { return static_cast<int*>(_attribs[SPLIT_METRIC]); }
		MCSize*				getMtCodes() { return _primArray.getMtCodes(); }

		void				clearExtNodes(int size);
		void				clearExtBvs(int size);
		int					buildExtNodes(int primsize);
		void				calcSplitMetrics(int extsize);
		void				calcRestrSplitMetrics(int extsize, const int* _leafRestrRoots);
	private:
		typedef enum { PAR, MARK, LCA, RCL, STIDX, SEGLEN, SPLIT_METRIC, NUM_ATTRIBS } EnumBvhExtNodeAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumBvhExtNodePorts;

		void*	&portptr(EnumBvhExtNodePorts no);
		template<int N>
		typename BvhExtNodePortType<N>::PortType* port() { return static_cast<typename BvhExtNodePortType<N>::PortType*>(_ports[N]); }

		BvhBvArray			_bvArray;
		BvhPrimitiveArray	_primArray;
		uint				_extSize;
	};

	__global__ void markPrimSplitPos(int size, BvhPrimitiveCompletePort _prims, uint *_mark);
	__global__ void collapsePrimitives(int primsize, BvhExtNodeCompletePort _lvs, int *_extIds);
	__global__ void calcExtNodeSplitMetrics(int extsize, const MCSize *_codes, int *_metrics);
	__global__ void calcExtNodeRestrSplitMetrics(int extsize, const int *_leafRestrRoots, const MCSize *_codes, int *_metrics);

}

#endif