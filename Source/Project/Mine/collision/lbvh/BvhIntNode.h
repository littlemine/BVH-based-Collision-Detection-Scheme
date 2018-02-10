/**	\file	BvhIntNode.h
*	\brief	BVH trunk layer
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVH_INT_NODE_H_
#define __BVH_INT_NODE_H_

#include "base\AggregatedAttribs.h"
#include "BvhBV.h"
#include "BvhExtNode.h"

namespace mn {

	typedef unsigned char uchar;

	/**
	* \brief	Port
	*/
	class BvhIntNodeCompletePort : public AttribPort<9> {
	public:
		__host__ __device__ BvhIntNodeCompletePort() {}
		__host__ BvhIntNodeCompletePort(BvhBvCompletePort bvPort) :
			_bvPort(bvPort) {}
		__host__ __device__ ~BvhIntNodeCompletePort() {}

		__device__ int& lc(int i) { return ((int*)_ddAttrs[LC])[i]; }
		__device__ int& rc(int i) { return ((int*)_ddAttrs[RC])[i]; }
		__device__ int& par(int i) { return ((int*)_ddAttrs[PAR])[i]; }
		__device__ int& rcd(int i) { return ((int*)_ddAttrs[RCD])[i]; }
		__device__ uint& mark(int i) { return ((uint*)_ddAttrs[MARK])[i]; }
		__device__ int& rangex(int i) { return ((int*)_ddAttrs[RANGEX])[i]; }
		__device__ int& rangey(int i) { return ((int*)_ddAttrs[RANGEY])[i]; }
		__device__ uint& flag(int i) { return ((uint*)_ddAttrs[FLAG])[i]; }
		__device__ ExtentType& metric(int i) { return ((ExtentType*)_ddAttrs[QUALITY_METRIC])[i]; }

		__device__ int2 range(int i) const { return make_int2(((int*)_ddAttrs[RANGEX])[i], ((int*)_ddAttrs[RANGEY])[i]); }
		__device__ int getlc(int i) const { return ((int*)_ddAttrs[LC])[i]; }
		__device__ int getrc(int i) const { return ((int*)_ddAttrs[RC])[i]; }
		__device__ int getpar(int i) const { return ((int*)_ddAttrs[PAR])[i]; }
		__device__ int getrcd(int i) const { return ((int*)_ddAttrs[RCD])[i]; }
		__device__ uint getmark(int i) const { return ((uint*)_ddAttrs[MARK])[i]; }
		__device__ int getrangex(int i) const { return ((int*)_ddAttrs[RANGEX])[i]; }
		__device__ int getrangey(int i) const { return ((int*)_ddAttrs[RANGEY])[i]; }
		__device__ ExtentType getmetric(int i) const { return ((ExtentType*)_ddAttrs[QUALITY_METRIC])[i]; }

		__device__ const BvhBvCompletePort& bvs() const { return _bvPort; }

		__device__ void refit(int i) {	///< make sure both lc, rc are int node indices
			int chl = lc(i), chr = rc(i);
			_bvPort.minx(i) = ::fmin(_bvPort.minx(chl), _bvPort.minx(chr));
			_bvPort.miny(i) = ::fmin(_bvPort.miny(chl), _bvPort.miny(chr));
			_bvPort.minz(i) = ::fmin(_bvPort.minz(chl), _bvPort.minz(chr));
			_bvPort.maxx(i) = ::fmax(_bvPort.maxx(chl), _bvPort.maxx(chr));
			_bvPort.maxy(i) = ::fmax(_bvPort.maxy(chl), _bvPort.maxy(chr));
			_bvPort.maxz(i) = ::fmax(_bvPort.maxz(chl), _bvPort.maxz(chr));
		}
		__device__ void refit(int i, BvhExtNodeCompletePort& lvs) {
			BvhBvCompletePort a, b;
			int chl = lc(i), chr = rc(i);
			switch(mark(i) & 3) {
			case 0: a = _bvPort, b = _bvPort; break;
			case 1: a = lvs.getExtBvs(), b = _bvPort; break;
			case 2: a = _bvPort, b = lvs.getExtBvs(); break;
			case 3: a = lvs.getExtBvs(), b = lvs.getExtBvs(); break;
			}
			_bvPort.minx(i) = ::fmin(a.minx(chl), b.minx(chr));
			_bvPort.miny(i) = ::fmin(a.miny(chl), b.miny(chr));
			_bvPort.minz(i) = ::fmin(a.minz(chl), b.minz(chr));
			_bvPort.maxx(i) = ::fmax(a.maxx(chl), b.maxx(chr));
			_bvPort.maxy(i) = ::fmax(a.maxy(chl), b.maxy(chr));
			_bvPort.maxz(i) = ::fmax(a.maxz(chl), b.maxz(chr));
		}
		__device__ void update(int i, BvhExtNodeCompletePort& lvs) {
			BvhBvCompletePort a, b;
			int chl = lc(i), chr = rc(i);
			switch (mark(i) & 3) {
			case 0: a = _bvPort, b = _bvPort; break;
			case 1: a = lvs.getExtBvs(), b = _bvPort; break;
			case 2: a = _bvPort, b = lvs.getExtBvs(); break;
			case 3: a = lvs.getExtBvs(), b = lvs.getExtBvs(); break;
			}
			_bvPort.minx(i) = ::fmin(a.minx(chl), b.minx(chr));
			_bvPort.miny(i) = ::fmin(a.miny(chl), b.miny(chr));
			_bvPort.minz(i) = ::fmin(a.minz(chl), b.minz(chr));
			_bvPort.maxx(i) = ::fmax(a.maxx(chl), b.maxx(chr));
			_bvPort.maxy(i) = ::fmax(a.maxy(chl), b.maxy(chr));
			_bvPort.maxz(i) = ::fmax(a.maxz(chl), b.maxz(chr));
			metric(i) = (a.volume(chl) + b.volume(chr)) / _bvPort.volume(i);
		}
		__device__ float checkUpdate(int i, BvhExtNodeCompletePort& lvs) {
			BvhBvCompletePort a, b;
			int chl = lc(i), chr = rc(i);
			switch (mark(i) & 3) {
			case 0: a = _bvPort, b = _bvPort; break;
			case 1: a = lvs.getExtBvs(), b = _bvPort; break;
			case 2: a = _bvPort, b = lvs.getExtBvs(); break;
			case 3: a = lvs.getExtBvs(), b = lvs.getExtBvs(); break;
			}
			_bvPort.minx(i) = ::fmin(a.minx(chl), b.minx(chr));
			_bvPort.miny(i) = ::fmin(a.miny(chl), b.miny(chr));
			_bvPort.minz(i) = ::fmin(a.minz(chl), b.minz(chr));
			_bvPort.maxx(i) = ::fmax(a.maxx(chl), b.maxx(chr));
			_bvPort.maxy(i) = ::fmax(a.maxy(chl), b.maxy(chr));
			_bvPort.maxz(i) = ::fmax(a.maxz(chl), b.maxz(chr));
			return (a.volume(chl) + b.volume(chr)) / _bvPort.volume(i);
		}

		__device__ bool overlaps(int i, const BOX&b) const { return _bvPort.overlaps(i, b); }
		__device__ void setBV(int i, const BvhIntNodeCompletePort& tks, int j) {
			_bvPort.setBV(i, tks.bvs(), j);
		}

	private:
		enum { LC, RC, PAR, RCD, MARK, RANGEX, RANGEY, FLAG, QUALITY_METRIC };	///< order & metric

		BvhBvCompletePort		_bvPort;
	};

	template<int N>
	struct BvhIntNodePortType;
	template<>
	struct BvhIntNodePortType<0> { typedef BvhIntNodeCompletePort PortType; };
	/**
	* \brief	Connector, two-phase constructor
	* \note	Not CRTP here!
	*/
	class BvhIntNodeArray : public AttribConnector<9, 1> {
	public:
		BvhIntNodeArray();
		~BvhIntNodeArray();
		void	setup(uint intSize);
		void	cleanup();

		template<int N>
		typename BvhIntNodePortType<N>::PortType portobj() { return *port<N>(); }

		void				scatter(int size, const int* scatterPos, BvhIntNodeArray& to);
		void				clear(int size);
		void				clearIntNodes(int size);
		void				clearFlags(int size);
		uint*				getMarks() { return static_cast<uint*>(_attribs[MARK]); }
		int*				getPars() { return static_cast<int*>(_attribs[PAR]); }
		int*				getLbds() { return static_cast<int*>(_attribs[RANGEX]); }
		int*				getRbds() { return static_cast<int*>(_attribs[RANGEY]); }
	private:
		typedef enum { LC, RC, PAR, RCD, MARK, RANGEX, RANGEY, FLAG, QUALITY_METRIC, NUM_ATTRIBS } EnumBvhIntNodeAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumBvhIntNodePorts;

		void*	&portptr(EnumBvhIntNodePorts no);
		template<int N>
		typename BvhIntNodePortType<N>::PortType* port() { return static_cast<typename BvhIntNodePortType<N>::PortType*>(_ports[N]); }

		BvhBvArray			_bvArray;
		uint				_intSize;
	};

	__global__ void scatterIntNodes(int size, const int* scatterPos, BvhIntNodeCompletePort from, BvhIntNodeCompletePort to);
}

#endif