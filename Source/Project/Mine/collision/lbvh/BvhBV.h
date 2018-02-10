/**	\file	BvhBv.h
*	\brief	Bounding box (SoA) class
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVH_BV_H_
#define __BVH_BV_H_

#include "base\Bv.h"
#include "base\AggregatedAttribs.h"
#include "utility\Utils.h"

namespace mn {

	/**
	* \brief	Port
	*/
	class BvhBvCompletePort : public AttribPort<6> {
	public:
		__host__ __device__ BvhBvCompletePort() {}
		__host__ __device__ ~BvhBvCompletePort() {}

		__device__ ExtentType& minx(int i) { return ((ExtentType*)_ddAttrs[MINX])[i]; }
		__device__ ExtentType& miny(int i) { return ((ExtentType*)_ddAttrs[MINY])[i]; }
		__device__ ExtentType& minz(int i) { return ((ExtentType*)_ddAttrs[MINZ])[i]; }
		__device__ ExtentType& maxx(int i) { return ((ExtentType*)_ddAttrs[MAXX])[i]; }
		__device__ ExtentType& maxy(int i) { return ((ExtentType*)_ddAttrs[MAXY])[i]; }
		__device__ ExtentType& maxz(int i) { return ((ExtentType*)_ddAttrs[MAXZ])[i]; }

		__device__ ExtentType getminx(int i) const { return ((ExtentType*)_ddAttrs[MINX])[i]; }
		__device__ ExtentType getminy(int i) const { return ((ExtentType*)_ddAttrs[MINY])[i]; }
		__device__ ExtentType getminz(int i) const { return ((ExtentType*)_ddAttrs[MINZ])[i]; }
		__device__ ExtentType getmaxx(int i) const { return ((ExtentType*)_ddAttrs[MAXX])[i]; }
		__device__ ExtentType getmaxy(int i) const { return ((ExtentType*)_ddAttrs[MAXY])[i]; }
		__device__ ExtentType getmaxz(int i) const { return ((ExtentType*)_ddAttrs[MAXZ])[i]; }

		__device__ ExtentType volume(int i) const {
			return (getmaxx(i) - getminx(i)) * (getmaxy(i) - getminy(i)) * (getmaxz(i) - getminz(i));
		}

		__device__ void setBV(int i, const BvhBvCompletePort& bvs, int j) {
			minx(i) = bvs.getminx(j), miny(i) = bvs.getminy(j), minz(i) = bvs.getminz(j);
			maxx(i) = bvs.getmaxx(j), maxy(i) = bvs.getmaxy(j), maxz(i) = bvs.getmaxz(j);
		}

		__device__ void setBV(int i, const BOX& bv) {
			minx(i) = bv._min.x, miny(i) = bv._min.y, minz(i) = bv._min.z;
			maxx(i) = bv._max.x, maxy(i) = bv._max.y, maxz(i) = bv._max.z;
		}
		__device__ BOX getBV(int i) const {
			return BOX{ getminx(i), getminy(i), getminz(i), getmaxx(i), getmaxy(i), getmaxz(i) };
		}
		__device__ bool overlaps(int i, const BOX&b) const {
			if (b._min.x >getmaxx(i) || b._max.x < getminx(i)) return false;
			if (b._min.y >getmaxy(i) || b._max.y < getminy(i)) return false;
			if (b._min.z >getmaxz(i) || b._max.z < getminz(i)) return false;
			return true;
		}
		__device__ bool overlaps(int i, int j) const {
			if (getminx(j) >getmaxx(i) || getmaxx(j) < getminx(i)) return false;
			if (getminy(j) >getmaxy(i) || getmaxy(j) < getminy(i)) return false;
			if (getminz(j) >getmaxz(i) || getmaxz(j) < getminz(i)) return false;
			return true;
		}
		__device__ bool contains(int i, const PointType &v) const  {
			return v.x <= getmaxx(i) && v.x >= getminx(i) &&
				v.y <= getmaxy(i) && v.y >= getminy(i) &&
				v.z <= getmaxz(i) && v.z >= getminz(i);
		}
		/// examine the max corner cover situation
		__device__  unsigned int examine_overlap(int i, const BOX &b) const  {
			unsigned int mark = getmaxx(i) > b._max.x | ((getmaxy(i) > b._max.y) << 1) | ((getmaxz(i) > b._max.z) << 2);
			return mark;
		}
	private:
		enum { MINX, MINY, MINZ, MAXX, MAXY, MAXZ };
	};

	template<int N>
	struct BvhBvPortType;
	template<>
	struct BvhBvPortType<0> { typedef BvhBvCompletePort PortType; };
	/**
	* \brief	Connector, two-phase constructor
	* \note	Not CRTP here!
	*/
	class BvhBvArray : public AttribConnector<6, 1> {
	public:
		BvhBvArray();
		~BvhBvArray();
		void	setup(uint count);
		void	cleanup();

		template<int N>
		typename BvhBvPortType<N>::PortType portobj() { return *port<N>(); }
		void	gather(int size, const int* gatherPos, BvhBvArray& to);
		void	scatter(int size, const int* scatterPos, BvhBvArray& to);

		void	clear(int size);
	private:
		typedef enum { MINX, MINY, MINZ, MAXX, MAXY, MAXZ, NUM_ATTRIBS } EnumBvhBvAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumBvhBvPorts;

		void*	&portptr(EnumBvhBvPorts no);
		template<int N>
		typename BvhBvPortType<N>::PortType* port() { return static_cast<typename BvhBvPortType<N>::PortType*>(_ports[N]); }

		uint	_count;
	};

	__global__ void gatherBVs(int size, const int* gatherPos, BvhBvCompletePort from, BvhBvCompletePort to);
	__global__ void scatterBVs(int size, const int* scatterPos, BvhBvCompletePort from, BvhBvCompletePort to);

}

#endif