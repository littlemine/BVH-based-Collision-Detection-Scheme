/**	\file	OrderLog.h
*	\brief	Base structure
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __ORDER_LOG_H_
#define __ORDER_LOG_H_

#include "base\AggregatedAttribs.h"

namespace mn {

	typedef unsigned int uint;

	/**
	 * \brief	Port
	 */
	class OrderCompletePort : public AttribPort<2> {
	public:
		__host__ __device__ OrderCompletePort() {}
		__host__ __device__ ~OrderCompletePort() {}

		__device__ uint& cnt(int i) { return ((uint*)_ddAttrs[RECORD_NUM])[i]; }
		__device__ uint* pcnt(int i) const { return ((uint*)_ddAttrs[RECORD_NUM]) + i; }
		__device__ uint& segbeg(int i) { return ((uint*)_ddAttrs[SEGMENT_POS])[i]; }
	private:
		enum { RECORD_NUM, SEGMENT_POS };
	};

	template<int N>
	struct OrderLogPortType;
	template<>
	struct OrderLogPortType<0> { typedef OrderCompletePort PortType; };

	/**
	 * \brief	Connector, two-phase constructor
	 * \note	Not CRTP here!
	 */
	class OrderLog : public AttribConnector<2, 1> {
	public:
		OrderLog();
		~OrderLog();
		void	setup(uint size);
		void	cleanup();

		void	clear(int size);
		void	prepare(int size);
		uint*	recordCnts() { return static_cast<uint*>(_attribs[RECORD_NUM]); }
		uint*	segmentPos() { return static_cast<uint*>(_attribs[SEGMENT_POS]); };

		template<int N>
		typename OrderLogPortType<N>::PortType portobj() { return *port<N>(); }
	private:
		typedef enum { RECORD_NUM, SEGMENT_POS, NUM_ATTRIBS } EnumOrderAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumOrderPorts;

		void*	&portptr(EnumOrderPorts no);
		template<int N>
		typename OrderLogPortType<N>::PortType* port() { return static_cast<typename OrderLogPortType<N>::PortType*>(_ports[N]); }

		uint	_size;
	};

}

#endif