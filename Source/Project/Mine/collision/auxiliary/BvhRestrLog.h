/**	\file	BvhRestrLog.h
*	\brief	Base structure
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVH_RESTR_LOG_H_
#define __BVH_RESTR_LOG_H_

#include "base\AggregatedAttribs.h"

namespace mn {

	typedef unsigned int uint;

	/**
	* \brief	Port
	*/
	class BvhRestrCompletePort : public AttribPort<7> {
	public:
		__host__ __device__ BvhRestrCompletePort() {}
		__host__ __device__ ~BvhRestrCompletePort() {}

		__device__ int& extrange(int i) { return ((int*)_ddAttrs[EXT_RANGE])[i]; }
		__device__ int& intrange(int i) { return ((int*)_ddAttrs[INT_RANGE])[i]; }
		__device__ int& extrt(int i) { return ((int*)_ddAttrs[EXT_RESTR])[i]; }
		__device__ int& intrt(int i) { return ((int*)_ddAttrs[INT_RESTR])[i]; }
		__device__ int& primrt(int i) { return ((int*)_ddAttrs[PRIM_RESTR])[i]; }
		__device__ int& rtroot(int i) { return ((int*)_ddAttrs[SUBBVH_ROOT_IDS])[i]; }
		__device__ int& islarge(int i) { return ((int*)_ddAttrs[SUBBVH_OVERSIZE])[i]; }
	private:
		enum { EXT_RANGE, INT_RANGE, EXT_RESTR, INT_RESTR, PRIM_RESTR, SUBBVH_ROOT_IDS, SUBBVH_OVERSIZE } ;
	};

	template<int N>
	struct BvhRestrPortType;
	template<>
	struct BvhRestrPortType<0> { typedef BvhRestrCompletePort PortType; };
	/**
	 * \brief	Connector, two-phase constructor
	 */
	class BvhRestrLog : public AttribConnector<7, 1> {
	public:
	/**
	 * \note	SUBBVH_ROOT_IDS: take ext-index as input
	 * \note	SUBBVH_OVERSIZE: take int-index (restr-subbvh's root) as input
	 */
		typedef enum { EXT_RANGE, INT_RANGE, EXT_RESTR, INT_RESTR, PRIM_RESTR, SUBBVH_ROOT_IDS, SUBBVH_OVERSIZE, NUM_ATTRIBS } EnumBvhRestrAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumBvhRestrPorts;
		BvhRestrLog();
		~BvhRestrLog();
		void	setup(uint extSize, uint primSize);
		void	cleanup();

		void	clear(int extSize);

		void	setUpdateTag(bool tag = true) { _updated = tag; }
		bool	getUpdateTag() const { return _updated; }
		void	setBvhOptTag(int tag = 0) { _bvhopt = tag; }	///< 0: refit 1: restr 2: build
		int 	getBvhOptTag() const { return _bvhopt; }

		int*	getExtRange() { return (int*)_attribs[EXT_RANGE]; }
		int*	getExtMark() { return (int*)_attribs[EXT_RESTR]; }
		int*	getIntRange() { return (int*)_attribs[INT_RANGE]; }
		int*	getIntMark() { return (int*)_attribs[INT_RESTR]; }
		int*	getPrimMark() { return (int*)_attribs[PRIM_RESTR]; }
		int*	getRestrBvhRoot() { return (int*)_attribs[SUBBVH_ROOT_IDS]; }

		template<int N>
		typename BvhRestrPortType<N>::PortType portobj() { return *port<N>(); }

	private:
		void*	&portptr(EnumBvhRestrPorts no);
		template<int N>
		typename BvhRestrPortType<N>::PortType* port() { return static_cast<typename BvhRestrPortType<N>::PortType*>(_ports[N]); }
		
		uint	_extSize, _primSize;
		bool	_updated{ false };	// 
		int		_bvhopt{ 0 };	// 
	};


}

#endif