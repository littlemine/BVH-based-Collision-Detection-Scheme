/**	\file	FlOrderLog.h
*	\brief	Base structure
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __FL_ORDER_LOG_H_
#define __FL_ORDER_LOG_H_

#include <base\AggregatedAttribs.h>

namespace mn {

	typedef unsigned int uint;

	/**
	 * \brief	Port
	 */
	class FlOrderCompletePort : public AttribPort<10> {
	public:	
		__host__ __device__ FlOrderCompletePort() {}
		__host__ __device__ ~FlOrderCompletePort() {}

		__device__ uint& intcnt(int i) { return ((uint*)_ddAttrs[INT_FL_NODE_NUM])[i]; }
		__device__ uint* pintcnt(int i) const { return ((uint*)_ddAttrs[INT_FL_NODE_NUM]) + i; }
		__device__ uint& extcnt(int i) { return ((uint*)_ddAttrs[EXT_FL_NODE_NUM])[i]; }
		__device__ uint* pextcnt(int i) const { return ((uint*)_ddAttrs[EXT_FL_NODE_NUM]) + i; }
		__device__ uint& intbeg(int i) { return ((uint*)_ddAttrs[INT_FL_BEG])[i]; }
		__device__ uint& extbeg(int i) { return ((uint*)_ddAttrs[EXT_FL_BEG])[i]; }

		__device__ uint& intbackcnt(int i) { return ((uint*)_ddAttrs[INT_FL_BACK_NUM])[i]; }
		__device__ uint& extbackcnt(int i) { return ((uint*)_ddAttrs[EXT_FL_BACK_NUM])[i]; }
		__device__ uint& intbegbak(int i) { return ((uint*)_ddAttrs[INT_FL_BEG_BAK])[i]; }
		__device__ uint& extbegbak(int i) { return ((uint*)_ddAttrs[EXT_FL_BEG_BAK])[i]; }

		__device__ uint& primCpCnt(int i) { return ((uint*)_ddAttrs[PRIM_CP_NUM])[i]; }
		__device__ uint& primCpOffset(int i) { return ((uint*)_ddAttrs[PRIM_CP_OFFSET])[i]; }
	private:
		enum {
			INT_FL_NODE_NUM, EXT_FL_NODE_NUM, INT_FL_BEG, EXT_FL_BEG,
			INT_FL_BACK_NUM, EXT_FL_BACK_NUM, INT_FL_BEG_BAK, EXT_FL_BEG_BAK, 
			PRIM_CP_NUM, PRIM_CP_OFFSET
		};
	};

	template<int N>
	struct FlOrderLogPortType;
	template<>
	struct FlOrderLogPortType<0> { typedef FlOrderCompletePort PortType; };
	/**
	 * \brief	Connector, two-phase constructor
	 * \note	Not CRTP here!
	 */
	class FlOrderLog : public AttribConnector<10, 1> {
	public:
		FlOrderLog();
		~FlOrderLog();
		void	setup(uint intSize, uint extSize, uint intFtSize, uint extFtSize);
		void	cleanup();

		void	clear(int size);
		void	clearCpCnts(int size);
		void	prepare(int size);
		void	prepareBak(int size);
		void	preserveCnts(int size);

		uint*	intNodeCnts() { return static_cast<uint*>(_attribs[INT_FL_NODE_NUM]); }
		uint*	extNodeCnts() { return static_cast<uint*>(_attribs[EXT_FL_NODE_NUM]); }
		uint*	intBegPos() { return static_cast<uint*>(_attribs[INT_FL_BEG]); }
		uint*	extBegPos() { return static_cast<uint*>(_attribs[EXT_FL_BEG]); }
		uint*	intNodeBackCnts() { return static_cast<uint*>(_attribs[INT_FL_BACK_NUM]); }
		uint*	extNodeBackCnts() { return static_cast<uint*>(_attribs[EXT_FL_BACK_NUM]); }
		uint*	intBegPosBak() { return static_cast<uint*>(_attribs[INT_FL_BEG_BAK]); }
		uint*	extBegPosBak() { return static_cast<uint*>(_attribs[EXT_FL_BEG_BAK]); }
		uint*	primCpNum() { return static_cast<uint*>(_attribs[PRIM_CP_NUM]); }
		uint*	primCpOffset() { return static_cast<uint*>(_attribs[PRIM_CP_OFFSET]); }

		template<int N>
		typename FlOrderLogPortType<N>::PortType portobj() { return *port<N>(); }
	private:
		typedef enum { INT_FL_NODE_NUM, EXT_FL_NODE_NUM, INT_FL_BEG, EXT_FL_BEG, 
			INT_FL_BACK_NUM, EXT_FL_BACK_NUM, INT_FL_BEG_BAK, EXT_FL_BEG_BAK, PRIM_CP_NUM, PRIM_CP_OFFSET, NUM_ATTRIBS } EnumFlOrderAttribs;
		typedef enum { COMPLETE, NUM_PORTS } EnumFlOrderPorts;

		void*	&portptr(EnumFlOrderPorts no);
		template<int N>
		typename FlOrderLogPortType<N>::PortType* port() { return static_cast<typename FlOrderLogPortType<N>::PortType*>(_ports[N]); }

		uint	_intSize, _extSize, _intFtSize, _extFtSize;
	};

	__global__ void filterCnts(uint size, int *_masks, uint *_validCnts, uint *_invalidCnts);
	__global__ void filterIntFrontCnts(uint size, int *_masks, const int* _prevLbds, const int* _leafRestrRoots,
		uint *_validIntCnts, uint *_invalidIntCnts);
	__global__ void filterExtFrontCnts(uint size, int *_masks, const int* _prevLbds, const int* _leafRestrRoots,
		uint *_validExtCnts, uint *_invalidExtCnts, uint *_invalidIntCnts);
}

#endif