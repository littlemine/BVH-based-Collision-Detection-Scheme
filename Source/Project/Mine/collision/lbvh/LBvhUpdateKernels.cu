#include "LBvhKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "BvhExtNode.h"
#include "BvhIntNode.h"
#include "setting\BvhSettings.h"

namespace mn {

#if MACRO_VERSION
	__global__ void refitExtNodeARCSim(int primsize, BvhExtNodeCompletePort _lvs, int* _primMap, uint3* _faces, g_box* _bxs) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= primsize) return;
		auto &_prims = _lvs.refPrimPort();
		//auto &_extBvs = _lvs.refExtBvs();
		///
		const g_box box = _bxs[idx];
		const BOX bv(box._min.x, box._min.y, box._min.z, box._max.x, box._max.y, box._max.z);
		int newIdx = _primMap[idx];
		//int extIdx = _prims.getextno(newIdx) - 1;
		_prims.setBV(newIdx, bv);
		_lvs.setBV(newIdx, bv);
		//_extBvs.setBV(extIdx, bv);	///< since it's one-to-one
		/*
		atomicMinD(&_extBvs.minx(extIdx), bv._min.x);
		atomicMinD(&_extBvs.miny(extIdx), bv._min.y);
		atomicMinD(&_extBvs.minz(extIdx), bv._min.z);
		atomicMaxD(&_extBvs.maxx(extIdx), bv._max.x);
		atomicMaxD(&_extBvs.maxy(extIdx), bv._max.y);
		atomicMaxD(&_extBvs.maxz(extIdx), bv._max.z);
		*/
	}
#endif

	__global__ void refitExtNode(int primsize, BvhExtNodeCompletePort _lvs, int* _primMap, int3* _faces, PointType* _vertices) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= primsize) return;
		auto &_prims = _lvs.refPrimPort();
		//auto &_extBvs = _lvs.refExtBvs();
		BOX bv{};
		auto v = _vertices[_faces[idx].x];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].y];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].z];
		bv.combines(v.x, v.y, v.z);

		int newIdx = _primMap[idx];
		//int extIdx = _prims.getextno(newIdx) - 1;
		_prims.setBV(newIdx, bv);
		_lvs.setBV(newIdx, bv);
		//_extBvs.setBV(extIdx, bv);	///< since it's one-to-one
		/*
		atomicMinD(&_extBvs.minx(extIdx), bv._min.x);
		atomicMinD(&_extBvs.miny(extIdx), bv._min.y);
		atomicMinD(&_extBvs.minz(extIdx), bv._min.z);
		atomicMaxD(&_extBvs.maxx(extIdx), bv._max.x);
		atomicMaxD(&_extBvs.maxy(extIdx), bv._max.y);
		atomicMaxD(&_extBvs.maxz(extIdx), bv._max.z);
		*/
	}

	__global__ void refitIntNode(int size, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int par = _lvs.getpar(idx);
			while (atomicAdd(&_tks.flag(par), 1) == 1) {
				_tks.refit(par, _lvs);
				if (par == 0) break;
				par = _tks.getpar(par);
			}
		//}
	}

	__global__ void refitIntNodeLayer(int numLeaf, int *_leafPars, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		for (; idx < numLeaf; idx += gridDim.x * blockDim.x) {
			int par = _leafPars[idx];
			while (atomicAdd(&_tks.flag(par), 1) == 1) {
				_tks.refit(par);
				par = _tks.par(par);
				if (par == -1) break;
			}
		}
	}

	/// deprecated. proved useless
	__global__ void updateIntNode(int size, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks, //FlOrderCompletePort _log,
		int* _lvMarks, int* _tkMarks, int* _restrLcas) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		for (; idx < size; idx += gridDim.x * blockDim.x) {
			int		par = _lvs.par(idx), l, r;
			int		markNode = -1;

			while (atomicAdd(&_tks.flag(par), 1) == 1) {
				float ratio = _tks.checkUpdate(par, _lvs);
				/// check mark degenerated sub-BVH
				/// only mark before ascending from right branch or reaches the root
				if ((_tks.mark(par) & 4 || par == 0) && markNode != -1) {
					_tks.metric(markNode) = ratio;
					l = _tks.rangex(markNode), r = _tks.rangey(markNode);
					_restrLcas[l] = markNode;
					/// mark leaves
					atomicAdd(_lvMarks + l, 1);
					atomicAdd(_lvMarks + r + 1, -1);
					/// mark trunks
					r = markNode + (r - l) - 1;
					l = markNode;
					atomicAdd(_tkMarks + l, 1);
					atomicAdd(_tkMarks + r + 1, -1);

					markNode = -1;
				}
				/// nothing to restructure yet, check the current node
				else {
					l = _tks.rangex(par), r = _tks.rangey(par);
					if (r - l >= SINGLE_SUBTREE_RESTR_SIZE_LOWER_THRESHOLD && 
						r - l < SINGLE_SUBTREE_RESTR_SIZE_UPPER_THRESHOLD &&
						ratio > RESTR_BASE_QUALITY_METRIC &&	///< considered degenerated
						ratio > _tks.getmetric(par) * RESTR_WARNING_QUALITY_METRIC)	///< degenerated enough to restructure it
						markNode = par;
				}
				if (ratio < RESTR_BASE_QUALITY_METRIC)
					_tks.metric(par) = ratio;
				par = _tks.par(par);
				if (par == -1) break;
			}
		}
	}

}
