#include "LBvhKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "BvhExtNode.h"
#include "BvhIntNode.h"
#include "setting\BvhSettings.h"

namespace mn {

#if MACRO_VERSION
	__global__ void calcMaxBVARCSim(int size, g_box *_bxs, BOX* _bv) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
		const g_box bv = _bxs[idx];
		/// could use aggregate atomic min/max
		atomicMinCustom<ExtentType>(&_bv->_min.x, bv._min.x);
		atomicMinCustom<ExtentType>(&_bv->_min.y, bv._min.y);
		atomicMinCustom<ExtentType>(&_bv->_min.z, bv._min.z);
		atomicMaxCustom<ExtentType>(&_bv->_max.x, bv._max.x);
		atomicMaxCustom<ExtentType>(&_bv->_max.y, bv._max.y);
		atomicMaxCustom<ExtentType>(&_bv->_max.z, bv._max.z);
		//}
	}

	__global__ void calcMCsARCSim(int size, g_box *_bxs, BOX scene, uint* codes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
		const g_box bv = _bxs[idx];
		const PointType c = MakePoint<ExtentType>::p((bv._min.x + bv._max.x) / 2, (bv._min.y + bv._max.y) / 2, (bv._min.z + bv._max.z) / 2);
		const PointType offset = c - scene._min;
		codes[idx] = morton3D(offset.x / scene.width(), offset.y / scene.height(), offset.z / scene.depth());
		//}
	}

	__global__ void buildPrimitivesARCSim(int size, BvhPrimitiveCompletePort _prims, int *_primMap, uint3 *_faces, g_box *_bxs) {	///< update idx-th _bxs to idx-th leaf
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
		const g_box box = _bxs[idx];
		const BOX bv(box._min.x, box._min.y, box._min.z, box._max.x, box._max.y, box._max.z);
		int newIdx = _primMap[idx];
		_prims.vida(newIdx) = _faces[idx].x;
		_prims.vidb(newIdx) = _faces[idx].y;
		_prims.vidc(newIdx) = _faces[idx].z;
		_prims.idx(newIdx) = idx;
		_prims.type(newIdx) = static_cast<uint>(ModelType::FixedDeformableType);
		_prims.setBV(newIdx, bv);
		//}
	}
#endif

	/// could be optimized by warp intrinsics
	__global__ void calcMaxBV(int size, const int3 *_faces, const PointType *_vertices, BOX* _bv) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
			BOX bv{};
			auto v = _vertices[_faces[idx].x];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].y];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].z];
			bv.combines(v.x, v.y, v.z);
			/// could use aggregate atomic min/max
			atomicMinCustom<ExtentType>(&_bv->_min.x, bv._min.x);
			atomicMinCustom<ExtentType>(&_bv->_min.y, bv._min.y);
			atomicMinCustom<ExtentType>(&_bv->_min.z, bv._min.z);
			atomicMaxCustom<ExtentType>(&_bv->_max.x, bv._max.x);
			atomicMaxCustom<ExtentType>(&_bv->_max.y, bv._max.y);
			atomicMaxCustom<ExtentType>(&_bv->_max.z, bv._max.z);
		//}
	}

	__global__ void calcMCs(int size, int3* _faces, PointType* _vertices, BOX scene, uint* codes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
			BOX bv{};
			auto v = _vertices[_faces[idx].x];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].y];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].z];
			bv.combines(v.x, v.y, v.z);
			const PointType c = bv.center();
			const PointType offset = c - scene._min;
			codes[idx] = morton3D(offset.x / scene.width(), offset.y / scene.height(), offset.z / scene.depth());
		//}
	}

	__global__ void calcMC64s(int size, int3* _faces, PointType* _vertices, BOX* scene, uint64* codes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		BOX bv{};
		auto v = _vertices[_faces[idx].x];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].y];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].z];
		bv.combines(v.x, v.y, v.z);
		const PointType c = bv.center();
		const PointType offset = c - scene->_min;
		codes[idx] = morton3D64(offset.x / scene->width(), offset.y / scene->height(), offset.z / scene->depth());
	}

	__global__ void copyBackCodes(int size, uint64* _primcodes, uint* _codes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		_primcodes[idx] = _codes[idx] << 32;
	}

	__global__ void buildPrimitives(int size, BvhPrimitiveCompletePort _prims, int *_primMap, int3 *_faces, PointType *_vertices) {	///< update idx-th _bxs to idx-th leaf
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int newIdx = _primMap[idx];
			BOX bv{};
			auto v = _vertices[_faces[idx].x];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].y];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].z];
			bv.combines(v.x, v.y, v.z);
			//_prims.vida(newIdx) = _faces[idx].x;
			//_prims.vidb(newIdx) = _faces[idx].y;
			//_prims.vidc(newIdx) = _faces[idx].z;
			_prims.idx(newIdx) = idx;
			_prims.type(newIdx) = static_cast<uint>(ModelType::FixedDeformableType);
			_prims.setBV(newIdx, bv);
		//}
	}

	__global__ void buildIntNodes(int size, uint *_depths, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		_lvs.lca(idx) = -1, _depths[idx] = 0;
		int		l = idx - 1, r = idx;	///< (l, r]
		bool	mark;

		if (l >= 0)	mark = _lvs.getmetric(l) < _lvs.getmetric(r);	///< true when right child, false otherwise
		else		mark = false;

		int		cur = mark ? l : r;
		_lvs.par(idx) = cur;
		if (mark)	_tks.rc(cur) = idx, _tks.rangey(cur) = idx, atomicOr(&_tks.mark(cur), 0x00000002), _lvs.mark(idx) = 0x00000007;
		else		_tks.lc(cur) = idx, _tks.rangex(cur) = idx, atomicOr(&_tks.mark(cur), 0x00000001), _lvs.mark(idx) = 0x00000003;

		while (atomicAdd(&_tks.flag(cur), 1) == 1) {
			//_tks.update(cur, _lvs);	/// Update
			_tks.refit(cur, _lvs);	/// Refit
			_tks.mark(cur) &= 0x00000007;

			l = _tks.rangex(cur) - 1, r = _tks.rangey(cur);
			_lvs.lca(l + 1) = cur/*, _tks.rcd(cur) = ++_lvs.rcl(r)*/, _depths[l + 1]++;
			if (l >= 0)	mark = _lvs.metric(l) < _lvs.metric(r);	///< true when right child, false otherwise
			else		mark = false;

			if (l + 1 == 0 && r == size - 1) {
				_tks.par(cur) = -1;
				_tks.mark(cur) &= 0xFFFFFFFB;
				break;
			}

			int par = mark ? l : r;
			_tks.par(cur) = par;
			if (mark)	_tks.rc(par) = cur, _tks.rangey(par) = r, atomicAnd(&_tks.mark(par), 0xFFFFFFFD), _tks.mark(cur) |= 0x00000004;
			else		_tks.lc(par) = cur, _tks.rangex(par) = l + 1, atomicAnd(&_tks.mark(par), 0xFFFFFFFE), _tks.mark(cur) &= 0xFFFFFFFB;
			cur = par;
		}
	}

	__global__ void calcIntNodeOrders(int size, BvhIntNodeCompletePort _tks, int* _lcas, uint* _depths, uint* _offsets, int* _tkMap) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
		int node = _lcas[idx], depth = _depths[idx], id = _offsets[idx];
		if (node != -1) {
			for (; depth--; node = _tks.lc(node)) {
				_tkMap[node] = id++;
			}
		}
		//}
	}

	__global__ void updateBvhExtNodeLinks(int size, const int *_mapTable, int* _lcas, int* _pars) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		int ori;
		_pars[idx] = _mapTable[_pars[idx]];
		if ((ori = _lcas[idx]) != -1)
			_lcas[idx] = _mapTable[ori] << 1;
		else
			_lcas[idx] = idx << 1 | 1;
		//if (_lvs.getrca(idx - (size - 1)) != -1)
		//	_lvs.rca(idx - (size - 1)) = _mapTable[_lvs.getrca(idx - (size - 1))] << 1;
		//else
		//	_lvs.rca(idx - (size - 1)) = idx - (size - 1) << 1 | 1;
	}

	__global__ void reorderIntNode(int intSize, const int* _tkMap, BvhIntNodeCompletePort _unorderedTks, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= intSize) return;
		int newId = _tkMap[idx];
		uint mark = _unorderedTks.getmark(idx);

		_tks.lc(newId) = mark & 1 ? _unorderedTks.getlc(idx) : _tkMap[_unorderedTks.getlc(idx)];
		_tks.rc(newId) = mark & 2 ? _unorderedTks.getrc(idx) : _tkMap[_unorderedTks.getrc(idx)];
		_tks.mark(newId) = mark;
		mark = _unorderedTks.getpar(idx);
		_tks.par(newId) = mark != -1 ? _tkMap[mark] : -1;
		_tks.rangex(newId) = _unorderedTks.getrangex(idx);
		_tks.rangey(newId) = _unorderedTks.getrangey(idx);
		//_tks.rcd(newId) = _rcls[mark] - _unorderedTks.getrcd(idx);
		_tks.setBV(newId, _unorderedTks, idx);
	}

	__global__ void checkRestrTrunkMap(int numRtIntNode, const int* _restrQueue, const int* _tkMap, const int * _restrIntMark, int *_rtIntCount) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numRtIntNode) return;
		int mappedVal = _tkMap[_restrQueue[idx]];
		if (atomicAdd(_rtIntCount + mappedVal, 1) != 0)
			printf("\n\t~~%d-th index(%d) mapped int index(%d) repeated\n", idx, _restrQueue[idx], mappedVal);
		if (_restrIntMark[mappedVal] == 0)
			printf("\n\t~~%d-th index(%d) mapped int index(%d) not requiring restructuring\n", idx, _restrQueue[idx], mappedVal);
	}
	__global__ void checkPrimmap(int size, int* _primmap, int* _cnt) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		int val;
		if ((val = atomicAdd(&_cnt[_primmap[idx]], 1)) != 0)
			printf("%d-th map record(%d) wrong %d\n", idx, _primmap[idx], val);
	}
	__global__ void checkBVHIntegrity(int size, BvhExtNodeCompletePort _leaves, BvhIntNodeCompletePort _trunks, int *tag) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		//if (idx >= size) return;
		if (idx == 0)
			if (_leaves.getlca(size) != -1) {
				printf("wrong tail sentinel.\n");
			}
		for (; idx < size; idx += gridDim.x * blockDim.x) {
			int par = _leaves.par(idx), dep = 1;
			bool right = _leaves.mark(idx) & 4;

			if (right) {
				if (_trunks.rc(par) != idx || (_trunks.mark(par) & 2) == 0 || _trunks.rangey(par) != idx || _leaves.lca(idx) & 1 == 0 || _leaves.lca(idx) / 2 != idx)
					printf("leaf %d(as right child) is wrong. type:%d mark: %u par: %d\n", idx, 
						   _trunks.rc(par) != idx | ((_trunks.mark(par) & 2) == 0) << 1 | (_trunks.rangey(par) != idx) << 2, _leaves.mark(idx), par),
						atomicAdd(tag, 1);
			}
			else {
				if (_trunks.lc(par) != idx || (_trunks.mark(par) & 1) == 0 || _trunks.rangex(par) != idx || _leaves.lca(idx) & 1 == 1 || _trunks.rangex(_leaves.lca(idx) / 2) != idx)
					printf("leaf %d(as left child) is wrong. type:%d mark: %u par: %d\n", idx, 
						   _trunks.lc(par) != idx | ((_trunks.mark(par) & 1) == 0) << 1 | (_trunks.rangex(par) != idx) << 2, _leaves.mark(idx), par),
						atomicAdd(tag, 1);
			}
			//if (idx == 171)
			//	printf("%d-th primitive: mark:%o\n", idx, _leaves.mark(idx));
			while (_trunks.par(par) != -1 && *tag < 30) {
				right = _trunks.mark(par) & 4;
				//if (par + _trunks.rangey(par) - _trunks.rangex(par) - 1 != _leaves.par(_trunks.rangey(par)))
				//	printf("\n\nsubtree %d[%d(%d), %d(%d)] is wrong.\n\n", par,
				//		_trunks.rangex(par), _leaves.par(_trunks.rangex(par)), 
				//		_trunks.rangey(par), _leaves.par(_trunks.rangey(par))), atomicAdd(tag, 1);

				if (right) {
					if (_trunks.rc(_trunks.par(par)) != par || (_trunks.mark(_trunks.par(par)) & 2) == 2 || _trunks.rangey(_trunks.par(par)) != _trunks.rangey(par)
						|| (_trunks.mark(_trunks.par(par)) & 1) == 0 && (_trunks.rangex(par) != _trunks.rangey(_trunks.lc(_trunks.par(par))) + 1)
						|| (_trunks.mark(_trunks.par(par)) & 1) == 1 && (_trunks.rangex(_trunks.par(par)) != _trunks.lc(_trunks.par(par))))
						printf("trunk %d(as right child) %d[%d, %d] %d[%d, %d] -> %d[%d, %d].\n", par,
							   _trunks.lc(_trunks.par(par)), _trunks.rangex(_trunks.lc(_trunks.par(par))), _trunks.rangey(_trunks.lc(_trunks.par(par))),
							   par, _trunks.rangex(par), _trunks.rangey(par),
							   _trunks.par(par), _trunks.rangex(_trunks.par(par)), _trunks.rangey(_trunks.par(par))), atomicAdd(tag, 1);
					break;
				}
				//else {
				if (_trunks.lc(_trunks.par(par)) != par || (_trunks.mark(_trunks.par(par)) & 1) == 1 || _trunks.rangex(_trunks.par(par)) != _trunks.rangex(par)
					|| _trunks.par(par) + 1 != par)
					printf("trunk %d(as left child) is wrong.\n", par), atomicAdd(tag, 1);
				//}

				//if (idx == 171)
				//	printf("%d-th primitive: %d level\t %d-th node [%d, %d]\t mark:%o\n", 
				//		idx, dep, par, _trunks.rangex(par), _trunks.rangey(par), _trunks.mark(par));
				++dep;
				par = _trunks.par(par);
			}
			if (dep >= 32) {
				printf("%d-th primitive depth: %d\n", idx, dep);
			}
		}
	}
	/*	parent computation different
	__global__ void reorderIntNode(int intSize, const int* _tkMap, BvhIntNodeCompletePort _unorderedTks, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= intSize) return;
		int newId = _tkMap[idx];
		uint mark = _unorderedTks.getmark(idx);

		/// left-branch
		if (mark & 1) 	_tks.lc(newId) = _unorderedTks.getlc(idx);
		else			_tks.par(_tks.lc(newId) = _tkMap[_unorderedTks.getlc(idx)]) = newId;
		/// right-branch
		if (mark & 2) 	_tks.rc(newId) = _unorderedTks.getrc(idx);
		else			_tks.par(_tks.rc(newId) = _tkMap[_unorderedTks.getrc(idx)]) = newId;
		/// 
		_tks.mark(newId) = mark;
		_tks.rangex(newId) = _unorderedTks.getrangex(idx);
		_tks.rangey(newId) = _unorderedTks.getrangey(idx);
		//_tks.metric(newId) = _unorderedTks.getmetric(tid);
		_tks.setBV(newId, _unorderedTks, idx);
	}
	*/
}
