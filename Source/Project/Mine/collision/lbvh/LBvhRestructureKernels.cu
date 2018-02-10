#include "LBvhKernels.cuh"
#include <cuda_runtime.h>
#include "utility\CudaDeviceUtils.h"
#include "BvhExtNode.h"
#include "BvhIntNode.h"

namespace mn {

	__global__ void calibrateLeafRangeMarks(int size, BvhIntNodeCompletePort _tks, const int* _leafRestrRoots, const int* _intRestrMarks, int* _leafRangeMarks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		int subrt = _leafRestrRoots[idx];
		if (subrt != INT_MAX && _intRestrMarks[subrt] == 1) {	///< second condition ensures picking the largest covering subtree
			atomicAdd(_leafRangeMarks + idx, 1);
			atomicAdd(_leafRangeMarks + _tks.getrangey(subrt) + 1, -1);
		}
	}

	__global__ void calibrateRestrRoots(int size, BvhIntNodeCompletePort _tks, const int* _leafRestrRoots, const int* _intRestrMarks,
		int* _leafRestrRootMarks, int* _numSubtree, uint* _subtreeSizes, int* _subtrees, int* _numRtIntNode) {	/// count number of affected int nodes
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		int root = _leafRestrRoots[idx];
		if (root != INT_MAX && _intRestrMarks[root] == 1) {
			//printf("\tadding %d-th [%d, %d] int node need restructure\n", root, _tks.rangex(root), _tks.rangey(root));
			int2 range{ _tks.getrangex(root), _tks.getrangey(root) };
			atomicAdd(_leafRestrRootMarks + idx, root);
			atomicAdd(_leafRestrRootMarks + range.y + 1, -root);
			atomicAdd(_numRtIntNode, range.y - range.x);

			int id = atomicAdd(_numSubtree, 1);
			_subtrees[id] = root;
			_subtreeSizes[id] = range.y - range.x + 1;
		}
	}

	__global__ void calcRestrMCs(int size, const int3* _faces, const PointType* _vertices, BOX scene, const int* _primRestrMarks, const int* _primmap, uint* codes) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) return;
		//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int pid = _primmap[idx];
			if (_primRestrMarks[pid]) {
				BOX bv{};
				auto v = _vertices[_faces[idx].x];
				bv.combines(v.x, v.y, v.z);
				v = _vertices[_faces[idx].y];
				bv.combines(v.x, v.y, v.z);
				v = _vertices[_faces[idx].z];
				bv.combines(v.x, v.y, v.z);
				const auto c = bv.center();
				const auto offset = c - scene._min;
				codes[pid] = morton3D(offset.x / scene.width(), offset.y / scene.height(), offset.z / scene.depth());
			}
		//}
	}

	__global__ void selectPrimitives(int primsize, const int* _leafRestrRoots, const int* _gatherMap, const MCSize* _mtcodes, uint64* _finalKeys, int* _scatterMap) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= primsize) return;
		if (_leafRestrRoots[idx] > 0) {
			int gatherPos = _gatherMap[idx];
			/// idx -> cptPos
			_finalKeys[gatherPos] = ((uint64)_leafRestrRoots[idx] << 32) | _mtcodes[idx];
			_scatterMap[gatherPos] = idx;	///< going to be sorted soon
		}
	}

	__global__ void updatePrimMap(int restrPrimNum, int* _primIds, int* _newPrimIds, int* _primToIdx, int* _primmap) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= restrPrimNum) return;
		int val;
		_primmap[val = _primToIdx[_primIds[idx]]] = _newPrimIds[idx];
		_primToIdx[_newPrimIds[idx]] = val;
		//_primmap[_primToIdx[_primIds[idx]]] = _primIds[idx];
		//_primmap[_primIds[idx]] = _primIds[idx];
		//_primToIdx[_primIds[idx]] = _primIds[idx];
	}

	__global__ void updatePrimAndExtNode(int primsize, const int *_primRestrMarks, const int * _primMap, const int3* _faces, 
		const PointType * _vertices, const BOX * scene, BvhExtNodeCompletePort _lvs) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= primsize) return;
		auto &_prims = _lvs.refPrimPort();
		//auto _prims = _lvs.getPrimPort();
		int pid = _primMap[idx];
		BOX bv{};
		auto v = _vertices[_faces[idx].x];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].y];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].z];
		bv.combines(v.x, v.y, v.z);
		// primitive layer

		_prims.setBV(pid, bv);
		// ext node layer
		//int extId = _prims.extno(idx) - 1;
		//const auto &primBvs = _lvs.primBvs();
		//auto &extBvs = _lvs.refExtBvs();	// issues

		_lvs.setBV(pid, bv);
		//atomicMinD(&extBvs.minx(extId), primBvs.getminx(idx));
		//atomicMinD(&extBvs.miny(extId), primBvs.getminy(idx));
		//atomicMinD(&extBvs.minz(extId), primBvs.getminz(idx));
		//atomicMaxD(&extBvs.maxx(extId), primBvs.getmaxx(idx));
		//atomicMaxD(&extBvs.maxy(extId), primBvs.getmaxy(idx));
		//atomicMaxD(&extBvs.maxz(extId), primBvs.getmaxz(idx));
		// restr primitive
		if (_primRestrMarks[pid]) {
			_prims.vida(pid) = _faces[idx].x;
			_prims.vidb(pid) = _faces[idx].y;
			_prims.vidc(pid) = _faces[idx].z;
			//_prims.idx(pid) = idx;
			//_prims.type(pid) = static_cast<uchar>(ModelType::FixedDeformableType);

			const auto c = bv.center();
			const auto offset = c - scene->_min;
			_prims.mtcode(pid) = morton3D(offset.x / scene->width(), offset.y / scene->height(), offset.z / scene->depth());
		}
	}

	__global__ void restrIntNodes(int extSize, int numRtExtNode, const int *_restrExtNodes, const uint *_prevTkMarks,
		const int *_leafRestrRoots, uint *_depths, int *_localLcas, BvhExtNodeCompletePort _lvs, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numRtExtNode) return;
		idx = _restrExtNodes[idx];

		_localLcas[idx] = -1; //, _lvs.rcl(idx) = 0;
		const int subtreeRoot = _leafRestrRoots[idx];
		if (subtreeRoot == 0) return;	///< actually not necessary, skipped from the beginning
		int		l = idx - 1, r = idx;	///< (l, r]
		bool	mark;

		char bdmark = (l < 0 || _leafRestrRoots[l] != subtreeRoot) << 1 | (r == extSize - 1 || _leafRestrRoots[r + 1] != subtreeRoot);
		if (bdmark)	mark = bdmark & 1;
		else 		mark = _lvs.getmetric(l) < _lvs.getmetric(r);	///< true when right child, false otherwise

		int		cur = mark ? l : r;
		_lvs.par(idx) = cur;

		if (mark)	_tks.rc(cur) = idx, _tks.rangey(cur) = idx, atomicOr(&_tks.mark(cur), 0x00000002) , _lvs.mark(idx) = 0x00000007;
		else		_tks.lc(cur) = idx, _tks.rangex(cur) = idx, atomicOr(&_tks.mark(cur), 0x00000001) , _lvs.mark(idx) = 0x00000003;
		while (atomicAdd(&_tks.flag(cur), 1) == 1) {
			_tks.refit(cur, _lvs);
			_tks.mark(cur) &= 0x00000007;

			l = _tks.rangex(cur) - 1, r = _tks.rangey(cur);
			_localLcas[l + 1] = cur, _depths[l + 1]++;
			bdmark = (l < 0 || _leafRestrRoots[l] != subtreeRoot) << 1 | (r == extSize - 1 || _leafRestrRoots[r + 1] != subtreeRoot);
			if (bdmark)	mark = bdmark & 1;
			else 		mark = _lvs.getmetric(l) < _lvs.getmetric(r);

			if (bdmark == 3) {
				/// relationship with father
				if (_prevTkMarks[subtreeRoot] & 4) _tks.mark(cur) |= 0x00000004;
				else _tks.mark(cur) &= 0xFFFFFFFB;
				_tks.par(cur) = -1;	///< sentinel mark, no need modify mapping
				break;
			}
			int par = mark ? l : r;
			_tks.par(cur) = par;
			if (mark)	_tks.rc(par) = cur, _tks.rangey(par) = r, atomicAnd(&_tks.mark(par), 0xFFFFFFFD), _tks.mark(cur) |= 0x00000004;
			else		_tks.lc(par) = cur, _tks.rangex(par) = l + 1, atomicAnd(&_tks.mark(par), 0xFFFFFFFE), _tks.mark(cur) &= 0xFFFFFFFB;
			cur = par;
		}
	}

	__global__ void calcRestrIntNodeOrders(int numRtExtNode, const int *_restrExtNodes, const uint *_depths, const uint *_offsets, const int *_subtreeRoots,
		const int* _prevLbds, const uint *_prevTkMarks, const int* _localLcas, int *_lcas, int *_pars, BvhIntNodeCompletePort _unorderedTks, int *_tkMap, int *_sequence) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numRtExtNode) return;
		idx = _restrExtNodes[idx];	///< leaf (ext node) index
		int node = _localLcas[idx];	///< index of the unsorted trunk
		if (node != -1) {
			int depth = _subtreeRoots[idx], pos = _offsets[idx];	///< first work as the root of the restr subtree
																	/// for now, the rangex of the original subtree root is that of the newly built _unorderedTks
			int newId = depth + pos - _offsets[_prevLbds[depth]];
			//if (_lcas[idx] != depth << 1)
			//if (!(_prevTkMarks[depth] & 4 == 0 && _prevLbds[_lcas[idx] >> 1] == _prevLbds[newId]))
			if (_prevTkMarks[depth] & 4 || idx != _prevLbds[depth])
				_lcas[idx] = newId << 1;

			for (depth = _depths[idx]; depth--; node = _unorderedTks.getlc(node)) {
				if (_unorderedTks.getmark(node) & 2) {
					_pars[_unorderedTks.getrc(node)] = newId;
				}
				_tkMap[node] = newId++;
				_sequence[pos++] = node;
			}
			_pars[idx] = newId - 1;
		}
		else {
			_lcas[idx] = idx << 1 | 1;
		}
	}

	__global__ void reorderRestrIntNodes(int numRtIntNode, const int *_restrIntNodes, const int *_tkMap,
		BvhIntNodeCompletePort _unorderedTks, BvhIntNodeCompletePort _tks) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numRtIntNode) return;
		
		idx = _restrIntNodes[idx];
		int newId = _tkMap[idx];
		uint mark = _unorderedTks.getmark(idx);

		_tks.mark(newId) = mark;
		_tks.lc(newId) = mark & 1 ? _unorderedTks.getlc(idx) : _tkMap[_unorderedTks.getlc(idx)];
		_tks.rc(newId) = mark & 2 ? _unorderedTks.getrc(idx) : _tkMap[_unorderedTks.getrc(idx)];
		if ((mark = _unorderedTks.getpar(idx)) != -1)
			_tks.par(newId) = _tkMap[mark];
		_tks.rangex(newId) = _unorderedTks.getrangex(idx);
		_tks.rangey(newId) = _unorderedTks.getrangey(idx);
		//_tks.rcd(newId) = _rcls[mark] - _unorderedTks.getrcd(idx);
		_tks.setBV(newId, _unorderedTks, idx);
	}

}