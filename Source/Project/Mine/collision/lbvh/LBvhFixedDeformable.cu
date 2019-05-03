#include "LBvhFixedDeformable.h"
#include <cstdio>
#include <thrust\sort.h>
#include <thrust\count.h>
#include <thrust\sequence.h>
#include <thrust\execution_policy.h>

#include "utility\CudaThrustUtils.hpp"
#include "utility\CudaKernelUtils.cuh"
#include "system\CudaDevice\CudaDevice.h"
#include "system\CudaDevice\CudaKernelLauncher.cu"

#include "utility\CudaDeviceUtils.h"
#include "utility\CudaThrustUtils.hpp"
#include "setting\CDBenchmarkSettings.h"

namespace mn {

	/// Definitions
	LBvhFixedDeformable::LBvhFixedDeformable(LBvhBuildConfig&& config) {
		TheCudaDevice = CudaDevice::getInstance();
		_bvh.setup(config);
		///
		_unsortedPrims.setup(config.primSize);
		_unsortedTks.setup(config.intSize);
		_restrLog.setup(config.extSize, config.primSize);
		_restrLog.setBvhOptTag(false);

		checkCudaErrors(cudaMalloc((void**)&d_vertices, sizeof(PointType)*config.primSize*3));
		checkCudaErrors(cudaMalloc((void**)&d_faces, sizeof(int3)*config.primSize));
		checkCudaErrors(cudaMalloc((void**)&d_numRtSubtree, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_numRtIntNode, sizeof(int)));

		checkThrustErrors(
		d_primMap.resize(config.primSize);
		d_tkMap.resize(config.intSize);
		d_count.resize(config.primSize);
		d_offsetTable.resize(config.primSize);

		d_keys32.resize(config.primSize);
		d_keys64.resize(config.primSize);
		d_vals.resize(config.primSize);

		/// the following only used in the restructuring scheme
		//d_prevLbds.resize(config.intSize);
		//d_prevRbds.resize(config.intSize);
		//d_gatherMap.resize(config.primSize);
		//d_taskSequence.resize(config.primSize);	///< task sequence for primitives, external nodes
		//d_sequence.resize(config.primSize);
		//d_rtSubtrees.resize(config.intSize << 1);	///< stores root nodes of subtree
		//d_sizePerSubtree.resize(config.intSize << 1);
		//d_begPerSubtree.resize(config.intSize << 1);
		);
		reportMemory();
	}

	LBvhFixedDeformable::~LBvhFixedDeformable() {
		checkCudaErrors(cudaFree(d_numRtIntNode));
		checkCudaErrors(cudaFree(d_numRtSubtree));
		checkCudaErrors(cudaFree(d_faces));
		checkCudaErrors(cudaFree(d_vertices));
		_restrLog.cleanup();
		_unsortedTks.cleanup();
		_unsortedPrims.cleanup();
		_bvh.cleanup();
	}

	void LBvhFixedDeformable::maintain(LBvhFixedDeformableMaintenance scheme, const SceneData& pdata) {
		/// 0: rebuild 1: refit 2: update
		updatePrimData(pdata);

		_restrLog.setBvhOptTag(0);
		if (CDBenchmarkSettings::enableRestr()) {
			if (logUpdated()) {
				restructure();
				//build();
				return;
			}
		}

		switch (scheme) {
		case LBvhFixedDeformableMaintenance::BUILD: build(); /*checkBvhValidity();*/ break;
		case LBvhFixedDeformableMaintenance::REFIT: refit(); /*checkBvhValidity();*/ break;
		case LBvhFixedDeformableMaintenance::UPDATE: update(); /*restructure(); checkBvhValidity();*/ break;
		default: break;
		}
	}

#if MACRO_VERSION
	void LBvhFixedDeformable::maintain(LBvhFixedDeformableMaintenance scheme, const ARCSimSceneData& pdata) {
		cbvh().primSize() = pdata.fsize;
		d_facesARCSim = pdata.faces;
		d_bxsARCSim = pdata.boxes;
		///
		_restrLog.setBvhOptTag(0);
		if (CDBenchmarkSettings::enableRestr()) {
			if (logUpdated()) {
				restructure();
				//build();
				return;	///< automatic measure taking over according to front status
			}
		}

		printf("begin lbvh maintain\n");
		switch (scheme) {
		case LBvhFixedDeformableMaintenance::BUILD: build(); /*checkBvhValidity();*/ break;
		case LBvhFixedDeformableMaintenance::REFIT: refit(); /*checkBvhValidity();*/ break;
		case LBvhFixedDeformableMaintenance::UPDATE_RESTR: update(); /*restructure(); checkBvhValidity();*/ break;
		case LBvhFixedDeformableMaintenance::UPDATE_BUILD: update(); /*build(); checkBvhValidity();*/ break;
		default: break;
		}
	}
#endif

	void LBvhFixedDeformable::updatePrimData(const SceneData& pdata) {
		cbvh().primSize() = pdata.fids.size();
		checkCudaErrors(cudaMemcpy(d_faces, pdata.fids.data(), sizeof(int3)*pdata.fids.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vertices, pdata.pos.data(), sizeof(PointType)*pdata.pos.size(), cudaMemcpyHostToDevice));
	}

	void LBvhFixedDeformable::build() {
		/// calculate scene bounding box
		BOX	bv{};
		checkCudaErrors(cudaMemcpy(cbvh().bv(), &bv, sizeof(BOX), cudaMemcpyHostToDevice));
#if MACRO_VERSION
			configuredLaunch({ "CalcBVARCSim", cbvh().primSize() }, calcMaxBVARCSim,
				cbvh().primSize(), d_bxsARCSim, cbvh().bv());
#else
			configuredLaunch({ "CalcBV", cbvh().primSize() }, calcMaxBV,
				cbvh().primSize(), (const int3*)d_faces, (const PointType*)d_vertices, cbvh().bv());
#endif
		checkCudaErrors(cudaMemcpy(&bv, cbvh().bv(), sizeof(BOX), cudaMemcpyDeviceToHost));

		//Logger::tick<TimerType::CPU>();
		//_mortonCoder.configureScene(bv._min, bv._max);
		//Logger::tock<TimerType::CPU>("ConfigureCoder");
		//configuredLaunch({ "CalcEMCs", cbvh().primSize() }, calcEMCs,
		//	cbvh().primSize(), d_faces, d_vertices, _mortonCoder.getCoder(), getRawPtr(d_keys64));
#if MACRO_VERSION
			configuredLaunch({ "CalcMCsARCSim", cbvh().primSize() }, calcMCsARCSim,
				cbvh().primSize(), d_bxsARCSim, bv, getRawPtr(d_keys32));
#else
			configuredLaunch({ "CalcMCs", cbvh().primSize() }, calcMCs,
				cbvh().primSize(), d_faces, d_vertices, bv, getRawPtr(d_keys32));
#endif
		//configuredLaunch({ "CalcMC64s", cbvh().primSize() }, calcMC64s,
		//	cbvh().primSize(), d_faces, d_vertices, cbvh().bv(), getRawPtr(d_keys64));

		reorderPrims();

		/// build primitives
#if MACRO_VERSION
			configuredLaunch({ "BuildPrimsARCSim", cbvh().primSize() }, buildPrimitivesARCSim,
				cbvh().primSize(), cbvh().lvs().getPrimitiveArray().portobj<0>(), getRawPtr(d_primMap), d_facesARCSim, d_bxsARCSim);
#else
			configuredLaunch({ "BuildPrims", cbvh().primSize() }, buildPrimitives,
				cbvh().primSize(), cbvh().lvs().getPrimitiveArray().portobj<0>(), getRawPtr(d_primMap), d_faces, d_vertices);
#endif

		/// build external nodes
		cbvh().intSize() = (cbvh().extSize() = cbvh().lvs().buildExtNodes(cbvh().primSize())) - 1;
		cbvh().lvs().calcSplitMetrics(cbvh().extSize());
		/// build internal nodes
		_unsortedTks.clearIntNodes(cbvh().intSize());
		configuredLaunch({ "BuildIntNodes", cbvh().extSize() }, buildIntNodes,
			cbvh().extSize(), getRawPtr(d_count), cbvh().lvs().portobj<0>(), _unsortedTks.portobj<0>());

Logger::recordSection<TimerType::GPU>("construct_bvh");

		/// first correct indices, then sort
		reorderIntNodes();

Logger::recordSection<TimerType::GPU>("sort_bvh");

		//_unsortedTks.scatter(cbvh().intSize(), getRawPtr(d_tkMap), cbvh().tks());

		printf("Primsize: %d Extsize: %d\n", cbvh().primSize(), cbvh().extSize());
	}

	void LBvhFixedDeformable::refit() {

		cbvh().lvs().clearExtBvs(cbvh().extSize());
#if MACRO_VERSION{
			configuredLaunch({ "RefitExtNodeARCSim", cbvh().primSize() }, refitExtNodeARCSim,
				cbvh().primSize(), cbvh().lvs().portobj<0>(), getRawPtr(d_primMap), d_facesARCSim, d_bxsARCSim);
#else
			configuredLaunch({ "RefitExtNode", cbvh().primSize() }, refitExtNode,
				cbvh().primSize(), cbvh().lvs().portobj<0>(), getRawPtr(d_primMap), d_faces, d_vertices);
#endif

		cbvh().tks().clearIntNodes(cbvh().intSize());
		configuredLaunch({ "RefitIntNode", cbvh().extSize() }, refitIntNode,
			cbvh().extSize(), cbvh().lvs().portobj<0>(), cbvh().tks().portobj<0>());
		
Logger::recordSection<TimerType::GPU>("refit_bvh");

	}

	void LBvhFixedDeformable::update() {
		cbvh().lvs().clearExtBvs(cbvh().extSize());
		configuredLaunch({ "RefitExtNode", cbvh().primSize() }, refitExtNode,
			cbvh().primSize(), cbvh().lvs().portobj<0>(), getRawPtr(d_primMap), d_faces, d_vertices);

		cbvh().tks().clearIntNodes(cbvh().intSize());
		configuredLaunch({ "UpdateIntNode", cbvh().extSize() }, updateIntNode,
			cbvh().extSize(), cbvh().lvs().portobj<0>(), cbvh().tks().portobj<0>(),
			_restrLog.getExtRange(), _restrLog.getIntRange(), _restrLog.getRestrBvhRoot());
	}

	bool LBvhFixedDeformable::restructure() {
		static bool lastRestr = false;

		// 0 preliminary mark restr root(done in Front::checkQuality)
		// 1 calc int node marks
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::inclusive_scan(getDevicePtr(_restrLog.getIntRange()), getDevicePtr(_restrLog.getIntRange()) + cbvh().intSize(), getDevicePtr(_restrLog.getIntMark())));
		Logger::tock<TimerType::GPU>("CalcIntNodeRestrMarks");

		// 2 preliminary mark ext nodes
		checkThrustErrors(thrust::fill(getDevicePtr(_restrLog.getExtRange()), getDevicePtr(_restrLog.getExtRange()) + cbvh().extSize() + 1, 0));
		configuredLaunch({ "CalibrateLeafRangeMarks", cbvh().extSize() }, calibrateLeafRangeMarks,
			cbvh().extSize(), cbvh().tks().portobj<0>(), (const int*)_restrLog.getRestrBvhRoot(), (const int*)_restrLog.getIntMark(), _restrLog.getExtRange());

		// 3 calc ext node marks
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::inclusive_scan(getDevicePtr(_restrLog.getExtRange()), getDevicePtr(_restrLog.getExtRange()) + cbvh().extSize() + 1, getDevicePtr(_restrLog.getExtMark())));
		Logger::tock<TimerType::GPU>("CalcExtNodeRestrMarks");

		// 4 mark restr root again, eliminate redundant marks
		checkThrustErrors(thrust::fill(getDevicePtr(_restrLog.getExtRange()), getDevicePtr(_restrLog.getExtRange()) + cbvh().extSize() + 1, 0));
		checkCudaErrors(cudaMemset(d_numRtSubtree, 0, sizeof(int)));
		checkCudaErrors(cudaMemset(d_numRtIntNode, 0, sizeof(int)));
		configuredLaunch({ "CalibrateRestrRoots", cbvh().extSize() }, calibrateRestrRoots,
			cbvh().extSize(), cbvh().tks().portobj<0>(), (const int*)_restrLog.getRestrBvhRoot(), (const int*)_restrLog.getIntMark(), _restrLog.getExtRange(),
			d_numRtSubtree, getRawPtr(d_sizePerSubtree), getRawPtr(d_rtSubtrees), d_numRtIntNode);

Logger::recordSection<TimerType::GPU>("check_bvh_restr");

		// check the extent of the degeneration according to the number of subtree
		checkCudaErrors(cudaMemcpy(&_numRtSubtree, d_numRtSubtree, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&_numRtIntNode, d_numRtIntNode, sizeof(int), cudaMemcpyDeviceToHost));

		std::string	str;
		/// restr threshold is user defined
		int			opt = _numRtIntNode > (cbvh().intSize() >> 1);// || _numRtSubtree > (cbvh().intSize() >> 3);
		if (opt) { ///< restructure couldn't handle this much degeneration efficiently
			str = string_format("%d subtrees and %d (%d) internal nodes need restructuring. build\n", _numRtSubtree, _numRtIntNode, cbvh().intSize());
			std::cout << str << '\n';
			Logger::message(str);

			build();
			// should notify the fronts to be reconstructed
			_restrLog.setBvhOptTag(2);
			lastRestr = false;
			return false;
		}
		opt = _numRtIntNode < (cbvh().intSize() >> 3);	/// could consider the quantity of related front nodes
		if (opt) {
			str = string_format("%d subtrees and %d (%d) internal nodes need restructuring. refit\n", _numRtSubtree, _numRtIntNode, cbvh().intSize());
			std::cout << str << '\n';
			Logger::message(str);

			refit();
			_restrLog.setBvhOptTag(0);
			lastRestr = false;
			return false;
		}

		checkCudaErrors(cudaMemcpy(getRawPtr(d_prevLbds), ctks().getLbds(), sizeof(int) * cbvh().intSize(), cudaMemcpyDeviceToDevice));

		// 5 calc restr roots
		Logger::tick<TimerType::GPU>();
		// plus 1 here convenient for split metric calculation
		checkThrustErrors(thrust::inclusive_scan(getDevicePtr(_restrLog.getExtRange()), getDevicePtr(_restrLog.getExtRange()) + cbvh().extSize() + 1, getDevicePtr(_restrLog.getRestrBvhRoot())));
		Logger::tock<TimerType::GPU>("CalcExtNodeRestrMarks");

		/// reordering restr primitives
		_numRtExtNode = _numRtPrim = _numRtIntNode + _numRtSubtree;
		// 6 calc primitive marks(simplified version, since prim and ext node map one to one)
		checkCudaErrors(cudaMemcpy(_restrLog.getPrimMark(), _restrLog.getExtMark(), sizeof(int) * cbvh().primSize(), cudaMemcpyDeviceToDevice));
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr(_restrLog.getPrimMark()), getDevicePtr(_restrLog.getPrimMark()) + cbvh().primSize() + 1, getDevicePtr(d_gatherMap)));
		Logger::tock<TimerType::GPU>("CalcPrimitiveGatherMap");

		BOX	bv{};
		checkCudaErrors(cudaMemcpy(cbvh().bv(), &bv, sizeof(BOX), cudaMemcpyHostToDevice));
		configuredLaunch({ "CalcBV", cbvh().primSize() }, calcMaxBV,
			cbvh().primSize(), (const int3 *)d_faces, (const PointType*)d_vertices, cbvh().bv());
		checkCudaErrors(cudaMemcpy(&bv, cbvh().bv(), sizeof(BOX), cudaMemcpyDeviceToHost));
		configuredLaunch({ "CalcRestrMCs", cbvh().primSize() }, calcRestrMCs,
			cbvh().primSize(), (const int3 *)d_faces, (const PointType*)d_vertices, bv,
			(const int*)_restrLog.getPrimMark(), (const int*)getRawPtr(d_primMap), getRawPtr(d_keys32));

		// 7 prepare restr keys(subtree id, morton code) and scatter map
		configuredLaunch({ "SelectPrimitives", cbvh().primSize() }, selectPrimitives,
			cbvh().primSize(), (const int*)_restrLog.getRestrBvhRoot(), (const int*)getRawPtr(d_gatherMap),
			(const MCSize*)getRawPtr(d_keys32), getRawPtr(d_keys64), getRawPtr(d_taskSequence));
		checkCudaErrors(cudaMemcpy(getRawPtr(d_vals), getRawPtr(d_taskSequence), sizeof(int) * _numRtPrim, cudaMemcpyDeviceToDevice));
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::sort_by_key(getDevicePtr(d_keys64), getDevicePtr(d_keys64) + _numRtPrim, getDevicePtr(d_vals)));
		Logger::tock<TimerType::GPU>("SortRestrPrimCodes");

		/// 8 build external nodes upon ordered primitives(extnode-primitive structure remains intact)
		// update primitive map
		configuredLaunch({ "UpdatePrimMap", _numRtPrim }, updatePrimMap,
			_numRtPrim, getRawPtr(d_taskSequence), getRawPtr(d_vals), cbvh().lvs().getPrimitiveArray().getIdx(), getRawPtr(d_primMap));

		/// 9 build primitives
		// update primitives and ext nodes
		configuredLaunch({ "UpdatePrimAndExtNode", cbvh().primSize() }, updatePrimAndExtNode,
			cbvh().primSize(), (const int*)_restrLog.getPrimMark(), (const int*)getRawPtr(d_primMap),
			(const int3*)d_faces, (const PointType*)d_vertices, (const BOX *)cbvh().bv(), clvs().portobj<0>());

		// should be faster than dealing with restr int nodes separately
		cbvh().tks().clearIntNodes(cbvh().intSize());
		configuredLaunch({ "RefitIntNode", cbvh().extSize() }, refitIntNode,
			cbvh().extSize(), clvs().portobj<0>(), ctks().portobj<0>());

		/// 10 build internal nodes
		//cbvh().lvs().calcRestrSplitMetrics(cbvh().extSize(), _restrLog.getRestrBvhRoot());
		cbvh().lvs().calcSplitMetrics(cbvh().extSize());

		str = string_format("%d subtrees and %d (%d) internal nodes need restructuring. restr\n", _numRtSubtree, _numRtIntNode, cbvh().intSize());
		std::cout << str << '\n';
		Logger::message(str);

		checkCudaErrors(cudaMemset(getRawPtr(d_count), 0, sizeof(uint) * (cbvh().extSize() + 1)));	///< storing lcl-values, used for ordering
		_unsortedTks.clearIntNodes(cbvh().intSize());

		configuredLaunch({ "RestrIntNodes", _numRtExtNode }, restrIntNodes,
			cbvh().extSize(), _numRtExtNode, (const int*)getRawPtr(d_taskSequence), (const uint*)cbvh().tks().getMarks(),
			(const int*)_restrLog.getRestrBvhRoot(), getRawPtr(d_count), getRawPtr(d_vals), clvs().portobj<0>(), _unsortedTks.portobj<0>());
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr(d_count), getDevicePtr(d_count) + cbvh().extSize() + 1, getDevicePtr(d_offsetTable)));
		Logger::tock<TimerType::GPU>("CalcRestrIntNodeSortOffsets");

		/// d_taskSequence here also works as the compacted restr ext node queue
		configuredLaunch({ "CalcRestrIntNodeOrders",  _numRtExtNode }, calcRestrIntNodeOrders,
			_numRtExtNode, (const int*)getRawPtr(d_taskSequence), (const uint*)getRawPtr(d_count), (const uint*)getRawPtr(d_offsetTable),
			(const int*)_restrLog.getRestrBvhRoot(), (const int*)ctks().getLbds(), (const uint*)cbvh().tks().getMarks(), 
			(const int*)getRawPtr(d_vals), clvs().getLcas(), clvs().getPars(), _unsortedTks.portobj<0>(), getRawPtr(d_tkMap), getRawPtr(d_sequence));
		configuredLaunch({ "ReorderRestrIntNodes",  _numRtIntNode }, reorderRestrIntNodes,
			_numRtIntNode, (const int*)getRawPtr(d_sequence), (const int*)getRawPtr(d_tkMap),
			_unsortedTks.portobj<0>(), ctks().portobj<0>());

Logger::recordSection<TimerType::GPU>("restr_bvh");

		_restrLog.setBvhOptTag(1);

		lastRestr = true;
		return true;
	}

	/// finer tasks
	void LBvhFixedDeformable::reorderPrims() {
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::sequence(getDevicePtr(d_vals), getDevicePtr(d_vals) + cbvh().primSize()));
		checkThrustErrors(thrust::sort_by_key(getDevicePtr(d_keys32), getDevicePtr(d_keys32) + cbvh().primSize(), getDevicePtr<int>(d_vals)));
		Logger::tock<TimerType::GPU>("SortCodes");
		configuredLaunch({ "CalcPrimMap", cbvh().primSize() }, calcInverseMapping,
			cbvh().primSize(), getRawPtr(d_vals), getRawPtr(d_primMap));
		checkCudaErrors(cudaMemcpy(cbvh().lvs().getPrimitiveArray().getMtCodes(), getRawPtr(d_keys32), sizeof(MCSize) * cbvh().primSize(), cudaMemcpyDeviceToDevice));
	}

	void LBvhFixedDeformable::reorderIntNodes() {
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr(d_count), getDevicePtr(d_count) + cbvh().extSize(), getDevicePtr(d_offsetTable)));
		Logger::tock<TimerType::GPU>("CalcIntNodeSortOffsets");

		configuredLaunch({ "CalcIntNodeOrders", cbvh().extSize() }, calcIntNodeOrders,
			cbvh().extSize(), _unsortedTks.portobj<0>(), cbvh().lvs().getLcas(), getRawPtr(d_count), getRawPtr(d_offsetTable), getRawPtr(d_tkMap));
		/// update lcas too
		checkThrustErrors(thrust::fill(getDevicePtr(clvs().getLcas()) + cbvh().extSize(), 
			getDevicePtr(clvs().getLcas()) + cbvh().extSize() + 1, -1));
		configuredLaunch({ "UpdateBvhExtNodeLinks", cbvh().extSize() }, updateBvhExtNodeLinks,
			cbvh().extSize(), (const int*)getRawPtr(d_tkMap), cbvh().lvs().getLcas(), cbvh().lvs().getPars());
		/// the above two kernels cannot easily do it in one megakernel cuz multiple link mappings should be pre-calculated.
		configuredLaunch({ "ReorderIntNode",  cbvh().intSize() }, reorderIntNode,
			cbvh().intSize(), (const int*)getRawPtr(d_tkMap), _unsortedTks.portobj<0>(), cbvh().tks().portobj<0>());
	}

	/// debug
	void LBvhFixedDeformable::checkPrimitiveMap() {
		/// prim idx
		checkCudaErrors(cudaMemset(getRawPtr(d_keys32), 0, sizeof(int)*cbvh().primSize()));
		recordLaunch("CheckPrimIdx", (cbvh().primSize() + 255) / 256, 256, (size_t)0, checkPrimmap,
			cbvh().primSize(), cbvh().lvs().getPrimitiveArray().getIdx(), (int*)getRawPtr(d_keys32));
		int invalid;
		checkThrustErrors(invalid = thrust::count_if(thrust::device, getDevicePtr(d_keys32), getDevicePtr(d_keys32) + cbvh().primSize(), NotOne()));
		if (invalid) {
			printf("\n\tprimIdx has %d errors\n\n", invalid);
			getchar();
		}
		/// prim map
		checkCudaErrors(cudaMemset(getRawPtr(d_keys32), 0, sizeof(int)*cbvh().primSize()));
		recordLaunch("CheckPrimMap", (cbvh().primSize() + 255) / 256, 256, 0, checkPrimmap,
			cbvh().primSize(), getRawPtr(d_primMap), (int*)getRawPtr(d_keys32));
		checkThrustErrors(invalid = thrust::count_if(thrust::device, getDevicePtr(d_keys32), getDevicePtr(d_keys32) + cbvh().primSize(), NotOne()));
		if (invalid) {
			printf("\n\tprimMap has %d errors\n\n", invalid);
			getchar();
		}
		/// link
		recordLaunch("CheckPrimMapIdxLink", (cbvh().primSize() + 255) / 256, 256, 0, checkLink,
			cbvh().primSize(), getRawPtr(d_primMap), cbvh().lvs().getPrimitiveArray().getIdx());
	}

	void LBvhFixedDeformable::checkBvhValidity() {
		int* d_tag, tag;
		checkCudaErrors(cudaMalloc((void**)&d_tag, sizeof(int)));
		checkCudaErrors(cudaMemset(d_tag, 0, sizeof(int)));
		recordLaunch("CheckBVHIntegrity", (cbvh().extSize() + 255) / 256, 256, 0, checkBVHIntegrity,
			cbvh().extSize(), cbvh().lvs().portobj<0>(), cbvh().tks().portobj<0>(), d_tag);
		checkCudaErrors(cudaMemcpy(&tag, d_tag, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_tag));
		if (tag)
			getchar();
	}
}
