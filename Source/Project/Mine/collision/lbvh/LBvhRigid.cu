#include "LBvhRigid.h"

#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "utility/CudaThrustUtils.hpp"
#include "utility/CudaKernelUtils.cuh"
#include "utility/CudaDeviceUtils.h"
#include "system/CudaDevice/CudaDevice.h"
#include "system/CudaDevice/CudaKernelLauncher.cu"
#include "setting/CDBenchmarkSettings.h"

namespace mn {

	/// Definitions
	LBvhRigid::LBvhRigid(LBvhBuildConfig&& config) {
		_lvs.setup(config.primSize, config.extSize);
		_tks.setup(config.intSize);
		///
		_unsortedPrims.setup(config.primSize);
		_unsortedTks.setup(config.intSize);

		checkCudaErrors(cudaMalloc((void**)&d_vertices, sizeof(PointType)*config.primSize*3));
		checkCudaErrors(cudaMalloc((void**)&d_faces, sizeof(int3)*config.primSize));
		checkCudaErrors(cudaMalloc((void**)&d_bv, sizeof(BOX)));

		checkThrustErrors(
		d_primMap.resize(config.primSize);
		d_tkMap.resize(config.intSize);
		d_offsetTable.resize(config.primSize);

		d_count.resize(config.primSize);
		d_keys32.resize(config.intSize << 1);
		d_vals.resize(config.intSize << 1);
		);

		checkThrustErrors(thrust::sequence(d_primMap.begin(), d_primMap.end()));
	}

	LBvhRigid::~LBvhRigid() {
		checkCudaErrors(cudaFree(d_bv));
		checkCudaErrors(cudaFree(d_faces));
		checkCudaErrors(cudaFree(d_vertices));
		_unsortedTks.cleanup();
		_unsortedPrims.cleanup();
		_lvs.cleanup();
		_tks.cleanup();
	}

	void LBvhRigid::maintain(LBvhRigidMaintenance scheme, const SceneData& pdata) {
		/// 0: rebuild 1: refit
		updatePrimData(pdata);
		switch (scheme) {
		case LBvhRigidMaintenance::BUILD: build(); break;
		case LBvhRigidMaintenance::REFIT: refit(); break;
		default: break;
		}
	}

#if MACRO_VERSION
	void LBvhRigid::maintain(LBvhRigidMaintenance scheme, const ARCSimSceneData& pdata) {
		_primSize = pdata.fsize;
		d_facesARCSim = pdata.faces;
		d_bxsARCSim = pdata.boxes;
		/// refresh pdata
		switch (scheme) {
		case LBvhRigidMaintenance::BUILD: build(); /*checkBvhValidity();*/ break;
		case LBvhRigidMaintenance::REFIT: refit(); /*checkBvhValidity();*/ break;
		default: break;
		}
	}
#endif

	void LBvhRigid::updatePrimData(const SceneData& pdata) {
		_primSize = pdata.fids.size();
		checkCudaErrors(cudaMemcpy(d_faces, pdata.fids.data(), sizeof(int3)*pdata.fids.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vertices, pdata.pos.data(), sizeof(PointType)*pdata.pos.size(), cudaMemcpyHostToDevice));
	}
	void LBvhRigid::reorderPrims() {
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::sequence(getDevicePtr(d_vals), getDevicePtr(d_vals) + _primSize));
		checkThrustErrors(thrust::sort_by_key(getDevicePtr(d_keys32), getDevicePtr(d_keys32) + _primSize, getDevicePtr<int>(d_vals)));
		Logger::tock<TimerType::GPU>("SortCodes");
		configuredLaunch({ "CalcPrimMap", _primSize }, calcInverseMapping,
			_primSize, getRawPtr(d_vals), getRawPtr(d_primMap));
		/// should copy back morton codes
		checkCudaErrors(cudaMemcpy(_unsortedPrims.getMtCodes(), getRawPtr(d_keys32), sizeof(MCSize) * _primSize, cudaMemcpyDeviceToDevice));
	}

	void LBvhRigid::reorderIntNodes() {
		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::exclusive_scan(getDevicePtr(d_count), getDevicePtr(d_count) + _extSize, getDevicePtr(d_offsetTable)));
		Logger::tock<TimerType::GPU>("CalcIntNodeSortOffsets");

		configuredLaunch({ "CalcIntNodeOrders", _extSize }, calcIntNodeOrders,
			_extSize, _unsortedTks.portobj<0>(), _lvs.getLcas(), getRawPtr(d_count), getRawPtr(d_offsetTable), getRawPtr(d_tkMap));
		checkThrustErrors(thrust::fill(getDevicePtr(_lvs.getLcas()) + _extSize, 
			getDevicePtr(_lvs.getLcas()) + _extSize + 1, -1));
		configuredLaunch({ "UpdateBvhExtNodeLinks", _extSize }, updateBvhExtNodeLinks,
			_extSize, (const int*)getRawPtr(d_tkMap), _lvs.getLcas(), _lvs.getPars());
		configuredLaunch({ "ReorderIntNode",  _intSize }, reorderIntNode,
			_intSize, (const int*)getRawPtr(d_tkMap), _unsortedTks.portobj<0>(), _tks.portobj<0>());
	}

	void LBvhRigid::build() {
		/// calculate scene bounding box
		BOX	bv{};
		checkCudaErrors(cudaMemcpy(d_bv, &bv, sizeof(BOX), cudaMemcpyHostToDevice));
#if MACRO_VERSION
			configuredLaunch({ "CalcBVARCSim", _primSize }, calcMaxBVARCSim,
				_primSize, d_bxsARCSim, d_bv);
#else
			configuredLaunch({ "CalcBV", _primSize }, calcMaxBV,
				_primSize, (const int3*)d_faces, (const PointType*)d_vertices, d_bv);
#endif
		checkCudaErrors(cudaMemcpy(&bv, d_bv, sizeof(BOX), cudaMemcpyDeviceToHost));

#if MACRO_VERSION
			configuredLaunch({ "CalcMCsARCSim", _primSize }, calcMCsARCSim,
				_primSize, d_bxsARCSim, bv, getRawPtr(d_keys32));
#else
			configuredLaunch({ "CalcMCs", _primSize }, calcMCs,
				_primSize, d_faces, d_vertices, bv, getRawPtr(d_keys32));
#endif

		reorderPrims();

		/// build primitives
#if MACRO_VERSION
			configuredLaunch({ "BuildPrimsARCSim", _primSize }, buildPrimitivesARCSim,
				_primSize, _lvs.getPrimitiveArray().portobj<0>(), getRawPtr(d_primMap), d_facesARCSim, d_bxsARCSim);
#else
			configuredLaunch({ "BuildPrims", _primSize }, buildPrimitives,
				_primSize, _lvs.getPrimitiveArray().portobj<0>(), getRawPtr(d_primMap), d_faces, d_vertices);
#endif

		/// build external nodes
		_intSize = (_extSize = _lvs.buildExtNodes(_primSize)) - 1;
		_lvs.calcSplitMetrics(_extSize);
		/// build internal nodes
		_unsortedTks.clearIntNodes(_intSize);
		configuredLaunch({ "BuildIntNodes", _extSize }, buildIntNodes,
			_extSize, getRawPtr(d_count), _lvs.portobj<0>(), _unsortedTks.portobj<0>());

Logger::recordSection<TimerType::GPU>("construct_bvh");

		/// first correct indices, then sort
		reorderIntNodes();

Logger::recordSection<TimerType::GPU>("sort_bvh");

		printf("Rigid Bvh: Primsize: %d Extsize: %d\n", _primSize, _extSize);
	}

	void LBvhRigid::refit() {
		Logger::tick<TimerType::GPU>();
		_lvs.clearExtBvs(_extSize);
		_tks.clearIntNodes(_intSize);
		Logger::tock<TimerType::GPU>("init_bvh_bvs");

#if MACRO_VERSION
			configuredLaunch({ "RefitExtNodeARCSim", _primSize }, refitExtNodeARCSim,
				_primSize, _lvs.portobj<0>(), getRawPtr(d_primMap), d_facesARCSim, d_bxsARCSim);
#else
			configuredLaunch({ "RefitExtNode", _primSize }, refitExtNode,
				_primSize, _lvs.portobj<0>(), getRawPtr(d_primMap), d_faces, d_vertices);
#endif

		configuredLaunch({ "RefitIntNode", _extSize }, refitIntNode,
			_extSize, _lvs.portobj<0>(), _tks.portobj<0>());
		
Logger::recordSection<TimerType::GPU>("refit_bvh");
	}

}
