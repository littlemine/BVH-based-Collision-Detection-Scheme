#include "BvttFrontLooseIntra.h"
#include "BvttFrontLooseKernels.cuh"
#include "collision\lbvh\LBvh.h"
#include "utility\CudaHostUtils.h"
#include "utility\CudaThrustUtils.hpp"
#include "utility\CudaKernelUtils.cuh"
#include "system\CudaDevice\CudaDevice.h"
#include "system\CudaDevice\CudaKernelLauncher.cu"
#include "setting\CDBenchmarkSettings.h"

#include <thrust\execution_policy.h>
#include "collision\narrow_phase\narrow_phase.cuh"

namespace mn {

	BvttFrontLooseIntra::BvttFrontLooseIntra(BvttFrontIntraBuildConfig<LBvh<ModelType::FixedDeformableType>> config) {
		TheCudaDevice = CudaDevice::getInstance();
		_pBvh = config.pbvh;
		std::array<uint, 2>	sizeConfig;
		sizeConfig[0] = config.intFrontSize;
		sizeConfig[1] = config.extFrontSize;
		_fronts.initBufs(sizeConfig.data(), (uchar)sizeConfig.size());
		_log.setup(config.intNodeSize, config.extNodeSize, config.intFrontSize, config.extFrontSize);

		checkThrustErrors(d_IntFtIndices.resize(config.intNodeSize));
		checkThrustErrors(d_ExtFtIndices.resize(config.extNodeSize));
		checkThrustErrors(d_snapshot.resize(config.intNodeSize));

		checkCudaErrors(cudaMalloc((void**)&d_extFtNodeCnt, sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_intFtNodeCnt, sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_cpNum, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_actualCpNum, sizeof(int)));
		checkThrustErrors(d_cpRes.resize(config.cpNum));
		checkThrustErrors(d_orderedCdpairs.resize(config.cpNum));


		reportMemory();
	}

	BvttFrontLooseIntra::~BvttFrontLooseIntra() {
		checkCudaErrors(cudaFree(d_intFtNodeCnt));
		checkCudaErrors(cudaFree(d_extFtNodeCnt));
		checkCudaErrors(cudaFree(d_actualCpNum));
		checkCudaErrors(cudaFree(d_cpNum));
		_log.cleanup();
		_fronts.destroyBufs();
		_pBvh = nullptr;
	}

	void BvttFrontLooseIntra::inspectResults() {
		_fronts.retrieveSizes();
		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&_actualCpNum, d_actualCpNum, sizeof(int), cudaMemcpyDeviceToHost));

		std::string str = string_format("front (%d, %d) cpNum %d actualCpNum %d", _fronts.cs(0), _fronts.cs(1), _cpNum, _actualCpNum);
		if (_fronts.cs(0) >= BvttFrontSettings::int_front_size() || _fronts.cs(1) >= BvttFrontSettings::ext_front_size() || _cpNum >= BvttFrontSettings::collision_pair_num()) {
			printf("exceed front length! %d, %d  %d\n", _fronts.cs(0), _fronts.cs(1), _cpNum);
		}
		std::cout << str << '\n';
		Logger::message(str);
	}

	void BvttFrontLooseIntra::applyCpResults(uint* _idx, uint2* _front) {
		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_idx, d_cpNum, sizeof(uint), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(_front, getRawPtr(d_cpRes), sizeof(uint2) * _cpNum, cudaMemcpyDeviceToDevice));
	}

	void BvttFrontLooseIntra::maintain(BvttFrontLooseIntraMaintenance scheme) {
		_pBvh->restrLog().setUpdateTag(false);
		static bool first = true;
		/**	
		 *	\note simplified version, adopt bvtt front opt according to bvh opt. Not robust. See supplementary file for more details.
		 */
		if (CDBenchmarkSettings::enableRestr()) {
			/// front maintenance override
			if (_pBvh->bvhOptTag() == 2) {		///< build
				Logger::message("build front(in restr). ");
				generate(); 
				proximityQuery();
				inspectResults();
				return;
			} 
			if (_pBvh->bvhOptTag() == 1) {		///< bvh restr
				checkCudaErrors(cudaMemset(d_extFtNodeCnt, 0, sizeof(uint)));
				checkCudaErrors(cudaMemset(d_intFtNodeCnt, 0, sizeof(uint)));
				configuredLaunch({ "CountRestrFrontNodes", (int)_pBvh->getExtNodeSize() + (int)_pBvh->getIntNodeSize() }, countRestrFrontNodes,
					make_uint2(_pBvh->getExtNodeSize(), _pBvh->getIntNodeSize()), _pBvh->restrLog().portobj<0>(), _log.portobj<0>(), d_intFtNodeCnt, d_extFtNodeCnt);
				uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));
				checkCudaErrors(cudaMemcpy(&_extFtNodeCnt, d_extFtNodeCnt, sizeof(int), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(&_intFtNodeCnt, d_intFtNodeCnt, sizeof(int), cudaMemcpyDeviceToHost));
				if ((_extFtNodeCnt + _intFtNodeCnt) * 1. / (osizes.x + osizes.y) > 0.30) {	///< front build
					Logger::message("build front(in restr). ");
					printf("bd front: restr(%d, %d) total(%d, %d) ratio: %.3f\n", _intFtNodeCnt, _extFtNodeCnt, osizes.x, osizes.y, (_extFtNodeCnt + _intFtNodeCnt) * 1. / (osizes.x + osizes.y));
					//getchar();
					generate();
					inspectResults();
					return;
				}
				else {			///< front restr
					Logger::message("restructure front(in restr) ");
					printf("rt front: restr(%d, %d) total(%d, %d) ratio: %.3f\n", _intFtNodeCnt, _extFtNodeCnt, osizes.x, osizes.y, (_extFtNodeCnt + _intFtNodeCnt) * 1. / (osizes.x + osizes.y));
					//getchar();
					generate(); 
					//restructure();
					inspectResults();
					_restructured = true;
					return;
				}
			}
			/// otherwise adopt regular front operation
		}

		switch (scheme) {
		case BvttFrontLooseIntraMaintenance::PURE_BVH_CD: Logger::message("pure bvhcd. "); pureBvhCd(); proximityQuery(); break;
		/// front based methods
		case BvttFrontLooseIntraMaintenance::GENERATE: Logger::message("generate front. "); generate(); proximityQuery(); break;
		case BvttFrontLooseIntraMaintenance::UPDATE: Logger::message("update front. "); pruneSprout(); proximityQuery(); break;
		case BvttFrontLooseIntraMaintenance::REORDER: Logger::message("balance front. "); balance(); proximityQuery(); break;
		case BvttFrontLooseIntraMaintenance::KEEP: Logger::message("preserve front. "); keep(); proximityQuery(); break;

		default: break;
		}
		inspectResults();
		first = false;
	}

	void BvttFrontLooseIntra::proximityQuery() {
		if (CDBenchmarkSettings::includeNarrowPhase()) {
			checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemset(d_actualCpNum, 0, sizeof(int)));
			configuredLaunch({ "SimpleNarrowPhase", _cpNum }, simpleNarrowPhase,
				(uint)_cpNum, getRawPtr(d_orderedCdpairs), _pBvh->getFaces(), _pBvh->getVertices(), d_actualCpNum);

			Logger::recordSection<TimerType::GPU>("narrow_phase");

		}
	}

	void BvttFrontLooseIntra::reorderFronts() {
		Logger::tick<TimerType::GPU>();
		_log.prepare(_pBvh->getExtNodeSize());
		Logger::tock<TimerType::GPU>("prepare_ordering_calc_offset");
		_log.clear(_pBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		checkCudaErrors(cudaMemcpy(_fronts.nsizes(), _fronts.csizes(), sizeof(uint) * 2, cudaMemcpyDeviceToDevice));
		
		uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));
		configuredLaunch({ "PureReorderLooseFrontsWithLog", (int)osizes.x + (int)osizes.y }, pureReorderLooseFrontsWithLog,
			osizes, _fronts.cbufs(), _fronts.nbufs(), _log.portobj<0>());

		//checkThrustErrors(thrust::copy(getDevicePtr(_fronts.cbuf(0)), getDevicePtr(_fronts.cbuf(0)) + osizes.x, getDevicePtr(_fronts.nbuf(0))));
		//checkThrustErrors(thrust::copy(getDevicePtr(_fronts.cbuf(1)), getDevicePtr(_fronts.cbuf(1)) + osizes.y, getDevicePtr(_fronts.nbuf(1))));
/*
		debugLaunch((int)(osizes.x + 255) / 256, 256, extractFrontKeys, (int)osizes.x, _fronts.cbuf(0), getRawPtr(d_IntFtIndices));
		debugLaunch((int)(osizes.y + 255) / 256, 256, extractFrontKeys, (int)osizes.y, _fronts.cbuf(1), getRawPtr(d_ExtFtIndices));

		Logger::tick<TimerType::GPU>();
		checkThrustErrors(thrust::sort_by_key(getDevicePtr(d_IntFtIndices), getDevicePtr(d_IntFtIndices) + osizes.x, getDevicePtr<int2>(_fronts.cbuf(0))));
		checkThrustErrors(thrust::sort_by_key(getDevicePtr(d_ExtFtIndices), getDevicePtr(d_ExtFtIndices) + osizes.y, getDevicePtr<int2>(_fronts.cbuf(1))));
		Logger::tock<TimerType::GPU>("radix_sort");

		checkThrustErrors(thrust::copy(getDevicePtr(_fronts.cbuf(0)), getDevicePtr(_fronts.cbuf(0)) + osizes.x, getDevicePtr(_fronts.nbuf(0))));
		checkThrustErrors(thrust::copy(getDevicePtr(_fronts.cbuf(1)), getDevicePtr(_fronts.cbuf(1)) + osizes.y, getDevicePtr(_fronts.nbuf(1))));
		*/

Logger::recordSection<TimerType::GPU>("broad_phase_front_ordering");

		_fronts.slide();
	}

	void BvttFrontLooseIntra::separateFronts() {
		/// compact valid front nodes
		_log.preserveCnts(_pBvh->getExtNodeSize());
		configuredLaunch({ "FilterIntFrontCnts", (int)_pBvh->getIntNodeSize() }, filterIntFrontCnts,
			_pBvh->getIntNodeSize(), _pBvh->restrLog().getIntMark(), (const int*)_pBvh->getPrevLbds(),
			(const int*)_pBvh->restrLog().getRestrBvhRoot(), _log.intNodeCnts(), _log.intNodeBackCnts());
		configuredLaunch({ "FilterExtFrontCnts", (int)_pBvh->getExtNodeSize() }, filterExtFrontCnts,
			_pBvh->getExtNodeSize(), _pBvh->restrLog().getExtMark(), (const int*)_pBvh->getPrevLbds(), 
			(const int*)_pBvh->restrLog().getRestrBvhRoot(), _log.extNodeCnts(), _log.extNodeBackCnts(), _log.intNodeBackCnts());

		_log.prepare(_pBvh->getExtNodeSize());
		_log.prepareBak(_pBvh->getExtNodeSize());
		checkCudaErrors(cudaMemcpy(_numValidFrontNodes, _log.intBegPos() + _pBvh->getIntNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_numValidFrontNodes + 1, _log.extBegPos() + _pBvh->getExtNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_numInvalidFrontNodes, _log.intBegPosBak() + _pBvh->getIntNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_numInvalidFrontNodes + 1, _log.extBegPosBak() + _pBvh->getExtNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));

		_log.clear(_pBvh->getExtNodeSize());
		checkThrustErrors(thrust::fill(getDevicePtr(_fronts.nsize(0)), getDevicePtr(_fronts.nsize(0) + 1),
			_numValidFrontNodes[0] + _numInvalidFrontNodes[0]));
		checkThrustErrors(thrust::fill(getDevicePtr(_fronts.nsize(1)), getDevicePtr(_fronts.nsize(1) + 1),
			_numValidFrontNodes[1]));

		_fronts.retrieveSizes();
		uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));
		configuredLaunch({ "SeparateIntLooseIntraFrontWithLog", (int)osizes.x }, separateIntLooseIntraFrontWithLog,
			osizes.x, make_uint2(_numValidFrontNodes[0], _numValidFrontNodes[1]), (const int2*)_fronts.cbuf(0), 
			_fronts.nsizes(), _fronts.nbufs(), (const int*)_pBvh->restrLog().getIntMark(), 
			(const int*)_pBvh->restrLog().getRestrBvhRoot(), (const int*)_pBvh->getPrevLbds(), 
			(const int*)_pBvh->clvs().getLcas(), _log.portobj<0>());
		configuredLaunch({ "SeparateExtLooseIntraFrontWithLog", (int)osizes.y }, separateExtLooseIntraFrontWithLog,
			osizes.y, make_uint2(_numValidFrontNodes[0], _numValidFrontNodes[1]), (const int2*)_fronts.cbuf(1), 
			_fronts.nsizes(), _fronts.nbufs(), (const int*)_pBvh->restrLog().getExtMark(), 
			(const int*)_pBvh->restrLog().getRestrBvhRoot(),
			(const int*)_pBvh->clvs().getLcas(), _log.portobj<0>());

		printf("\n#original front(%d, %d) valid(%d, %d) invalid(%d, %d)#\n\n", osizes.x, osizes.y,
			_numValidFrontNodes[0], _numValidFrontNodes[1], osizes.x - _numValidFrontNodes[0], osizes.y - _numValidFrontNodes[1]);
		_fronts.slide();
	}

	void BvttFrontLooseIntra::calcSnapshot() {
		configuredLaunch({ "FrontSnapshot", (int)_pBvh->getIntNodeSize() }, frontSnapshot,
			_pBvh->getIntNodeSize(), _pBvh->ctks().portobj<0>(), _log.portobj<0>(), getRawPtr(d_snapshot));
Logger::recordSection<TimerType::GPU>("broad_phase_quality_gen");
	}

	void BvttFrontLooseIntra::checkQuality() {
		_pBvh->restrLog().clear(_pBvh->getExtNodeSize());

		configuredLaunch({ "CheckFrontQuality", (int)_pBvh->getIntNodeSize() }, checkFrontQuality,
			_pBvh->getIntNodeSize(), _pBvh->ctks().portobj<0>(), _log.portobj<0>(), getRawPtr(d_snapshot), _pBvh->restrLog().portobj<0>());
		
Logger::recordSection<TimerType::GPU>("broad_phase_quality_check");

		_pBvh->restrLog().setUpdateTag(true);
	}

	void BvttFrontLooseIntra::generate() {
		_log.clear(_pBvh->getExtNodeSize());
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		configuredLaunch({ "GenLooseIntraFrontsWithLog", (int)_pBvh->getPrimNodeSize() - 1 }, genLooseIntraFrontsWithLog,
			_pBvh->getPrimNodeSize(), _pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), 
			_fronts.nsizes(), _fronts.nbufs(), _log.portobj<0>(), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();

Logger::recordSection<TimerType::GPU>("broad_phase_cd_gen");

		reorderFronts();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

		if (CDBenchmarkSettings::enableRestr())
			calcSnapshot();
		
		_restructured = false;
	}

	void BvttFrontLooseIntra::pruneSprout() {
		_log.clear(_pBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		uint osize;

		osize = _fronts.cs(0);
		configuredLaunch({ "MaintainIntLooseIntraFrontsWithLog", (int)osize }, maintainIntLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));
		osize = _fronts.cs(1);
		configuredLaunch({ "MaintainExtLooseIntraFrontsWithLog", (int)osize }, maintainExtLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(1),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();

Logger::recordSection<TimerType::GPU>("broad_phase_cd_update");

		if (CDBenchmarkSettings::enableRestr()) {
			checkQuality();
		}

		reorderFronts();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));
	}

	void BvttFrontLooseIntra::balance() {
		_log.prepare(_pBvh->getExtNodeSize());
		_log.clear(_pBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		checkCudaErrors(cudaMemcpy(_fronts.nsizes(), _fronts.csizes(), sizeof(uint) * 2, cudaMemcpyDeviceToDevice));

		uint osize = _fronts.cs(0);
		configuredLaunch({ "ReorderIntLooseIntraFrontsWithLog", (int)osize }, reorderIntLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), osize, _fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nbuf(0), d_cpNum, getRawPtr(d_cpRes));
		osize = _fronts.cs(1);
		//configuredLaunch({ "ReorderExtLooseIntraFrontsWithLog", (int)osize }, reorderExtLooseIntraFrontsWithLog,
		//	_pBvh->clvs().portobj<0>(), osize, _fronts.cbuf(1),
		//	_log.portobj<0>(), _fronts.nbuf(1), d_cpNum, getRawPtr(d_cpRes));
		checkCudaErrors(cudaMemcpy(_fronts.nbuf(1), _fronts.cbuf(1), sizeof(int2) * osize, cudaMemcpyDeviceToDevice));
		configuredLaunch({ "KeepExtLooseIntraFronts", (int)osize }, keepExtLooseIntraFronts,
			_pBvh->clvs().portobj<0>(), osize, (const int2*)_fronts.cbuf(1), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

Logger::recordSection<TimerType::GPU>("broad_phase_cd_balance");
	}

	void BvttFrontLooseIntra::keep() {
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		uint osize = _fronts.cs(0);
		configuredLaunch({ "KeepIntLooseIntraFronts", (int)osize }, keepIntLooseIntraFronts,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(0), d_cpNum, getRawPtr(d_cpRes));
		osize = _fronts.cs(1);
		configuredLaunch({ "KeepExtLooseIntraFronts", (int)osize }, keepExtLooseIntraFronts,
			_pBvh->clvs().portobj<0>(), osize, (const int2*)_fronts.cbuf(1), d_cpNum, getRawPtr(d_cpRes));

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

Logger::recordSection<TimerType::GPU>("broad_phase_cd_preserve");
	}

	void BvttFrontLooseIntra::restructure() {
		/// prune is disabled in front restructuring
		separateFronts();

Logger::recordSection<TimerType::GPU>("broad_phase_prep_front_restr");

		_log.clear(_pBvh->getExtNodeSize());
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));
		_fronts.retrieveSizes();
		uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));

		/// valid parts
		checkThrustErrors(thrust::fill(getDevicePtr(_fronts.nsize(1)), getDevicePtr(_fronts.nsize(1)) + 1, _numValidFrontNodes[1]));
		checkCudaErrors(cudaMemcpy(_fronts.nbuf(1), _fronts.cbuf(1), sizeof(int2) * _numValidFrontNodes[1], cudaMemcpyDeviceToDevice));
		configuredLaunch({ "SproutExtLooseIntraFrontsWithLog", _numValidFrontNodes[1] }, sproutExtLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), (uint)_numValidFrontNodes[1], (const int2*)_fronts.cbuf(1),
			_log.portobj<0>(), d_cpNum, getRawPtr(d_cpRes));
		configuredLaunch({ "SproutIntLooseIntraFrontsWithLog", _numValidFrontNodes[0] }, sproutIntLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), (uint)_numValidFrontNodes[0], (const int2*)_fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));

		printf("\n#restr front(%d, %d) valid(%d, %d) invalid(%d, %d)#\n\n", osizes.x, osizes.y,
			_numValidFrontNodes[0], _numValidFrontNodes[1], osizes.x - _numValidFrontNodes[0], osizes.y - _numValidFrontNodes[1]);
		/// invalid parts
		// int fronts & ext fronts, including self CD in restr subtrees
		if (osizes.x - _numValidFrontNodes[0] > 0)
			configuredLaunch({ "RestructureIntLooseIntraFrontWithLog", (int)osizes.x - _numValidFrontNodes[0] }, restructureIntLooseIntraFrontWithLog,
				_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(),
				osizes.x - _numValidFrontNodes[0], (const int2*)_fronts.cbuf(0) + _numValidFrontNodes[0],
				_log.portobj<0>(), 
				_fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));
		if (osizes.y - _numValidFrontNodes[1] > 0)
			configuredLaunch({ "RestructureExtLooseIntraFrontWithLog", (int)osizes.y - _numValidFrontNodes[1] }, restructureExtLooseIntraFrontWithLog,
				_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(),
				osizes.y - _numValidFrontNodes[1], (const int2*)_fronts.cbuf(1) + _numValidFrontNodes[1],
				_log.portobj<0>(), (const int*)_pBvh->restrLog().getRestrBvhRoot(),
				_fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();
		reorderFronts();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

		/// prune after restr
		_log.clear(_pBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		_fronts.resetNextSizes();

		uint osize;
		osize = _fronts.cs(0);
		configuredLaunch({ "PruneIntLooseIntraFrontsWithLog", (int)osize }, pruneIntLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs());
		osize = _fronts.cs(1);
		configuredLaunch({ "PruneExtLooseIntraFrontsWithLog", (int)osize }, pruneExtLooseIntraFrontsWithLog,
			_pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(1),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs());

		_fronts.slide();

Logger::recordSection<TimerType::GPU>("broad_phase_restr_front");

		if (CDBenchmarkSettings::enableRestr()) {
			checkQuality();
		}
		reorderFronts();
 
		_restructured = true;
	}

	void BvttFrontLooseIntra::pureBvhCd() {
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		configuredLaunch({ "PureBvhSelfCD", (int)_pBvh->getPrimNodeSize() - 1 }, pureBvhSelfCD,
			_pBvh->getPrimNodeSize(), _pBvh->clvs().portobj<0>(), _pBvh->ctks().portobj<0>(), d_cpNum, getRawPtr(d_cpRes));

Logger::recordSection<TimerType::GPU>("broad_phase_cd_bvh");

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));
		
	}

}
