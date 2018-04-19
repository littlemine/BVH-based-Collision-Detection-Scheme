#include "BvttFrontLooseInter.h"
#include "BvttFrontLooseKernels.cuh"
#include "collision\lbvh\LBvh.h"
#include "utility\CudaHostUtils.h"
#include "utility\CudaThrustUtils.hpp"
#include "utility\CudaKernelUtils.cuh"
#include "system\CudaDevice\CudaDevice.h"
#include "system\CudaDevice\CudaKernelLauncher.cu"
#include "setting\CDBenchmarkSettings.h"

#include <thrust/execution_policy.h>
#include "collision/narrow_phase/narrow_phase.cuh"

namespace mn {

	BvttFrontLooseInter::BvttFrontLooseInter(BvttFrontInterBuildConfig<LBvh<ModelType::FixedDeformableType>, LBvh<ModelType::RigidType>> config) {
		TheCudaDevice = CudaDevice::getInstance();
		_pFixedDeformableBvh = config.pbvha;
		_pRigidBvh = config.pbvhb;
		std::array<uint, 2>	sizeConfig;
		sizeConfig[0] = config.intFrontSize;
		sizeConfig[1] = config.extFrontSize;
		_fronts.initBufs(sizeConfig.data(), (uchar)sizeConfig.size());
		_log.setup(config.intNodeSize, config.extNodeSize, config.intFrontSize, config.extFrontSize);

		checkThrustErrors(d_IntFtIndices.resize(config.intNodeSize));
		checkThrustErrors(d_ExtFtIndices.resize(config.extNodeSize));

		checkCudaErrors(cudaMalloc((void**)&d_cpNum, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_actualCpNum, sizeof(int)));
		checkThrustErrors(d_cpRes.resize(config.cpNum));
		checkThrustErrors(d_orderedCdpairs.resize(config.cpNum));

		reportMemory();
	}

	BvttFrontLooseInter::~BvttFrontLooseInter() {
		checkCudaErrors(cudaFree(d_actualCpNum));
		checkCudaErrors(cudaFree(d_cpNum));
		_log.cleanup();
		_fronts.destroyBufs();
		_pFixedDeformableBvh = nullptr;
		_pRigidBvh = nullptr;
	}

	void BvttFrontLooseInter::inspectResults() {
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

	void BvttFrontLooseInter::applyCpResults(uint* _idx, uint2* _front) {
		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_idx, d_cpNum, sizeof(uint), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(_front, getRawPtr(d_cpRes), sizeof(uint2) * _cpNum, cudaMemcpyDeviceToDevice));
	}

	void BvttFrontLooseInter::maintain(BvttFrontLooseInterMaintenance scheme) {
		/**
		 *	\note simplified version, adopt
		 */
		if (CDBenchmarkSettings::enableRestr()) {
			/// front maintenance override
			if (_pFixedDeformableBvh->bvhOptTag() == 2) {		///< build
				Logger::message("build front(in restr). ");
				generate();
				inspectResults();
				return;
			}
			if (_pFixedDeformableBvh->bvhOptTag() == 1) {		///< restr
				Logger::message("restructure front(in restr) ");
				//generate(); 
				restructure();
				inspectResults();
				_restructured = true;
				return;
			}
			/// otherwise adopt regular front operation
		}

		switch (scheme) {
		case BvttFrontLooseInterMaintenance::PURE_BVH_CD: Logger::message("pure bvhcd. "); pureBvhCd(); break;
			/// front based methods
		case BvttFrontLooseInterMaintenance::GENERATE: Logger::message("generate front. "); generate(); break;
		case BvttFrontLooseInterMaintenance::UPDATE: Logger::message("update front. "); pruneSprout(); break;
		case BvttFrontLooseInterMaintenance::REORDER: Logger::message("balance front. "); balance(); break;
		case BvttFrontLooseInterMaintenance::KEEP: Logger::message("preserve front. "); keep(); break;

		default: break;
		}
		inspectResults();
	}

	void BvttFrontLooseInter::reorderFronts() {
		Logger::tick<TimerType::GPU>();
		_log.prepare(_pFixedDeformableBvh->getExtNodeSize());
		Logger::tock<TimerType::GPU>("prepare_ordering_calc_offset");
		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		checkCudaErrors(cudaMemcpy(_fronts.nsizes(), _fronts.csizes(), sizeof(uint)* 2, cudaMemcpyDeviceToDevice));

		uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));
		configuredLaunch({ "PureReorderLooseFrontsWithLog", (int)osizes.x + (int)osizes.y }, pureReorderLooseFrontsWithLog,
			osizes, _fronts.cbufs(), _fronts.nbufs(), _log.portobj<0>());

		Logger::recordSection<TimerType::GPU>("broad_phase_front_ordering");

		_fronts.slide();
	}

	void BvttFrontLooseInter::separateFronts() {
		/// compact valid front nodes
		_log.preserveCnts(_pFixedDeformableBvh->getExtNodeSize());
		configuredLaunch({ "FilterIntFrontCnts", (int)_pFixedDeformableBvh->getIntNodeSize() }, filterIntFrontCnts,
			_pFixedDeformableBvh->getIntNodeSize(), _pFixedDeformableBvh->restrLog().getIntMark(), (const int*)_pFixedDeformableBvh->getPrevLbds(),
			(const int*)_pFixedDeformableBvh->restrLog().getRestrBvhRoot(), _log.intNodeCnts(), _log.intNodeBackCnts());
		configuredLaunch({ "FilterExtFrontCnts", (int)_pFixedDeformableBvh->getExtNodeSize() }, filterExtFrontCnts,
			_pFixedDeformableBvh->getExtNodeSize(), _pFixedDeformableBvh->restrLog().getExtMark(), (const int*)_pFixedDeformableBvh->getPrevLbds(),
			(const int*)_pFixedDeformableBvh->restrLog().getRestrBvhRoot(), _log.extNodeCnts(), _log.extNodeBackCnts(), _log.intNodeBackCnts());

		_log.prepare(_pFixedDeformableBvh->getExtNodeSize());
		_log.prepareBak(_pFixedDeformableBvh->getExtNodeSize());
		checkCudaErrors(cudaMemcpy(_numValidFrontNodes, _log.intBegPos() + _pFixedDeformableBvh->getIntNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_numValidFrontNodes + 1, _log.extBegPos() + _pFixedDeformableBvh->getExtNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_numInvalidFrontNodes, _log.intBegPosBak() + _pFixedDeformableBvh->getIntNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(_numInvalidFrontNodes + 1, _log.extBegPosBak() + _pFixedDeformableBvh->getExtNodeSize(), sizeof(int), cudaMemcpyDeviceToHost));

		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		checkThrustErrors(thrust::fill(getDevicePtr(_fronts.nsize(0)), getDevicePtr(_fronts.nsize(0) + 1),
			_numValidFrontNodes[0] + _numInvalidFrontNodes[0]));
		checkThrustErrors(thrust::fill(getDevicePtr(_fronts.nsize(1)), getDevicePtr(_fronts.nsize(1) + 1),
			_numValidFrontNodes[1]));

		_fronts.retrieveSizes();
		uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));
		configuredLaunch({ "SeparateIntLooseInterFrontWithLog", (int)osizes.x }, separateIntLooseInterFrontWithLog,
			osizes.x, make_uint2(_numValidFrontNodes[0], _numValidFrontNodes[1]), (const int2*)_fronts.cbuf(0),
			_fronts.nsizes(), _fronts.nbufs(), (const int*)_pFixedDeformableBvh->restrLog().getIntMark(),
			(const int*)_pFixedDeformableBvh->restrLog().getRestrBvhRoot(), (const int*)_pFixedDeformableBvh->getPrevLbds(),
			(const int*)_pFixedDeformableBvh->clvs().getLcas(), _log.portobj<0>());
		configuredLaunch({ "SeparateExtLooseInterFrontWithLog", (int)osizes.y }, separateExtLooseInterFrontWithLog,
			osizes.y, make_uint2(_numValidFrontNodes[0], _numValidFrontNodes[1]), (const int2*)_fronts.cbuf(1),
			_fronts.nsizes(), _fronts.nbufs(), (const int*)_pFixedDeformableBvh->restrLog().getExtMark(),
			(const int*)_pFixedDeformableBvh->restrLog().getRestrBvhRoot(),
			(const int*)_pFixedDeformableBvh->clvs().getLcas(), _log.portobj<0>());

		printf("\n#original front(%d, %d) valid(%d, %d) invalid(%d, %d)#\n\n", osizes.x, osizes.y,
			_numValidFrontNodes[0], _numValidFrontNodes[1], osizes.x - _numValidFrontNodes[0], osizes.y - _numValidFrontNodes[1]);
		_fronts.slide();
	}

	void BvttFrontLooseInter::generate() {
		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		configuredLaunch({ "GenLooseInterFrontsWithLog", (int)_pRigidBvh->getPrimNodeSize() - 1 }, genLooseInterFrontsWithLog,
			_pRigidBvh->getPrimNodeSize(), _pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(),
			_fronts.nsizes(), _fronts.nbufs(), _log.portobj<0>(), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();

		Logger::recordSection<TimerType::GPU>("broad_phase_cd_gen");

		reorderFronts();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

		_restructured = false;
	}

	void BvttFrontLooseInter::pruneSprout() {
		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		uint osize;

		osize = _fronts.cs(0);
		configuredLaunch({ "MaintainIntLooseInterFrontsWithLog", (int)osize }, maintainIntLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));
		osize = _fronts.cs(1);
		configuredLaunch({ "MaintainExtLooseInterFrontsWithLog", (int)osize }, maintainExtLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(1),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();

		Logger::recordSection<TimerType::GPU>("broad_phase_cd_update");

		reorderFronts();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));
	}

	void BvttFrontLooseInter::balance() {
		_log.prepare(_pFixedDeformableBvh->getExtNodeSize());
		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		checkCudaErrors(cudaMemcpy(_fronts.nsizes(), _fronts.csizes(), sizeof(uint)* 2, cudaMemcpyDeviceToDevice));

		uint osize = _fronts.cs(0);
		configuredLaunch({ "ReorderIntLooseInterFrontsWithLog", (int)osize }, reorderIntLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), osize, _fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nbuf(0), d_cpNum, getRawPtr(d_cpRes));
		osize = _fronts.cs(1);
		//configuredLaunch({ "ReorderExtLooseInterFrontsWithLog", (int)osize }, reorderExtLooseInterFrontsWithLog,
		//	_pRigidBvh->cprim().portobj<0>(),_pFixedDeformableBvh->clvs().portobj<0>(), osize, _fronts.cbuf(1),
		//	_log.portobj<0>(), _fronts.nbuf(1), d_cpNum, getRawPtr(d_cpRes));
		checkCudaErrors(cudaMemcpy(_fronts.nbuf(1), _fronts.cbuf(1), sizeof(int2)* osize, cudaMemcpyDeviceToDevice));
		configuredLaunch({ "KeepExtLooseInterFronts", (int)osize }, keepExtLooseInterFronts,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), osize, (const int2*)_fronts.cbuf(1), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

		Logger::recordSection<TimerType::GPU>("broad_phase_cd_balance");
	}

	void BvttFrontLooseInter::keep() {
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		uint osize = _fronts.cs(0);
		configuredLaunch({ "KeepIntLooseInterFronts", (int)osize }, keepIntLooseInterFronts,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(0), d_cpNum, getRawPtr(d_cpRes));
		osize = _fronts.cs(1);
		configuredLaunch({ "KeepExtLooseInterFronts", (int)osize }, keepExtLooseInterFronts,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), osize, (const int2*)_fronts.cbuf(1), d_cpNum, getRawPtr(d_cpRes));

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

		Logger::recordSection<TimerType::GPU>("broad_phase_cd_preserve");
	}

	void BvttFrontLooseInter::restructure() {
		/// prune is disabled in front restructuring
		separateFronts();

		Logger::recordSection<TimerType::GPU>("broad_phase_prep_front_restr");

		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		_fronts.resetNextSizes();
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));
		_fronts.retrieveSizes();
		uint2 osizes = make_uint2(_fronts.cs(0), _fronts.cs(1));

		/// valid parts
		checkThrustErrors(thrust::fill(getDevicePtr(_fronts.nsize(1)), getDevicePtr(_fronts.nsize(1)) + 1, _numValidFrontNodes[1]));
		checkCudaErrors(cudaMemcpy(_fronts.nbuf(1), _fronts.cbuf(1), sizeof(int2)* _numValidFrontNodes[1], cudaMemcpyDeviceToDevice));
		configuredLaunch({ "SproutExtLooseInterFrontsWithLog", _numValidFrontNodes[1] }, sproutExtLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), (uint)_numValidFrontNodes[1], (const int2*)_fronts.cbuf(1),
			_log.portobj<0>(), d_cpNum, getRawPtr(d_cpRes));
		configuredLaunch({ "SproutIntLooseInterFrontsWithLog", _numValidFrontNodes[0] }, sproutIntLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), (uint)_numValidFrontNodes[0], (const int2*)_fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));

		printf("\n#restr front(%d, %d) valid(%d, %d) invalid(%d, %d)#\n\n", osizes.x, osizes.y,
			_numValidFrontNodes[0], _numValidFrontNodes[1], osizes.x - _numValidFrontNodes[0], osizes.y - _numValidFrontNodes[1]);
		/// invalid parts
		// int fronts & ext fronts
		if (osizes.x - _numValidFrontNodes[0] > 0)
			configuredLaunch({ "RestructureIntLooseInterFrontWithLog", (int)osizes.x - _numValidFrontNodes[0] }, restructureIntLooseInterFrontWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(),
			osizes.x - _numValidFrontNodes[0], (const int2*)_fronts.cbuf(0) + _numValidFrontNodes[0],
			_log.portobj<0>(),
			_fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));
		if (osizes.y - _numValidFrontNodes[1] > 0)
			configuredLaunch({ "RestructureExtLooseInterFrontWithLog", (int)osizes.y - _numValidFrontNodes[1] }, restructureExtLooseInterFrontWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(),
			osizes.y - _numValidFrontNodes[1], (const int2*)_fronts.cbuf(1) + _numValidFrontNodes[1],
			_log.portobj<0>(), (const int*)_pFixedDeformableBvh->restrLog().getRestrBvhRoot(),
			_fronts.nsizes(), _fronts.nbufs(), d_cpNum, getRawPtr(d_cpRes));

		_fronts.slide();
		reorderFronts();

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

		/// prune after restr
		_log.clear(_pFixedDeformableBvh->getExtNodeSize());
		_fronts.retrieveSizes();
		_fronts.resetNextSizes();

		uint osize;

		osize = _fronts.cs(0);
		configuredLaunch({ "PruneIntLooseInterFrontsWithLog", (int)osize }, pruneIntLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(0),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs());
		osize = _fronts.cs(1);
		configuredLaunch({ "PruneExtLooseInterFrontsWithLog", (int)osize }, pruneExtLooseInterFrontsWithLog,
			_pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), osize, (const int2*)_fronts.cbuf(1),
			_log.portobj<0>(), _fronts.nsizes(), _fronts.nbufs());

		_fronts.slide();
		reorderFronts();
		Logger::recordSection<TimerType::GPU>("broad_phase_restr_front");

		_restructured = true;
	}

	void BvttFrontLooseInter::pureBvhCd() {
		checkCudaErrors(cudaMemset(d_cpNum, 0, sizeof(int)));

		configuredLaunch({ "PureBvhInterCD", (int)_pRigidBvh->getPrimNodeSize() - 1 }, pureBvhInterCD,
			_pRigidBvh->getPrimNodeSize(), _pRigidBvh->cprim().portobj<0>(), _pFixedDeformableBvh->clvs().portobj<0>(), _pFixedDeformableBvh->ctks().portobj<0>(), d_cpNum, getRawPtr(d_cpRes));

		Logger::recordSection<TimerType::GPU>("broad_phase_cd_bvh");

		checkCudaErrors(cudaMemcpy(&_cpNum, d_cpNum, sizeof(int), cudaMemcpyDeviceToHost));
		checkThrustErrors(thrust::copy(getDevicePtr(d_cpRes), getDevicePtr(d_cpRes) + _cpNum, getDevicePtr(d_orderedCdpairs)));

	}

}
