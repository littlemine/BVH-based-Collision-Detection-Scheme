#include "CudaProjectApp.h"
#include <utility\CudaHostUtils.h>

// systems
#include "system\Log\Logger.hpp"
#include "system\CudaDevice\CudaDevice.h"
#include "system\Logic\BenchmarkLogic.hpp"

namespace mn {

	void SimpleApp::Initialize() {
		SimpleApp* app;
		AppBase::Run = &SimpleApp::Run;
		AppBase::TheApplication = app = new SimpleApp;
		app->TheCudaDevice = CudaDevice::getInstance();
		app->TheLogic = BenchmarkLogic::getInstance();
		printf("* Finishing App initialization!\n");
	}

	int SimpleApp::Run(int iQuantity, char** apcArguments) {
		auto pkTheApp = dynamic_cast<SimpleApp*>(TheApplication);
		return pkTheApp->Main(iQuantity, apcArguments);
	}

	int SimpleApp::Main(int iQuantity, char** apcArguments) {
		reportMemory();
		printf("Begin allocating memory for BVH\n");
		_bvh = std::make_unique<LBvh<ModelType::FixedDeformableType>>(LBvhBuildConfig{});
		printf("Begin allocating memory for BVTT fronts\n");
		_fl = std::make_unique<BvttFront<BvttFrontType::LooseIntraType>>(mn::BvttFrontIntraBuildConfig<mn::LBvh<mn::ModelType::FixedDeformableType>>(
			 _bvh.get(), BvttFrontType::LooseIntraType,
				BvttFrontSettings::ext_front_size(), BvttFrontSettings::int_front_size(),
				BvhSettings::ext_node_size(), BvhSettings::int_node_size() 
		));
		printf("End GPU memory allocations\n");

		// Main loop
		while (true) {
			if (!TheLogic->next()) {
				if (TheLogic->isBenchmarkFinished())
					Logger::record(TheLogic->previousOutputFile());
				if (getchar() == 'q') break;
				continue;
			}
			if (!_scene.update()) continue;
			if (TheLogic->isBenchmarkFinished())
				Logger::record(TheLogic->previousOutputFile());

			auto maintainOpts = TheLogic->maintainScheme();

			_bvh->maintain(maintainOpts.first, _scene.getFormatedData());
			_fl->maintain(maintainOpts.second);

			Logger::blankLine<TimerType::GPU>();
		}
		_fl.reset();
		_bvh.reset();
		return 0;
	}

	void SimpleApp::Terminate() {
	}

}
