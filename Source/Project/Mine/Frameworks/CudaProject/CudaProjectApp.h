#ifndef __CUDA_PROJECT_APP_H_
#define __CUDA_PROJECT_APP_H_

#include <string>
#include "base\Singleton.h"
#include "Frameworks\AppBase\AppBase.h"

// framework modules
#include "world\Scene.h"
#include "collision\lbvh\LBvh.h"
#include "collision\bvtt_front\BVTTFront.h"

class GLFWwindow;

namespace mn {

	class BenchmarkLogic;	///< or should be context?
	class CudaDevice;
	/*
	 *	@note Engine Framework, manages loop/ resources
	 */
	class SimpleApp : public AppBase {
	public:
		static void Initialize();
		static int Run(int iQuantity, char** apcArguments);
		static void Terminate();

		virtual int Main(int iQuantity, char** apcArguments);

	protected:
		SimpleApp() {}
		virtual ~SimpleApp() {}

	private:
		/// systems (initialized)
		CudaDevice*		TheCudaDevice;
		BenchmarkLogic* TheLogic;

		// for CD benchmark testing
		Scene								_scene;
		LBvhKernelRegister					_kLBvhKernelRegister;
		BvttFrontKernelRegister				_kFrontKernelRegister;
		std::unique_ptr<LBvh<ModelType::FixedDeformableType>>	
											_bvh;
		std::unique_ptr<BvttFront<BvttFrontType::LooseIntraType>>
											_fl;
	};

}

#endif