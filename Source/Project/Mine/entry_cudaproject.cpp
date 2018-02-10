#include "Frameworks\AppBase\Main.h"

#include "Frameworks\CudaProject\CudaProjectModuleRegister.h"	///< pick one application
#include "Frameworks\AppBase\AppBase.h"

bool mn::ModuleRegister::s_bInitializeRegistered = mn::ModuleRegister::RegisterInitialize();
bool mn::ModuleRegister::s_bTerminateRegistered = mn::ModuleRegister::RegisterTerminate();

int main(int argc, char** argv) {
	mn::Main::Initialize();

	int iExitCode = mn::AppBase::Run(argc, argv);	///< class type derived from AppBase

	mn::Main::Terminate();

	printf("Press any key to exit system.");
	getchar();
	return 0;
}