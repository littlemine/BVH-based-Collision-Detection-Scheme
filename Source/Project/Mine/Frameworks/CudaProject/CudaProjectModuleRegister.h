#ifndef __MODULE_REGISTER_H_
#define __MODULE_REGISTER_H_

namespace mn {

	class ModuleRegister {
	public:
		static bool RegisterInitialize();
		static bool RegisterTerminate();

	private:
		static bool s_bInitializeRegistered;
		static bool s_bTerminateRegistered;
	};

}

#endif