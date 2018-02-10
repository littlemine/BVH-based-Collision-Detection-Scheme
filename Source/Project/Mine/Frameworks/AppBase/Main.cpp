#include "Main.h"

namespace mn {

	Main::InitializerArray*	Main::s_pkInitializers = nullptr;
	Main::TerminatorArray*	Main::s_pkTerminators = nullptr;
	int						Main::s_iStartObjects = 0;
	int						Main::s_iFinalObjects = 0;

	void Main::AddInitializer(Initializer oInitialize) {
		if (!s_pkInitializers)
			s_pkInitializers = new InitializerArray;
		s_pkInitializers->push_back(oInitialize);
	}

	void Main::Initialize() {
		if (s_pkInitializers)
			for (int i = 0; i < s_pkInitializers->size(); i++)
				(*s_pkInitializers)[i]();
		delete s_pkInitializers;
		s_pkInitializers = nullptr;
	}

	void Main::AddTerminator(Terminator oTerminate) {
		if (!s_pkTerminators)
			s_pkTerminators = new TerminatorArray;
		s_pkTerminators->push_back(oTerminate);
	}

	void Main::Terminate() {
		if (s_pkTerminators)
			for (int i = 0; i < s_pkTerminators->size(); i++)
				(*s_pkTerminators)[i]();
		delete s_pkTerminators;
		s_pkTerminators = nullptr;
	}
}
