#ifndef __MAIN_H_
#define __MAIN_H_

#include <vector>

namespace mn {

	class Main {
	public:
		typedef void(*Initializer)(void);
		typedef std::vector<Initializer>	InitializerArray;

		static void AddInitializer(Initializer oInitialize);	///< sub-system registration
		static void Initialize();

		typedef void(*Terminator)(void);
		typedef std::vector<Terminator>		TerminatorArray;

		static void AddTerminator(Terminator oTerminate);
		static void Terminate();

	private:
		static InitializerArray*	s_pkInitializers;
		static TerminatorArray*		s_pkTerminators;
		static int					s_iStartObjects;
		static int					s_iFinalObjects;
	};

}

#endif