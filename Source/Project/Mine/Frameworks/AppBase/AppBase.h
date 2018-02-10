#ifndef __APP_BASE_H_
#define __APP_BASE_H_


namespace mn {

	class AppBase {
	public:
		virtual ~AppBase();

		static AppBase* TheApplication;

		typedef int(*EntryPoint)(int, char**);
		static EntryPoint Run;

	protected:
		AppBase();

	};

}

#endif