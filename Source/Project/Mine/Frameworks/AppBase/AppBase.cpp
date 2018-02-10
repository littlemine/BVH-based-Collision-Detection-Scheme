#include "AppBase.h"

namespace mn {

	AppBase* AppBase::TheApplication = nullptr;
	AppBase::EntryPoint AppBase::Run = nullptr;

	AppBase::AppBase() {}
	AppBase::~AppBase() {}

}