#ifndef __CD_BENCHMARK_CONFIGURATION_H_
#define __CD_BENCHMARK_CONFIGURATION_H_

#include "CDBenchmarkSettings.h"
#include <string>
#include <vector>

namespace mn {

	class CDBenchmarkConfiguration {
	public:
		CDBenchmarkConfiguration(std::string iptAddr, std::string optAddr) :
			_inputBaseAddr(std::move(iptAddr)), _outputBaseAddr(std::move(optAddr)) {}
			

	private:
		std::string		_inputBaseAddr;
		std::string		_outputBaseAddr;
		bool			_bSchemeOverride {false};
	};

}

#endif