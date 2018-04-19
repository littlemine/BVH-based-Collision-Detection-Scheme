#ifndef __CD_BENCHMARK_SETTINGS_H_
#define __CD_BENCHMARK_SETTINGS_H_

#include <vector>

namespace mn {

	enum class CDSchemeType { STATIC_MANDATORY, REFIT_ONLY_FRONT, GENERATE, FRONT_GENERATE };

	#define MACRO_VERSION 0

	struct Scheme {
		CDSchemeType	type;
		std::string		tag;
		Scheme() {}
		Scheme(CDSchemeType type, std::string tag) : type(type), tag(std::move(tag)) {}
	};

	struct Benchmark {
		int				stIdx, len;
		std::string		inputFileFormat;
		std::string		expName;
		std::string		outputFileFormat;
		Scheme			schemeOpt;
	};

	class CDBenchmarkSettings {
	public:
		static bool enableRestr() { return _enableRestr; }
		static bool enableDivergentMark() { return _enableDivergentMark; }
		static bool includeNarrowPhase() { return _includeNarrowPhase; }
		static int version() { return _version; }
		static int benchmarkNum() { return _benchmarks.size(); }
		static Benchmark benchmark(int i) { return _benchmarks[i]; }
		static std::pair<std::string, std::string> file_name(int expNo, int frameNo) {
			char tmp[500];
			std::pair<std::string, std::string>	pair;
			sprintf(tmp, benchmark(expNo).inputFileFormat.c_str(), frameNo);
			pair.first = _inputBaseAddr + std::string(tmp);
			sprintf(tmp, benchmark(expNo).outputFileFormat.c_str(), benchmark(expNo).schemeOpt.tag.c_str(), benchmark(expNo).expName.c_str());
			pair.second = _outputBaseAddr + std::string(tmp);
			return pair;
		}
		
	private:
		/// only allow access to static methods
		CDBenchmarkSettings() = delete;
		~CDBenchmarkSettings() = delete;

		static std::string		_inputBaseAddr;
		static std::string		_outputBaseAddr;
		static std::vector<Benchmark>	_benchmarks;
		static  const bool		_enableRestr{ true };
		static  const bool		_enableDivergentMark{ true };
		static  const bool		_includeNarrowPhase{ true };

		static	const int		_version{ MACRO_VERSION };	///< 0: standalone 1: arcsim
	};

}

#endif