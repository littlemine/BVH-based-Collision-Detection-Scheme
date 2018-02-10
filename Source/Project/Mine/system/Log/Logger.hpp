#ifndef __LOGGER_HPP_
#define __LOGGER_HPP_

#include "base\Singleton.h"
#include "utility\CudaTimer.hpp"
#include "utility\Timer.hpp"
#include <iostream>
#include <filesystem>

namespace mn {

	enum class TimerType { CPU, GPU };

	class Logger : public ManagedSingleton<Logger> {
	public:
		Logger()	{}
		~Logger()	{}

		template<TimerType>
		static void tick();
		template<TimerType>
		static void tock();
		template<TimerType>
		static void tock(std::string message);
		template<TimerType>
		static void blankLine();
		template<TimerType>
		static void recordSection(std::string msg);

		static void message(const std::string& filename);

		static void record(const std::string& filename);
		static void insertRecord(const std::string& filename);

	private:
		CudaTimer	_kGPUTimer;
		Timer		_kCPUTimer;
		std::vector<std::string>	_kInfos;
	};

	template <>
	inline void Logger::tick<TimerType::GPU>() { getInstance()->_kGPUTimer.tick(); }
	template <>
	inline void Logger::tick<TimerType::CPU>() { getInstance()->_kCPUTimer.tick(); }

	template <>
	inline void Logger::tock<TimerType::GPU>(std::string message) { 
		//getInstance()->_kGPUTimer.tock();
		getInstance()->_kGPUTimer.record(message);
		//std::cout << message.c_str() << ": " << getInstance()->_kGPUTimer.elapsed() << std::endl; 
		std::cout << message.c_str() << ": " << getInstance()->_kGPUTimer << std::endl; 
	}
	template <>
	inline void Logger::tock<TimerType::CPU>(std::string message) { 
		getInstance()->_kCPUTimer.record(message);
		std::cout << message.c_str() << ": " << getInstance()->_kCPUTimer << std::endl; 
	}
	template<>
	inline void Logger::recordSection<TimerType::GPU>(std::string msg) { getInstance()->_kGPUTimer.recordSection(msg); }
	template<>
	inline void Logger::blankLine<TimerType::GPU>() { getInstance()->_kGPUTimer.blankLine(); }
	template<>
	inline void Logger::blankLine<TimerType::CPU>() { getInstance()->_kCPUTimer.blankLine(); }

	inline void Logger::record(const std::string& filename) {
		using namespace std::experimental::filesystem;
		path outputTarget(filename);
		if (outputTarget.empty()) return;
		if (!exists(outputTarget.parent_path()))
			create_directory(outputTarget.parent_path());

		std::ofstream ofs;
		while (exists(outputTarget)) {
			outputTarget = outputTarget.parent_path().string() + "\\" + outputTarget.stem().string() + "_" + outputTarget.extension().string();
		}
		ofs.open(outputTarget.string());
		if (ofs.is_open()) {
			getInstance()->_kCPUTimer.log(ofs, true);
			ofs << '\n';

			for (auto& str : getInstance()->_kInfos)
				ofs << str << '\n';
			ofs << '\n';
			getInstance()->_kInfos.clear();

			getInstance()->_kGPUTimer.log(ofs, true);
			ofs << '\n';

			ofs.close();
		}
	}

	inline void Logger::insertRecord(const std::string& filename) {
		std::ofstream ofs;
		ofs.open(filename, std::ios_base::app);
		printf("writing to %s\n", filename.c_str());
		if (ofs.is_open()) {
			getInstance()->_kCPUTimer.log(ofs, true);
			ofs << '\n';

			for (auto& str : getInstance()->_kInfos)
				ofs << str << '\n';
			ofs << '\n';
			getInstance()->_kInfos.clear();

			getInstance()->_kGPUTimer.log(ofs, true);
			ofs << '\n';

			ofs.close();
		}
	}

	inline void Logger::message(const std::string& filename) {
		getInstance()->_kInfos.emplace_back(filename);
	}
}

#endif