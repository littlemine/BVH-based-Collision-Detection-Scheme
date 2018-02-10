#ifndef __BENCHMARK_LOGIC_HPP_
#define __BENCHMARK_LOGIC_HPP_

#include "base\Singleton.h"
#include "setting\CDBenchmarkSettings.h"
#include "collision\bvtt_front\BvttFrontLooseIntra.h"
#include "collision\bvtt_front\BvttFrontLooseInter.h"

namespace mn {

	class BenchmarkLogic : public ManagedSingleton<BenchmarkLogic> {
	public:
		BenchmarkLogic() {}
		~BenchmarkLogic() {}

		bool next() {
			if (cnt - 1 > 0) {
				--cnt;
				++frameid;

				std::pair<std::string, std::string> pair = CDBenchmarkSettings::file_name(benchmarkid, frameid);
				inputFile = pair.first;
				outputFile = pair.second;
				benchmarkFinished = false;
				return true;
			}
			if (benchmarkid + 1 < CDBenchmarkSettings::benchmarkNum()) {	///< next benchmark
				++benchmarkid;
				exp = CDBenchmarkSettings::benchmark(benchmarkid);
				frameid = exp.stIdx;
				cnt = exp.len;

				std::pair<std::string, std::string> pair = CDBenchmarkSettings::file_name(benchmarkid, frameid);
				inputFile = pair.first;
				prevOutputFile = outputFile;
				outputFile = pair.second;
				benchmarkFinished = true;
				return true;
			}
			prevOutputFile = outputFile;
			inputFile = outputFile = std::string();
			if (!prevOutputFile.empty())
				benchmarkFinished = true;
			else
				benchmarkFinished = false;
			return false;
		}
		const std::string& currentInputFile() const { return inputFile; }
		const std::string& currentOutputFile() const { return outputFile; }
		const std::string& previousOutputFile() const { return prevOutputFile; }
		int currentFrameId() const { return frameid; }
		int currentBenchmarkId() const { return benchmarkid; }
		bool isBenchmarkFinished() const { return benchmarkFinished; }

		/// used in ARCSim version
		std::pair<LBvhFixedDeformableMaintenance, BvttFrontLooseIntraMaintenance> getSchemeOpt(int frameid, int schemeid = 0) const {
			///	scheme ID: 0 static, 1 periodic
			LBvhFixedDeformableMaintenance bvhOpt;
			BvttFrontLooseIntraMaintenance frontOpt;
			int frame;
			switch (schemeid) {
			case 0: 
				bvhOpt = frameid ? LBvhFixedDeformableMaintenance::REFIT : LBvhFixedDeformableMaintenance::BUILD;
				frame = frameid % BvttFrontSettings::mandatoryRebuildCycle();
				if (frame == 0)
					frontOpt = frameid == 0 ? BvttFrontLooseIntraMaintenance::GENERATE : BvttFrontLooseIntraMaintenance::UPDATE;
				else if (frame == BvttFrontSettings::mandatoryRebuildCycle() - 1)
					frontOpt = BvttFrontLooseIntraMaintenance::REORDER;
				else
					frontOpt = BvttFrontLooseIntraMaintenance::KEEP;
				break;
			case 1:
				bvhOpt = frameid % BvhSettings::mandatoryRebuildCycle() ? LBvhFixedDeformableMaintenance::REFIT : LBvhFixedDeformableMaintenance::BUILD;
				switch (frameid % BvttFrontSettings::mandatoryRebuildCycle()) {
				case 0: frontOpt = BvttFrontLooseIntraMaintenance::GENERATE; break;
				default: frontOpt = BvttFrontLooseIntraMaintenance::KEEP;
				}
				break;
			default:
				__assume(false);
			}
			return std::make_pair(bvhOpt, frontOpt);
		}

		/// used in standalone benchmarks
		std::pair<LBvhFixedDeformableMaintenance, BvttFrontLooseIntraMaintenance> maintainScheme() const {
			LBvhFixedDeformableMaintenance bvhOpt;
			BvttFrontLooseIntraMaintenance frontOpt;
			switch (exp.schemeOpt.type) {
			case CDSchemeType::GENERATE: 
				bvhOpt = LBvhFixedDeformableMaintenance::BUILD;
				frontOpt = BvttFrontLooseIntraMaintenance::PURE_BVH_CD;
				break;
			case CDSchemeType::FRONT_GENERATE: 
				bvhOpt = LBvhFixedDeformableMaintenance::BUILD;
				frontOpt = BvttFrontLooseIntraMaintenance::GENERATE;
				break;
			case CDSchemeType::STATIC_MANDATORY: ///< bvh cycle should sync with front cycle
				if (CDBenchmarkSettings::enableDivergentMark()) {
					bvhOpt = (frameid - exp.stIdx) % BvhSettings::mandatoryRebuildCycle() ? LBvhFixedDeformableMaintenance::REFIT : LBvhFixedDeformableMaintenance::BUILD;
					int frame = (frameid - exp.stIdx) % BvttFrontSettings::mandatoryRebuildCycle();
					if (frame == 0) 
						frontOpt = frameid == exp.stIdx ? BvttFrontLooseIntraMaintenance::GENERATE : BvttFrontLooseIntraMaintenance::UPDATE;
					else if (frame == BvttFrontSettings::mandatoryRebuildCycle() - 1) 
						frontOpt = BvttFrontLooseIntraMaintenance::REORDER;
					else 
						frontOpt = BvttFrontLooseIntraMaintenance::KEEP;

					if (bvhOpt == LBvhFixedDeformableMaintenance::BUILD)
						frontOpt = BvttFrontLooseIntraMaintenance::GENERATE;
					if ((frameid - exp.stIdx + 1) % BvhSettings::mandatoryRebuildCycle() == 0)
						frontOpt = BvttFrontLooseIntraMaintenance::KEEP;
					break;
				}
				else {
					bvhOpt = (frameid - exp.stIdx) % BvhSettings::mandatoryRebuildCycle() ? LBvhFixedDeformableMaintenance::REFIT : LBvhFixedDeformableMaintenance::BUILD;
					switch ((frameid - exp.stIdx) % BvttFrontSettings::mandatoryRebuildCycle()) {
					case 0: frontOpt = frameid == exp.stIdx ? BvttFrontLooseIntraMaintenance::GENERATE : BvttFrontLooseIntraMaintenance::UPDATE; break;
					default: frontOpt = BvttFrontLooseIntraMaintenance::KEEP;
					}
					if (bvhOpt == LBvhFixedDeformableMaintenance::BUILD)
						frontOpt = BvttFrontLooseIntraMaintenance::GENERATE;
					break;
				}
			case CDSchemeType::REFIT_ONLY_FRONT: ///< bvh cycle should sync with front cycle
				if (CDBenchmarkSettings::enableDivergentMark()) {
					bvhOpt = (frameid - exp.stIdx) ? LBvhFixedDeformableMaintenance::REFIT : LBvhFixedDeformableMaintenance::BUILD;
					int frame = (frameid - exp.stIdx) % BvttFrontSettings::mandatoryRebuildCycle();
					if (frame == 0) 
						frontOpt = frameid == exp.stIdx ? BvttFrontLooseIntraMaintenance::GENERATE : BvttFrontLooseIntraMaintenance::UPDATE; 
					else if (frame == BvttFrontSettings::mandatoryRebuildCycle() - 1) 
						frontOpt = BvttFrontLooseIntraMaintenance::REORDER; 
					else
						frontOpt = BvttFrontLooseIntraMaintenance::KEEP;
				}
				else {
					bvhOpt = (frameid - exp.stIdx) ? LBvhFixedDeformableMaintenance::REFIT : LBvhFixedDeformableMaintenance::BUILD;
					switch ((frameid - exp.stIdx) % BvttFrontSettings::mandatoryRebuildCycle()) {
					case 0: frontOpt = frameid == exp.stIdx ? BvttFrontLooseIntraMaintenance::GENERATE : BvttFrontLooseIntraMaintenance::UPDATE; break;
					default: frontOpt = BvttFrontLooseIntraMaintenance::KEEP;
					}
					break;
				}
			default:
				__assume(false);
				//__builtin_unreachable();
			}
			return std::make_pair(bvhOpt, frontOpt);
		}

	private:
		Benchmark	exp;
		int			benchmarkid{ -1 };
		int			frameid{};
		int			cnt{ 1 };
		std::string	inputFile{}, outputFile{}, prevOutputFile{};
		bool		benchmarkFinished{ false };
	};

}

#endif