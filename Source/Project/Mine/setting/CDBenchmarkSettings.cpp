#include "CDBenchmarkSettings.h"

namespace mn {
	//std::string("E:\\data\\wxl-data\\cloth_ball.plys\\cloth_ball%d.ply"), 
	//94, std::string("clothball-92k"), 
	//std::string("E:\\data\\wxl-data\\ExpResults\\bvh-cd-benchmarks\\%s-%.3f.txt")
	int					stIdx, len;
	const std::string	inputFileFormat;
	std::string			expName;
	const std::string	outputFileFormat;

	/// default address for standalone benchmarks
	std::string	CDBenchmarkSettings::_inputBaseAddr("C:\\Mine\\Assets\\");
	std::string	CDBenchmarkSettings::_outputBaseAddr("C:\\Mine\\ExpResults\\standaloneCD\\gtx1060\\");
	//std::string	CDBenchmarkSettings::_inputBaseAddr("E:\\data\\wxl-data\\");
	//std::string	CDBenchmarkSettings::_outputBaseAddr("E:\\data\\wxl-data\\ExpResults\\bvh-cd-benchmarks\\");

	std::vector<Benchmark>	CDBenchmarkSettings::_benchmarks({
		//Benchmark{0, 76, "balls16_.plys\\balls16_%d.ply", "nbody-146k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		//Benchmark{0, 76, "balls16_.plys\\balls16_%d.ply", "nbody-146k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		//Benchmark{0, 76, "balls16_.plys\\balls16_%d.ply", "nbody-146k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		/*
		Benchmark{ 0, 252, "flowing_cloth\\%04d_00.obj", "flowingcloth","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		Benchmark{ 0, 252, "flowing_cloth\\%04d_00.obj", "flowingcloth","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		Benchmark{ 0, 252, "flowing_cloth\\%04d_00.obj", "flowingcloth","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		Benchmark{ 0, 252, "flowing_cloth\\%04d_00.obj", "flowingcloth","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		Benchmark{ 16, 45, "breakinglion\\%d.ply", "breakinglion","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		Benchmark{ 16, 45, "breakinglion\\%d.ply", "breakinglion","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		Benchmark{ 16, 45, "breakinglion\\%d.ply", "breakinglion","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		Benchmark{ 16, 45, "breakinglion\\%d.ply", "breakinglion","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		Benchmark{ 0, 337, "squishy_ball\\%04d_00.obj", "squishyball","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		Benchmark{ 0, 337, "squishy_ball\\%04d_00.obj", "squishyball","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		Benchmark{ 0, 337, "squishy_ball\\%04d_00.obj", "squishyball","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		Benchmark{ 0, 337, "squishy_ball\\%04d_00.obj", "squishyball","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		Benchmark{ 0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		Benchmark{ 1, 501, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		Benchmark{ 0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },

		Benchmark{0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		Benchmark{1, 501, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		Benchmark{0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },

		//Benchmark{ 0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_CLUSTERED_PAIRS, "bvhcdclusteredpairs" } },
		//Benchmark{ 1, 501, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_CLUSTERED_PAIRS, "bvhcdclusteredpairs" } },
		//Benchmark{ 0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_CLUSTERED_PAIRS, "bvhcdclusteredpairs" } },
		*/
		Benchmark{ 0, 76, "balls16_.plys\\balls16_%d.ply", "nbody-146k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		//Benchmark{ 0, 16, "dragbun.plys\\dragbun%d.ply", "dragbun-252k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		//Benchmark{ 0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },

		Benchmark{ 0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::REFIT_ONLY_FRONT, "refit_only" } },
		Benchmark{ 1, 500, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::REFIT_ONLY_FRONT, "refit_only" } },
		//Benchmark{ 0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::REFIT_ONLY_FRONT, "refit_only" } },
		//Benchmark{ 0, 210, "princess.plys\\princess%d.ply", "princess-40k","%s\\%s.txt",{ CDSchemeType::REFIT_ONLY_FRONT, "refit_only" } },

		Benchmark{ 0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },
		Benchmark{ 1, 500, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },
		//Benchmark{ 0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },
		//Benchmark{ 0, 210, "princess.plys\\princess%d.ply", "princess-40k","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		//Benchmark{ 0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::FRONT_GENERATE, "front_generate" } },
		//Benchmark{ 1, 501, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::FRONT_GENERATE, "front_generate" } },
		//Benchmark{ 0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::FRONT_GENERATE, "front_generate" } },
		//Benchmark{ 0, 210, "princess.plys\\princess%d.ply", "princess-40k","%s\\%s.txt",{ CDSchemeType::FRONT_GENERATE, "front_generate" } },

		//Benchmark{ 16, 45, "breakinglion\\%d.ply", "breakinglion","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		//Benchmark{ 16, 45, "breakinglion\\%d.ply", "breakinglion","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },
		//Benchmark{ 0, 101, "lion\\lion%d.ply", "lion","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		//Benchmark{ 0, 101, "lion\\lion%d.ply", "lion","%s\\%s.txt",{ CDSchemeType::REFIT_ONLY_FRONT, "refit_only" } },
		//Benchmark{ 0, 101, "lion\\lion%d.ply", "lion","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		/*
		Benchmark{0, 94, "cloth_ball.plys\\cloth_ball%d.ply", "clothball-92k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		Benchmark{1, 501, "funnel.plys\\%03d.ply", "funnel-18k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		Benchmark{0, 706, "flamenco5\\flamenco5.%d.ply", "flamenco-49k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },
		*/

		//Benchmark{0, 667, "cloth\\%04d_00.obj", "cloth-8k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		//Benchmark{0, 667, "cloth\\%04d_00.obj", "cloth-8k","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		//Benchmark{0, 210, "princess.plys\\princess%d.ply", "princess-40k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		//Benchmark{0, 210, "princess.plys\\princess%d.ply", "princess-40k","%s\\%s.txt",{ CDSchemeType::BVHCD_FROM_PAIRS, "bvhcdfrompairs" } },
		//Benchmark{0, 210, "princess.plys\\princess%d.ply", "princess-40k","%s\\%s.txt",{ CDSchemeType::STATIC_MANDATORY, "static_mandatory" } },

		//Benchmark{0, 101, "lion\\lion%d.ply", "lion-1470k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },

		//Benchmark{0, 101, "lion\\lion%d.ply", "lion-147k","%s\\%s.txt",{ CDSchemeType::GENERATE, "generate" } },

		//Benchmark{0, 16, "breakinglion\\%d.ply", "dragbun-252k","%s\\%s.txt",{ CDSchemeType::BVHCD, "bvhcd" } },
		//(make_tuple(std::string("E:\\Data\\CD\\data -lzh\\qman-gpu-better\\%04d_00.obj"), 632, std::string("qman-157k"), std::string("E:\\ExpResults\\bvh-cd-benchmarks\\%s-%.3f.txt")));
		//(make_tuple(std::string("E:\\Data\\CD\\data -lzh\\bridson-diff-res\\b1-256k\\%04d_00.obj"), 484, std::string("bridson-262k"), std::string("E:\\ExpResults\\bvh-cd-benchmarks\\%s-%.3f.txt")));
	});
}