/**	\file	BvttFrontSettings.h
*	\brief	BVTT front configuration
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVTT_FRONT_SETTINGS_H_
#define __BVTT_FRONT_SETTINGS_H_

#include "BvhSettings.h"
#include <memory>

namespace mn {

	using uint = unsigned int;
	using uchar = unsigned char;

	enum struct BvttFrontType { LooseInterType, LooseIntraType, BvttFrontTypeNum };

	class BvttFrontSettings {				///< static (pre-defined) setting
	public:
		static  int ext_front_size() { return _maxExtFrontSize; }
		static  int int_front_size() { return _maxIntFrontSize; }
		static  int collision_pair_num() { return _maxCollisionPairNum; }
		static  int mandatoryRebuildCycle() { return _mandatoryRebuildCycle; }

	private:
		/// only allow access to static methods
		BvttFrontSettings() = delete;
		~BvttFrontSettings() = delete;

		//static const int	_maxExtFrontSize = 33554432;	//(1 << 25)
		//static const int	_maxExtFrontSize = 16777216;	//(1 << 24)
		static const int	_maxExtFrontSize = 2300000;
		//static const int	_maxExtFrontSize = 50000000;

		//static const int	_maxIntFrontSize = 16777216;	//(1 << 24)
		static const int	_maxIntFrontSize = 2300000;
		//static const int	_maxIntFrontSize = 33554432;
		//static const int	_maxIntFrontSize = 50000000;
		
		//static const int	_maxCollisionPairNum = 8388608;		//(1 << 23)
		static const int	_maxCollisionPairNum = 1600000;
		//static const int	_maxCollisionPairNum = 4194304;		//(1 << 22)
		//static const int	_maxCollisionPairNum = 2000000;
		//static const int	_maxCollisionPairNum = 35000000;
		//static const int	_maxCollisionPairNum = 25000000;

		static const int	_mandatoryRebuildCycle = 6;
	};

	template<typename Bvh>
	struct BvttFrontIntraBuildConfig {
		Bvh* pbvh{ nullptr };
		BvttFrontType type{ BvttFrontType::LooseIntraType };
		int extFrontSize{ BvttFrontSettings::ext_front_size() };
		int intFrontSize{ BvttFrontSettings::int_front_size() };
		int cpNum{ BvttFrontSettings::collision_pair_num() };
		int extNodeSize{ BvhSettings::ext_node_size() };
		int intNodeSize{ BvhSettings::int_node_size() };
		BvttFrontIntraBuildConfig(Bvh* bvh, BvttFrontType type, int extSize, int intSize, int enNum, int inNum) :
			pbvh(bvh), type(type), extFrontSize(extSize), intFrontSize(intSize), extNodeSize(enNum), intNodeSize(inNum) {}
	};

	template<typename BvhA, typename BvhB>
	struct BvttFrontInterBuildConfig {
		BvhA* pbvha{ nullptr };	///< should directly be primitive
		BvhB* pbvhb{ nullptr };
		BvttFrontType type{ BvttFrontType::LooseInterType };
		int extFrontSize{ BvttFrontSettings::ext_front_size() };
		int intFrontSize{ BvttFrontSettings::int_front_size() };
		int cpNum{ BvttFrontSettings::collision_pair_num() };
		int extNodeSize{ BvhSettings::ext_node_size() };
		int intNodeSize{ BvhSettings::int_node_size() };
		BvttFrontInterBuildConfig(BvhA* a, BvhB* b, BvttFrontType type, int extSize, int intSize, int enNum, int inNum) :
			pbvha(a), pbvhb(b), type(type), extFrontSize(extSize), intFrontSize(intSize), extNodeSize(enNum), intNodeSize(inNum) {}
	};
}

#endif