/**	\file	BvhSettings.h
*	\brief	BVH configuration
*	\author	littlemine
*	\ref	article - Efficient BVH-based collision detection scheme for deformable models with ordering and restructuring
*/

#ifndef __BVH_SETTINGS_H_
#define __BVH_SETTINGS_H_

namespace mn {

	using uint = unsigned int;
	using uchar = unsigned char;

	enum struct ModelType { FixedDeformableType, AdaptiveDeformableType, RigidType, ParticleType, ModelTypeNum };

#define SINGLE_SUBTREE_RESTR_SIZE_UPPER_THRESHOLD 100
#define SINGLE_SUBTREE_RESTR_SIZE_LOWER_THRESHOLD 4
#define RESTR_BASE_QUALITY_METRIC 1.111
#define RESTR_WARNING_QUALITY_METRIC 1.25

	class BvhSettings {				///< static (pre-defined) setting
	public:
		static const int prim_size() { return _maxPrimitiveSize; }
		static const int ext_node_size() { return _maxExtNodeSize; }
		static const int int_node_size() { return _maxIntNodeSize; }

		static  int mandatoryRebuildCycle() { return _mandatoryRebuildCycle; }
		static uchar& maintainScheme() { return _maintainScheme; }
	private:
		/// only allow access to static methods
		BvhSettings() = delete;
		~BvhSettings() = delete;
		
		static const int	_maxPrimitiveSize = 150000;				///< (1 << 19);
		static const int	_maxExtNodeSize = 150000;				///< (1 << 19);
		//static const int	_maxPrimitiveSize = 524288;				///< (1 << 19);
		//static const int	_maxExtNodeSize = 524288;				///< (1 << 19);
		//static const int	_maxPrimitiveSize = 4194304;			///< (1 << 22);
		//static const int	_maxExtNodeSize = 4194304;				///< (1 << 22);
		//static const int	_maxPrimitiveSize = 2097152;			///< (1 << 21);
		//static const int	_maxExtNodeSize = 2097152;				///< (1 << 21);
		static const int	_maxIntNodeSize = _maxExtNodeSize - 1;	///< (1 << 19);

		static const int	_mandatoryRebuildCycle = 12;
		static uchar		_maintainScheme;
	};

	struct LBvhBuildConfig {
		int primSize{ BvhSettings::prim_size() };
		int extSize{ BvhSettings::ext_node_size() };
		int intSize{ BvhSettings::int_node_size() };
		ModelType type{ ModelType::FixedDeformableType };
	};
}

#endif