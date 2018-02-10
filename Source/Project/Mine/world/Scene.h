#ifndef __SCENE_H_
#define __SCENE_H_

#include <cuda_runtime.h>
#include "setting\CDBenchmarkSettings.h"

#if MACRO_VERSION
#include "cuda\box.cuh"
#else
#include "utility\Meta.h"
#include "utility\ModelReader.hpp"
#endif

namespace mn {

	struct SceneData {
		int						offset = 0;
		std::vector<PointType>	pos;
		std::vector<int3>		fids;
	};

	class Scene {
	public:
		Scene();
		Scene(ModelReader loader);
		~Scene();

		const SceneData& getFormatedData() const { return _kFormatedData; }

		bool update();
		void addMesh(const std::string& fn);
		void clearMeshes();

	private:
		ModelReader		_kModelLoader;
		SceneData		_kFormatedData;
	};

#if MACRO_VERSION
	struct ARCSimSceneData {
		int fsize, psize;
		uint3 *faces;
		//double3 *points;
		g_box *boxes;	///< 2*n boxes, bounding boxes(min + max)
	};
#endif

}

#endif