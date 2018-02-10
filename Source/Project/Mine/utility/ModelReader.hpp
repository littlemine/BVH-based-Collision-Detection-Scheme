#ifndef __MODEL_READER_HPP_
#define __MODEL_READER_HPP_

#include <vector>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace mn {

	class ModelReader {
	public:
		ModelReader() { _scenes.clear(); }
		~ModelReader() { release(); }

		const aiScene* getMesh(int no = -1) {
			if (no == -1) return _scenes.back();
			else {
				assert(no < _scenes.size());
				return _scenes[no];
			}
		}
		void loadMesh(const char* str) {
			release();
			const aiScene* scene = aiImportFile(str, aiProcess_Triangulate | aiProcess_GenSmoothNormals);
			_scenes.push_back(scene);
			
		}
		void release() {
			if (!_scenes.empty()) {
				for (auto scene : _scenes)
					aiReleaseImport(scene);
				_scenes.clear();
			}
		}

	private:
		std::vector<const aiScene*> _scenes;

	};

}

#endif