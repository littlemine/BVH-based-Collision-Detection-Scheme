#include "Scene.h"
#include <filesystem>
#include "system\Logic\BenchmarkLogic.hpp"

namespace mn {

	Scene::Scene() { clearMeshes(); }
	Scene::Scene(ModelReader loader) : _kModelLoader(loader) { clearMeshes(); }
	Scene::~Scene() {}

	bool Scene::update() {
		using namespace std::experimental::filesystem;
		clearMeshes();
		const std::string &inputFile = BenchmarkLogic::getInstance()->currentInputFile();
		path inputTarget(inputFile);
		if (!exists(inputTarget)) {
			printf("\nFail loading frame %d [%s]\n", BenchmarkLogic::getInstance()->currentFrameId(), inputFile.c_str());
			return false;
		}
		printf("\nLoaded frame %d [%s]\n", BenchmarkLogic::getInstance()->currentFrameId(), inputFile.c_str());
		addMesh(inputFile);
		return true;
	}

	void Scene::addMesh(const std::string& fn) {
		_kModelLoader.loadMesh(fn.c_str());
		const aiScene* scene = _kModelLoader.getMesh();
		for (int i = 0; i < scene->mNumMeshes; i++) {
			for (int j = 0; j < scene->mMeshes[i]->mNumVertices; j++) {
				_kFormatedData.pos.emplace_back(MakePoint<ExtentType>::p(scene->mMeshes[i]->mVertices[j].x, scene->mMeshes[i]->mVertices[j].y, scene->mMeshes[i]->mVertices[j].z));
			}
			for (int j = 0; j < scene->mMeshes[i]->mNumFaces; j++) {
				_kFormatedData.fids.emplace_back(make_int3(_kFormatedData.offset + scene->mMeshes[i]->mFaces[j].mIndices[0],
					_kFormatedData.offset + scene->mMeshes[i]->mFaces[j].mIndices[1], _kFormatedData.offset + scene->mMeshes[i]->mFaces[j].mIndices[2]));
			}
			_kFormatedData.offset += scene->mMeshes[i]->mNumVertices;
		}
		printf("Loaded %d meshes\n", scene->mNumMeshes);
	}

	void Scene::clearMeshes() {
		_kFormatedData.offset = 0;
		_kFormatedData.fids.clear();
		_kFormatedData.pos.clear();
	}

}
