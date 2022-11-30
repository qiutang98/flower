#include "Pch.h"
#include "SceneManager.h"
#include "SceneNode.h"
#include "Scene.h"
#include "SceneArchive.h"
#include "Project.h"
#include "../Renderer/Renderer.h"

namespace Flower
{
	SceneManager::SceneManager(ModuleManager* in)
		: IRuntimeModule(in, "SceneManager")
	{

	}

	bool SceneManager::init()
	{
		return true;
	}

	void SceneManager::tick(const RuntimeModuleTickData& tickData)
	{
		if (auto* scene = m_scene.get())
		{
			scene->tick(tickData);
		}
	}

	void SceneManager::release()
	{
		releaseScene();
	}

	void SceneManager::releaseScene()
	{
		m_scene = nullptr;
	}

	void SceneManager::saveScene(const std::filesystem::path& savePath)
	{
		getScenes();
		m_scene->setSavePath(savePath.string());

		std::ofstream os(ProjectContext::get()->path / savePath);
		cereal::JSONOutputArchive oarchive(os);

		oarchive(m_scene);

		
		m_scene->setDirty(false);
	}

	void SceneManager::loadScene(const std::filesystem::path& loadPath)
	{
		getScenes();

		std::ifstream is(ProjectContext::get()->path / loadPath);
		cereal::JSONInputArchive iarchive(is);

		iarchive(m_scene);
		m_scene->setDirty(false);
	}

	Scene* SceneManager::getScenes()
	{
		if (m_scene == nullptr)
		{
			createEmptyScene();
		}

		return m_scene.get();
	}

	Scene* SceneManager::createEmptyScene()
	{
		CHECK(m_scene == nullptr);

		m_scene = Scene::create();

		m_scene->init();
		m_scene->setDirty(false);

		return m_scene.get();;
	}
}