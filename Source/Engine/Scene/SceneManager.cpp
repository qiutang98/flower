#include "Pch.h"
#include "SceneManager.h"
#include "SceneNode.h"
#include "Scene.h"

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