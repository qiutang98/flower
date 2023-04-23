#include "scene.h"
#include <asset/asset_system.h>

namespace engine
{
	void SceneManager::registerCheck(Engine* engine)
	{

	}

	bool SceneManager::init()
	{
		return true;
	}

	bool SceneManager::tick(const RuntimeModuleTickData& tickData)
	{
		getActiveScene()->tick(tickData);

		return true;
	}

	void SceneManager::release()
	{
		releaseScene();
	}

	std::shared_ptr<Scene> SceneManager::getActiveScene()
	{
		if (m_scene == nullptr)
		{
			m_scene = Scene::create();
			m_scene->init();
			m_scene->setDirty(false);
		}

		return m_scene;
	}

	void SceneManager::releaseScene()
	{
		m_scene = nullptr;
	}


	bool SceneManager::saveScene(bool bBinary, const std::filesystem::path& relativeProjectRootPath)
	{
		return true;
	}

	bool SceneManager::loadScene(const std::filesystem::path& loadPath)
	{
		// Reload active scene.
		if (!getActiveScene()->savePathUnvalid())
		{
			getAssetSystem()->reloadAsset<Scene>(getActiveScene());
		}


		auto copyPath = loadPath;
		const auto relativePath = buildRelativePathUtf8(getAssetSystem()->getProjectRootPath(), copyPath.replace_extension());

		auto newScene = std::static_pointer_cast<Scene>(getAssetSystem()->getAssetByRelativeMap(relativePath));
		if (newScene)
		{
			m_scene = newScene;
			return true;
		}


		return false;
	}
}