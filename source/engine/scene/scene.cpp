#include "scene.h"
#include <asset/asset_system.h>

namespace engine
{
	void SceneManager::registerCheck(Engine* engine)
	{

	}

	bool SceneManager::init()
	{
		m_onGameBeginHandle = m_engine->onGameStart.addLambda([this] { onGameBegin(); });
		m_onGamePauseHandle = m_engine->onGamePause.addLambda([this] { onGamePause(); });
		m_onGameContinueHandle = m_engine->onGameContinue.addLambda([this] { onGameContinue(); });
		m_onGameEndHandle = m_engine->onGameStop.addLambda([this] { onGameStop(); });
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

		m_engine->onGameStart.remove(m_onGameBeginHandle);
		m_engine->onGamePause.remove(m_onGamePauseHandle);
		m_engine->onGameContinue.remove(m_onGameContinueHandle);
		m_engine->onGameStop.remove(m_onGameEndHandle);

	}

	void SceneManager::onGameBegin()
	{
		getActiveScene()->onGameBegin();
	}

	void SceneManager::onGamePause()
	{
		getActiveScene()->onGamePause();
	}

	void SceneManager::onGameStop()
	{
		getActiveScene()->onGameStop();
	}


	void SceneManager::onGameContinue()
	{
		getActiveScene()->onGameContinue();
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