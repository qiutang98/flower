#include "scene_manager.h"
#include "../asset/asset_manager.h"

namespace engine
{
	SceneManager* engine::getSceneManager()
	{
		static SceneManager* sceneManager = Engine::get()->getRuntimeModule<SceneManager>();
		return sceneManager;
	}

	void SceneManager::registerCheck(Engine* engine)
	{
		ASSERT(engine->existRegisteredModule<AssetManager>(),
			"When scene enable, you must register asset manager module before scene.");
	}

	bool SceneManager::init()
	{
		m_onGameBeginHandle    = m_engine->onGameStart   .addLambda([this] { onGameBegin();    });
		m_onGamePauseHandle    = m_engine->onGamePause   .addLambda([this] { onGamePause();    });
		m_onGameContinueHandle = m_engine->onGameContinue.addLambda([this] { onGameContinue(); });
		m_onGameEndHandle      = m_engine->onGameStop    .addLambda([this] { onGameStop();     });

		return true;
	}

	bool SceneManager::tick(const RuntimeModuleTickData& tickData)
	{
		if (getAssetManager()->isProjectSetup())
		{
			getActiveScene()->tick(tickData);
		}

		return true;
	}

	bool SceneManager::beforeRelease()
	{
		return true;
	}

	bool SceneManager::release()
	{
		releaseScene();

		m_engine->onGameStart.remove(m_onGameBeginHandle);
		m_engine->onGamePause.remove(m_onGamePauseHandle);
		m_engine->onGameContinue.remove(m_onGameContinueHandle);
		m_engine->onGameStop.remove(m_onGameEndHandle);

		return true;
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
		if (!m_scene.lock())
		{
			// Build a default scene.
			AssetSaveInfo saveInfo = AssetSaveInfo::buildTemp("untitled" + Scene::getCDO()->getSuffix());
			m_scene = getAssetManager()->createAsset<Scene>(saveInfo);

			// Active scene switch now.
			onActiveSceneChange.broadcast(nullptr, m_scene.lock().get());
		}

		return m_scene.lock();
	}

	void SceneManager::releaseScene()
	{
		if (m_scene.lock())
		{
			// Active scene switch now.
			onActiveSceneChange.broadcast(m_scene.lock().get(), nullptr);
			m_scene.lock()->unload();
			m_scene = { };
		}
	}

	bool SceneManager::loadScene(const std::filesystem::path& loadPath)
	{
		if (m_scene.lock())
		{
			releaseScene();
		}

		if (auto newScene = getAssetManager()->getOrLoadAsset<Scene>(loadPath).lock())
		{
			m_scene = newScene;

			// Active scene switch now.
			onActiveSceneChange.broadcast(nullptr, m_scene.lock().get());

			return true;
		}

		return false;
	}
}