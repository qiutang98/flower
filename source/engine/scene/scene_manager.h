#pragma once
#include "scene.h"
#include "scene_node.h"
#include "component.h"
#include "../engine.h"

namespace engine
{
	class SceneManager final : public IRuntimeModule
	{
	public:
		SceneManager(Engine* engine) : IRuntimeModule(engine) { }
		~SceneManager() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual bool beforeRelease() override;
		virtual bool release() override;

		void onGameBegin();
		void onGamePause();
		void onGameContinue();
		void onGameStop();

		// Get current active scene.
		std::shared_ptr<Scene> getActiveScene();

		// Release active scene.
		void releaseScene();

		// Load scene from path into active scene.
		bool loadScene(const std::filesystem::path& loadPath);

		// Event when active scene change.
		MulticastDelegate<Scene*/*old*/, Scene*/*new*/> onActiveSceneChange;

	private:
		std::weak_ptr<Scene> m_scene;

		DelegateHandle m_onGameBeginHandle;
		DelegateHandle m_onGameEndHandle;
		DelegateHandle m_onGamePauseHandle;
		DelegateHandle m_onGameContinueHandle;
	};

	extern SceneManager* getSceneManager();
}