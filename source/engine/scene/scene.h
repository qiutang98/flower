#pragma once

#include "scene_graph.h"
#include "scene_node.h"
#include "component.h"
#include "scene_archive.h"

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
		virtual void release() override;

		// Get current active scene.
		std::shared_ptr<Scene> getActiveScene();

		// Release active scene.
		void releaseScene();

		// Save current active scene to path.
		bool saveScene(bool bBinary, const std::filesystem::path& relativeProjectRootPath);

		// Load scene from path into active scene.
		bool loadScene(const std::filesystem::path& loadPath);

	private:
		std::shared_ptr<Scene> m_scene = nullptr;
	};
}