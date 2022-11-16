#pragma once
#include "../RuntimeModule.h"
#include "../Core/Core.h"

namespace Flower
{
	class Scene;

	class SceneManager : public IRuntimeModule
	{
	public:
		SceneManager(ModuleManager* in);

		virtual bool init() override;
		virtual void release() override;
		virtual void tick(const RuntimeModuleTickData& tickData) override;

		Scene* getScenes();
		

		void releaseScene();

	private:
		std::shared_ptr<Scene> m_scene = nullptr;

	private:
		Scene* createEmptyScene();
	};

	
}