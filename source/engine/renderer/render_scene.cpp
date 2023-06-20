#include "render_scene.h"
#include <scene/component/static_mesh.h>
#include <scene/scene.h>

namespace engine
{
	RenderScene::RenderScene(VulkanContext* context, SceneManager* sceneManager)
		: m_context(context), m_sceneManager(sceneManager)
	{

	}

	RenderScene::~RenderScene()
	{

	}

	void RenderScene::tick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd)
	{
		auto activeScene = m_sceneManager->getActiveScene();

		renderObjectCollect(tickData, activeScene.get(), cmd);

		tlasPrepare(tickData, activeScene.get(), cmd);

		lightCollect(activeScene.get(), cmd);



		// TODO: post process volunme lerp.
		activeScene->loopComponents<PostprocessVolumeComponent>([&](std::shared_ptr<PostprocessVolumeComponent> comp) -> bool
		{
			m_postprocessVolumeInfo = comp->getSetting();

			// NOTE: current only use first postprocess volume.
			return true;
		});
	}

	bool RenderScene::shouldRenderSDSM() const
	{
		bool bExistMeshNeedRender = isStaticMeshExist() || isTerrainExist();
		return isSkyExist() && (bExistMeshNeedRender);
	}


	bool RenderScene::isASValid() const
	{
		return !m_cacheASInstances.empty() && m_tlas.isInit();
	}

	// All dynamic, no cache yet, I don't want to make the logic too complex, keep simple make life easy.
	void RenderScene::renderObjectCollect(const RuntimeModuleTickData& tickData, Scene* scene, VkCommandBuffer cmd)
	{
		// Clear AS instance.
		m_cacheASInstances.clear();
		m_staticmeshObjects.clear();
		m_collectPMXes.clear();

		// Collect all terrain object.
		m_terrainComponents.clear();
		scene->loopComponents<TerrainComponent>([&](std::shared_ptr<TerrainComponent> comp) -> bool
		{
			m_terrainComponents.push_back(comp);
			return false;
		});

		// Collect all pmx mesh object.
		scene->loopComponents<PMXComponent>([&](std::shared_ptr<PMXComponent> comp) -> bool
		{
			comp->onRenderTick(tickData, cmd, m_staticmeshObjects, m_cacheASInstances);
			m_collectPMXes.push_back(comp);
			return false;
		});

		// Static mesh.
		scene->loopComponents<StaticMeshComponent>([&](std::shared_ptr<StaticMeshComponent> comp) -> bool
		{
			comp->renderObjectCollect(m_staticmeshObjects, m_cacheASInstances);

			// We need to loop all static mesh component.
			return false;
		});

		// Now update all static mesh info.
		if (!m_staticmeshObjects.empty())
		{
			m_staticmeshObjectsGPU = getContext()->getBufferParameters().getStaticStorage("StaticMeshObjects", sizeof(m_staticmeshObjects[0]) * m_staticmeshObjects.size());
			m_staticmeshObjectsGPU->updateDataPtr((void*)m_staticmeshObjects.data());
		}
	}

	void RenderScene::lightCollect(Scene* scene, VkCommandBuffer cmd)
	{
		// Sky component.
		scene->loopComponents<SkyComponent>([&](std::shared_ptr<SkyComponent> comp) -> bool
		{
			// Cache weak pointer. when move to multi-thread rendering, will require cache owner pointer or copy resource. but current just use weak ptr is enough.
			m_sky = comp;

			m_skyGPU.intensity = comp->getIntensity();
			m_skyGPU.color = comp->getColor();
			m_skyGPU.direction = comp->getDirection();

			m_skyGPU.cacsadeConfig = comp->getCacsadeConfig();
			m_skyGPU.atmosphereConfig = comp->getAtmosphereConfig();
			m_skyGPU.rayTraceShadow = comp->isRayTraceShadow() ? 1 : 0;

			// Current we only support one sky light, so pre-return when first sky light collect finish.
			return true;
		});
	}


	void RenderScene::tlasPrepare(const RuntimeModuleTickData& tickData, Scene* scene, VkCommandBuffer cmd)
	{
		// When instance is empty, destroy tlas and pre-return.
		if (m_cacheASInstances.empty())
		{
			m_tlas.destroy();
			return;
		}

		// Sometimes need rebuild. clear here.
		if (false)
		{

			m_tlas.destroy();
		}

		// Update or build TLAS.
		m_tlas.buildTlas(cmd, m_cacheASInstances, m_tlas.isInit());
	}
}