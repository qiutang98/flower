#include "Pch.h"
#include "RenderSceneData.h"
#include "../Scene/SceneArchive.h"
#include "../Scene/SceneManager.h"
#include "RenderSettingContext.h"

namespace Flower
{
	// Loop all static mesh.
	// This function is slow when static mesh is a lot.
	void RenderSceneData::staticMeshCollect(Scene* scene)
	{
		// TODO: Don't collect every tick, add some cache method.
		m_collectStaticMeshes.clear();
		scene->loopComponents<StaticMeshComponent>([&](std::shared_ptr<StaticMeshComponent> comp) -> bool
		{
			comp->renderObjectCollect(m_collectStaticMeshes);

			// We need to loop all static mesh component.
			return false;
		});

		// Now update all static mesh info.
		if (!m_collectStaticMeshes.empty())
		{
			m_staticMeshesObjectsPtr = m_bufferParametersRing->getStaticStorage("StaticMeshObjects", sizeof(GPUPerObjectData) * m_collectStaticMeshes.size());
			m_staticMeshesObjectsPtr->buffer.updateDataPtr((void*)m_collectStaticMeshes.data());
		}
	}

	void RenderSceneData::pmxCollect(Scene* scene, VkCommandBuffer cmd)
	{
		m_collectPMXes.clear();
		m_cachePlayingPMXWithCamera = nullptr;

		scene->loopComponents<PMXComponent>([&](std::shared_ptr<PMXComponent> comp) -> bool
		{
			if(comp->pmxReady())
			{
				m_collectPMXes.push_back(comp); // shared_ptr to keep component alive.
				comp->onRenderTick(cmd);

				if (!m_cachePlayingPMXWithCamera && comp->isPMXCameraPlaying())
				{
					m_cachePlayingPMXWithCamera = comp;
				}
			}

			// We need to loop all static mesh component.
			return false;
		});
	}


	void RenderSceneData::lightCollect(Scene* scene)
	{
		// Load first directional light.
		std::vector<GPUDirectionalLightInfo> directionalLights = {};
		scene->loopComponents<DirectionalLightComponent>([&](std::shared_ptr<DirectionalLightComponent> comp) -> bool
		{
			GPUDirectionalLightInfo newDirectionalLight{};
			newDirectionalLight.intensity = comp->getIntensity();
			newDirectionalLight.color = comp->getColor();
			newDirectionalLight.direction = comp->getDirection();
			newDirectionalLight.shadowFilterSize = comp->getShadowFilterSize();
			newDirectionalLight.cascadeCount = comp->getCascadeCount();
			newDirectionalLight.perCascadeXYDim = comp->getPerCascadeDimXY();
			newDirectionalLight.splitLambda = comp->getCascadeSplitLambda();
			newDirectionalLight.shadowBiasConst = comp->getShadowBiasConst();
			newDirectionalLight.shadowBiasSlope = comp->getShadowBiasSlope();
			newDirectionalLight.cascadeBorderAdopt = comp->getCascadeBorderAdopt();
			newDirectionalLight.cascadeEdgeLerpThreshold = comp->getCascadeEdgeLerpThreshold();
			newDirectionalLight.maxDrawDistance = comp->getMaxDrawDepthDistance();
			newDirectionalLight.maxFilterSize = comp->getMaxFilterSize();
			directionalLights.push_back(newDirectionalLight);

			// NOTE: current we only support one directional light, so pre-return when first directional light collect finish.
			return true;
		});

		// Fill directional light infos. Current only support one directional light.
		{
			m_importanceLights.directionalLightCount = 0;

			if (directionalLights.size() > 0)
			{
				// Current use first directional light.
				m_importanceLights.directionalLights = directionalLights[0];

				m_earthAtmosphereInfo = RenderSettingManager::get()->earthAtmosphere.earthAtmosphere;

				m_importanceLights.directionalLightCount ++;
			}

			// Prepare cascade infos for directional lights.
			uint32_t cascadeCount = 1u;
			if (m_importanceLights.directionalLightCount > 0)
			{
				cascadeCount = m_importanceLights.directionalLights.cascadeCount;
			}

			// At least we create one cascade count buffer for feedback set.
			m_cascsadeBufferInfos = m_bufferParametersRing->getStaticStorageGPUOnly("CascadeInfos", sizeof(GPUCascadeInfo) * cascadeCount);
		}

		// TODO: Other light type gather.
	}

	void RenderSceneData::postprocessVolumeCollect(Scene* scene)
	{
		// TODO: post process volunme lerp.
		scene->loopComponents<PostprocessVolumeComponent>([&](std::shared_ptr<PostprocessVolumeComponent> comp) -> bool
		{
			m_postprocessVolumeInfo = comp->getSetting();

			// NOTE: current only use first postprocess volume.
			return true;
		});
	}

	RenderSceneData::RenderSceneData()
	{
		m_bufferParametersRing = std::make_unique<BufferParametersRing>();
	}

	void RenderSceneData::tick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd)
	{
		// Update buffer ring.
		m_bufferParametersRing->tick();

		// Find active scene.
		Scene* activeScene = GEngine->getRuntimeModule<SceneManager>()->getScenes();

		// Collect static mesh.
		staticMeshCollect(activeScene);

		pmxCollect(activeScene, cmd);

		// Collect light.
		lightCollect(activeScene);

		// Collect post process volume.
		postprocessVolumeCollect(activeScene);
	}
}

