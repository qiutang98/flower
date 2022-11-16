#include "Pch.h"
#include "RenderSceneData.h"
#include "../Scene/Scene.h"
#include "../Scene/Component/StaticMesh.h"
#include "../Scene/Component/DirectionalLight.h"
#include "../Scene/SceneManager.h"
#include "../Scene/Component/DirectionalLight.h"
#include "RenderSettingContext.h"

namespace Flower
{
	// Loop all static mesh.
	// This function is slow when static mesh is a lot.
	void RenderSceneData::staticMeshCollect(Scene* scene)
	{
		// TODO: Don't collect every tick, add some cache method.
		m_collectStaticMeshes.clear();
		scene->loopComponents<StaticMeshComponent>([&](std::shared_ptr<StaticMeshComponent> comp) 
		{
			comp->renderObjectCollect(m_collectStaticMeshes);
		});

		if (!m_collectStaticMeshes.empty())
		{
			m_staticMeshesObjectsPtr = m_bufferParametersRing->getStaticStorage("StaticMeshObjects", sizeof(GPUPerObjectData) * m_collectStaticMeshes.size());
			m_staticMeshesObjectsPtr->buffer.updateDataPtr((void*)m_collectStaticMeshes.data());
		}
	}

	void RenderSceneData::lightCollect(Scene* scene)
	{
		// Load all directional light.
		std::vector<GPUDirectionalLightInfo> directionalLights = {};
		scene->loopComponents<DirectionalLightComponent>([&](std::shared_ptr<DirectionalLightComponent> comp)
		{
			GPUDirectionalLightInfo newDirectionalLight{};
			newDirectionalLight.intensity = comp->getIntensity();
			newDirectionalLight.color = colorspace::srgb_2_rec2020(comp->getColor());
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
		});

		// Fill directional light infos.
		{
			m_importanceLights.directionalLightCount = 0;

			if (directionalLights.size() > 0)
			{
				// Current use first directional light.
				m_importanceLights.directionalLights = directionalLights[0];

				m_earthAtmosphereInfo = RenderSettingManager::get()->earthAtmosphere.earthAtmosphere;
				m_earthAtmosphereInfo.absorptionExtinction = colorspace::srgb_2_rec2020(m_earthAtmosphereInfo.absorptionExtinction);
				m_earthAtmosphereInfo.rayleighScattering = colorspace::srgb_2_rec2020(m_earthAtmosphereInfo.rayleighScattering);
				m_earthAtmosphereInfo.mieScattering = colorspace::srgb_2_rec2020(m_earthAtmosphereInfo.mieScattering);
				m_earthAtmosphereInfo.mieAbsorption = colorspace::srgb_2_rec2020(m_earthAtmosphereInfo.mieAbsorption);
				m_earthAtmosphereInfo.groundAlbedo = colorspace::srgb_2_rec2020(m_earthAtmosphereInfo.groundAlbedo);

				m_importanceLights.directionalLightCount ++;
			}

			// Prepare cascade infos for directional lights.
			uint32_t cascadeCount = 1u;
			if (m_importanceLights.directionalLightCount > 0)
			{
				cascadeCount = m_importanceLights.directionalLights.cascadeCount;

				
			}

			// At least we create one cascade count buffer for feedback set.
			m_cascsadeBufferInfos = m_bufferParametersRing->getStaticStorageGPUOnly("CascadeInfos",
				sizeof(GPUCascadeInfo)* cascadeCount);
		}

		// TODO: Other light type gather.
	}

	RenderSceneData::RenderSceneData()
	{
		m_bufferParametersRing = std::make_unique<BufferParametersRing>();
	}

	void RenderSceneData::tick(const RuntimeModuleTickData& tickData)
	{
		// Update buffer ring.
		m_bufferParametersRing->tick();

		// Find active scene.
		Scene* activeScene = GEngine->getRuntimeModule<SceneManager>()->getScenes();

		// Collect static mesh.
		staticMeshCollect(activeScene);

		// Collect light.
		lightCollect(activeScene);
	}
}

