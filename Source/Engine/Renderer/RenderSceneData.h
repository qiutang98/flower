#pragma once
#include "RendererCommon.h"
#include "Parameters.h"
#include "BufferParameter.h"
#include "../Scene/Component/PostprocessingVolume.h"

// Cross multi renderer scene data.
// Update every frame.
namespace Flower
{
	struct SceneImportLightInfos
	{
		uint32_t directionalLightCount;
		GPUDirectionalLightInfo directionalLights;
	};

	class Scene;
	class PMXComponent;

	class RenderSceneData : NonCopyable
	{
	private:
		

		// Scene static mesh collect data.
		std::vector<GPUPerObjectData> m_collectStaticMeshes;

		std::vector<std::shared_ptr<PMXComponent>> m_collectPMXes;
		std::shared_ptr<PMXComponent> m_cachePlayingPMXWithCamera = nullptr;

		// Importance light infos.
		SceneImportLightInfos m_importanceLights;

		// Earth atmosphere info.
		EarthAtmosphere m_earthAtmosphereInfo;

		// Scene postprocessing volume info.
		PostprocessVolumeSetting m_postprocessVolumeInfo;

		std::unique_ptr<BufferParametersRing> m_bufferParametersRing;

		BufferParamRefPointer m_cascsadeBufferInfos;
		BufferParamRefPointer m_staticMeshesObjectsPtr;

		std::array<GPULocalSpotLightInfo, GMaxImportanceLocalSpotLightNum> m_importanceSpotLights = { };
		uint32_t m_cacheImportanceLocalSpotLitNum = 0;

	private:
		// Collect scne static mesh.
		void staticMeshCollect(Scene* scene);

		void pmxCollect(Scene* scene, VkCommandBuffer cmd);

		void lightCollect(Scene* scene);
		
		void postprocessVolumeCollect(Scene* scene);

	public:
		RenderSceneData();

		~RenderSceneData()
		{
			m_cascsadeBufferInfos = nullptr;
			m_staticMeshesObjectsPtr = nullptr;
		}

		uint32_t getImpotanceLocalSpotLightNum() const { return m_cacheImportanceLocalSpotLitNum; }
		const auto& getCacheImportanceLocalSpotLight() const {
			return m_importanceSpotLights;
		}

		// Get collect static meshes infos.
		const std::vector<GPUPerObjectData>& getCollectStaticMeshes() const
		{
			return m_collectStaticMeshes;
		}

		BufferParamRefPointer getStaticMeshesObjectsPtr() const
		{
			return m_staticMeshesObjectsPtr;
		}

		BufferParamRefPointer getCascadeInfoPtr() const
		{
			return m_cascsadeBufferInfos;
		}

		const auto& getImportanceLights() const
		{
			return m_importanceLights;
		}

		const auto& getEarthAtmosphere() const
		{
			return m_earthAtmosphereInfo;
		}

		// Current scene exist some static mesh?
		bool isStaticMeshExist() const
		{
			return !m_collectStaticMeshes.empty();
		}

		const auto& getPostprocessVolumeSetting() const
		{
			return m_postprocessVolumeInfo;
		}

		bool isPMXExist() const
		{
			return !m_collectPMXes.empty();
		}

		bool isPMXPlayWithCamera() const { return m_cachePlayingPMXWithCamera != nullptr;  }
		auto getPlayingPMXWithCamera() { return m_cachePlayingPMXWithCamera; }

		const auto& getPMXes() const
		{
			return m_collectPMXes;
		}

		// Upadte collect scene infos. often call before all renderer logic.
		void tick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd);
	};
}