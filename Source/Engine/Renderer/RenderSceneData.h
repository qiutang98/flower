#pragma once
#include "RendererCommon.h"
#include "Parameters.h"
#include "BufferParameter.h"

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
	class RenderSceneData : NonCopyable
	{
	private:
		// Scene static mesh collect data.
		std::vector<GPUPerObjectData> m_collectStaticMeshes;

		// Importance light infos.
		SceneImportLightInfos m_importanceLights;

		// Earth atmosphere info.
		EarthAtmosphere m_earthAtmosphereInfo;

		std::unique_ptr<BufferParametersRing> m_bufferParametersRing;

		BufferParamRefPointer m_cascsadeBufferInfos;
		BufferParamRefPointer m_staticMeshesObjectsPtr;

	private:
		// Collect scne static mesh.
		void staticMeshCollect(Scene* scene);

		void lightCollect(Scene* scene);
		

	public:
		RenderSceneData();

		~RenderSceneData()
		{
			m_cascsadeBufferInfos = nullptr;
			m_staticMeshesObjectsPtr = nullptr;
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

		// Upadte collect scene infos. often call before all renderer logic.
		void tick(const RuntimeModuleTickData& tickData);
	};
}