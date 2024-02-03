#pragma once
#include "../utils/utils.h"
#include "../graphics/context.h"
#include "../utils/camera_interface.h"
#include <common_header.h>

namespace engine
{
	class SkyComponent;
	class PostprocessComponent;
	class ReflectionProbeComponent;
	class LandscapeComponent;

	extern int32_t getShadowDepthDimTerrain();
	extern int32_t getShadowDepthDimCloud();

	struct AABBBounds
	{
		AABBBounds()
		{
			min = vec3(std::numeric_limits<float>::max());
			max = vec3(std::numeric_limits<float>::lowest());
		}

		void transform(mat4 transformMatrix);

		vec3 min;
		vec3 max;
	};

	// Render scene collect perframe scene info.
	class RenderScene : NonCopyable
	{
	public:
		void tick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd);

		// Get objects.
		auto& getObjectCollector() { return m_perFrameCollect.objects; }

		const auto& getObjectCollector() const { return m_perFrameCollect.objects; }

		BufferParameterHandle getObjectBufferGPU() const { return m_perFrameCollect.objectsBufferGPU; }

		// Get sky.
		SkyComponent* getSkyComponent() const { return m_perFrameCollect.skyComponent; }

		// Get post.
		PostprocessComponent* getPostprocessComponent() const 
		{ 
			return m_perFrameCollect.postprocessingComponent; 
		}

		// Get landscape.
		LandscapeComponent* getLandscape() const
		{
			return m_perFrameCollect.landscape;
		}

		const auto& getReflections() const { return m_perFrameCollect.reflections; }

		// Accelerate struct valid or not.
		bool isTLASValid() const;

		TLASBuilder* getTLAS() { return &m_tlas; }

		// Unvalid Accelerate struct.
		void unvalidTLAS() { m_tlas.destroy(); }

		auto& getBLASObjectCollector() { return m_perFrameCollect.cacheASInstances; }

		void markTLASFullRebuild() { m_perFrameCollect.bTLASFullRebuild = true; }

		void fillPerframe(PerFrameData& inout, const RuntimeModuleTickData& tickData);
		auto& getAABBBounds() { return m_perFrameCollect.sceneStaticMeshAABB; }

		auto& getDebugLineDrawer() { return m_perFrameCollect.drawLineCPU; }

		void drawAABB(vec3 center, vec3 extents);
		void drawAABBminMax(vec3 min, vec3 max);

		void clearAllReflectionCapture() { m_bClearAllRelfectionInThisLoop = true; }

	private:
		void tlasPrepare(const RuntimeModuleTickData& tickData, class Scene* scene, VkCommandBuffer cmd);


	private:
		struct PerFrameCollect
		{
			// Collect scene objects.
			std::vector<PerObjectInfo> objects = {};
			AABBBounds sceneStaticMeshAABB = {};

			std::vector<vec4> drawLineCPU = {};

			BufferParameterHandle objectsBufferGPU = nullptr;

			// Sky component used for rendering.
			SkyComponent* skyComponent = nullptr;

			// Postprocessing component.
			PostprocessComponent* postprocessingComponent = nullptr;

			bool bTLASFullRebuild = false;
			std::vector<VkAccelerationStructureInstanceKHR> cacheASInstances = {};

			std::vector<ReflectionProbeComponent*> reflections;

			LandscapeComponent* landscape;

		} m_perFrameCollect;


		TLASBuilder m_tlas;
		bool m_bClearAllRelfectionInThisLoop = true;

	};
}