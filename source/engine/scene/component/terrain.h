#pragma once
#include "../component.h"

namespace engine
{
	struct TerrainSetting
	{
		int32_t gpuSubd = 3;
		float primitivePixelLengthTarget = 7.0f;
		float minLodStdev = 0.1f;
		int32_t maxDepth = 25;
		float dumpFactor = 1.0f;

		auto operator<=>(const TerrainSetting&) const = default;
		template<class Archive> void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(gpuSubd, primitivePixelLengthTarget, minLodStdev, maxDepth, dumpFactor);
		}
	};

	struct TerrainCommonPassPush
	{
		math::mat4  u_ModelMatrix = glm::mat4(1.0f);

		float u_LodFactor;
		float u_DmapFactor = 1.0f;
		float u_MinLodVariance;
		uint32_t sceneNodeId;

		uint32_t bSelected;
		uint32_t cascadeId;
	};

	class TerrainComponent : public Component
	{
	public:
		TerrainComponent() {}
		virtual ~TerrainComponent();

		TerrainComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

		const TerrainSetting& getSetting() { return m_setting; }
		bool changeSetting(const TerrainSetting& in);

		void render(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, class GBufferTextures* inGBuffers, class RenderScene* scene, class RendererInterface* renderer);

		void renderSDSMDepth(
			VkCommandBuffer cmd, 
			BufferParameterHandle perFrameGPU, 
			class GBufferTextures* inGBuffers, 
			class RenderScene* scene, 
			class RendererInterface* renderer, 
			struct SDSMInfos& sdsmInfo,
			uint32_t cascadeId);

		void setHeightField(const UUID& in);
		void setMask(const UUID& in);

		bool isHeightfieldSet() const { return !m_terrainHeightfieldId.empty(); }
		bool isMaskSet() const { return !m_terrainGrassSandMudMaskId.empty(); }
		uint32_t getHeightfieldWidth() const;
		uint32_t getHeightfieldHeight() const;

		VulkanImage& getHeightfiledImage();

		void loadTexturesByUUID(bool bSync);

	protected:
		bool allBufferValid() const;

		bool loadBuffers();
		bool loadLebBuffer();
		bool loadCbtNodeCountBuffer();
		bool loadRenderCmdBuffer();
		bool loadMeshletBuffers();

		void updateLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, class GBufferTextures* inGBuffers, class RenderScene* scene, class RendererInterface* renderer);
		void reductionLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, class GBufferTextures* inGBuffers, class RenderScene* scene, class RendererInterface* renderer);

		void batchLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, class GBufferTextures* inGBuffers, class RenderScene* scene, class RendererInterface* renderer);
		void renderLeb(VkCommandBuffer cmd, BufferParameterHandle perFrameGPU, class GBufferTextures* inGBuffers, class RenderScene* scene, class RendererInterface* renderer);

		struct
		{
			int lebPingpong = 0;
			TerrainCommonPassPush commonPushConst{};
			std::shared_ptr<GPUImageAsset> heightFieldImage = nullptr;

			// Cache sand mud mask
			std::shared_ptr<GPUImageAsset> grassSandMudMaskImage = nullptr;
		} m_renderContext;

	protected:
		BufferParameterHandle m_lebBuffer = nullptr;
		BufferParameterHandle m_cbtNodeCountBuffer = nullptr;

		BufferParameterHandle m_terrainDrawCmdBuffer = nullptr;
		BufferParameterHandle m_dispatchCmdBuffer = nullptr;

		BufferParameterHandle m_verticesBuffer = nullptr;
		BufferParameterHandle m_indicesBuffer = nullptr;

		math::mat4 m_localMatrix = math::mat4{1.0f};
		math::mat4 m_localMatrixPrev = math::mat4{1.0f};

	protected:
		ARCHIVE_DECLARE;

		UUID m_terrainHeightfieldId = {};
		UUID m_terrainGrassSandMudMaskId = {};
		TerrainSetting m_setting;
	};
}