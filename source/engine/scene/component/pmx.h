#pragma once
#include "../component.h"
#include <rhi/rhi.h>
#include <Saba/Model/MMD/PMXModel.h>
#include <Saba/Model/MMD/VMDFile.h>
#include <Saba/Model/MMD/VMDAnimation.h>
#include <Saba/Model/MMD/VMDCameraAnimation.h>

namespace engine
{
	struct PMXGpuParams
	{
		glm::mat4 modelMatrix;
		glm::mat4 modelMatrixPrev;

		uint32_t texId;
		uint32_t spTexID;
		uint32_t toonTexID;
		uint32_t sceneNodeId;

		float pixelDepthOffset;
		float shadingModel;
		uint32_t  bSelected;
		uint32_t indicesArrayId;

		uint32_t positionsArrayId;
		uint32_t positionsPrevArrayId;
		uint32_t normalsArrayId;
		uint32_t uv0sArrayId;

	};

	struct PerFrameMMDCamera
	{
		bool bValidData = false;
		glm::mat4 viewMat;
		glm::mat4 projMat;
		float fovy;
		glm::vec3 worldPos;
	};

	struct PMXInitTrait
	{
		std::string pmxPath;
		std::vector<std::string> vmdPath;
	};



	class PMXMeshProxy
	{
	public:
		struct Vertex
		{
			glm::vec3 position;
			glm::vec3 normal;
			glm::vec2 uv;
			glm::vec3 positionLast;
		};

		inline static const auto kInputAttris = std::vector<VkVertexInputAttributeDescription>
		{
			{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 0 }, // pos
			{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3 }, // normal
			{ 2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6 }, // uv0
			{ 3, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 8 }, // pos prev.
		};

		 PMXMeshProxy(const UUID& uuid);
		~PMXMeshProxy();

		bool isInit() const { return m_bInit; }


		void onRenderCollect(
			class RendererInterface* renderer,
			VkCommandBuffer cmd,
			VkPipelineLayout pipelinelayout,
			const glm::mat4& modelMatrix,
			const glm::mat4& modelMatrixPrev,
			bool bTranslucentPass,
			uint32_t sceneNodeId,
			bool bSelected);

		void collectObjectInfos(std::vector<GPUStaticMeshPerObjectData>& collector, std::vector<VkAccelerationStructureInstanceKHR>& asInstances, uint32_t sceneNodeId,
			bool bSelected,
			const glm::mat4& modelMatrix,
			const glm::mat4& modelMatrixPrev);

		void updateVertex(VkCommandBuffer cmd);
		void updateBLAS(VkCommandBuffer cmd);

		
		
	private:
		bool m_bInit = false;
		
		VkIndexType m_indexType;
		std::unique_ptr<VulkanBuffer> m_indexBuffer  = nullptr;

		std::unique_ptr<VulkanBuffer> m_positionBuffer = nullptr;
		std::unique_ptr<VulkanBuffer> m_positionPrevFrameBuffer = nullptr;
		std::unique_ptr<VulkanBuffer> m_normalBuffer = nullptr;
		std::unique_ptr<VulkanBuffer> m_uvBuffer = nullptr;

		std::unique_ptr<VulkanBuffer> m_stageBufferPosition  = nullptr;
		std::unique_ptr<VulkanBuffer> m_stageBufferPositionPrevFrame = nullptr;
		std::unique_ptr<VulkanBuffer> m_stageBufferNormal = nullptr;
		std::unique_ptr<VulkanBuffer> m_stageBufferUv = nullptr;

		uint32_t m_indicesBindless = ~0;
		uint32_t m_positionBindless = ~0;
		uint32_t m_positionPrevBindless = ~0;
		uint32_t m_normalBindless = ~0;
		uint32_t m_uvBindless = ~0;

		std::unique_ptr<saba::MMDModel>	m_mmdModel = nullptr;
		std::shared_ptr<class AssetPMX> m_pmxAsset = nullptr;

		BLASBuilder m_blasBuilder;
	};

	class PMXComponent : public Component
	{
	public:
		PMXComponent() = default;
		virtual ~PMXComponent();

		PMXComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

		virtual void tick(const RuntimeModuleTickData& tickData) override;

		const UUID& getPmxUUID() const { return m_pmxUUID; }
		bool setPMX(const UUID& in);


		void onRenderCollect(
			class RendererInterface* renderer,
			VkCommandBuffer cmd,
			VkPipelineLayout pipelinelayout,
			bool bTranslucentPass);

		void onRenderTick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, 
			std::vector<GPUStaticMeshPerObjectData>& collector, 
			std::vector<VkAccelerationStructureInstanceKHR>& asInstances);

	private:
		std::unique_ptr<PMXMeshProxy> m_proxy = nullptr;

	public:
		ARCHIVE_DECLARE;

		UUID m_pmxUUID = {};
	};
}
