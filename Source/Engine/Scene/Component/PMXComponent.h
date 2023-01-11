#pragma once
#include "../Component.h"
#include "../../RHI/RHI.h"
#include <Saba/Model/MMD/PMXModel.h>
#include <Saba/Model/MMD/VMDFile.h>
#include <Saba/Model/MMD/VMDAnimation.h>
#include <Saba/Model/MMD/VMDCameraAnimation.h>
#include "../../Renderer/ShadingModel.h"

namespace Flower
{
	struct PMXGpuParams
	{
		glm::mat4 modelMatrix;
		glm::mat4 modelMatrixPrev;

		uint32_t texId;
		uint32_t spTexID;
		uint32_t toonTexID;
		uint32_t pmxObjectID;

		float pixelDepthOffset;
		float shadingModel;
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

	class PMXDrawMaterial
	{
		ARCHIVE_DECLARE;

	public:
#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
		saba::MMDMaterial material;

		bool bTranslucent = false;
		bool bHide = false;
		float pixelDepthOffset = 0.0f;
		uint32_t pmxShadingModel = uint32_t(EPMXShadingModel::Basic);

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

		// Runtime build info.
		uint32_t mmdTex;
		uint32_t mmdSphereTex;
		uint32_t mmdToonTex;
	};

	class RendererInterface;
	class PMXComponent;

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

		struct Material
		{
			uint32_t mmdTex;
			uint32_t mmdToonTex;
			uint32_t mmdSphereTex;
		};
		explicit PMXMeshProxy(PMXComponent* InComp);
		bool Ready();

	private:
		PMXComponent* m_component = nullptr;

		std::string m_cameraPath;
		std::unique_ptr<saba::VMDCameraAnimation> vmdCameraAnim;

		std::string m_pmxPath;
		std::vector<std::string> m_vmdPath;

		std::shared_ptr<saba::MMDModel>	m_mmdModel;
		std::unique_ptr<saba::VMDAnimation>	m_vmdAnim;

		PerFrameMMDCamera m_currentFrameCameraData;

		std::shared_ptr<VulkanBuffer> m_indexBuffer = nullptr;
		std::shared_ptr<VulkanBuffer> m_vertexBuffer = nullptr;

		std::shared_ptr<VulkanBuffer> m_stageBuffer = nullptr;

		VkIndexType m_indexType;

		

		void release();
		bool prepareVMD();
		bool preparePMX();
		bool prepareMaterial();
		bool prepareVertexBuffer();
		void UpdateAnimation(float vmdFrameTime, float physicElapsed);
		
		void UpdateVertex(VkCommandBuffer cmd);

	public:
		void OnRenderTick(VkCommandBuffer cmd);
		void Setup(PMXInitTrait initTrait);
		void SetupCamera(std::string cameraPath);
		void OnSceneTick(float vmdFrameTime, float physicElapsed);

		PerFrameMMDCamera GetCurrentFrameCameraData(
			float width, 
			float height, 
			float zNear, 
			float zFar, 
			glm::mat4 worldMatrix);

		void OnRenderCollect(
			RendererInterface* renderer, 
			VkCommandBuffer cmd, 
			VkPipelineLayout pipelinelayout, 
			const glm::mat4& modelMatrix, 
			const glm::mat4& modelMatrixPrev,
			bool bTranslucentPass);

		void OnShadowRenderCollect(
			RendererInterface* renderer, 
			VkCommandBuffer cmd, 
			VkPipelineLayout pipelinelayout, 
			uint32_t cascadeIndex, 
			const glm::mat4& modelMatrix, 
			const glm::mat4& modelMatrixPrev);
	};

	// TODO: Seperate camera and PMX.
	class PMXComponent : public Component
	{
		ARCHIVE_DECLARE;
		friend PMXMeshProxy;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	public:
		std::string m_pmxPath = "";
		std::string m_vmdPath = "";
		std::string m_wavPath = "";
		std::string m_cameraPath = "";

		// PMX material can keep same name :(
		std::vector<PMXDrawMaterial> m_materials;

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		PMXComponent() = default;
		virtual ~PMXComponent();

		PMXComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

		const std::string& getPmxPath() const { return m_pmxPath; }
		const std::string& getVmdPath() const { return m_vmdPath; }
		const std::string& getWavPath() const { return m_wavPath; }
		const std::string& getCameraPath() const { return m_cameraPath; }

		void setPmxPath(std::string path);
		void setVmdPath(std::string vmdPath);
		void setWavPath(std::string wavPath);
		void setCameraPath(std::string cameraPath);

		///
		void onRenderCollect(
			RendererInterface* renderer, 
			VkCommandBuffer cmd, 
			VkPipelineLayout pipelinelayout, 
			bool bTranslucentPass);
		void onShadowRenderCollect(RendererInterface* renderer, VkCommandBuffer cmd, VkPipelineLayout pipelinelayout, uint32_t cascadeIndex);
		void onRenderTick(VkCommandBuffer cmd);
		///

		void resetAnimation();

		void setPlayAnimationState(bool bState);
		bool getPlayAnimationState() const { return m_bPlayAnimation; }

		bool isPMXCameraPlaying() const;

		PerFrameMMDCamera getCurrentFrameCameraData(float width, float height, float zNear, float zFar);

		//
		virtual void tick(const RuntimeModuleTickData& tickData) override;
		//

		PMXMeshProxy* getProxy();

		bool pmxReady() { return getProxy()->Ready(); }
	private:

		bool m_bPMXMeshChanged = true;
		bool m_bWaveChanged = true;
		bool m_bCameraPathChanged = true;
		bool bCameraSetupReady = false;

		std::unique_ptr<PMXMeshProxy> m_proxy;

		float m_animationPlayTime = 0.0f;
		float m_elapsed = 0.0f;

		bool m_bPlayAnimation = false;
	};
}
