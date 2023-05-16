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
	public:
		// Runtime build info.
		uint32_t mmdTex;
		uint32_t mmdSphereTex;
		uint32_t mmdToonTex;

	public:
		saba::MMDMaterial material;

		bool     bTranslucent = false;
		bool     bHide = false;
		float    pixelDepthOffset = 0.0f;
		EShadingModelType pmxShadingModel = EShadingModelType::StandardPBR;

		auto operator<=>(const PMXDrawMaterial&) const = default;
		template<class Archive> void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(material);

			archive(bTranslucent, bHide, pixelDepthOffset);

			uint32_t pmxShadingModelValue = uint32_t(pmxShadingModel);
			archive(pmxShadingModelValue);
			pmxShadingModel = EShadingModelType(pmxShadingModelValue);
		}
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

		const UUID& getPmxUUID() const { return m_pmxUUID; }
		bool setPMX(const UUID& in);

	public:
		ARCHIVE_DECLARE;

		UUID m_pmxUUID = {};
	};
}
