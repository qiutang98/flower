#include "camera.h"
#include "scene/scene_archive.h"

namespace engine
{
	void MMDCameraComponent::tick(const RuntimeModuleTickData& tickData)
	{
		auto initVmd = [&]()
		{
			auto vmdAsset = std::dynamic_pointer_cast<AssetVMD>(getAssetSystem()->getAsset(m_vmdUUID));

			if (vmdAsset->m_bCamera)
			{
				auto vmdPath = vmdAsset->getVMDFilePath().string();
				saba::VMDFile vmdFile;

				if (!saba::ReadVMDFile(&vmdFile, vmdPath.c_str()))
				{
					LOG_ERROR("Failed to read VMD file {0}.", vmdPath);
					return;
				}
				if (vmdFile.m_cameras.empty())
				{
					LOG_ERROR("Vmd file {0} no contain camera.", vmdPath);
					return;
				}

				m_vmdCameraAnim = std::make_unique<saba::VMDCameraAnimation>();
				if (!m_vmdCameraAnim->Create(vmdFile))
				{
					LOG_ERROR("Failed to create VMDCameraAnimation.");

					m_vmdCameraAnim = nullptr;
					return;
				}
			}
			else
			{
				LOG_ERROR("Vmd is not camera vmd!");
				return;
			}
		};

		if (!m_vmdCameraAnim && !m_vmdUUID.empty())
		{
			initVmd();
		}

		if (m_vmdCameraAnim)
		{
			m_vmdCameraAnim->Evaluate(tickData.gameTime * 30.0f);


		}
	}

	bool MMDCameraComponent::setVmd(const UUID& id)
	{
		if (id != m_vmdUUID)
		{
			m_vmdUUID = id;

			m_vmdCameraAnim = nullptr;
			return true;
		}
		return false;
	}

	PerFrameMMDCamera MMDCameraComponent::getCameraPerframe(float width, float height, float zNear, float zFar) const
	{
		glm::mat4 worldMatrix = m_node.lock()->getTransform()->getWorldMatrix();
		PerFrameMMDCamera result{};

		const auto mmdCam = m_vmdCameraAnim->GetCamera();
		saba::MMDLookAtCamera lookAtCam(mmdCam);

		const bool bReverseZ = true;

		// update current Frame camera data.
		glm::vec3 eyeWorldPos = worldMatrix * glm::vec4(lookAtCam.m_eye, 1.0f);
		glm::vec3 centerWorldPos = worldMatrix * glm::vec4(lookAtCam.m_center, 1.0f);
		glm::vec3 upWorld = worldMatrix * glm::vec4(lookAtCam.m_up, 0.0f);

		result.viewMat = glm::lookAt(eyeWorldPos, centerWorldPos, upWorld);


		auto fov = mmdCam.m_fov;

		result.projMat = glm::perspectiveFovRH(
			fov,
			width,
			height,
			bReverseZ ? zFar : zNear,
			bReverseZ ? zNear : zFar
		);

		result.fovy = fov;
		result.worldPos = eyeWorldPos;

		result.front = math::normalize(centerWorldPos - eyeWorldPos);

		return result;
	}
}