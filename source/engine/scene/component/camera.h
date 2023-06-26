#pragma once

#include "../component.h"

namespace engine
{
	struct PerFrameMMDCamera
	{
		math::mat4 viewMat;
		math::mat4 projMat;
		float fovy;
		math::vec3 worldPos;
		math::vec3 front;
	};

	class MMDCameraComponent : public Component
	{
	public:
		MMDCameraComponent() = default;
		virtual ~MMDCameraComponent() = default;

		MMDCameraComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

		virtual void tick(const RuntimeModuleTickData& tickData) override;

		bool setVmd(const UUID& id);
		const UUID& getVmdUUID() const { return m_vmdUUID; }

		PerFrameMMDCamera getCameraPerframe(float width, float height, float zNear, float zFar) const;

	protected:
		std::unique_ptr<saba::VMDCameraAnimation> m_vmdCameraAnim = nullptr;
	protected:
		ARCHIVE_DECLARE;

		UUID m_vmdUUID = {};
	};
}


