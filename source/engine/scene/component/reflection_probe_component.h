#pragma once
#include "../component.h"
#include "../shader/common_header.h"
#include "../../graphics/pool.h"

namespace engine
{
	class RenderScene;

	class ReflectionProbeComponent : public Component
	{
	

		friend class ReflectionCaptureManager;
		REGISTER_BODY_DECLARE(Component);

	public:

		ReflectionProbeComponent() = default;
		ReflectionProbeComponent(std::shared_ptr<SceneNode> sceneNode) : Component(sceneNode)
		{

		}

		virtual ~ReflectionProbeComponent();

		virtual bool uiDrawComponent() override;
		static const UIComponentReflectionDetailed& uiComponentReflection();

	public:
		void collectReflectionProbe(RenderScene& renderScene);

		void updateReflectionCapture(
			VkCommandBuffer graphicsCmd, 
			const RuntimeModuleTickData& tickData);

		virtual void tick(const RuntimeModuleTickData& tickData) override;

		auto getSceneCapture() const { return m_sceneCapture; }


		bool isDrawExtent() const { return m_bDrawExtent; }
		vec3 getMinExtent() const { return m_minExtent; }
		vec3 getMaxExtent() const { return m_maxExtent; }
		bool isCaptureValid() const { return !!m_sceneCapture; }
		bool isCaptureOutOfDate() const { return m_bCaptureOutOfDate; }

		void markOutOfDate() 
		{ 
			m_bCaptureOutOfDate = true; 
		}

		void clearCapture()
		{
			m_sceneCapture = nullptr;
			m_bCaptureOutOfDate = false;
		}

		void updateActiveFrameNumber(uint64_t in) { m_prevActiveFrameNumber = in; }
		uint64_t getPreActiveFrameNumber() const { return m_prevActiveFrameNumber; }
		const auto& getCapturePos() const { return m_capturePos; }
	protected:
		vec3 m_capturePos = {};
		uint64_t m_prevActiveFrameNumber = 0;
		bool m_bCaptureOutOfDate = false;
		PoolImageSharedRef m_sceneCapture = nullptr;


		bool m_bDrawExtent = false;

	protected:
		int m_dimension = 256;

		vec3 m_minExtent = vec3(-50.0f);
		vec3 m_maxExtent = vec3( 50.0f);

	};
}
