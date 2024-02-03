#pragma once
#include "../component.h"
#include "../shader/common_header.h"

namespace engine
{
	inline math::vec4 getBloomPrefilter(float threshold, float thresholdSoft)
	{
		float knee = threshold * thresholdSoft;
		math::vec4 prefilter{ };

		prefilter.x = threshold;
		prefilter.y = prefilter.x - knee;
		prefilter.z = 2.0f * knee;
		prefilter.w = 0.25f / (knee + 0.00001f);

		return prefilter;
	}

	extern PostprocessVolumeSetting defaultPostprocessVolumeSetting();

	extern PostprocessVolumeSetting computePostprocessSettingDetail(
		const PerFrameData& perframe,
		const PostprocessVolumeSetting& in, 
		float deltaTime);

	class PostprocessComponent : public Component
	{
		REGISTER_BODY_DECLARE(Component);

	public:
		PostprocessComponent() = default;
		PostprocessComponent(std::shared_ptr<SceneNode> sceneNode) : Component(sceneNode)
		{

		}

		virtual ~PostprocessComponent() = default;

		virtual bool uiDrawComponent() override;
		static const UIComponentReflectionDetailed& uiComponentReflection();

	public:
		const PostprocessVolumeSetting& getSetting() const { return m_setting; }


	protected:


	protected:
		PostprocessVolumeSetting m_setting = defaultPostprocessVolumeSetting();
	};
}
