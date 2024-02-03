#pragma once
#include "../component.h"
#include "../shader/common_header.h"

namespace engine
{
	class SkyComponent;

	struct TimeOfDay
	{
		DEFAULT_COMPARE_ARCHIVE(TimeOfDay)

	public:
		TimeOfDay();
		bool uiDraw(bool bLocalTime);

	public:
		int year;
		int month;
		int day;
		int hour;
		int minute;
		int second;
	};

	extern AtmosphereParametersInputs defaultAtmosphereParameters();
	extern CloudParametersInputs defaultCloudParameters(float earthRadius = -1.0f);
	extern CascadeShadowConfig defaultCascadeConfig();
	extern RaytraceShadowConfig defaultRaytraceShadowConfig();

	extern SkyLightInfo defaultSun();

	class SkyComponent : public Component
	{
		REGISTER_BODY_DECLARE(Component);

	public:
		SkyComponent() = default;
		SkyComponent(std::shared_ptr<SceneNode> sceneNode) : Component(sceneNode)
		{

		}

		virtual ~SkyComponent() = default;

		virtual void tick(const RuntimeModuleTickData& tickData) override;

		virtual bool uiDrawComponent() override;
		static const UIComponentReflectionDetailed& uiComponentReflection();

		math::vec3 getSunDirection() const;

		SkyLightInfo getSunInfo() const { return m_sun; }
		const AtmosphereParametersInputs& getAtmosphereParameters() const { return m_atmosphere; }
		const CloudParametersInputs& getCloudParameters() const { return m_cloud; }
		bool collectSkyLight(class RenderScene& renderScene);

	protected:
		bool m_bLocalTime = true;
		TimeOfDay m_tod;

		AtmosphereParametersInputs m_atmosphere = defaultAtmosphereParameters();
		CloudParametersInputs m_cloud = defaultCloudParameters();

		SkyLightInfo m_sun = defaultSun();
		SkyLightInfo m_prevFrameSun = defaultSun();

	};
}
