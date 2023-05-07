#pragma once
#include "light.h"

namespace engine
{
	class SkyComponent : public LightComponent
	{
	public:
		SkyComponent()
		{
			init();
		}

		virtual ~SkyComponent() = default;

		SkyComponent(std::shared_ptr<SceneNode> sceneNode);

		const CascadeShadowConfig& getCacsadeConfig() const { return m_cascadeConfig; }
		CascadeShadowConfig& getCacsadeConfig() { return m_cascadeConfig; }
		bool setCascadeConfig(const CascadeShadowConfig& newValue);

		const AtmosphereConfig& getAtmosphereConfig() const { return m_atmosphereConfig; }
		AtmosphereConfig& getAtmosphereConfig() { return m_atmosphereConfig; }

	private:
		void init();

	protected:
		ARCHIVE_DECLARE;
		CascadeShadowConfig m_cascadeConfig;
		AtmosphereConfig m_atmosphereConfig;
	};
}