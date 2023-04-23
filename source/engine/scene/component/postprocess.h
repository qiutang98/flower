#pragma once
#include "../component.h"

namespace engine
{
	struct PostprocessVolumeSetting
	{
		bool bAutoExposure = true;
		float fixExposure = 10.0f;
		float autoExposureLowPercent           = 0.5f;
		float autoExposureHighPercent          = 0.95f;
		float autoExposureMinBrightness        = -3.0f;
		float autoExposureMaxBrightness        = 0.0f;
		float autoExposureSpeedDown            = 1.0f;
		float autoExposureSpeedUp              = 2.0f;
		float autoExposureExposureCompensation = 1.0f;

		float bloomIntensity = 1.0f;
		float bloomRadius = 0.80f;
		float bloomThreshold = 0.80f;
		float bloomThresholdSoft = 0.6f;

		int   gtaoSliceNum       = 2;
		int   gtaoStepNum        = 8;
		float gtaoRadius         = 2.0f;
		float gtaoThickness      = 1.0f;
		float gtaoPower          = 1.0f;
		float gtaoIntensity      = 1.0f;

		auto operator<=>(const PostprocessVolumeSetting&) const = default;
		template<class Archive> void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(gtaoSliceNum, gtaoStepNum, gtaoRadius, gtaoThickness, gtaoPower, gtaoIntensity);

			archive(
				bAutoExposure,
				fixExposure,
				autoExposureLowPercent, 
				autoExposureHighPercent, 
				autoExposureMinBrightness, 
				autoExposureMaxBrightness, 
				autoExposureSpeedDown, 
				autoExposureSpeedUp, 
				autoExposureExposureCompensation);

			archive(bloomIntensity, bloomRadius, bloomThreshold, bloomThresholdSoft);
		}
	};

	class PostprocessVolumeComponent : public Component
	{
	public:
		PostprocessVolumeComponent() = default;
		virtual ~PostprocessVolumeComponent() = default;

		PostprocessVolumeComponent(std::shared_ptr<SceneNode> sceneNode)
			: Component(sceneNode)
		{

		}

	public:
		const PostprocessVolumeSetting& getSetting() const { return m_settings; }
		bool changeSetting(const PostprocessVolumeSetting& in);

	protected:
		ARCHIVE_DECLARE;
		PostprocessVolumeSetting m_settings;
	};
}

