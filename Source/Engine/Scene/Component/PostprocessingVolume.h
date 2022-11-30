#pragma once
#include "../Component.h"

namespace Flower
{
	struct PostprocessVolumeSetting
	{
		float bloomIntensity = 1.0f;
		float bloomRadius = 0.75f;
		float bloomThreshold = 0.4f;
		float bloomThresholdSoft = 0.6f;

		float autoExposureLowPercent = 0.5f;
		float autoExposureHighPercent = 0.95f;
		float autoExposureMinBrightness = -5.0f;
		float autoExposureMaxBrightness = 7.0f;
		float autoExposureSpeedDown = 1.0f;
		float autoExposureSpeedUp = 2.0f;
		float autoExposureExposureCompensation = 0.0f;

		int   gtaoSliceNum = 2;
		int   gtaoStepNum = 8;
		float gtaoRadius = 2.0f;
		float gtaoThickness = 1.0f;
		float gtaoPower = 1.0f;
		float gtaoIntensity = 1.0f;

		float tonemapper_P = 10000.0f;  // Max brightness.
		float tonemapper_a = 1.4f;  // contrast
		float tonemapper_m = 0.22f; // linear section start
		float tonemapper_l = 0.4f;  // linear section length
		float tonemapper_c = 1.33f; // black
		float tonemapper_b = 0.0f;  // pedestal
		float tonemmaper_s = 4.0f; // scale 

		// C++ 20 save my life.
		auto operator<=>(const PostprocessVolumeSetting&) const = default;
	};

	class PostprocessVolumeComponent : public Component
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
		////////////////////////////// Serialize area //////////////////////////////
	protected:
		PostprocessVolumeSetting m_settings;

		////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

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
	};
}


