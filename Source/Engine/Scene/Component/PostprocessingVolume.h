#pragma once
#include "../Component.h"

namespace Flower
{
	struct PostprocessVolumeSetting
	{
		bool bEnableVignette = true;
		float vignette_falloff = 0.35f;

		int32_t bEnableFringeMode = 0; // 0 off, 1 Conrady, 2 barrel
		float fringe_barrelStrength = 1.0f;
		float fringe_zoomStrength = 0.1f;
		float fringe_lateralShift = 0.5f;

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
		float tonemapper_a = 1.0f;  // contrast
		float tonemapper_m = 0.22f; // linear section start
		float tonemapper_l = 0.4f;  // linear section length
		float tonemapper_c = 1.33f; // black
		float tonemapper_b = 0.0f;  // pedestal
		float tonemmaper_s = 4.0f; // scale 

		bool bDofEnable = false;
		bool dof_bNearBlur = true;
		int32_t dof_focusMode = 0; // Dof focusMode. 0 is use focus point, 1 is use focus distance, 2 is pmx charater
		glm::vec3 dof_focusPoint = glm::vec3(0.0f); //
		float dof_focusDistance = 10.0f;
		int32_t dof_trackPMXMode = 0; // 0 is min depth, 1 is max depth, 2 is avg depth

		float dof_aperture      = 1.4f;
		bool  dof_bUseCameraFOV = true;
		float dof_focusLength   = 50.0f; // mm
		int32_t dof_kernelSize  = 2;
		float dof_FilmHeight = 0.024f;// Height of the 35mm full-frame format (36mm x 24mm)


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


