#pragma once
#include "RendererCommon.h"
#include "Parameters.h"

namespace Flower
{
	class IBLLightingContext
	{
	private:
		bool m_bIBLDirty = false;

	public:
		float intensity = 1.0f;

		bool bEnableIBLLight = false;

		// TODO: When ibl bake ready, should release this owner.
		std::shared_ptr<GPUImageAsset> hdrSrc = nullptr;

		bool iblEnable() const;

		void reset()
		{
			bEnableIBLLight = false;
			hdrSrc = nullptr;
		}

		void release()
		{
			hdrSrc = nullptr;
		}

		void setDirty(bool bState)
		{
			m_bIBLDirty = bState;
		}

		bool needRebuild() const;
			
	};

	struct RenderSetting
	{
		IBLLightingContext ibl;

		RHI::DisplayMode displayMode = RHI::DisplayMode::DISPLAYMODE_SDR;

		void reset()
		{
			ibl.reset();
		}

		void release()
		{
			ibl.release();
		}
	};

	using RenderSettingManager = Singleton<RenderSetting>;
}