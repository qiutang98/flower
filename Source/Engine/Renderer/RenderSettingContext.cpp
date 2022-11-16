#include "Pch.h"
#include "RenderSettingContext.h"
#include "../AssetSystem/TextureManager.h"

namespace Flower
{
	bool IBLLightingContext::iblEnable() const
	{
		return hdrSrc && bEnableIBLLight && hdrSrc->isAssetReady();
	}

	bool IBLLightingContext::needRebuild() const
	{
		return iblEnable() && m_bIBLDirty;
	}
}