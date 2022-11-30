#include "Pch.h"
#include "PostprocessingVolume.h"

namespace Flower
{
	bool PostprocessVolumeComponent::changeSetting(const PostprocessVolumeSetting& in)
	{
		if (in != m_settings)
		{
			m_settings = in;
			return true;
		}

		return false;
	}

}