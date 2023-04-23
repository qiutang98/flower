#include "postprocess.h"

namespace engine
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