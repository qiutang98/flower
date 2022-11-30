#include "Pch.h"
#include "SkyLight.h"

namespace Flower
{
	bool SkyLightComponent::setRealtimeCapture(bool bState)
	{
		if (m_bRealtimeCapture != bState)
		{
			m_bRealtimeCapture = bState;
			return true;
		}

		return false;
	}

}

