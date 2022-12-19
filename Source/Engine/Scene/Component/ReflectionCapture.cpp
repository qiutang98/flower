#include "Pch.h"
#include "ReflectionCapture.h"

namespace Flower
{
	bool ReflectionCaptureComponent::setUseIBLTexture(bool bState)
	{
		if (bState != m_bUseIBLTexture)
		{
			m_bUseIBLTexture = bState;
			return true;
		}
		return false;
	}

	bool ReflectionCaptureComponent::setRadius(float in)
	{
		if (in != m_radius)
		{
			m_radius = in;
			return true;
		}
		return false;
	}

}

