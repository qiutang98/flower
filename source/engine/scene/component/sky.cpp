#include "sky.h"


namespace engine
{
	bool SkyComponent::setCascadeConfig(const CascadeShadowConfig& newValue)
	{
        if (newValue != m_cascadeConfig)
        {
            m_cascadeConfig = newValue;
            markDirty();
            return true;
        }

        return false;
	}
}
