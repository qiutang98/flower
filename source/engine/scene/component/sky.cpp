#include "sky.h"


namespace engine
{
    SkyComponent::SkyComponent(std::shared_ptr<SceneNode> sceneNode)
        : LightComponent(sceneNode)
    {
        init();
    }

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
    void SkyComponent::init()
    {
        m_color = temperature2Color(6500.0f);
    }
}
