#include "terrain.h"

namespace engine
{
    TerrainComponent::~TerrainComponent()
    {
        getContext()->waitDeviceIdle();
    }

    bool TerrainComponent::changeSetting(const TerrainSetting& in)
    {
        if (in != m_setting)
        {
            m_setting = in;
            markDirty();
            return true;
        }

        return false;
    }

    uint32_t TerrainComponent::getHeightfieldWidth() const
    {
        if (m_renderContext.heightFieldImage)
        {
            return m_renderContext.heightFieldImage->getImage().getExtent().width;
        }
        return 0;
    }

    uint32_t TerrainComponent::getHeightfieldHeight() const
    {
        if (m_renderContext.heightFieldImage)
        {
            return m_renderContext.heightFieldImage->getImage().getExtent().height;
        }
        return 0;
    }

    VulkanImage& TerrainComponent::getHeightfiledImage()
    {
        if(m_renderContext.heightFieldImage) return m_renderContext.heightFieldImage->getImage();


        return getContext()->getEngineTextureTranslucent()->getImage();
    }

}
