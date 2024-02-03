#include "landscape_component.h"

#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include <editor/widgets/content.h>
#include <editor/editor.h>
#include <renderer/render_scene.h>
#include <asset/asset_texture.h>
#include <asset/asset_manager.h>

namespace engine
{

    bool LandscapeComponent::uiDrawComponent()
    {
        bool bChangedValue = false;
        static const auto& meta = LandscapeComponent::uiComponentReflection();

        if (ImGui::DragInt("Dimension", &m_dimension))
        {
            m_dimension = getNextPOT(m_dimension);
            m_dimension = math::clamp(m_dimension, 1024, 8192);

            bChangedValue = true;
        }

        if (ImGui::DragFloat2("Offset", &m_offset.x))
        {
            bChangedValue = true;
        }

        if (ImGui::DragFloat("MinHeight", &m_minHeight))
        {
            bChangedValue = true;
            if (m_minHeight >= m_maxHeight)
            {
                m_minHeight = m_maxHeight - 1.0f;
            }
        }

        if (ImGui::DragFloat("MaxHeight", &m_maxHeight))
        {
            bChangedValue = true;
            if (m_minHeight >= m_maxHeight)
            {
                m_maxHeight = m_maxHeight + 1.0f;
            }
        }

        auto drawHeightFieldImageSelect = [&]()
        {
            static const auto suffix = AssetTexture::getCDO()->getSuffix();
            ImGui::Spacing();

            const auto& set = getAssetManager()->getAssetTypeMap(suffix);

            for (const auto& texId : set)
            {
                auto asset = getAssetManager()->getAsset(texId);
                if (ImGui::MenuItem((std::string("  ") + ICON_FA_IMAGE"   " + asset->getSaveInfo().getName()).c_str()))
                {
                    setAssetUUID(texId);
                }

                ui::hoverTip(asset->getSaveInfo().getStorePathU8().c_str());
            }
        };


        static const std::string selectButtonName = meta.iconCreated + " Chose ";
        if (ImGui::Button(selectButtonName.c_str()))
            ImGui::OpenPopup("HeightFieldTexSelectPopUp");
        if (ImGui::BeginPopup("HeightFieldTexSelectPopUp"))
        {
            ImGui::TextDisabled("Select HeightField...");
            ImGui::Spacing();
            drawHeightFieldImageSelect();
            ImGui::EndPopup();
        }

        return bChangedValue;
    }

    const UIComponentReflectionDetailed& LandscapeComponent::uiComponentReflection()
    {
        static const UIComponentReflectionDetailed reflection =
        {
            .bOptionalCreated = true,
            .iconCreated = ICON_FA_MOUNTAIN + std::string("  Landscape"),
        };
        return reflection;
    }

    uint32_t LandscapeComponent::getLODCount() const
    {
        uint32_t validDim = math::max(uint32_t(m_dimension), kTerrainLowestNodeDim);

        // Get max lod count.
        int32_t count = 
            int32_t(std::bit_width(validDim)) - int32_t(std::bit_width(kTerrainLowestNodeDim)) + 2 - std::bit_width(kTerrainCoarseNodeDim);

        // Valid check.
        return uint32_t(math::max(count, 1));
    }

    uint32_t LandscapeComponent::getRenderDimension() const
    {
        return kTerrainLowestNodeDim * kTerrainCoarseNodeDim * math::pow(2, getLODCount() - 1);
    }

    const GPUImageAsset* LandscapeComponent::getGPUImage() const
    {
        return m_heightmapImage.get();
    }

    GPUImageAsset* LandscapeComponent::getGPUImage()
    {
        return m_heightmapImage.get();
    }

    bool LandscapeComponent::setAssetUUID(const UUID& in)
    {
        if (m_heightmapTextureUUID != in)
        {
            clearCache();
            m_heightmapTextureUUID = in;

            buildCache();

            return true;
        }
        return false;
    }
}


