#include "Pch.h"
#include "DirectionalLight.h"
#include "../../Renderer/Parameters.h"

namespace Flower
{
    bool DirectionalLightComponent::setShadowFilterSize(float newValue)
    {
        if (newValue != m_shadowFilterSize)
        {
            m_shadowFilterSize = newValue;
            markDirty();
            return true;
        }

        return false;
    }

    bool DirectionalLightComponent::setCascadeSplitLambda(float newValue)
    {
        if (newValue != m_cascadeSplitLambda)
        {
            m_cascadeSplitLambda = newValue;
            markDirty();
            return true;
        }

        return false;
    }

    bool DirectionalLightComponent::setCascadeCount(uint32_t newValue)
    {
        const auto cascadeCount = glm::clamp(
            newValue,
            1u, 
            GMaxCascadePerDirectionalLight);
        
        if (cascadeCount != m_cascadeCount)
        {
            m_cascadeCount = cascadeCount;
            markDirty();
            return true;
        }

        return false;
    }

    bool DirectionalLightComponent::setPerCascadeDimXY(uint32_t newValue)
    {
        const auto pageSize = glm::clamp(
            getNextPOT(newValue),
            GMinCascadePerDirectionalLightDimXY,
            GMaxCascadePerDirectionalLightDimXY);
        if (pageSize != m_percascadeDimXY)
        {
            m_percascadeDimXY = pageSize;
            markDirty();
            return true;
        }
        return false;
    }

    bool DirectionalLightComponent::setShadowBiasConst(float newValue)
    {
        if (newValue != m_shadowBiasConst)
        {
            m_shadowBiasConst = newValue;
            markDirty();
            return true;
        }

        return false;
    }

    bool DirectionalLightComponent::setShadowBiasSlope(float newValue)
    {
        if (newValue != m_shadowBiasSlope)
        {
            m_shadowBiasSlope = newValue;
            markDirty();
            return true;
        }
        return false;
    }

    bool DirectionalLightComponent::setCascadeBorderAdopt(float newValue)
    {
        if (newValue != m_cascadeBorderAdopt)
        {
            m_cascadeBorderAdopt = newValue;
            markDirty();
            return true;
        }
        return false;
    }

    bool DirectionalLightComponent::setCascadeEdgeLerpThreshold(float newValue)
    {
        if (newValue != m_cascadeEdgeLerpThreshold)
        {
            m_cascadeEdgeLerpThreshold = newValue;
            markDirty();
            return true;
        }
        return false;
    }
    bool DirectionalLightComponent::setMaxDrawDepthDistance(float newValue)
    {
        if (newValue != m_maxDrawDepthDistance)
        {
            m_maxDrawDepthDistance = newValue;
            markDirty();
            return true;
        }
        return false;
    }
    bool DirectionalLightComponent::setMaxFilterSize(float newValue)
    {
        if (newValue != m_maxFilterSize)
        {
            m_maxFilterSize = newValue;
            markDirty();
            return true;
        }
        return false;
    }
}