#pragma once
#include "Light.h"
#include "../../Renderer/Parameters.h"

namespace Flower
{
	class SunSkyComponent : public LightComponent
	{
		ARCHIVE_DECLARE;

#pragma region SerializeField
	////////////////////////////// Serialize area //////////////////////////////
	protected:
		uint32_t m_percascadeDimXY = 2048;
		uint32_t m_cascadeCount = 4;

		float m_shadowFilterSize = 0.5f;
		float m_maxFilterSize = 1.0f;

		float m_cascadeSplitLambda = 1.0f;

		float m_shadowBiasConst = -1.25f; // We reverse z, so bias const should be negative.
		float m_shadowBiasSlope = -1.75f; // We reverse z, so bias slope should be negative.

		float m_cascadeBorderAdopt = 0.006f;
		float m_cascadeEdgeLerpThreshold = 0.8f;
		float m_maxDrawDepthDistance = 200.0f;

		EarthAtmosphere m_earthAtmosphere;

	////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField

	public:
		SunSkyComponent() {}
		virtual ~SunSkyComponent() = default;

		SunSkyComponent(std::shared_ptr<SceneNode> sceneNode)
			: LightComponent(sceneNode)
		{

		}

		uint32_t getCascadeCount() const { return m_cascadeCount; }
		bool setCascadeCount(uint32_t newValue);

		uint32_t getPerCascadeDimXY() const { return m_percascadeDimXY; }
		bool setPerCascadeDimXY(uint32_t newValue);

		
		float getShadowFilterSize() const { return m_shadowFilterSize; }
		bool setShadowFilterSize(float newValue);

		float getCascadeSplitLambda() const { return m_cascadeSplitLambda; }
		bool setCascadeSplitLambda(float newValue);

		float getShadowBiasConst() const { return m_shadowBiasConst; } 
		bool setShadowBiasConst(float newValue);

		float getShadowBiasSlope() const { return m_shadowBiasSlope; } 
		bool setShadowBiasSlope(float newValue);

		float getCascadeBorderAdopt() const { return m_cascadeBorderAdopt; }
		bool setCascadeBorderAdopt(float newValue);

		float getCascadeEdgeLerpThreshold() const { return m_cascadeEdgeLerpThreshold; }
		bool setCascadeEdgeLerpThreshold(float newValue);

		float getMaxDrawDepthDistance() const { return m_maxDrawDepthDistance; }
		bool setMaxDrawDepthDistance(float newValue);

		float getMaxFilterSize() const { return m_maxFilterSize; }
		bool setMaxFilterSize(float newValue);

	public:
		const EarthAtmosphere& getAtmosphere() const { return m_earthAtmosphere; }
		bool changeAtmosphere(const EarthAtmosphere& in);

	
	};
}


