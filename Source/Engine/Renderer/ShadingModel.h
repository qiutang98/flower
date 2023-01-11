#pragma once

namespace Flower
{
	constexpr float kShadingModelDelta = 0.02f;

	constexpr float kShadingModelUnvalid = 0.0f;
	constexpr float kShadingModelStandardPBR = 0.02f;

	constexpr float kShadingModelPMXBasic = 0.04f; // Start.
	constexpr float kShadingModelPMXCharacterBasic = 0.06f;


	enum class EPMXShadingModel
	{
		Basic = 0,
		CharacterBasic = 1,
	};

	inline float PMXShadingModelToParam(EPMXShadingModel model)
	{
		return kShadingModelPMXBasic + int(model) * kShadingModelDelta;
	}

	inline bool IsPMXCharacter(EPMXShadingModel model)
	{
		return int(model) >= int(EPMXShadingModel::CharacterBasic);
	}
}