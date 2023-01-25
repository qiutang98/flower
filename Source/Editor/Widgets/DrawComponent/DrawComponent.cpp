#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

const std::string GIconLandscape = std::string("  ") + ICON_FA_MOUNTAIN_SUN + std::string("  Landscape");
const std::string GIconSunSky = std::string("  ") + ICON_FA_SUN + std::string("  SunSky");
const std::string GIconSpotLight = std::string("  ") + ICON_FA_SUN + std::string("  SpotLight");
const std::string GIconStaticMesh = std::string("   ") + ICON_FA_BUILDING + std::string("   StaticMesh");
const std::string GIconPMX = std::string("   ") + ICON_FA_M + ICON_FA_I + ICON_FA_K + ICON_FA_U + std::string("   PMX");
const std::string GIconReflectionCapture = std::string("   ") + ICON_FA_CAMERA + std::string("   ReflectionCapture");
const std::string GIconPostprocessVolume = std::string("   ") + ICON_FA_CUBE + std::string("   PostProcessVolume");

std::unordered_map<std::string, ComponentDrawer> GDrawComponentMap = 
{
	{ GIconPMX, { typeid(PMXComponent).name(), &ComponentDrawer::drawPMX }},
	{ GIconLandscape, { typeid(LandscapeComponent).name(), &ComponentDrawer::drawLandscape }},
	{ GIconSunSky, { typeid(SunSkyComponent).name(), &ComponentDrawer::drawSunSky }},
	{ GIconSpotLight, { typeid(SpotLightComponent).name(), &ComponentDrawer::drawSpotLight }},
	{ GIconStaticMesh, { typeid(StaticMeshComponent).name(), &ComponentDrawer::drawStaticMesh }},
	{ GIconReflectionCapture, { typeid(ReflectionCaptureComponent).name(), &ComponentDrawer::drawReflectionCapture }},
	{ GIconPostprocessVolume, { typeid(PostprocessVolumeComponent).name(), &ComponentDrawer::drawPostprocessVolume }},
};