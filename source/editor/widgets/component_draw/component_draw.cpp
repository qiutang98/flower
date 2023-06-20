#include "component_draw.h"
#include <imgui/ui.h>
#include <scene/component/static_mesh.h>
#include <scene/component/postprocess.h>
#include <scene/component/terrain.h>

using namespace engine;
using namespace engine::ui;

const std::string kIconStaticMesh = ICON_FA_BUILDING + std::string("   StaticMesh");
const std::string kIconSky = ICON_FA_SUN + std::string("  Sky");
const std::string kIconPostprocess = ICON_FA_STAR + std::string("  Postprocess");
const std::string kIconTerrain = ICON_FA_MOUNTAIN_SUN + std::string("  Terrain");
const std::string kIconPMX = std::string("     PMX");

std::unordered_map<std::string, ComponentDrawer> kDrawComponentMap =
{
	{ kIconStaticMesh, { typeid(StaticMeshComponent).name(), &ComponentDrawer::drawStaticMesh }},
	{ kIconSky, { typeid(SkyComponent).name(), &ComponentDrawer::drawSky }},
	{ kIconPostprocess, { typeid(PostprocessVolumeComponent).name(), &ComponentDrawer::drawPostprocess }},
	{ kIconTerrain, { typeid(TerrainComponent).name(), &ComponentDrawer::drawTerrain }},
	{ kIconPMX, {typeid(PMXComponent).name(), &ComponentDrawer::drawPMX }}
};


void ComponentDrawer::drawLight(std::shared_ptr<engine::LightComponent> comp)
{
	ImGui::PushID("DrawLight");
	{
		ImGui::Spacing();

		math::vec3 color = comp->getColor();
		ImGui::ColorEdit3("Color", &color[0]);
		comp->setColor(color);

		ImGui::PushItemWidth(100.0f);

		float intensity = comp->getIntensity();
		ImGui::DragFloat("Intensity", &intensity, 0.25f, 0.0f, 1000.0f);
		comp->setIntensity(intensity);

		ImGui::SameLine();
		bool bRayTraceShadow = comp->isRayTraceShadow();
		ImGui::Checkbox("Raytrace shadow", &bRayTraceShadow);
		comp->setRayTraceShadow(bRayTraceShadow);

		ImGui::PopItemWidth();

		ImGui::Spacing();
	}
	ImGui::PopID();
}