#pragma once

#include <util/util.h>
#include <scene/scene.h>

extern const std::string kIconStaticMesh;
extern const std::string kIconSky;
extern const std::string kIconPostprocess;
extern const std::string kIconTerrain;
extern const std::string kIconPMX;

struct ComponentDrawer
{
	const char* typeName;
	std::function<void(std::shared_ptr<engine::SceneNode> node)> drawFunc;

	static void drawLight(std::shared_ptr<engine::LightComponent> comp);

	static void drawStaticMesh(std::shared_ptr<engine::SceneNode> node);
	static void drawSky(std::shared_ptr<engine::SceneNode> node);
	static void drawPostprocess(std::shared_ptr<engine::SceneNode> node);
	static void drawTerrain(std::shared_ptr<engine::SceneNode> node);
	static void drawPMX(std::shared_ptr<engine::SceneNode> node);
};

extern std::unordered_map<std::string, ComponentDrawer> kDrawComponentMap;