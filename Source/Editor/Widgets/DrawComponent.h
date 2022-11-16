#pragma once
#include "Pch.h"

extern const std::string GIconLandscape;
extern const std::string GIconDirectionalLight;
extern const std::string GIconSpotLight;
extern const std::string GIconStaticMesh;
extern const std::string GIconPMX;

struct ComponentDrawer
{
	const char* typeName;
	std::function<void(std::shared_ptr<Flower::SceneNode> node)> drawFunc;

	static void drawStaticMesh(std::shared_ptr<Flower::SceneNode> node);

	static void drawLight(std::shared_ptr<Flower::LightComponent> comp);
	static void drawDirectionalLight(std::shared_ptr<Flower::SceneNode> node);
	static void drawSpotLight(std::shared_ptr<Flower::SceneNode> node);
	static void drawPMX(std::shared_ptr<Flower::SceneNode> node);

};

extern std::unordered_map<std::string, ComponentDrawer> GDrawComponentMap;