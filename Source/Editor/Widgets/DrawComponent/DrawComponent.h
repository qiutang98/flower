#pragma once
#include "Pch.h"

extern const std::string GIconLandscape;
extern const std::string GIconSunSky;
extern const std::string GIconSpotLight;
extern const std::string GIconStaticMesh;
extern const std::string GIconPMX;
extern const std::string GIconPostprocessVolume;
extern const std::string GIconReflectionCapture;

struct ComponentDrawer
{
	const char* typeName;
	std::function<void(std::shared_ptr<Flower::SceneNode> node)> drawFunc;

	static void drawStaticMesh(std::shared_ptr<Flower::SceneNode> node);

	static void drawLight(std::shared_ptr<Flower::LightComponent> comp);
	static void drawSunSky(std::shared_ptr<Flower::SceneNode> node);
	static void drawSpotLight(std::shared_ptr<Flower::SceneNode> node);
	static void drawPMX(std::shared_ptr<Flower::SceneNode> node);
	static void drawReflectionCapture(std::shared_ptr<Flower::SceneNode> node);
	static void drawLandscape(std::shared_ptr<Flower::SceneNode> node);
	static void drawPostprocessVolume(std::shared_ptr<Flower::SceneNode> node);
};

extern std::unordered_map<std::string, ComponentDrawer> GDrawComponentMap;