#include "reflection.h"
#include "../asset/asset.h"
#include "../asset/asset_common.h"
#include "../asset/asset_texture.h"
#include "../scene/scene.h"
#include "../asset/asset_staticmesh.h"
#include "../asset/asset_material.h"
#include <engine/scene/component/staticmesh_component.h>
#include <engine/scene/component/sky_component.h>
#include <engine/scene/component/postprocess_component.h>
#include <engine/scene/component/reflection_probe_component.h>
#include <engine/scene/component/landscape_component.h>

size_t engine::kRelfectionCompilePlayHolder = buildRuntimeUUID64u();

using namespace rttr;
using namespace engine;

RTTR_REGISTRATION
{
	registration::class_<AssetInterface>("engine::AssetInterface");

	registration::class_<AssetMaterial>("engine::AssetMaterial")
		.method("uiGetAssetReflectionInfo", &AssetMaterial::uiGetAssetReflectionInfo);

	registration::class_<AssetTexture>("engine::AssetTexture")
		.method("uiGetAssetReflectionInfo", &AssetTexture::uiGetAssetReflectionInfo);

	registration::class_<AssetStaticMesh>("engine::AssetStaticMesh")
		.method("uiGetAssetReflectionInfo", &AssetStaticMesh::uiGetAssetReflectionInfo);

	registration::class_<Scene>("engine::Scene");

	registration::class_<Component>("engine::Component")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &Component::uiComponentReflection);

	registration::class_<RenderableComponent>("engine::RenderableComponent")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &RenderableComponent::uiComponentReflection);

	registration::class_<Transform>("engine::Transform")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &Transform::uiComponentReflection);

	registration::class_<StaticMeshComponent>("engine::StaticMeshComponent")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &StaticMeshComponent::uiComponentReflection);

	registration::class_<SkyComponent>("engine::SkyComponent")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &SkyComponent::uiComponentReflection);

	registration::class_<PostprocessComponent>("engine::PostprocessComponent")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &PostprocessComponent::uiComponentReflection);

	registration::class_<ReflectionProbeComponent>("engine::ReflectionProbeComponent")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &ReflectionProbeComponent::uiComponentReflection);

	registration::class_<LandscapeComponent>("engine::LandscapeComponent")
		.constructor<>()(policy::ctor::as_std_shared_ptr)
		.method("uiComponentReflection", &LandscapeComponent::uiComponentReflection);
}