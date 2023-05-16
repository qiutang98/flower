#pragma once

#include "scene_graph.h"
#include "scene_node.h"
#include "component.h"
#include "component/static_mesh.h"
#include "component/transform.h"
#include "component/light.h"
#include "component/sky.h"
#include "component/postprocess.h"
#include "component/terrain.h"
#include "component/pmx.h"

ASSET_ARCHIVE_IMPL(Component)
{
    ARCHIVE_NVP_DEFAULT(m_node);
}
ASSET_ARCHIVE_END


ASSET_ARCHIVE_IMPL_INHERIT(Transform, Component)
{
    ARCHIVE_NVP_DEFAULT(m_translation);
    ARCHIVE_NVP_DEFAULT(m_rotation);
    ARCHIVE_NVP_DEFAULT(m_scale);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(TerrainComponent, Component)
{
    ARCHIVE_NVP_DEFAULT(m_terrainHeightfieldId);
    ARCHIVE_NVP_DEFAULT(m_terrainGrassSandMudMaskId);
    ARCHIVE_NVP_DEFAULT(m_setting);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(StaticMeshComponent, Component)
{
    ARCHIVE_NVP_DEFAULT(m_bEngineAsset);
    ARCHIVE_NVP_DEFAULT(m_staticMeshUUID);
    ARCHIVE_NVP_DEFAULT(m_staticMeshAssetRelativeRoot);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(PMXComponent, Component)
{
    ARCHIVE_NVP_DEFAULT(m_pmxUUID);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(LightComponent, Component)
{
    ARCHIVE_NVP_DEFAULT(m_color);
    ARCHIVE_NVP_DEFAULT(m_forward);
    ARCHIVE_NVP_DEFAULT(m_intensity);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(SkyComponent, LightComponent)
{
    ARCHIVE_NVP_DEFAULT(m_cascadeConfig);
    ARCHIVE_NVP_DEFAULT(m_atmosphereConfig);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(PostprocessVolumeComponent, Component)
{
    ARCHIVE_NVP_DEFAULT(m_settings);
}
ASSET_ARCHIVE_END


ASSET_ARCHIVE_IMPL(SceneNode)
{
    ARCHIVE_NVP_DEFAULT(m_bVisibility);
    ARCHIVE_NVP_DEFAULT(m_bStatic);
    ARCHIVE_NVP_DEFAULT(m_id);
    ARCHIVE_NVP_DEFAULT(m_runTimeIdName);
    ARCHIVE_NVP_DEFAULT(m_depth);
    ARCHIVE_NVP_DEFAULT(m_name);
    ARCHIVE_NVP_DEFAULT(m_parent);
    ARCHIVE_NVP_DEFAULT(m_scene);
    ARCHIVE_NVP_DEFAULT(m_components);
    ARCHIVE_NVP_DEFAULT(m_children);
    ARCHIVE_ENUM_CLASS(m_type);
}
ASSET_ARCHIVE_END

ASSET_ARCHIVE_IMPL_INHERIT(Scene, AssetInterface)
{
    ARCHIVE_NVP_DEFAULT(m_currentId);
    ARCHIVE_NVP_DEFAULT(m_root);
    ARCHIVE_NVP_DEFAULT(m_initName);
    ARCHIVE_NVP_DEFAULT(m_cacheSceneComponents);
    ARCHIVE_NVP_DEFAULT(m_nodeCount);
    ARCHIVE_NVP_DEFAULT(m_cacheSceneNodeMaps);
}
ASSET_ARCHIVE_END