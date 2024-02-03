#include "../asset/asset_common.h"
#include "../asset/asset_texture.h"
#include "../scene/scene.h"
#include "../scene/scene_node.h"
#include "../scene/component.h"
#include "../scene/component/transform.h"
#include "../asset/asset_material.h"
#include "../asset/asset_staticmesh.h"
#include "../scene/component/staticmesh_component.h"
#include "../scene/component/sky_component.h"
#include "../scene/component/postprocess_component.h"
#include "../scene/component/reflection_probe_component.h"
#include "../scene/component/landscape_component.h"

registerPODClassMember(AssetSaveInfo)
{
	archive(m_name, m_storeFolder, m_storePath);
}

registerClassMember(AssetInterface)
{
	archive(m_saveInfo);
    archive(m_rawAssetPath);
}

registerPODClassMember(StaticMeshRenderBounds)
{
    archive(origin, extents, radius);
}
registerPODClassMember(TimeOfDay)
{
    archive(year, month, day, hour, minute, second);
}

registerPODClassMember(CloudParametersInputs)
{
    archive(
        cloudAreaStartHeight,
        cloudAreaThickness,
        cloudGodRayScale,
        cloudShadowExtent,
        cloudWeatherUVScale,
        cloudCoverage,
        cloudDensity,
        cloudShadingSunLightScale,
        cloudFogFade,
        cloudMaxTraceingDistance,
        cloudTracingStartMaxDistance,
        cloudDirection,
        cloudSpeed,
        cloudMultiScatterExtinction,
        cloudMultiScatterScatter,
        cloudBasicNoiseScale,
        cloudDetailNoiseScale,
        cloudAlbedo,
        cloudPhaseForward,
        cloudPhaseBackward,
        cloudPhaseMixFactor,
        cloudPowderScale,
        cloudPowderPow,
        cloudLightStepMul,
        cloudLightBasicStep,
        cloudLightStepNum,
        cloudEnableGroundContribution,
        cloudMarchingStepNum,
        cloudSunLitMapOctave,
        cloudNoiseScale,
        cloudGodRay,
        cloudAmbientScale);
}

registerPODClassMember(AtmosphereParametersInputs)
{
    archive(
        absorptionColor,
        absorptionLength,
        rayleighScatteringColor,
        rayleighScatterLength,
        multipleScatteringFactor,
        miePhaseFunctionG,
        bottomRadius,
        topRadius,
        mieScatteringColor,
        mieScatteringLength,
        mieAbsColor,
        mieAbsLength,
        mieAbsorption,
        viewRayMarchMinSPP,
        groundAlbedo,
        viewRayMarchMaxSPP);

    constexpr int rayleighDensityCount = sizeof(rayleighDensity) / sizeof(rayleighDensity[0]);
    for (int i = 0; i < rayleighDensityCount; i++)
    {
        archive(rayleighDensity[i]);
        archive(mieDensity[i]);
        archive(absorptionDensity[i]);
    }
}

registerPODClassMember(CascadeShadowConfig)
{
    archive(
        bSDSM,
        bContactShadow,
        cascadeCount,
        percascadeDimXY,
        maxDrawDepthDistance,
        splitLambda,
        shadowBiasConst,
        shadowBiasSlope,
        filterSize,
        cascadeMixBorder,
        contactShadowLen,
        contactShadowSampleNum);
}

registerPODClassMember(RaytraceShadowConfig)
{
    archive(rayMinRange, rayMaxRange, lightRadius);
}

registerPODClassMember(SkyLightInfo)
{
    archive(
        color,
        intensity,
        direction,
        shadowType,
        cascadeConfig);

    if (version > 0)
    {
        archive(shadowColorIntensity, shadowColor, rayTraceConfig);
    }
}

registerPODClassMember(PostprocessVolumeSetting)
{
    // Auto exposure.
    archive(
        bAutoExposureEnable,
        autoExposureFixExposure,
        autoExposureLowPercent,
        autoExposureHighPercent,
        autoExposureMinBrightness,
        autoExposureMaxBrightness,
        autoExposureSpeedDown,
        autoExposureSpeedUp,
        autoExposureExposureCompensation);

    archive(bTAAEnableColorFilter, taaAntiFlicker, taaHistorySharpen, taaBaseBlendFactor);

    archive(bloomIntensity, bloomRadius, bloomThreshold, bloomThresholdSoft);

    archive(expandGamutFactor);

    // Tonemapper.
    archive(
        tonemapper_P,
        tonemapper_a,
        tonemapper_m,
        tonemapper_l,
        tonemapper_c,
        tonemapper_b);

    archive(
        tonemapper_type,
        tonemapper_filmACESSlope,
        tonemapper_filmACESToe,
        tonemapper_filmACESShoulder,
        tonemapper_filmACESBlackClip,
        tonemapper_filmACESWhiteClip,
        tonemapper_filmACESPreDesaturate,
        tonemapper_filmACESPostDesaturate,
        tonemapper_filmACESRedModifier,
        tonemapper_filmACESGlowScale);

    archive(
        tonemapper_P,
        tonemapper_a,
        tonemapper_m,
        tonemapper_l,
        tonemapper_c,
        tonemapper_b);


    archive(
        ssao_sliceCount,
        ssao_falloff,
        ssao_stepCount,
        ssao_intensity,
        ssao_power,
        ssao_viewRadius);
    if (version > 3)
    {
        archive(gtao_radius, gtao_thickness, ssao_bGTAO, gtao_falloffEnd);
    }
}

registerPODClassMember(StaticMeshSubMesh)
{
    archive(indicesStart, indicesCount, material, bounds);
}

registerPODClassMember(StaticMeshBin)
{
    archive(normals, tangents, uv0s, positions, indices);
}

registerClassMemberInherit(AssetMaterial, AssetInterface)
{
    archive(baseColorTexture);
    archive(normalTexture);
    archive(metalRoughnessTexture);
    archive(emissiveTexture);
    archive(aoTexture);

    archive(baseColorMul);
    archive(baseColorAdd);

    archive(metalMul);
    archive(metalAdd);
    archive(roughnessMul);
    archive(roughnessAdd);

    archive(emissiveMul);
    archive(emissiveAdd);

    archive(cutoff);
    ARCHIVE_ENUM_CLASS(shadingModelType);
}}

registerClassMemberInherit(AssetTexture, AssetInterface)
{
    archive(m_bSRGB);
    archive(m_mipmapCount);
    archive(m_dimension);
    archive(m_format);
    archive(m_alphaMipmapCutoff);
}}

registerClassMemberInherit(AssetStaticMesh, AssetInterface)
{
    archive(m_subMeshes);
    archive(m_indicesCount);
    archive(m_verticesCount);

    if (version > 0)
    {
        archive(m_minPosition);
        archive(m_maxPosition);
    }
}}

registerClassMember(Component)
{
    archive(m_node);
}

registerClassMemberInherit(Transform, Component)
{
    archive(m_translation, m_rotation, m_scale);
}}

registerClassMemberInherit(LandscapeComponent, Component)
{
    if (version > 4)
    {
        archive(m_heightmapTextureUUID, m_dimension, m_maxHeight, m_minHeight, m_offset);
    }
}}

registerClassMemberInherit(RenderableComponent, Component)
{

}}

registerClassMemberInherit(StaticMeshComponent, RenderableComponent)
{
    archive(m_assetUUID);
}}

registerClassMemberInherit(ReflectionProbeComponent, Component)
{
    archive(m_dimension);
    if (version > 2)
    {
        archive(m_minExtent, m_maxExtent);
    }
}}

registerClassMemberInherit(SkyComponent, Component)
{
    archive(m_tod, m_bLocalTime, m_atmosphere, m_sun);

    if (version > 5)
    {
        archive(m_cloud);
    }
}}

registerClassMemberInherit(PostprocessComponent, Component)
{
    archive(m_setting);
}}

registerClassMember(SceneNode)
{
    archive(m_bVisibility);
    archive(m_bStatic);
    archive(m_id);
    archive(m_name);
    archive(m_parent);
    archive(m_scene);
    archive(m_components);
    archive(m_children);
}

registerClassMemberInherit(Scene, AssetInterface)
{	
	archive(m_currentId);
	archive(m_root);
	archive(m_components);
	archive(m_sceneNodes);
}}