#ifndef COMMON_HEADER_H_202309161308
#define COMMON_HEADER_H_202309161308

#define AP1_COLOR_SPACE 0

#ifdef __cplusplus

#include <engine/utils/utils.h>
#include <engine/asset/asset_common.h>

namespace engine 
{
    using vec2  = glm::vec2;
    using vec3  = glm::vec3;
    using vec4  = glm::vec4;
    using ivec2 = glm::ivec2;
    using ivec3 = glm::ivec3;
    using ivec4 = glm::ivec4;
    using uvec2 = glm::uvec2;
    using uvec3 = glm::uvec3;
    using uvec4 = glm::uvec4;
    using mat2  = glm::mat2;
    using mat3  = glm::mat3; 
    using mat4  = glm::mat4;

    using uint  = uint32_t;
    #define DEFAUTT_COMPARE(X) auto operator<=>(const X&) const = default;

    #define DEFAULT_COMPARE_ARCHIVE(X) \
        ARCHIVE_DECLARE \
        DEFAUTT_COMPARE(X)

    #define CHECK_SIZE_GPU_SAFE(X) \
        static_assert(sizeof(X) % (4 * sizeof(float)) == 0);
#else
    #define ARCHIVE_DECLARE
    #define constexpr const
    #define DEFAUTT_COMPARE(X)
    #define DEFAULT_COMPARE_ARCHIVE(X)
    #define CHECK_SIZE_GPU_SAFE(X)
    #define static_assert(X)
#endif

    struct AtmosphereParametersInputs
    {
        DEFAULT_COMPARE_ARCHIVE(AtmosphereParametersInputs)

        vec3  absorptionColor;
        float absorptionLength;

        vec3  rayleighScatteringColor;
        float rayleighScatterLength;

        vec3  mieScatteringColor;
        float mieScatteringLength;

        vec3  mieAbsColor;
        float mieAbsLength;

        vec3 mieAbsorption;
        int  viewRayMarchMinSPP;

        vec3 groundAlbedo;
        int  viewRayMarchMaxSPP;

        vec4 rayleighDensity[3];
        vec4 mieDensity[3];
        vec4 absorptionDensity[3];

        float multipleScatteringFactor;
        float miePhaseFunctionG;
        float bottomRadius;
        float topRadius;
    };
    CHECK_SIZE_GPU_SAFE(AtmosphereParametersInputs)

    struct LandscapeParametersInputs
    {
        mat4 sunFarShadowViewProj;
        mat4 sunFarShadowViewProjInverse;

        uint terrainObjectId;
        uint bLandscapeValid;
        uint bLandscapeSelect;
        uint lodCount;

        uint terrainDimension;
        float offsetX; // Offset of terrain.
        float offsetY; // Offset of terrain.
        float minHeight;

        float maxHeight;
        uint  heightmapUUID;
        uint  hzbUUID;
        uint  terrainShadowValid;
    };
    CHECK_SIZE_GPU_SAFE(LandscapeParametersInputs)

    struct CloudParametersInputs
    {
        DEFAULT_COMPARE_ARCHIVE(CloudParametersInputs)

        float cloudAreaStartHeight; // km
        float cloudAreaThickness;
        float cloudGodRayScale;
        float cloudShadowExtent; // x4

        vec3 camWorldPos; // cameraworld Position, in atmosphere space unit.
        uint updateFaceIndex; // update face index for cloud cubemap capture

        // World space to cloud space view project matrix. Unit also is km.
        mat4 cloudSpaceViewProject;
        mat4 cloudSpaceViewProjectInverse;

        // Cloud settings.
        vec2  cloudWeatherUVScale;
        float cloudCoverage;
        float cloudDensity;

        float cloudShadingSunLightScale;
        float cloudFogFade;
        float cloudMaxTraceingDistance;
        float cloudTracingStartMaxDistance;

        vec3 cloudDirection;
        float cloudSpeed;

        float cloudMultiScatterExtinction;
        float cloudMultiScatterScatter;
        float cloudBasicNoiseScale;
        float cloudDetailNoiseScale;

        vec3  cloudAlbedo;
        float cloudPhaseForward;

        float cloudPhaseBackward;
        float cloudPhaseMixFactor;
        float cloudPowderScale;
        float cloudPowderPow;

        float cloudLightStepMul;
        float cloudLightBasicStep;
        int  cloudLightStepNum;
        int cloudEnableGroundContribution;

        int cloudMarchingStepNum;
        int cloudSunLitMapOctave;
        float cloudNoiseScale;
        int cloudGodRay;

        float cloudAmbientScale;
        float cloudPad0;
        float cloudPad1;
        float cloudPad2;
    };
    CHECK_SIZE_GPU_SAFE(CloudParametersInputs)

   // Enum mesh type area.
#ifdef __cplusplus
    enum ETonemapperType
    {
        ETonemapperType_GT = 0,
        ETonemapperType_FilmACES = 1,

        ETonemapperType_Max,
    };
#else
    #define ETonemapperType              int
    #define ETonemapperType_GT           0
    #define ETonemapperType_FilmACES     1
#endif 

    struct PostprocessVolumeSetting
    {
        DEFAULT_COMPARE_ARCHIVE(PostprocessVolumeSetting)

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // Auto exposure.
        uint  bAutoExposureEnable;
        float autoExposureFixExposure;
        float autoExposureLowPercent;
        float autoExposureHighPercent;

        float autoExposureMinBrightness;
        float autoExposureMaxBrightness;
        float autoExposureSpeedDown;
        float autoExposureSpeedUp;

        float autoExposureExposureCompensation;
        float autoExposureScale;
        float autoExposureOffset;
        float autoExposureDeltaTime;
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // TAA
        uint  bTAAEnableColorFilter;
        float taaAntiFlicker;
        float taaHistorySharpen;
        float taaBaseBlendFactor;

        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        float bloomIntensity;
        float bloomRadius;
        float bloomThreshold;
        float bloomThresholdSoft;
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        float expandGamutFactor;
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        float tonemapper_P;  // Max brightness.
        float tonemapper_a;  // contrast
        float tonemapper_m;  // linear section start
        float tonemapper_l;  // linear section length
        float tonemapper_c;  // black
        float tonemapper_b;  // pedestal

        float tonemapper_l0;
        float tonemapper_L1;
        float tonemapper_S0;
        float tonemapper_S1;
        float tonemapper_C2;
        float tonemapper_CP;

        ETonemapperType tonemapper_type;
        float tonemapper_filmACESSlope;
        float tonemapper_filmACESToe;
        float tonemapper_filmACESShoulder;

        float tonemapper_filmACESBlackClip;
        float tonemapper_filmACESWhiteClip;
        float tonemapper_filmACESPreDesaturate;
        float tonemapper_filmACESPostDesaturate;

        float tonemapper_filmACESRedModifier;
        float tonemapper_filmACESGlowScale;
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        float ssao_uvRadius;
        int   ssao_sliceCount;
        float ssao_falloffMul;
        float ssao_falloffAdd;

        int   ssao_stepCount;
        float ssao_intensity;
        float ssao_power;
		float ssao_viewRadius;
        float ssao_falloff;
        
        int   ssao_bGTAO;
        float gtao_radius;
        float gtao_thickness;
        int   ssao_enable;

        float gtao_falloffEnd;
        float ssao_pad0;
        float ssao_pad1;
        float ssao_pad2;
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    };
    CHECK_SIZE_GPU_SAFE(PostprocessVolumeSetting)

    // Enum mesh type area.
    #define EMeshType                       int
    #define EMeshType_StaticMesh            0
    #define EMeshType_ReflectionCaptureMesh 1

    #define ERendererType                   int
    #define ERendererType_Viewport          0
    #define ERendererType_ReflectionCapture 1        

    // Enum shading model type here.
#ifdef __cplusplus
    enum EShadingModelType
    {
        EShadingModelType_DefaultLit = 0,

        EShadingModelType_Max,
    };
    static_assert((int)EShadingModelType_Max <= 255);
#else
    #define EShadingModelType            int
    #define EShadingModelType_DefaultLit 0
#endif 


#ifdef __cplusplus
    enum EShadowType
    {
        EShadowType_CascadeShadowMap = 0,
        EShadowType_Raytrace = 1,

        EShadowType_Max,
    };
#else
    #define EShadowType                  int
    #define EShadowType_CascadeShadowMap 0
#endif 



    // Const config values.
    constexpr uint kPositionStrip = 3U;
    constexpr uint kNormalStrip   = 3U;
    constexpr uint kUv0Strip      = 2U;
    constexpr uint kTangentStrip  = 4U;

    // Max cascade count is 8.
    constexpr uint kMaxCascadeNum = 8U;

    constexpr uint kMaxPushConstSize = 128U;

    constexpr float kMaxHalfFloat   = 65504.0f;
    constexpr float kMax11BitsFloat = 65024.0f;
    constexpr float kMax10BitsFloat = 64512.0f;

    constexpr uint kMaxObjectId   = 64000U; 
    static_assert(kMaxObjectId < kMaxHalfFloat);

    // 
    constexpr uint kSkyObjectId = kMaxObjectId + 1U;

    // Altitude in GuangZhou: 21m
    constexpr float kAtmospherePlanetRadiusOffset = 21.0f * 0.001f;     // Offset 1 m. When offset height is big, this value should be more bigger. TODO: Compute this by camera height.


    constexpr float kAtmosphereCameraOffset       = kAtmospherePlanetRadiusOffset; 
    constexpr float kAtmosphereAirPerspectiveKmPerSlice = 4.0f; // Total 32 * 4 = 128 km.
    constexpr float kAtmosphereUnitScale          = 1000.0f;    // Km to meter.

    constexpr uint kTerrainLowestNodeDim = 64; // 64 * 64 meter
    constexpr uint kTerrainPatchNum = 8; // 8 * 8 patch per lowest node.
    constexpr uint kTerrainCoarseNodeDim = 4; // 4 * 4 in coarse level

    struct CascadeShadowConfig
    {
        DEFAULT_COMPARE_ARCHIVE(CascadeShadowConfig)

        int   bSDSM;
        int   bContactShadow;
        int   cascadeCount;
        int   percascadeDimXY;

        float maxDrawDepthDistance;
        float splitLambda;
        float shadowBiasConst;
        float shadowBiasSlope;

        float filterSize;
        float cascadeMixBorder;
        float contactShadowLen;
        int contactShadowSampleNum;
    };
    CHECK_SIZE_GPU_SAFE(CascadeShadowConfig)

    struct RaytraceShadowConfig
    {
        DEFAULT_COMPARE_ARCHIVE(RaytraceShadowConfig)

        float rayMinRange;
        float rayMaxRange;
        float lightRadius;
        float pad0; 
    };
    CHECK_SIZE_GPU_SAFE(RaytraceShadowConfig)

    struct SkyLightInfo
    {
        DEFAULT_COMPARE_ARCHIVE(SkyLightInfo)

        vec3  color;
        float intensity;
        vec3  direction;
        EShadowType shadowType;

        vec3 shadowColor;
        float shadowColorIntensity;

        CascadeShadowConfig cascadeConfig;
        RaytraceShadowConfig rayTraceConfig;
    };
    CHECK_SIZE_GPU_SAFE(SkyLightInfo)

    // PerFrameData data.
    struct PerFrameData 
    {
        // .x is app runtime, .y is sin(.x), .z is cos(.x), .w is pad
        vec4 appTime;

        // .x is frame count, .y is frame count % 8, .z is frame count % 16, .w is frame count % 32
        uvec4 frameIndex;

        // Camera world space position.
        vec4 camWorldPos;

        // .x fovy, .y aspectRatio, .z nearZ, .w farZ
        vec4 camInfo;

        // prev-frame's cam info.
        vec4 camInfoPrev;

        // Camera forward direction vector.
        vec4 camForward;

        // Halton sequence jitter data, .xy is current frame jitter data, .zw is prev frame jitter data.
        vec4 jitterData;

        // Camera infos.
        mat4 camView;
        mat4 camProj;
        mat4 camViewProj;

        // Camera inverse matrixs.
        mat4 camInvertView;
        mat4 camInvertProj;
        mat4 camInvertViewProj;

        // Camera matrix remove jitter effects.
        mat4 camProjNoJitter;
        mat4 camViewProjNoJitter;

        // Camera invert matrixs no jitter effects.
        mat4 camInvertProjNoJitter;
        mat4 camInvertViewProjNoJitter;

        // Prev-frame camera infos.
        mat4 camViewProjPrev;
        mat4 camViewProjPrevNoJitter; 

        mat4 camViewPrev;

        // Camera frustum planes for culling.
        vec4 frustumPlanes[6];

        float renderWidth;
        float renderHeight;
        float postWidth;
        float postHeight;

        uint  jitterPeriod;        // jitter period for jitter data.
        uint  bEnableJitter;       // Is main camera enable jitter in this frame.
        float basicTextureLODBias; // Lod basic texture bias when render mesh, used when upscale need.
        uint  bCameraCut;          // Camera cut in this frame or not.

        uint  bSkyComponentValid;
        uint  bTAAU;
        uint  skyComponentSceneNodeId; 
        uint  bSkyComponentSelected; 

        ERendererType renderType;
        uint pad0;
        uint pad1;
        uint pad2;

        // Sun light info.
        SkyLightInfo sunLightInfo;
        SkyLightInfo moonLightInfo;

        // Atmosphere info.
        AtmosphereParametersInputs atmosphere;
        CloudParametersInputs cloud;

        // Postprocess settings.
        PostprocessVolumeSetting postprocessing;

        // Landscape settings.
        LandscapeParametersInputs landscape;
    };

    // Perobject bsdf material info.
    struct BSDFMaterialInfo
    {
        EShadingModelType shadingModel;
        // Base color texture alpha cutoff value.
        float cutoff;
        uint emissiveTexId;
        uint emissiveSampler;

        uint baseColorId;
        uint baseColorSampler;
        uint normalTexId;
        uint normalSampler;

        uint metalRoughnessTexId;
        uint metalRoughnessSampler;
        uint occlusionTexId;
        uint occlusionSampler;

        vec4 baseColorMul;
        vec4 baseColorAdd;

        float metalMul;
        float metalAdd;
        float roughnessMul;
        float roughnessAdd;

        vec4 emissiveMul;
        vec4 emissiveAdd; 
    };
    CHECK_SIZE_GPU_SAFE(BSDFMaterialInfo)
         
    // Perobject mesh info.
    struct MeshInfo
    {
        // Mesh type.
        EMeshType meshType;
        // Indices count.
        uint indicesCount;
        // Index start offset position.
        uint indexStartPosition;
        // Indices buffer in bindless buffer id.
        uint indicesArrayId;

        // Normal array id.
        uint normalsArrayId;
        // Tangent array id.
        uint tangentsArrayId;
        // Positions buffer in bindless buffer id.
        uint positionsArrayId;
        // Vertices buffer in bindless buffer id.    
        uint uv0sArrayId;

        // .xyz is center local position, .w is radius.
        vec4 sphereBounds;
        // aabb extents.
        vec3 extents; 
        // Index of submeshes in the mesh.
        uint submeshIndex;         
    };
    CHECK_SIZE_GPU_SAFE(MeshInfo)

    // Perobject info.
    struct PerObjectInfo
    {
        uint sceneNodeId;
        uint bSelected;
        uint pad0;
        uint pad1;

        mat4 modelMatrix;
        mat4 modelMatrixPrev;

        MeshInfo meshInfoData;
        BSDFMaterialInfo materialInfoData;
    };
    CHECK_SIZE_GPU_SAFE(PerObjectInfo)

    struct StaticMeshDrawCommand
    {
        uint vertexCount;
        uint instanceCount;
        uint firstVertex;
        uint objectId;
    };

    struct CascadeInfo
    {
        mat4 viewProj;
        vec4 frustumPlanes[6];

        // .xy is cascade texel alias scale.
        // .z is cascade split value in view space.
        vec4 cascadeScale;
    };

    struct LineDrawVertex
	{
		vec3 worldPos;
		uint color;
	};

    struct TerrainPatch
    {
        vec2 position;
        uint lod; // use for scale.
        uint patchCrossLOD;
    };


#ifdef __cplusplus
} // namespace engine
#endif

#endif // COMMON_HEADER_H_202309161308