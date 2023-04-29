#pragma once
#include "math.h"

namespace engine
{
    constexpr int32_t kMinCascadeDimSize = 512;
    constexpr int32_t kMaxCascadeDimSize = 4096;
    constexpr int32_t kMaxCascadeNum = 8;

    struct CascadeShadowConfig
    {
        int32_t cascadeCount = 4;
        int32_t percascadeDimXY = 2048;
        float cascadeSplitLambda = 1.0f;
        float maxDrawDepthDistance = 2000.0f;

        float shadowBiasConst = -1.25f; // We reverse z, so bias const should be negative.
        float shadowBiasSlope = -1.75f; // We reverse z, so bias slope should be negative.
        float shadowFilterSize = 0.5f;
        float maxFilterSize = 1.0f;

        float cascadeBorderAdopt = 0.006f;
        float cascadeEdgeLerpThreshold = 0.8f;
        float pad0;
        float pad1;

        auto operator<=>(const CascadeShadowConfig&) const = default;
        template<class Archive> void serialize(Archive& archive)
        {
            archive(
                cascadeCount,
                percascadeDimXY,
                cascadeSplitLambda,
                maxDrawDepthDistance,
                shadowBiasConst,
                shadowBiasSlope,
                shadowFilterSize,
                maxFilterSize,
                cascadeBorderAdopt,
                cascadeEdgeLerpThreshold);
        }
    };
    static_assert(sizeof(CascadeShadowConfig) % (4 * sizeof(float)) == 0);

    // All distance units in kilometers
    struct AtmosphereConfig
    {
        AtmosphereConfig() 
        { 
            resetAtmosphere(); 
            resetCloud();
        }
        void resetAtmosphere();
        void resetCloud();

        float atmospherePreExposure;
        float pad0;
        float pad1;
        float pad2;

        math::vec3 absorptionColor;
        float absorptionLength;

        math::vec3 rayleighScatteringColor;
        float rayleighScatterLength;

        float multipleScatteringFactor;
        float miePhaseFunctionG;
        float bottomRadius;
        float topRadius;

        math::vec3 mieScatteringColor;
        float mieScatteringLength;

        math::vec3 mieAbsColor;
        float mieAbsLength;

        math::vec3 mieAbsorption;
        int32_t viewRayMarchMinSPP;

        math::vec3 groundAlbedo;
        int32_t viewRayMarchMaxSPP;

        float rayleighDensity[12];
        float mieDensity[12];
        float absorptionDensity[12];

        // Clout infos.
        float cloudAreaStartHeight; // km
        float cloudAreaThickness;
        float pad3;
        float cloudShadowExtent; // x4

        math::vec3 camWorldPos; // cameraworld Position, in atmosphere space unit.
        uint32_t updateFaceIndex; // update face index for cloud cubemap capture

        // World space to cloud space view project matrix. Unit also is km.
        math::mat4 cloudSpaceViewProject;
        math::mat4 cloudSpaceViewProjectInverse;

        // Cloud settings.
        math::vec2  cloudWeatherUVScale;
        float cloudCoverage;
        float cloudDensity;

        float cloudShadingSunLightScale;
        float cloudFogFade;
        float cloudMaxTraceingDistance;
        float cloudTracingStartMaxDistance;

        math::vec3 cloudDirection;
        float cloudSpeed;

        float cloudMultiScatterExtinction;
        float cloudMultiScatterScatter;
        float cloudBasicNoiseScale;
        float cloudDetailNoiseScale;

        math::vec3  cloudAlbedo;
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
        float pad4;

        auto operator<=>(const AtmosphereConfig&) const = default;
        template<class Archive> void serialize(Archive& archive)
        {
            archive(
                atmospherePreExposure,
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

            for (uint32_t i = 0; i < 12; i++)
            {
                archive(rayleighDensity[i]);
                archive(mieDensity[i]);
                archive(absorptionDensity[i]);
            }


            archive(
                cloudAreaStartHeight,
                cloudAreaThickness,
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
                cloudEnableGroundContribution);
        }
    };
    static_assert(sizeof(AtmosphereConfig) % (4 * sizeof(float)) == 0);

    struct GPUSkyInfo
    {
        math::vec3  color;
        float intensity;

        math::vec3  direction;
        int32_t  shadowType; // Shadow type of this sky light.

        CascadeShadowConfig cacsadeConfig;
        AtmosphereConfig atmosphereConfig;
    };
    static_assert(sizeof(GPUSkyInfo) % (4 * sizeof(float)) == 0);

    struct GPUPerFrameData
    {
        // .x is app runtime, .y is sin(.x), .z is cos(.x), .w is pad
        math::vec4 appTime;

        // .x is frame count, .y is frame count % 8, .z is frame count % 16, .w is frame count % 32
        math::uvec4 frameIndex;

        // Camera world space position.
        math::vec4 camWorldPos;

        // .x fovy, .y aspectRatio, .z nearZ, .w farZ
        math::vec4 camInfo;

        // prev-frame's cam info.
        math::vec4 camInfoPrev;

        // Camera matrixs.
        math::mat4 camView;
        math::mat4 camProj;
        math::mat4 camViewProj;

        // Camera inverse matrixs.
        math::mat4 camInvertView;
        math::mat4 camInvertProj;
        math::mat4 camInvertViewProj;

        // Camera matrix remove jitter effects.
        math::mat4 camProjNoJitter;
        math::mat4 camViewProjNoJitter;

        // Camera invert matrixs no jitter effects.
        math::mat4 camInvertProjNoJitter;
        math::mat4 camInvertViewProjNoJitter;

        // Prev-frame camera infos.
        math::mat4 camViewProjPrev;
        math::mat4 camViewProjPrevNoJitter;

        // Camera frustum planes for culling.
        math::vec4 frustumPlanes[6];

        // Halton sequence jitter data, .xy is current frame jitter data, .zw is prev frame jitter data.
        math::vec4 jitterData;

        uint32_t  jitterPeriod;        // jitter period for jitter data.
        uint32_t  bEnableJitter;       // Is main camera enable jitter in this frame.
        float     basicTextureLODBias; // Lod basic texture bias when render mesh, used when upscale need.
        uint32_t  bCameraCut;          // Camera cut in this frame or not.

        uint32_t skyValid; // sky is valid.
        uint32_t skySDSMValid;
        float fixExposure;
        uint32_t bAutoExposure;

        GPUSkyInfo sky;
    };
    static_assert(sizeof(GPUPerFrameData) % (4 * sizeof(float)) == 0);

    // Keep same size with shared_struct.glsl
    struct GPUMaterialStandardPBR
    {
        uint32_t baseColorId;
        uint32_t baseColorSampler;
        uint32_t normalTexId;
        uint32_t normalSampler;

        uint32_t specTexId;
        uint32_t specSampler;
        uint32_t occlusionTexId;
        uint32_t occlusionSampler;

        uint32_t emissiveTexId;
        uint32_t emissiveSampler;
        float cutoff = 0.5f;
        // > 1.0f is backface cut, < -1.0f is frontface cut, [-1.0f, 1.0f] is no face cut.
        float faceCut = 0.0f;

        math::vec4 baseColorMul = math::vec4{ 1.0f };
        math::vec4 baseColorAdd = math::vec4{ 0.0f };

        float metalMul = 1.0f;
        float metalAdd = 0.0f;
        float roughnessMul = 1.0f;
        float roughnessAdd = 0.0f;

        math::vec4 emissiveMul = math::vec4{ 1.0f };
        math::vec4 emissiveAdd = math::vec4{ 0.0f };

        static GPUMaterialStandardPBR getDefault();
    };
    static_assert(sizeof(GPUMaterialStandardPBR) % (4 * sizeof(float)) == 0);

    struct GPUStaticMeshPerObjectData
    {
        // Material for static mesh.
        GPUMaterialStandardPBR material;

        // Current-frame model matrix.
        math::mat4 modelMatrix;

        // Prev-frame model matrix.
        math::mat4 modelMatrixPrev;

        uint32_t uv0sArrayId;    // Vertices buffer in bindless buffer id.    
        uint32_t positionsArrayId;   // Positions buffer in bindless buffer id.
        uint32_t indicesArrayId;     // Indices buffer in bindless buffer id.
        uint32_t indexStartPosition; // Index start offset position.

        math::vec4 sphereBounds;

        math::vec3 extents;
        uint32_t indexCount;         // Mesh object info, used to build draw calls.

        uint32_t objectId; // Object id of scene node.
        uint32_t bSelected;
        uint32_t tangentsArrayId;
        uint32_t normalsArrayId;
    };
    static_assert(sizeof(GPUStaticMeshPerObjectData) % (4 * sizeof(float)) == 0);

    struct GPUStaticMeshDrawCommand
    {
        // Build draw call data for VkDrawIndirectCommand
        uint32_t vertexCount;
        uint32_t instanceCount;
        uint32_t firstVertex;
        uint32_t firstInstance;

        // Object id for StaticMeshPerObjectData array indexing.
        uint32_t objectId;
        uint32_t pad0;
        uint32_t pad1;
        uint32_t pad2;
    };
    static_assert(sizeof(GPUStaticMeshDrawCommand) % (4 * sizeof(float)) == 0);


    struct GPUCascadeInfo
    {
        math::mat4 viewProj;
        math::vec4 frustumPlanes[6];
        math::vec4 cascadeScale;
    };
    static_assert(sizeof(GPUCascadeInfo) % (4 * sizeof(float)) == 0);

    struct GPUDispatchIndirectCommand
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t pad;
    };
    static_assert(sizeof(GPUDispatchIndirectCommand) % (4 * sizeof(float)) == 0);
}