: StaticMesh Culling Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/StaticMeshCulling.glsl -O -o Spirv/StaticMeshCulling.comp.spv

: StaticMesh GBuffer Vertex Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER Source/StaticMeshGBuffer.glsl -O -o Spirv/StaticMeshGBuffer.vert.spv

: StaticMesh GBuffer Pixel Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER Source/StaticMeshGBuffer.glsl -O -o Spirv/StaticMeshGBuffer.frag.spv

: Tonemapper Compute Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Tonemapper.glsl -O -o Spirv/Tonemapper.comp.spv

: BasicLighting Compute Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/BasicLighting.glsl -O -o Spirv/BasicLighting.comp.spv

%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/BlueNoiseGenerate.glsl -O -o Spirv/BlueNoiseGenerate.comp.spv

: IBL Compute Shader.
call CompileIBL.cmd

: Hiz build compute Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SceneHizBuild.glsl -O -o Spirv/SceneHizBuild.comp.spv

: Histogram lum
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AdaptiveExposureHistogramLumiance.glsl -O -o Spirv/AdaptiveExposureHistogramLumiance.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AdaptiveExposureAverageLumiance.glsl -O -o Spirv/AdaptiveExposureAverageLumiance.comp.spv

: GTAO
call CompileGTAO.cmd

: Bloom build compute shader.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/BasicBloomDownsample.glsl -O -o Spirv/BasicBloomDownsample.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/BasicBloomUpscale.glsl -O -o Spirv/BasicBloomUpscale.comp.spv

: Volumetric cloud.
call CompileVolumetricCloud.cmd

: SDSM Shaders.
call CompileSDSM.cmd

: Atmosphere
call CompileAtmosphere.cmd

: SSR 
call CompileSSR.cmd