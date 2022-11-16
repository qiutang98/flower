: Atmosphere lut.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/UE4_AtmosphereTransmittanceLut.glsl -O -o Spirv/AtmosphereTransmittanceLut.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/UE4_AtmosphereMultiScatterLut.glsl -O -o Spirv/AtmosphereMultiScatterLut.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/UE4_AtmosphereSkyViewLut.glsl -O -o Spirv/AtmosphereSkyViewLut.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/UE4_AtmosphereFroxelLut.glsl -O -o Spirv/AtmosphereFroxelLut.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/UE4_AtmosphereComposition.glsl -O -o Spirv/AtmosphereComposition.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/UE4_AtmosphereEnvironmentCapture.glsl -O -o Spirv/AtmosphereEnvironmentCapture.comp.spv