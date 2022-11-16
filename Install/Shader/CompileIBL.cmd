%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/IBL_Irradiance.glsl -O -o Spirv/IBLIrradiance.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/IBL_Prefilter.glsl -O -o Spirv/IBLPrefilter.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/IBL_BRDFLut.glsl -O -o Spirv/BRDFLut.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/IBL_SphericalToCube.glsl -O -o Spirv/SphericalToCube.comp.spv