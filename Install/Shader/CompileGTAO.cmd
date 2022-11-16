%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/GTAOEvaluate.glsl -O -o Spirv/GTAOEvaluate.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/GTAOSpatial.glsl -O -o Spirv/GTAOSpatialFilter.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/GTAOTemporal.glsl -O -o Spirv/GTAOTemporalFilter.comp.spv