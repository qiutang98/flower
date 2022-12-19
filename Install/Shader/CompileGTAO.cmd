%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/GTAO_Evaluate.glsl -O -o Spirv/GTAOEvaluate.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/GTAO_Spatial.glsl -O -o Spirv/GTAOSpatialFilter.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/GTAO_Temporal.glsl -O -o Spirv/GTAOTemporalFilter.comp.spv