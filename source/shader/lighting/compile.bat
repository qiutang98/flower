%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/brdf_lut.glsl -O -o %~dp0/../../../install/shader/brdf_lut.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/deferred_lighting.glsl -O -o %~dp0/../../../install/shader/deferred_lighting.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/skylight.glsl -O -o %~dp0/../../../install/shader/skylight.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/skyreflection.glsl -O -o %~dp0/../../../install/shader/skyreflection.comp.spv