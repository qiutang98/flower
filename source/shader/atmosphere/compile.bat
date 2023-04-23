%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/air_perspective.glsl -O -o %~dp0/../../../install/shader/sky_air_perspective.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/composition.glsl -O -o %~dp0/../../../install/shader/sky_composition.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/skyview_lut.glsl -O -o %~dp0/../../../install/shader/skyview_lut.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/transmittance_lut.glsl -O -o %~dp0/../../../install/shader/transmittance_lut.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/multi_scatter_lut.glsl -O -o %~dp0/../../../install/shader/multi_scatter_lut.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/bake_capture.glsl -O -o %~dp0/../../../install/shader/bake_capture.comp.spv