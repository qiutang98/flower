%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/point.glsl -O -o %~dp0/../../../install/shader/down_point.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/gaussian.glsl -O -o %~dp0/../../../install/shader/gaussian.comp.spv

%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/point4.glsl -O -o %~dp0/../../../install/shader/down_point4.comp.spv