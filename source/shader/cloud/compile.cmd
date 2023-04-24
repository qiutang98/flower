%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/cloud_basicnoise.glsl -O -o %~dp0/../../../install/shader/cloud_basicnoise.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/cloud_detailnoise.glsl -O -o %~dp0/../../../install/shader/cloud_detailnoise.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/cloud_raymarching.glsl -O -o %~dp0/../../../install/shader/cloud_raymarching.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/cloud_reconstruct.glsl -O -o %~dp0/../../../install/shader/cloud_reconstruct.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/cloud_composite.glsl -O -o %~dp0/../../../install/shader/cloud_composite.comp.spv