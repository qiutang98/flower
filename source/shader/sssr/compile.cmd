%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_apply.glsl -O -o %~dp0/../../../install/shader/sssr_apply.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_intersect_args.glsl -O -o %~dp0/../../../install/shader/sssr_intersect_args.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_intersect.glsl -O -o %~dp0/../../../install/shader/sssr_intersect.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_prefilter.glsl -O -o %~dp0/../../../install/shader/sssr_prefilter.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_reproject.glsl -O -o %~dp0/../../../install/shader/sssr_reproject.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_temporal.glsl -O -o %~dp0/../../../install/shader/sssr_temporal.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sssr_tile.glsl -O -o %~dp0/../../../install/shader/sssr_tile.comp.spv