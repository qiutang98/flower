%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sdsm/sdsm_cascade.glsl -O -o %~dp0/../../../install/shader/sdsm_cascade.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sdsm/sdsm_cull.glsl -O -o %~dp0/../../../install/shader/sdsm_cull.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sdsm/sdsm_range.glsl -O -o %~dp0/../../../install/shader/sdsm_range.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/sdsm/sdsm_resolve.glsl -O -o %~dp0/../../../install/shader/sdsm_resolve.comp.spv

%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/sdsm/sdsm_depth.glsl -O -o %~dp0/../../../install/shader/sdsm_depth.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/sdsm/sdsm_depth.glsl -O -o %~dp0/../../../install/shader/sdsm_depth.frag.spv

