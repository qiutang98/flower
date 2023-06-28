%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/pmx_gbuffer.glsl -O -o %~dp0/../../../install/shader/pmx_gbuffer.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/pmx_gbuffer.glsl -O -o %~dp0/../../../install/shader/pmx_gbuffer.frag.spv

%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/pmx_outline_depth.glsl -O -o %~dp0/../../../install/shader/pmx_outline_depth.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/pmx_outline_depth.glsl -O -o %~dp0/../../../install/shader/pmx_outline_depth.frag.spv


%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/pmx_sdsm_depth.glsl -O -o %~dp0/../../../install/shader/pmx_sdsm_depth.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/pmx_sdsm_depth.glsl -O -o %~dp0/../../../install/shader/pmx_sdsm_depth.frag.spv


%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/pmx_translucency.glsl -O -o %~dp0/../../../install/shader/pmx_translucency.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/pmx_translucency.glsl -O -o %~dp0/../../../install/shader/pmx_translucency.frag.spv