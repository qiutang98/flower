%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/batcher.glsl -O -o %~dp0/../../../install/shader/terrain_batcher.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/split.glsl -O -o %~dp0/../../../install/shader/terrain_split.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/merge.glsl -O -o %~dp0/../../../install/shader/terrain_merge.comp.spv

%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/render.glsl -O -o %~dp0/../../../install/shader/terrain_render.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/render.glsl -O -o %~dp0/../../../install/shader/terrain_render.frag.spv

%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/render_sdsm_depth.glsl -O -o %~dp0/../../../install/shader/render_sdsm_depth.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/render_sdsm_depth.glsl -O -o %~dp0/../../../install/shader/render_sdsm_depth.frag.spv