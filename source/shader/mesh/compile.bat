%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/staticmesh_after_prepass_cull.glsl -O -o %~dp0/../../../install/shader/staticmesh_cull.comp.spv
%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/staticmesh_gbuffer.glsl -O -o %~dp0/../../../install/shader/staticmesh_gbuffer.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/staticmesh_gbuffer.glsl -O -o %~dp0/../../../install/shader/staticmesh_gbuffer.frag.spv

%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/staticmesh_prepass_cull.glsl -O -o %~dp0/../../../install/shader/staticmesh_prepass_cull.comp.spv
%~dp0/../glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/staticmesh_prepass.glsl -O -o %~dp0/../../../install/shader/staticmesh_prepass.vert.spv
%~dp0/../glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/staticmesh_prepass.glsl -O -o %~dp0/../../../install/shader/staticmesh_prepass.frag.spv