call %~dp0/mesh/compile.bat
call %~dp0/postprocess/compile.bat
call %~dp0/lighting/compile.bat
call %~dp0/atmosphere/compile.bat
call %~dp0/shadow/compile.cmd
call %~dp0/gtao/compile.cmd
call %~dp0/sssr/compile.cmd
call %~dp0/ssgi/compile.cmd
call %~dp0/autoexposure/compile.cmd
call %~dp0/bloom/compile.cmd
call %~dp0/cbt/compile.cmd
call %~dp0/terrain/compile.cmd

%~dp0/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/hzb.glsl -O -o %~dp0/../../install/shader/hzb.comp.spv
%~dp0/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/pick.glsl -O -o %~dp0/../../install/shader/pick.comp.spv
%~dp0/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/selection_outline.glsl -O -o %~dp0/../../install/shader/selection_outline.comp.spv


%~dp0/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER %~dp0/grid.glsl -O -o %~dp0/../../install/shader/grid.vert.spv
%~dp0/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER  %~dp0/grid.glsl -O -o %~dp0/../../install/shader/grid.frag.spv
