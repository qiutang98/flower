%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/tone_mapper.glsl -O -o %~dp0/../../../install/shader/tone_mapper.comp.spv