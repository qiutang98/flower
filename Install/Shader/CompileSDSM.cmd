%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SDSM_DepthRange.glsl -O -o Spirv/SDSMDepthRange.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SDSM_PrepareCascade.glsl -O -o Spirv/SDSMPrepareCascade.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SDSM_Culling.glsl -O -o Spirv/SDSMCulling.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER Source/SDSM_Depth.glsl -O -o Spirv/SDSMDepth.vert.spv
%~dp0/../Tool/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER Source/SDSM_Depth.glsl -O -o Spirv/SDSMDepth.frag.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SDSM_EvaluateSoftShadow.glsl -O -o Spirv/SDSMEvaluateSoftShadow.comp.spv
