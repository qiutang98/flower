
: SSR 
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRTileClassify.glsl -o Spirv/SSRTileClassify.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRApply.glsl -o Spirv/SSRApply.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRIntersect.glsl -o Spirv/SSRIntersect.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRIntersectArgs.glsl -o Spirv/SSRIntersectArgs.comp.spv

%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRReproject.glsl -o Spirv/SSRReproject.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRPrefilter.glsl -o Spirv/SSRPrefilter.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/AMD_SSRTemporalFilter.glsl -o Spirv/SSRTemporalFilter.comp.spv