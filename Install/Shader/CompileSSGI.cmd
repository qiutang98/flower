
: SSGI
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SSGI_Apply.glsl -o Spirv/SSGI_Apply.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SSGI_Intersect.glsl -o Spirv/SSGI_Intersect.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SSGI_Reproject.glsl -o Spirv/SSGI_Reproject.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SSGI_Prefilter.glsl -o Spirv/SSGI_Prefilter.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/SSGI_Temporal.glsl -o Spirv/SSGI_Temporal.comp.spv