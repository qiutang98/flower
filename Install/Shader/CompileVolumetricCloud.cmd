%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Cloud_RayMarching.glsl -O -o Spirv/VolumetricCloudRayMarching.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Cloud_CompositeWithScreen.glsl -O -o Spirv/VolumetricCompositeWithScreen.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Cloud_NoiseBasic.glsl -O -o Spirv/VolumetricCloudNoiseBasic.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Cloud_NoiseDetail.glsl -O -o Spirv/VolumetricCloudNoiseWorley.comp.spv

%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Cloud_ShadowMap.glsl -O -o Spirv/Cloud_ShadowMap.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/Cloud_Reconstruction.glsl -O -o Spirv/Cloud_Reconstruction.comp.spv


