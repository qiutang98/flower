%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/tonemapper.glsl -O -o %~dp0/../../../install/shader/pp_tonemapper.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/combine.glsl -O -o %~dp0/../../../install/shader/pp_combine.comp.spv

%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/exposure_apply.glsl -O -o %~dp0/../../../install/shader/pp_exposure_apply.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/exposure_weight.glsl -O -o %~dp0/../../../install/shader/pp_exposure_weight.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/fusion_gaussian.glsl -O -o %~dp0/../../../install/shader/pp_fusion_gaussian.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/fusion_laplace4.glsl -O -o %~dp0/../../../install/shader/pp_fusion_laplace4.comp.spv


%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/fusion_blend.glsl -O -o %~dp0/../../../install/shader/pp_fusion_blend.comp.spv
%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/fusion.glsl -O -o %~dp0/../../../install/shader/pp_fusion.comp.spv

%~dp0/../glslc.exe -fshader-stage=comp --target-env=vulkan1.3 %~dp0/gaussian4.glsl -O -o %~dp0/../../../install/shader/gaussian4.comp.spv