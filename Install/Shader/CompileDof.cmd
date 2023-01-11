: Atmosphere lut.
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/KinoBokehDof_Prepare.glsl -O -o Spirv/Dof_Prepare.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/KinoBokehDof_Combine.glsl -O -o Spirv/Dof_Combine.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/KinoBokehDof_Blur.glsl -O -o Spirv/Dof_ExpandFill.comp.spv
%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/KinoBokehDof_Gather.glsl -O -o Spirv/Dof_Gather.comp.spv

%~dp0/../Tool/glslc.exe -fshader-stage=comp --target-env=vulkan1.3 Source/KinoBokehDof_FocusDepthEvaluate.glsl -O -o Spirv/Dof_FocusEvaluate.comp.spv