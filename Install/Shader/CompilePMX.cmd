: StaticMesh GBuffer Vertex Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER Source/PMX_GBuffer.glsl -O -o Spirv/PMX_GBuffer.vert.spv

: StaticMesh GBuffer Pixel Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER Source/PMX_GBuffer.glsl -O -o Spirv/PMX_GBuffer.frag.spv

: StaticMesh GBuffer Vertex Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER Source/PMX_Translucent.glsl -O -o Spirv/PMX_Translucent.vert.spv

: StaticMesh GBuffer Pixel Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER Source/PMX_Translucent.glsl -O -o Spirv/PMX_Translucent.frag.spv


: SDSM Vertex Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER Source/PMX_SDSMDepthDraw.glsl -O -o Spirv/PMX_SDSMDepthDraw.vert.spv

: SDSM Pixel Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER Source/PMX_SDSMDepthDraw.glsl -O -o Spirv/PMX_SDSMDepthDraw.frag.spv
