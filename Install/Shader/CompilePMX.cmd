: StaticMesh GBuffer Vertex Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=vert --target-env=vulkan1.3 -DVERTEX_SHADER Source/PMX_LightingDraw.glsl -O -o Spirv/PMX_LightingDraw.vert.spv

: StaticMesh GBuffer Pixel Shader.
%~dp0/../Tool/glslc.exe -fshader-stage=frag --target-env=vulkan1.3 -DPIXEL_SHADER Source/PMX_LightingDraw.glsl -O -o Spirv/PMX_LightingDraw.frag.spv
