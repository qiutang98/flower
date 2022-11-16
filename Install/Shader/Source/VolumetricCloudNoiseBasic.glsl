#version 460

#extension GL_GOOGLE_include_directive : enable

layout (set = 0, binding = 0, rgba8) uniform image2D imageBasicNoise; // 1024 x 1024
#include "Noise.glsl"

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 texSize = imageSize(imageBasicNoise);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= texSize.x || workPos.y >= texSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(texSize);

    vec4 noiseColor = vec4(1.0);
    {
        vec3 coord = fract(vec3(uv + vec2(.2, 0.62), .5));
    
        float mfbm = 0.9;
        float mvor = 0.7;
        
        noiseColor.r = 
            mix(1., tilableFbm(coord, 7, 4.), mfbm) * 
            mix(1., tilableVoronoi(coord, 8, 9. ), mvor); // < 1.0

        noiseColor.g = 
            0.625 * tilableVoronoi(coord + 0., 3, 15. ) +
        	0.250 * tilableVoronoi(coord + 0., 3, 19. ) +
        	0.125 * tilableVoronoi(coord + 0., 3, 23. ); // < 1.0

        noiseColor.b = 1. - tilableVoronoi(coord + 0.5, 6, 9.); // < 1.0
    }
	imageStore(imageBasicNoise, workPos, noiseColor);
}