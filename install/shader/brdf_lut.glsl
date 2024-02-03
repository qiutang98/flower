#version 460
#extension GL_GOOGLE_include_directive : enable

// Evaluate standard brdf lut for environment reflection lighting.

#include "common_lighting.glsl"

layout (set = 0, binding = 0, rg16f) uniform image2D brdflutImage;

// Standard disney brdf lut.
vec2 computeBRDFLut(float NoV, float alphaRoughness)
{
    // Sample count.
    const uint kNumSamples = 1024;

	// Normal always points along z-axis for the 2D lookup 
	const vec3 N = vec3(0.0, 0.0, 1.0);

    // (sin, 0, cos).
	vec3 V = vec3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
    
    vec2 lut = vec2(0.0);
	for(uint i = 0; i < kNumSamples; i++) 
    {
		vec2 Xi = hammersley2d(i, kNumSamples);

        // Sample direction build from importance sample GGX.
		vec3 H = importanceSampleGGX(Xi, alphaRoughness, N);

        // Then compute light direction.
		vec3 L = normalize(2.0 * dot(V, H) * H - V);

		float NoL = max(dot(N, L), 0.0);
		float VoH = max(dot(V, H), 0.0); 
		float NoH = max(dot(N, H), 0.0);

		if (NoL > 0.0) 
        {
            float G =  V_SmithGGXCorrelated(NoV, NoL, alphaRoughness);
            float Gv = 4.0 * NoL * G * VoH / NoH;
            float Fc = pow(1.0 - VoH, 5.0);

			lut += vec2((1.0 - Fc) * Gv, Fc * Gv);
		}
	}
	return lut / float(kNumSamples);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main() 
{
    ivec2 lutSize = imageSize(brdflutImage);
    ivec2 workPos = ivec2(gl_GlobalInvocationID.xy);

    if(workPos.x >= lutSize.x || workPos.y >= lutSize.y)
    {
        return;
    }

    const vec2 uv = (vec2(workPos) + vec2(0.5f)) / vec2(lutSize);
	imageStore(brdflutImage, workPos, vec4(computeBRDFLut(uv.x, uv.y), 0.0f, 0.0f));
}