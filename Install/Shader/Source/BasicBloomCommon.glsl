#ifndef BasicBloomCommon_GLSL
#define BasicBloomCommon_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#extension GL_GOOGLE_include_directive : enable

#define MIX_BLOOM_UPSCALE 1
#define MIX_BLOOM_OUTPUT 0

vec3 prefilter(vec3 c, vec4 prefilterFactor) 
{
    float brightness = max(c.r, max(c.g, c.b));

    float soft = brightness - prefilterFactor.y;

    soft = clamp(soft, 0, prefilterFactor.z);
    soft = soft * soft * prefilterFactor.w;
    
    float contribution = max(soft, brightness - prefilterFactor.x);
    contribution /= max(brightness, 0.00001);

    return c * contribution;
}

#endif