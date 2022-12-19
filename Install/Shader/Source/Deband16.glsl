#ifndef DEBAND_16_GLSL
#define DEBAND_16_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) 
{
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash(uvec3 v) 
{ 
    return hash(v.x ^ hash(v.y) ^ hash(v.z)); 
}

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) 
{
    // const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeMantissa = 0x00007FFFu;
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                   // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                        // Add fractional part to 1.0

    float  f = uintBitsToFloat(m);       // Range [1:2]
    return f - 1.0;                      // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
// NL because of >>8 mantissa returns in range [0:1/256] which is perfect for quantising
float random3(vec3  v) 
{ 
    return floatConstruct(hash(floatBitsToUint(v))); 
}

/* stuff by nomadic lizard */
vec3 quantise(in vec3 fragColor, in vec2 fragCoord, in const FrameData frameData)
{
    return fragColor + random3(vec3(fragCoord, frameData.appTime.x));
}

#endif