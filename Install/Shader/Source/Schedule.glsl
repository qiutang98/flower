#ifndef SCHEDULE_GLSL
#define SCHEDULE_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

// Compute shader schedule function.

// Quad schedule style, from AMD FSR.
// Input-> [0, 63]
//
// Output:
//  00 01 08 09 10 11 18 19
//  02 03 0a 0b 12 13 1a 1b
//  04 05 0c 0d 14 15 1c 1d
//  06 07 0e 0f 16 17 1e 1f
//  20 21 28 29 30 31 38 39
//  22 23 2a 2b 32 33 3a 3b
//  24 25 2c 2d 34 35 3c 3d
//  26 27 2e 2f 36 37 3e 3f
uvec2 remap8x8(uint lane) // gl_LocalInvocationIndex in 8x8 threadgroup.
{
    return uvec2(
        (((lane >> 2) & 0x0007) & 0xFFFE) | lane & 0x0001,
        ((lane >> 1) & 0x0003) | (((lane >> 3) & 0x0007) & 0xFFFC)
    );
}

#endif