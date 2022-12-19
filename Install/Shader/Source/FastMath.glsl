#ifndef FAST_MATH_GLSL
#define FAST_MATH_GLSL

/*
** Physical based render code, develop by engineer: qiutanguu.
*/

#include "Common.glsl"

// Relative error : ~3.4% over full
// Precise format : ~small float
// 2 ALU
float rsqrtFast(float x)
{
	int i = floatBitsToInt(x);
	i = 0x5f3759df - (i >> 1);
	return intBitsToFloat (i);
}

// max absolute error 9.0x10^-3
// Eberly's polynomial degree 1 - respect bounds
// 4 VGPR, 12 FR (8 FR, 1 QR), 1 scalar
// input [-1, 1] and output [0, PI]
float acosFast(float inX) 
{
    float x = abs(inX);
    float res = -0.156583f * x + (0.5 * kPI);
    res *= sqrt(1.0f - x);
    return (inX >= 0) ? res : kPI - res;
}

// Approximates acos(x) with a max absolute error of 9.0x10^-3.
// Input [0, 1]
float acosFastPositive(float x) 
{
    float p = -0.1565827f * x + 1.570796f;
    return p * sqrt(1.0 - x);
}


#endif