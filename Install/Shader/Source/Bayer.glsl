#ifndef BAYER_GLSL
#define BAYER_GLSL

// 4x4 bayer filter, use for cloud reconstruction.
const int bayerFilter4x4[16] = int[]
(
	0,   8,  2, 10,
	12,  4, 14,  6,
	3,  11,  1,  9,
	15,  7, 13,  5
);

// ivec2 offset = ivec2(bayerFilter4x4[frameId] % 4, bayerFilter4x4[frameId] / 4);
// 
#endif