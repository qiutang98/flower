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

float bayerDither(float grayscale, ivec2 pixelCoord)
{    
    int bayerMatrix16[16] = int[16](
		0, 8, 2, 10, 
		12, 4, 14, 6, 
		3, 11, 1, 9, 
		15, 7, 13, 5
	);

    int pixelIndex16 = (pixelCoord.x % 4) + (pixelCoord.y % 4) * 4;
    return grayscale > (float(bayerMatrix16[pixelIndex16]) + 0.5) / 16.0 ? 1.0 : 0.0;
}

#endif