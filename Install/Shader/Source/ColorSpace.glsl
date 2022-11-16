// srgb primaries
// Also same with Rec.709
const mat3 sRGB_2_XYZ = mat3(
    0.4124564, 0.2126729, 0.0193339,
    0.3575761, 0.7151522, 0.1191920,
    0.1804375, 0.0721750, 0.9503041
);
const mat3 XYZ_2_sRGB = mat3(
     3.2404542,-0.9692660, 0.0556434,
    -1.5371385, 1.8760108,-0.2040259,
    -0.4985314, 0.0415560, 1.0572252
);

// REC 2020 primaries
const mat3 XYZ_2_Rec2020 =  mat3(
	 1.7166084, -0.6666829, 0.0176422,
	-0.3556621,  1.6164776, -0.0427763,
	-0.2533601,  0.0157685, 0.94222867	
);

const mat3 Rec2020_2_XYZ = mat3(
	0.6369736, 0.2627066, 0.0000000,
	0.1446172, 0.6779996, 0.0280728,
	0.1688585, 0.0592938, 1.0608437
);


const mat3 sRGB_2_Rec2020 = XYZ_2_Rec2020 * sRGB_2_XYZ;
const mat3 Rec2020_2_sRGB = XYZ_2_sRGB * Rec2020_2_XYZ;

vec3 inputColorPrepare(vec3 src)
{
    return sRGB_2_Rec2020 * src;
}