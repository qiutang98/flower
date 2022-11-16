#ifndef SAMPLE_GLSL
#define SAMPLE_GLSL

#include "Common.glsl"

// Catmull Rom 9 tap sampler.
// sTex: linear clamp sampler2D.
// uv: sample uv.
// resolution: working rt resolution.
// https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
/**********************************************************************
	MIT License

	Copyright(c) 2019 MJP

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files(the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions :

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
********************************************************************/
vec4 catmullRom9Sample(texture2D sTex, sampler linearClampEdge, vec2 uv, vec2 resolution)
{
    vec2 samplePos = uv * resolution;
    vec2 texPos1   = floor(samplePos - 0.5f) + 0.5f;

    vec2 f = samplePos - texPos1;

    vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    vec2 w3 = f * f * (-0.5f + 0.5f * f);

    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    vec2 texPos0 = texPos1 - 1.0f;
    vec2 texPos3 = texPos1 + 2.0f;

    vec2 texPos12 = texPos1 + offset12;

    texPos0  /= resolution;
    texPos3  /= resolution;
    texPos12 /= resolution;

    vec4 result = vec4(0.0f);

    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos0.x,  texPos0.y),  0) * w0.x  * w0.y;
    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos12.x, texPos0.y),  0) * w12.x * w0.y;
    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos3.x,  texPos0.y),  0) * w3.x  * w0.y;

    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos0.x,  texPos12.y), 0) * w0.x  * w12.y;
    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos12.x, texPos12.y), 0) * w12.x * w12.y;
    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos3.x,  texPos12.y), 0) * w3.x  * w12.y;

    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos0.x,  texPos3.y),  0) * w0.x  * w3.y;
    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos12.x, texPos3.y),  0) * w12.x * w3.y;
    result += textureLod(sampler2D(sTex, linearClampEdge), vec2(texPos3.x,  texPos3.y),  0) * w3.x  * w3.y;

    return max(result, vec4(0.0f));
}


#endif