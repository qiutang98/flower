#ifndef PHASE_GLSL
#define PHASE_GLSL

#include "Common.glsl"

float getUniformPhase()
{
	return 1.0f / (4.0f * kPI);
}

// https://www.shadertoy.com/view/Mtc3Ds
// rayleigh phase function.
float rayleighPhase(float cosTheta)
{
	const float factor = 3.0f / (16.0f * kPI);
	return factor * (1.0f + cosTheta * cosTheta);
}

// Schlick approximation
float cornetteShanksMiePhaseFunction(float g, float cosTheta)
{
	float k = 3.0 / (8.0 * kPI) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

float hgPhase(float g, float cosTheta)
{
	return cornetteShanksMiePhaseFunction(g, cosTheta);
}

float dualLobPhase(float g0, float g1, float w, float cosTheta)
{
	return mix(hgPhase(g0, cosTheta), hgPhase(g1, cosTheta), w);
}

float beersLaw(float density, float stepLength, float densityScale)
{
	return exp(-density * stepLength * densityScale);
}

float henyeyGreenstein(float sundotrd, float g) 
{
	float gg = g * g;
	return (1. - gg) / pow(1. + gg - 2. * g * sundotrd, 1.5);
}

#endif