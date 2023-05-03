#ifndef SHARED_ATMOSPHERE_GLSL
#define SHARED_ATMOSPHERE_GLSL

#include "shared_struct.glsl"
#include "shared_functions.glsl"

// All distance units in kilometers

struct AtmosphereParameters
{
    float atmospherePreExposure;

	// Radius of the planet (center to ground)
	float bottomRadius;

	// Maximum considered atmosphere height (center to atmosphere top)
	float topRadius;

	// Rayleigh scattering exponential distribution scale in the atmosphere
	float rayleighDensityExpScale;

	// Rayleigh scattering coefficients
	vec3 rayleighScattering;

	// Mie scattering exponential distribution scale in the atmosphere
	float mieDensityExpScale;

	// Mie scattering coefficients
	vec3 mieScattering;

	// Mie extinction coefficients
	vec3 mieExtinction;

	// Mie absorption coefficients
	vec3 mieAbsorption;

	// Mie phase function excentricity
	float miePhaseG;

	// Another medium type in the atmosphere
	float absorptionDensity0LayerWidth;
	float absorptionDensity0ConstantTerm;
	float absorptionDensity0LinearTerm;
	float absorptionDensity1ConstantTerm;
	float absorptionDensity1LinearTerm;

	// This other medium only absorb light, e.g. useful to represent ozone in the earth atmosphere
	vec3 absorptionExtinction;

	// The albedo of the ground.
	vec3 groundAlbedo;

	float multipleScatteringFactor; 

	uint viewRayMarchMinSPP;
	uint viewRayMarchMaxSPP;

	float cloudAreaStartHeight; // km
    float cloudAreaThickness;
    mat4 cloudShadowViewProj;
    mat4 cloudShadowViewProjInverse;
};

// Build atmosphere parameters from frame data.
AtmosphereParameters getAtmosphereParameters(in const PerFrameData frameData)
{
	AtmosphereParameters parameters;

    const AtmosphereConfig config = frameData.sky.atmosphereConfig;

	parameters.absorptionExtinction = config.absorptionColor * config.absorptionLength;

    // Copy parameters.
    parameters.groundAlbedo             = config.groundAlbedo;
	parameters.bottomRadius             = config.bottomRadius;
	parameters.topRadius                = config.topRadius;
    parameters.viewRayMarchMinSPP       = config.viewRayMarchMinSPP;
	parameters.viewRayMarchMaxSPP       = config.viewRayMarchMaxSPP;
	parameters.miePhaseG                = config.miePhaseFunctionG;
    parameters.atmospherePreExposure    = config.atmospherePreExposure;
	parameters.multipleScatteringFactor = config.multipleScatteringFactor;

	// Traslation from Bruneton2017 parameterisation.
	parameters.rayleighDensityExpScale        = config.rayleighDensity[1].w;
	parameters.mieDensityExpScale             = config.mieDensity[1].w;
	parameters.absorptionDensity0LayerWidth   = config.absorptionDensity[0].x;
	parameters.absorptionDensity0ConstantTerm = config.absorptionDensity[1].x;
	parameters.absorptionDensity0LinearTerm   = config.absorptionDensity[0].w;
	parameters.absorptionDensity1ConstantTerm = config.absorptionDensity[2].y;
	parameters.absorptionDensity1LinearTerm   = config.absorptionDensity[2].x;

	parameters.rayleighScattering = config.rayleighScatteringColor * config.rayleighScatterLength;
    parameters.mieAbsorption      = config.mieAbsorption;
	parameters.mieScattering      = config.mieScatteringColor * config.mieScatteringLength;
	parameters.mieExtinction      = parameters.mieScattering + config.mieAbsColor * config.mieAbsLength;

	parameters.cloudAreaStartHeight       = config.cloudAreaStartHeight;
    parameters.cloudAreaThickness         = config.cloudAreaThickness;
    parameters.cloudShadowViewProj        = config.cloudSpaceViewProject;
    parameters.cloudShadowViewProjInverse = config.cloudSpaceViewProjectInverse;

	return parameters;
}

// https://github.com/sebh/UnrealEngineSkyAtmosphere
// Transmittance LUT function parameterisation from Bruneton 2017 https://github.com/ebruneton/precomputed_atmospheric_scattering
// Detail also in video https://www.youtube.com/watch?v=y-oBGzDCZKI at 08:35.
void lutTransmittanceParamsToUv(
    in const AtmosphereParameters atmosphere, 
    in float viewHeight, // [bottomRAdius, topRadius]
    in float viewZenithCosAngle, // [-1,1]
    out vec2 uv) // [0,1]
{
	float H = sqrt(max(0.0f, atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius));
	float rho = sqrt(max(0.0f, viewHeight * viewHeight - atmosphere.bottomRadius * atmosphere.bottomRadius));

	uv.y = rho / H;

	// Distance to atmosphere boundary
	float discriminant = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0) + atmosphere.topRadius * atmosphere.topRadius;
	float d = max(0.0, (-viewHeight * viewZenithCosAngle + sqrt(discriminant))); 

	float dMin = atmosphere.topRadius - viewHeight;
	float dMax = rho + H;

	uv.x = (d - dMin) / (dMax - dMin);
}

void uvToLutTransmittanceParams(
    in const AtmosphereParameters atmosphere, 
    out float viewHeight, // [bottomRAdius, topRadius]
    out float viewZenithCosAngle, // [-1,1]
    in vec2 uv) // [0,1]
{ 
	float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
	float rho = H * uv.y;
	viewHeight = sqrt(rho * rho + atmosphere.bottomRadius * atmosphere.bottomRadius);

	float dMin = atmosphere.topRadius - viewHeight;
	float dMax = rho + H;

	// Distance to atmosphere boundary
	float d = dMin + uv.x * (dMax - dMin);

	viewZenithCosAngle = (d == 0.0) ? 1.0f : (H * H - rho * rho - d * d) / (2.0 * viewHeight * d);
	viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0, 1.0);
}

float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

void skyViewLutParamsToUv(
	in const AtmosphereParameters atmosphere, 
	in bool  bIntersectGround, 
	in float viewZenithCosAngle, 
	in float lightViewCosAngle, 
	in float viewHeight, 
    in vec2 lutSize,
	out vec2 uv)
{
	float vHorizon = sqrt(viewHeight * viewHeight - atmosphere.bottomRadius * atmosphere.bottomRadius);

	// Ground to horizon cos.
	float cosBeta = vHorizon / viewHeight;		

	float beta = acos(cosBeta);
	float zenithHorizonAngle = kPI - beta;

	if (!bIntersectGround)
	{
		float coord = acos(viewZenithCosAngle) / zenithHorizonAngle;
		coord = 1.0 - coord;
		coord = sqrt(coord); // Non-linear sky view lut.

		coord = 1.0 - coord;
		uv.y = coord * 0.5f;
	}
	else
	{
		float coord = (acos(viewZenithCosAngle) - zenithHorizonAngle) / beta;
		coord = sqrt(coord); // Non-linear sky view lut.

		uv.y = coord * 0.5f + 0.5f;
	}

	// UV x remap.
	{
		float coord = -lightViewCosAngle * 0.5f + 0.5f;
		coord = sqrt(coord);
		uv.x = coord;
	}

	// Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible)
	uv = vec2(fromUnitToSubUvs(uv.x, lutSize.x), fromUnitToSubUvs(uv.y, lutSize.y));
}

// Air perspective.
const float kAirPerspectiveKmPerSlice = 4.0f; // total 32 * 4 = 128 km.
float aerialPerspectiveDepthToSlice(float depth) { return depth * (1.0f / kAirPerspectiveKmPerSlice); }
float aerialPerspectiveSliceToDepth(float slice) { return slice * kAirPerspectiveKmPerSlice; }

// Camera unit to atmosphere unit convert. meter -> kilometers.
vec3 convertToAtmosphereUnit(vec3 o, in const PerFrameData frame)
{
	const float cameraOffset = 0.5f;
	return o * 0.001f + vec3(0.0, cameraOffset, 0.0);
}  

// atmosphere unit to camera unit convert. kilometers -> meter.
vec3 convertToCameraUnit(vec3 o, in const PerFrameData frame)
{
	const float cameraOffset = 0.5f;
	return (o - vec3(0.0, cameraOffset, 0.0)) * 1000.0f;
}  

#endif