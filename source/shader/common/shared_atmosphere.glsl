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

// NOTE: When offset height is big, this value should be more bigger.
// TODO: Compute this by camera height.
const float kPlanetRadiusOffset = 0.001f; // Offset 1 m.

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

// Participating media functions and struct.
// From https://github.com/sebh/UnrealEngineSkyAtmosphere
struct MediumSampleRGB
{
	vec3 scattering;
	vec3 absorption;
	vec3 extinction;

	vec3 scatteringMie;
	vec3 absorptionMie;
	vec3 extinctionMie;

	vec3 scatteringRay;
	vec3 absorptionRay;
	vec3 extinctionRay;

	vec3 scatteringOzo;
	vec3 absorptionOzo;
	vec3 extinctionOzo;

	vec3 albedo;
};

float getAlbedo(float scattering, float extinction)
{
	return scattering / max(0.001, extinction);
}

vec3 getAlbedo(vec3 scattering, vec3 extinction)
{
	return scattering / max(vec3(0.001), extinction);
}

float getViewHeight(vec3 worldPos, in const AtmosphereParameters atmosphere)
{
	// Current default set planet center is (0, 0, 0).
    // And we start from horizontal plane which treat as 0 plane on height.
	return length(worldPos) - atmosphere.bottomRadius;
}

bool moveToTopAtmosphere(inout vec3 worldPos, in const vec3 worldDir, in const float atmosphereTopRadius)
{
	float viewHeight = length(worldPos);
	if (viewHeight > atmosphereTopRadius)
	{
		float tTop = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0), atmosphereTopRadius);
		if (tTop >= 0.0f)
		{
			vec3 upVector = worldPos / viewHeight;
			vec3 upOffset = upVector * -kPlanetRadiusOffset;
			worldPos = worldPos + worldDir * tTop + upOffset;
		}
		else
		{
			// Ray is not intersecting the atmosphere
			return false;
		}
	}
	return true; // ok to start tracing
}

MediumSampleRGB sampleMediumRGB(in vec3 worldPos, in const AtmosphereParameters atmosphere)
{
	const float viewHeight = getViewHeight(worldPos, atmosphere);

    // Get mie and ray density.
	const float densityMie = exp(atmosphere.mieDensityExpScale * viewHeight);
	const float densityRay = exp(atmosphere.rayleighDensityExpScale * viewHeight);

    // Get ozone density.
	const float densityOzo = saturate(viewHeight < atmosphere.absorptionDensity0LayerWidth ?
		atmosphere.absorptionDensity0LinearTerm * viewHeight + atmosphere.absorptionDensity0ConstantTerm :
		atmosphere.absorptionDensity1LinearTerm * viewHeight + atmosphere.absorptionDensity1ConstantTerm);

    // Build medium sample.
	MediumSampleRGB s;

    // Mie term.
	s.scatteringMie = densityMie * atmosphere.mieScattering;
	s.absorptionMie = densityMie * atmosphere.mieAbsorption;
	s.extinctionMie = densityMie * atmosphere.mieExtinction;

    // Ray term.
	s.scatteringRay = densityRay * atmosphere.rayleighScattering;
	s.absorptionRay = vec3(0.0);
	s.extinctionRay = s.scatteringRay + s.absorptionRay;

    // Ozone term.
	s.scatteringOzo = vec3(0.0);
	s.absorptionOzo = densityOzo * atmosphere.absorptionExtinction;
	s.extinctionOzo = s.scatteringOzo + s.absorptionOzo;

    // Composite.
	s.scattering = s.scatteringMie + s.scatteringRay + s.scatteringOzo;
	s.absorption = s.absorptionMie + s.absorptionRay + s.absorptionOzo;
	s.extinction = s.extinctionMie + s.extinctionRay + s.extinctionOzo;
	s.albedo = getAlbedo(s.scattering, s.extinction);
	return s;
}

#endif