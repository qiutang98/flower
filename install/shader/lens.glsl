#ifndef LENS_GLSL
#define LENS_GLSL



//https://www.shadertoy.com/view/MdGSWy
#define ORB_FLARE_COUNT	   6.0
#define DISTORTION_BARREL  0.5
#define GLARE_BRIGHTNESS   0.1
#define FLARE_BRIGHTNESS   1.0
#define HOLO_RADIUS_INNER  0.12
#define HOLO_RADIUS_OUTER  0.50

vec2 getDistOffset(vec2 uv, vec2 pxoffset)
{
    vec2 tocenter = uv.xy;
    vec3 prep = normalize(vec3(tocenter.y, -tocenter.x, 0.0));
    float angle = length(tocenter.xy) * 2.221 * DISTORTION_BARREL;
    vec3 oldoffset = vec3(pxoffset, 0.0);
    vec3 rotated = oldoffset * cos(angle) + cross(prep, oldoffset) * sin(angle) + prep * dot(prep, oldoffset) * (1.0 - cos(angle));
    return rotated.xy;
}

vec3 flare(vec2 uv, vec2 pos, float dist, float chromaOffset, float size)
{
    pos = getDistOffset(uv, pos);
    float r = max(0.01 - pow(length(uv + (dist - chromaOffset) * pos), 2.4) *( 1.0 / (size * 2.0)), 0.0) * 0.85;
    float g = max(0.01 - pow(length(uv +  dist                 * pos), 2.4) * (1.0 / (size * 2.0)), 0.0) * 1.0;
    float b = max(0.01 - pow(length(uv + (dist + chromaOffset) * pos), 2.4) * (1.0 / (size * 2.0)), 0.0) * 1.5;
    return vec3(r, g, b);
}

vec3 orb(vec2 uv, vec2 pos, float dist, float size)
{
    vec3 c = vec3(0.0);

    for (float i = 0.0; i < ORB_FLARE_COUNT; i++)
    {
        float j = i + 1;
        float offset = j / (j + 0.1);
        float colOffset = j / ORB_FLARE_COUNT * 0.5;

        float ss = size / (j + 1.0);

        c += flare(uv, pos, dist + offset, ss * 2.0, ss) * vec3(1.0 - colOffset, 1.0, 0.5 + colOffset) * j;
    }

    c += flare(uv, pos, dist + 0.8, 0.05, 3.0 * size) * 0.5;
    return c;
}

vec3 ring(vec2 uv, vec2 pos, float dist, float chromaOffset, float blur)
{
    vec2 uvd = uv * length(uv);
    float r = max(1.0 / (1.0 + 250.0 * pow(length(uvd + (dist - chromaOffset) * pos), blur)), 0.0) * 0.8;
    float g = max(1.0 / (1.0 + 250.0 * pow(length(uvd +  dist                 * pos), blur)), 0.0) * 1.0;
    float b = max(1.0 / (1.0 + 250.0 * pow(length(uvd + (dist + chromaOffset) * pos), blur)), 0.0) * 1.5;
    return vec3(r, g, b);
}

float safety_sin( in float x ) { return sin( mod( x, kPI ) ); }

vec2 rotate( in vec2 p, float r )
{
	float s = sin( r );
	float c = cos( r );
	return mat2( c, -s, s, c ) * p;
}

float rand( in vec2 p ) { return fract( safety_sin( dot(p, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ); }
float rand( in vec2 p, in float t ) { return fract( safety_sin( dot(p, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 + t ); }

float noise( in vec2 x )
{
	vec2 i = floor( x );
	vec2 f = fract( x );
	vec4 h;
	// Smooth Interpolation
	f = f * f * ( f * -2.0 + 3.0 );
	// Four corners in 2D of a tile
	h.x = rand( i + vec2( 0., 0. ) );
	h.y = rand( i + vec2( 1., 0. ) );
	h.z = rand( i + vec2( 0., 1. ) );
	h.w = rand( i + vec2( 1., 1. ) );
	// Mix 4 coorners percentages
	return mix( mix( h.x, h.y, f.x ), mix( h.z, h.w, f.x ), f.y );
}

float noise( in vec2 x, in float t )
{
	vec2 i = floor( x );
	vec2 f = fract( x );
	vec4 h;
	// Smooth Interpolation
	f = f * f * ( f * -2.0 + 3.0 );
	// Four corners in 2D of a tile
	h.x = rand( i + vec2( 0., 0. ), t );
	h.y = rand( i + vec2( 1., 0. ), t );
	h.z = rand( i + vec2( 0., 1. ), t );
	h.w = rand( i + vec2( 1., 1. ), t );
	// Mix 4 coorners percentages
	return mix( mix( h.x, h.y, f.x ), mix( h.z, h.w, f.x ), f.y );
}

float sdCircle( in vec2 p, in float r )
{
    return length(p) - r;
}

float sdHexagon( in vec2 p, in float r )
{
    const vec3 k = vec3(-0.866025404,0.5,0.577350269);
    p = abs(p);
    p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
    p -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
    return length(p)*sign(p.y);
}


float halo( in vec2 p, in vec2 center, in float r, in float offset )
{
    float c0 = 0.5;
    float c1 = 32.0;
    float c2 = 12.0;
    float c3 = 2.0;
    float c4 = 0.7;// 1.0 == blur
    float c5 = 4.0;// 0.0 == like glare
    float c6 = 5.2;
    float c7 = 0.5;
    float c8 = 1.5;
    float c9 = 0.25;

    float t  = dot( center, vec2( c9 ) );
    float l  = length( p );
    float l1 = abs( l - r );
    float l2 = pow( l, c0 );
    float n0 = noise( vec2( atan(  p.y,  p.x ) * c1, l2 ) * c2, t );
    float n1 = noise( vec2( atan( -p.y, -p.x ) * c1, l2 ) * c2, t );
    float n  = mix( pow( max( n0, n1 ), c3 ), 1.0, c4 ) * pow( saturate( 1.0 - l1 * c5 ), c6 );
    return n * 0.2 * saturate( pow(
        1.0 - saturate( pow( length( center ), c7 ) ),
        ( length( p - center ) / r ) * c8
    ) );
}

vec3 ghost3( in vec2 p, in vec2 center, float focus, in float r, in float offset )
{
    float shape_factor = 0.4;   // 0.0 == Circular aperture(like digital camera)
                                // 1.0 == Six blades aperture(like old camera)
    p -= center * offset;
    vec2 p2 = rotate( p, 0.25 );
    float d0 = mix( sdCircle( p2 * 0.85, r ), sdHexagon( p2 * 0.85, r ), shape_factor );
    float d1 = mix( sdCircle( p2,        r ), sdHexagon( p2,        r ), shape_factor );
    float d2 = mix( sdCircle( p2 * 1.15, r ), sdHexagon( p2 * 1.15, r ), shape_factor );
    return mix(
        vec3(
            halo( p * 1.05, center, r, offset ),
            halo( p * 1.0,  center, r, offset ),
            halo( p * 0.95, center, r, offset )
        ) * vec3( 2.0, 1.5, 1.0 ),
        pow( saturate( 1.0 - vec3( d0, d1, d2 ) ), vec3( 200. ) ),
        focus
    );
}

float getSun(vec2 uv){
    return length(uv) < 0.009 ? 1.0 : 0.0;
}


vec3 anflares2(vec2 uv, float intensity, float stretch, float brightness)
{
    vec3 r = vec3(0.0);

    {
        vec2 uv_1 = uv;
        uv_1.y *= 1.0/ (intensity * stretch);
        uv_1.x *= 0.6;
        r += vec3(smoothstep(0.005, 0.0, length(uv_1)))*brightness;
    }

    {
        vec2 uv_2 = uv;
        uv_2.x *= 1.0/ (intensity * stretch);
        uv_2.y *= 0.6;
        r += vec3(smoothstep(0.009, 0.0, length(uv_2)))*brightness;
    }

    return r;
}

vec3 lensFlare(vec2 texcoord, vec2 sunCoord, out vec3 ghostColor)
{
    vec2 coord = texcoord - 0.5;
    vec2 sunPos = sunCoord - 0.5;

    coord.x *= frameData.camInfo.y;
    sunPos.x *= frameData.camInfo.y;

    vec2 v = coord - sunPos;

    float dist = length(v);

    const float invTanHalfFovy = 1.0f / tan(frameData.camInfo.x * 0.5f);
    float fovFactor = max(invTanHalfFovy, 2.0);
    float gDist = dist * 25.0 / fovFactor;
    float phase = atan2(v) + 0.131;

    float gl = 2.0 - saturate(gDist) + sin(phase * 12.0) * saturate(gDist * 2.5 - 0.2);
    gl = gl * gl;
    gDist = gDist * gDist;
    gl *= 3e-4 / (gDist * gDist);



    float size = 0.5 * fovFactor;
    vec3 fl = vec3(0.0);

    fl += orb(coord, sunPos, 0.0, size * 0.02) * 0.15;
    fl += ring(coord, sunPos,  1.0, 0.02, 1.4) * 0.02;
    fl += ring(coord, sunPos, -1.0, 0.02, 1.4) * 0.01;

    fl += flare(coord, sunPos, -2.00, 0.05, size * 0.05) * 0.5;
    fl += flare(coord, sunPos, -0.90, 0.02, size * 0.03) * 0.25;
    fl += flare(coord, sunPos, -0.70, 0.01, size * 0.06) * 0.5;
    fl += flare(coord, sunPos, -0.55, 0.02, size * 0.02) * 0.25;
    fl += flare(coord, sunPos, -0.35, 0.02, size * 0.04) * 1.0;
    fl += flare(coord, sunPos, -0.25, 0.01, size * 0.15) * vec3(0.3, 0.4, 0.38);
    fl += flare(coord, sunPos, -0.25, 0.02, size * 0.08) * 0.3;
    fl += flare(coord, sunPos,  0.05, 0.01, size * 0.03) * 0.1;
    fl += flare(coord, sunPos,  0.30, 0.02, size * 0.20) * vec3(0.3, 0.25, 0.15);
    fl += flare(coord, sunPos,  1.20, 0.03, size * 0.10) * 0.5;

    fl += 0.3 * ghost3(coord, sunPos, 0.0,  HOLO_RADIUS_INNER,   1.0  );// halo
    fl += 0.1 * ghost3(coord, sunPos, 0.0,  HOLO_RADIUS_OUTER,   0.85 / 1.5 );
    ghostColor = frameData.sunLightInfo.color * frameData.sunLightInfo.intensity * vec3(gl * GLARE_BRIGHTNESS);
    vec3 lf = frameData.sunLightInfo.color * frameData.sunLightInfo.intensity * fl * FLARE_BRIGHTNESS;

#if 0
    vec3 anflare2 = pow(anflares2(coord- sunPos, 400.0, 0.5, .55), vec3(4.0));
    anflare2 += smoothstep(0.0025, 1.0, anflare2) * 10.0;
    anflare2 *= smoothstep(0.0, 1.0, anflare2);
    ghostColor += anflare2;

    vec3 anflare1 = pow(anflares2(coord- sunPos, 400.0, 0.5, .55), vec3(3.0));
    anflare1 += smoothstep(0.0025, 1.0, anflare1) * 10.0;
    anflare1 *= smoothstep(0.5, 1.0, anflare1);
    ghostColor += anflare1 * vec3(1.0, 0.8, 0.7);
#endif

    return lf;
}


#endif 