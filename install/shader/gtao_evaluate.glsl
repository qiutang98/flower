#version 460
#extension GL_GOOGLE_include_directive : enable

#include "gtao_common.glsl"

const float kRots[6] = { 60.0f, 300.0f, 180.0f, 240.0f, 120.0f, 0.0f };
const float kOffsets[4] = { 0.1f, 0.6f, 0.35f, 0.85f };

float interleavedGradientNoise(vec2 iPos)
{
	return fract(52.9829189f * fract((iPos.x * 0.06711056) + (iPos.y * 0.00583715)));
}

vec3 getRandomVector(uvec2 iPos)
{
	iPos.y = 16384 - iPos.y;

    float temporalAngle = kRots[frameData.frameIndex.x % 6] * (kPI / 360.0f);
	float temporalCos = cos(temporalAngle);
	float temporalSin = sin(temporalAngle);

	float gradientNoise = interleavedGradientNoise(vec2(iPos));

    vec2 randomTexVec = vec2(0);
	randomTexVec.x = cos(gradientNoise * kPI);
	randomTexVec.y = sin(gradientNoise * kPI);

	float scaleOffset = (1.0 / 4.0) * ((iPos.y - iPos.x) & 3);

    vec3 randomVec = vec3(0);
	randomVec.x = dot(randomTexVec.xy, vec2(temporalCos, -temporalSin));
    randomVec.y = dot(randomTexVec.xy, vec2(temporalSin,  temporalCos));
	randomVec.z = fract(scaleOffset + kOffsets[(frameData.frameIndex.x / 6) % 4] * 0.25);

	return randomVec;
}


vec3 getViewPos(vec2 uv)
{
    float deviceZ = textureLod(sampler2D(inHiz, pointClampEdgeSampler), uv, 0.0).r;
    return getViewPos(uv, deviceZ, frameData);
}

vec3 getViewPos(vec2 uv, float hzbLevel)
{
    const float kBasicHZBOffset = 1.0;
    float deviceZ = textureLod(sampler2D(inHiz, pointClampEdgeSampler), uv, hzbLevel + kBasicHZBOffset).r;
    return getViewPos(uv, deviceZ, frameData);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec2 gtaoSize = imageSize(GTAOImage);

    uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 workPos = ivec2(dispatchId);

    if(workPos.x >= gtaoSize.x || workPos.y >= gtaoSize.y)
    {
        return;
    }

    const vec2 texelSize = 1.0f / vec2(gtaoSize);
    const vec2 uv = (vec2(workPos) + vec2(0.5f)) * texelSize;
    
    // Pre-return if no valid shading model id.
    float sceneZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;
    if(sceneZ <= 0.0f)
    {
        imageStore(GTAOImage, workPos, vec4(1.0));
        return;
    }

    const uint kGTAOSliceNum = GTAOPush.sliceNum;
    const uint kGTAOStepNum = GTAOPush.stepNum;
    
    const float kGTAORadius = GTAOPush.radius;
    const float kGTAOThickness = GTAOPush.thickness;

    const float kGTAOMaxPixelScreenRadius = 256.0f; // Max offset 256 pixel.

    const vec3 viewSpacePos = getViewPos(uv);
    const vec3 worldNormal = unpackWorldNormal(texture(sampler2D(inGbufferB, pointClampEdgeSampler), uv).rgb);
    const vec3 viewSpaceNormal = normalize((frameData.camView * vec4(worldNormal, 0.0)).rgb);
    const vec3 viewDir = normalize(-viewSpacePos); // camera pos is 0.

    // Compute screen space step lenght, using simple similar triangles math.
    // Same tech detail from HBAO.
    const float invTanHalfFovy = 1.0f / tan(frameData.camInfo.x * 0.5f);
    const float fovScale = gtaoSize.y * invTanHalfFovy;
    const float zRadiusAdj = fovScale * kGTAORadius;
    float pixelRadius = zRadiusAdj / (-viewSpacePos.z);

    // Clamp range to a valid range.
    pixelRadius = min(pixelRadius, float(kGTAOMaxPixelScreenRadius)); // Max pixel search radius is 256 pixel, avoid large search step when view is close.
    pixelRadius = max(pixelRadius, float(kGTAOStepNum)); // At least step one pixel.
    
    const float stepRadius = pixelRadius / (float(kGTAOStepNum) + 1.0f); // Divide by steps + 1 so that the farthest samples are not fully attenuated
    const float attenFactor = 2.0 / max(GTAOPush.kFalloffRadius * GTAOPush.kFalloffRadius, 0.001);

    float occlusion = 0.0;
    vec3 bounceColor = vec3(0.0);
    {
        vec3 randomAndOffset = getRandomVector(workPos);

        vec2 sliceDir = randomAndOffset.xy;
        float offset = randomAndOffset.z;

        // Unreal magic code, rotate by step sin/cos.
        float stepAngle = kPI / float(kGTAOSliceNum);
        float sinDeltaAngle	= sin(stepAngle);
	    float cosDeltaAngle	= cos(stepAngle);

        for(uint i = 0; i < kGTAOSliceNum; i ++) 
        {
            vec2 sliceDirection = sliceDir * texelSize; // slice direction in texel uint.

            // Horizontal search best angle for this direction.
            vec2 bestAng = vec2(-1.0, -1.0);
            for(int j = 1; j < kGTAOStepNum + 1; j++)
            {
                float fi = float(j);
                
                // stepRadius > 1.0, and jitterStep is range [0, 1]
                vec2 uvOffset = sliceDirection * max(stepRadius * (fi + offset), fi + 1.0);

                // build two conversely sample direction.
                uvOffset.y *= -1.0;
                vec4 uv2 = uv.xyxy + vec4(uvOffset.xy, -uvOffset.xy);

                // Use hzb to imporve L2 cache hit rate.
                float mipLevel = 0.0;
                if(j == 2)
                {
                    mipLevel ++;
                }
                if(j > 3)
                {
                    mipLevel += 2;
                }

                vec3 h1 = getViewPos(uv2.xy, mipLevel) - viewSpacePos; // H1 is positive direction.
                vec3 h2 = getViewPos(uv2.zw, mipLevel) - viewSpacePos; // H2 is negative direction.

                
                float h1LenSq = dot(h1, h1);
                float falloffH1 = saturate(h1LenSq * attenFactor);
                if(falloffH1 < 1.0)
                {
                    float angH1 = dot(h1, viewDir) * rsqrtFast(h1LenSq + 0.0001);
                    angH1 = mix(angH1, bestAng.x, falloffH1);
                    bestAng.x = (angH1 > bestAng.x) ? angH1 : mix(angH1, bestAng.x, kGTAOThickness);
                }

                float h2LenSq = dot(h2, h2);
                float falloffH2 = saturate(h2LenSq * attenFactor);
                if(falloffH2 < 1.0)
                {
                    float angH2 = dot(h2, viewDir) * rsqrtFast(h2LenSq + 0.0001);
                    angH2 = mix(angH2, bestAng.y, falloffH2);
                    bestAng.y = (angH2 > bestAng.y) ? angH2 : mix(angH2, bestAng.y, kGTAOThickness);
                }
            }
            bestAng.x = acosFast(clamp(bestAng.x, -1.0, 1.0));
            bestAng.y = acosFast(clamp(bestAng.y, -1.0, 1.0));
            
            // Compute inner integral.
            {
                // Given the angles found in the search plane we need to project the View Space Normal onto the plane defined by the search axis and the View Direction and perform the inner integrate
                vec3 planeNormal = normalize(cross(vec3(sliceDir, 0.0), viewDir));
                vec3 perp = cross(viewDir, planeNormal);
                vec3 projNormal = viewSpaceNormal - planeNormal * dot(viewSpaceNormal, planeNormal);

                float lenProjNormal = length(projNormal) + 0.000001f;
                float recipMag = 1.0 / lenProjNormal;

                float cosAng = dot(projNormal, perp) * recipMag;	
	            float gamma = acosFast(cosAng) - 0.5 * kPI;				
	            float cosGamma = dot(projNormal, viewDir) * recipMag;
	            float sinGamma = cosAng * (-2.0);					

	            // clamp to normal hemisphere 
	            bestAng.x = gamma + max(-bestAng.x - gamma, -(0.5 * kPI));
	            bestAng.y = gamma + min( bestAng.y - gamma,  (0.5 * kPI));

                // See Activision GTAO paper: https://www.activision.com/cdn/research/s2016_pbs_activision_occlusion.pptx
                // Integrate arcCos weight.
	            float ao = ((lenProjNormal) * 0.25 * 
					((bestAng.x * sinGamma + cosGamma - cos(2.0 * bestAng.x - gamma)) +
				  	 (bestAng.y * sinGamma + cosGamma - cos(2.0 * bestAng.y - gamma))));

                occlusion += ao;
            }

            // Unreal magic code, rotate by sin/cos step.
            // Rotate for the next angle
            vec2 tempScreenDir = sliceDir.xy;
            sliceDir.x = (tempScreenDir.x * cosDeltaAngle) + (tempScreenDir.y * -sinDeltaAngle);
            sliceDir.y = (tempScreenDir.x * sinDeltaAngle) + (tempScreenDir.y *  cosDeltaAngle);
            offset = fract(offset + 0.617);
        }

        occlusion = occlusion / float(kGTAOSliceNum);
        occlusion *= 2.0 / kPI;
    }

    imageStore(GTAOImage, workPos, vec4(occlusion, 1.0f, 1.0f, 1.0f));
}
