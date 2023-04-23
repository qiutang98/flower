#version 460
#extension GL_GOOGLE_include_directive : enable

#include "gtao_common.glsl"

vec3 reprojectPos(vec2 uv, float depth)
{
    vec3 worldPos = getWorldPos(uv, depth, frameData);
    vec4 prevClip = frameData.camViewProjPrev * vec4(worldPos, 1.0);

    float prevZContruct = prevClip.z / prevClip.w;

    vec2 velocity = texture(sampler2D(inGbufferV, pointClampBorder0000Sampler), uv).rg;
    if(dot(velocity, velocity) > 0.0)
    {
        return vec3(uv + velocity, prevZContruct);
    }
    else
    {
        // When out of bounds or lost velocity, reconstruct from prev view matrix.
        vec2 prevScreen = prevClip.xy / prevClip.w;
        vec2 prevUV = prevScreen * 0.5 + 0.5;
        prevUV.y = 1.0 - prevUV.y;

        return vec3(prevUV, prevZContruct);
    }
}

float readHistoryClamp(vec2 uv, float minAO, float maxAO)
{
	float bilinearWeights[4];

    vec2 historyTextureSize = vec2(textureSize(inGTAOTemporalHistory, 0));
    vec2 historyTexturePixelSize = 1.0 / historyTextureSize;

	vec2 pixUV = (uv * historyTextureSize) - 0.5;
	vec2 floorUV = floor(pixUV);
	vec2 fracUV = (pixUV - floorUV); 
	uv = (floorUV * historyTexturePixelSize) + (historyTexturePixelSize * 0.5);

	bilinearWeights[0] = (1.0 -	fracUV.x) * ( 1.0 -	fracUV.y);
	bilinearWeights[1] = (		fracUV.x) * ( 1.0 -	fracUV.y);
	bilinearWeights[2] = (1.0 -	fracUV.x) * (       fracUV.y);
	bilinearWeights[3] = (		fracUV.x) * (       fracUV.y);

	// Read the 4 previous depths and History
	float historyAO[4];

	vec2 dUV = historyTexturePixelSize;
	historyAO[0] = texture(sampler2D(inGTAOTemporalHistory, pointClampBorder1111Sampler), uv + vec2(    0,     0)).r;
	historyAO[1] = texture(sampler2D(inGTAOTemporalHistory, pointClampBorder1111Sampler), uv + vec2(dUV.x,     0)).r;
	historyAO[2] = texture(sampler2D(inGTAOTemporalHistory, pointClampBorder1111Sampler), uv + vec2(    0, dUV.y)).r;
	historyAO[3] = texture(sampler2D(inGTAOTemporalHistory, pointClampBorder1111Sampler), uv + vec2(dUV.x, dUV.y)).r;

	float visHistory = 0;
	for(int i = 0; i < 4; i++)
	{
		historyAO[i] = clamp(historyAO[i], minAO, maxAO);
		visHistory += bilinearWeights[i] * historyAO[i];
	}

	return visHistory;
}

float compareVeloc(vec2 v1, vec2 v2)
{
	vec2 v12 = v1 - v2;
	return 1.0 - saturate(abs(v12.x + v12.y) * 100);
}

void temporalFilter(vec2 uv, inout float outAO)
{
	float blendWeight = 0.1; // 1.0 / 6.0;

	// Latest AO value
	float newAO	= texture(sampler2D(inGTAOFilterImage, linearClampEdgeSampler), uv).r;

	// Current depth of the rendered Scene
	float currDepthDeviceZ = texture(sampler2D(inDepth, pointClampEdgeSampler), uv).r;

	// Previous uv value
	vec3 prevUVDepth = reprojectPos(uv, currDepthDeviceZ);
	vec2 prevUV	= prevUVDepth.xy;

	vec2 pixVelocity = uv - prevUV;
	float velocityMag = saturate(length(pixVelocity) * 100.0);
	
	// Compare velocities 
	vec2 destVeloc = vec2(0);
	{
		float destDeviceZ = texture(sampler2D(inPrevDepth, pointClampEdgeSampler), prevUVDepth.xy).r;
		vec3 reproj	= reprojectPos(prevUVDepth.xy,  destDeviceZ); 
		destVeloc = prevUVDepth.xy - reproj.xy;
	}

	float velocCompare = compareVeloc(pixVelocity, destVeloc);

	// Get an acceptable range of values we care about from the current AO
	float rangeVal	= mix(0.1, 0.00, velocityMag);
	float minAO = saturate(newAO - rangeVal);
	float maxAO = saturate(newAO + rangeVal);

	// Simple history value
	float historyPrevUV	= readHistoryClamp(prevUV, minAO, maxAO);
	float historyThisUV	= texture(sampler2D(inGTAOTemporalHistory, pointClampBorder1111Sampler), uv).r;
	historyThisUV = clamp(historyThisUV, minAO, maxAO);

	float historyAO = mix(historyThisUV, historyPrevUV, velocCompare);
	outAO = mix(historyAO, newAO, blendWeight);
}

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
	uvec2 groupThreadId = remap8x8(gl_LocalInvocationIndex);
    uvec2 dispatchId = groupThreadId + gl_WorkGroupID.xy * 8;
    ivec2 pixelPos = ivec2(dispatchId);

    ivec2 workSize = imageSize(GTAOTempFilter);
    if(pixelPos.x >= workSize.x || pixelPos.y >= workSize.y)
    {
        return;
    }

    vec2 texelSize = 1.0f / vec2(workSize);
    vec2 uv = (vec2(pixelPos) + vec2(0.5)) * texelSize; 

    // Handle camera cut case, need to reset GTAO accumulation.
	if(frameData.bCameraCut != 0)
    {
        vec4 centerAO = texelFetch(inGTAOFilterImage, pixelPos, 0);
        imageStore(GTAOTempFilter, pixelPos, centerAO);
        return;
    }

    float outAO = 0;
	temporalFilter(uv, outAO);
	imageStore(GTAOTempFilter, pixelPos, vec4(outAO, 0.0, 0.0, 0.0));
}