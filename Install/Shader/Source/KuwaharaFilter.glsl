#ifndef KUWAHARA_FILTER_GLSL
#define KUWAHARA_FILTER_GLSL

#ifndef ROTATE_FILTER
#define ROTATE_FILTER 1
#endif

vec4 kuwaharaMeanKernelVariance(
    in texture2D inSrcTexture,
    in sampler inSampler,
    in vec2 uv,
    in ivec2 sobeWindow,
    in mat2 rotationMatrix,
    in vec2 quadrants,
    out float outVariance
)
{
    vec2 texelSize = 1.0 / vec2(textureSize(inSrcTexture, 0));

    vec4 mean = vec4(0);
    vec4 variance = vec4(0);
    for(int i = 0; i < sobeWindow.x; i++)
    {
        for(int j = 0; j < sobeWindow.y; j++)
        {
            vec2 sampleOffset = ivec2(i, j) * texelSize * quadrants;
        #if ROTATE_FILTER
            sampleOffset = rotationMatrix * sampleOffset;
        #endif

            vec4 pixelColor = texture(sampler2D(inSrcTexture, inSampler), uv + sampleOffset);
            mean += pixelColor;
            variance += pixelColor * pixelColor;
        }
    }

    mean /= float(sobeWindow.x * sobeWindow.y);

    variance = variance / float(sobeWindow.x * sobeWindow.y) - mean * mean;
    outVariance = variance.r + variance.g + variance.b + variance.a;
    
    return mean;
}

float kuwaharaPixelAngle(
    in texture2D inSrcTexture,
    in sampler inSampler,
    in vec2 uv
)
{
    vec2 texelSize = 1.0f / vec2(textureSize(inSrcTexture, 0));

    float gradientX = 0;
    float gradientY = 0;
    float sobelX[9] = float[](-1, -2, -1, 0, 0, 0, 1, 2, 1);
    float sobelY[9] = float[](-1, 0, 1, -2, 0, 2, -1, 0, 1);
    int i = 0;

    for(int x = -1; x <= 1; x++)
    {
        for(int y = -1; y <= 1; y++)
        {
            vec2 sampleOffset = vec2(x, y) * texelSize;
            vec3 pixelColor = texture(sampler2D(inSrcTexture, inSampler), uv + sampleOffset).xyz;
            float pixelValue = dot(pixelColor, vec3(0.3, 0.59, 0.11));
        
            gradientX += pixelValue * sobelX[i];
            gradientY += pixelValue * sobelY[i];
            i++;
        }
    }

    return atan(gradientY / gradientX);
}


vec4 kuwaharaFilter(
    in texture2D inSrcTexture,
    in sampler inSampler,
    in vec2 uv
)
{
    vec4 result = vec4(0.0f);
    
    const ivec2 sobeWindow = ivec2(3, 3);

#if ROTATE_FILTER
    float angle = kuwaharaPixelAngle(inSrcTexture, inSampler, uv);
    mat2 rotationMatrix = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
#else
    mat2 rotationMatrix = mat2(1);
#endif

    

    vec4 means[4];
    vec4 variances;
    means[0] = kuwaharaMeanKernelVariance(inSrcTexture, inSampler, uv, sobeWindow, rotationMatrix, 2.0f * vec2(-1, -1), variances.x);
    means[1] = kuwaharaMeanKernelVariance(inSrcTexture, inSampler, uv, sobeWindow, rotationMatrix, 2.0f * vec2( 1, -1), variances.y);
    means[2] = kuwaharaMeanKernelVariance(inSrcTexture, inSampler, uv, sobeWindow, rotationMatrix, 2.0f * vec2(-1,  1), variances.z);
    means[3] = kuwaharaMeanKernelVariance(inSrcTexture, inSampler, uv, sobeWindow, rotationMatrix, 2.0f * vec2( 1,  1), variances.w);
    
    vec4 finalColor = means[0];
    float minimumVariance = variances[0];
    for(int i = 1; i < 4; i++)
    {
        if (variances[i] < minimumVariance)
        {
            finalColor = means[i];
            minimumVariance = variances[i];
        }
    }

    return finalColor;
}

#endif