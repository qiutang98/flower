#ifndef SSGI_COMMON_GLSL
#define SSGI_COMMON_GLSL

layout (set = 0, binding = 0)  uniform texture2D inHiz;
layout (set = 0, binding = 1)  uniform texture2D inDepth;
layout (set = 0, binding = 2)  uniform texture2D inGbufferA;
layout (set = 0, binding = 3)  uniform texture2D inGbufferB;
layout (set = 0, binding = 4)  uniform texture2D inGbufferS;
layout (set = 0, binding = 5)  uniform texture2D inGbufferV;
layout (set = 0, binding = 6)  uniform texture2D inPrevDepth;
layout (set = 0, binding = 7)  uniform texture2D inPrevGbufferB;



#endif