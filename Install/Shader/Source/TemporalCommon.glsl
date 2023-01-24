#ifndef TEMPORAL_COMMON_GLSL
#define TEMPORAL_COMMON_GLSL

const vec2 kSample3x3[9] = vec2[]
(
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2( 0.0,  0.0),
    vec2( 1.0,  0.0),
    vec2(-1.0,  0.0),
    vec2( 0.0, -1.0),
    vec2( 0.0,  1.0)
);

const vec2 kSample3x3NoCenter[8] = vec2[]
(
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2( 1.0,  0.0),
    vec2(-1.0,  0.0),
    vec2( 0.0, -1.0),
    vec2( 0.0,  1.0)
);

#endif