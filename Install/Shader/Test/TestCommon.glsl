#ifndef TEST_COMMON_GLSL
#define TEST_COMMON_GLSL

// C++ engine test shader and developing iteration is pretty hard.

// So try some code like shader-toy using visual studio code to debug here.

// Some template code store here.

/*
    #iChannel0 "file://duck.png"
    #iChannel1 "https://66.media.tumblr.com/tumblr_mcmeonhR1e1ridypxo1_500.jpg"
    #iChannel2 "file://other/shader.glsl"
    #iChannel2 "self"
    #iChannel4 "file://music/epic.mp3"

    #iChannel0::MinFilter "NearestMipMapNearest"
    #iChannel0::MagFilter "Nearest"
    #iChannel0::WrapMode "Repeat"

    #iChannel0 "file://cubemaps/yokohama_{}.jpg" // Note the wildcard '{}'
    #iChannel0::Type "CubeMap"


    #pragma glslify: snoise = require('glsl-noise/simplex/2d')

    float noise(in vec2 pt) 
    {
        return snoise(pt) * 0.5 + 0.5;
    }

    void main () 
    {
        float r = noise(gl_FragCoord.xy * 0.01);
        float g = noise(gl_FragCoord.xy * 0.01 + 100.0);
        float b = noise(gl_FragCoord.xy * 0.01 + 300.0);
        gl_FragColor = vec4(r, g, b, 1);
    }
*/

#endif


