#version 460

// Bake mesh to sdf 3d texture. brute implement.

// https://iquilezles.org/articles/distfunctions/
// https://github.com/diharaw/sdf-baking

layout (set = 0, binding = 0, r32f) uniform image3D imageSDF;

struct Vertex
{
    vec3 position;
    vec3 normal;
};

layout (set = 1, binding = 0) buffer Indices { uint   indices[]; };
layout (set = 2, binding = 0) buffer Vertices{ Vertex vertices[]; };

layout(push_constant) uniform PushConsts
{   
    vec3 gridOrigin;
    float closestDistInit;

    vec3 gridStepSize;
    uint numTriangles;

} push;


float dot2( in vec2 v ) { return dot(v, v); }
float dot2( in vec3 v ) { return dot(v, v); }
float ndot( in vec2 a, in vec2 b ) { return a.x * b.x - a.y * b.y; }



float sdfTriangle( vec3 p, vec3 a, vec3 b, vec3 c)
{
    vec3 ba = b - a; vec3 pa = p - a;
    vec3 cb = c - b; vec3 pb = p - b;
    vec3 ac = a - c; vec3 pc = p - c;

    vec3 nor = cross(ba, ac);

    return 
        sqrt(
            (sign(dot(cross(ba,nor),pa)) +
             sign(dot(cross(cb,nor),pb)) +
             sign(dot(cross(ac,nor),pc)) < 2.0)
            ?
             min( min(
             dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
             dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb)),
             dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc))
            :
             dot(nor,pa)*dot(nor,pa)/dot2(nor) 
        );
}

bool isFrontFacing(vec3 p, Vertex v0, Vertex v1, Vertex v2)
{
    return dot(normalize(p - v0.position), v0.normal) >= 0.0f || dot(normalize(p - v1.position), v1.normal) >= 0.0f || dot(normalize(p - v2.position), v2.normal) >= 0.0f;
}

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    ivec3 voxelIndex = ivec3(gl_GlobalInvocationID.xyz);
    ivec3 workSize = ivec3(imageSize(imageSDF));

    if (all(lessThan(voxelIndex, workSize)))
    {
        vec3 p = push.gridOrigin + push.gridStepSize * vec3(voxelIndex);

        float closestDist = push.closestDistInit;
        bool bFrontFacing = true;

        for (int i = 0; i < push.numTriangles; i++)
        {
            Vertex v0 = vertices[indices[3 * i]];
            Vertex v1 = vertices[indices[3 * i + 1]];
            Vertex v2 = vertices[indices[3 * i + 2]];

            float h = sdfTriangle(p, v0.position, v1.position, v2.position);

            if (h < closestDist)
            {
                closestDist = h;
                bFrontFacing = isFrontFacing(p, v0, v1, v2);
            }
        }

        imageStore(imageSDF, voxelIndex, vec4(bFrontFacing ? closestDist : -closestDist));
    }
}