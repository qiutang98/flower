#ifndef BIT_FILED_GLSL
#define BIT_FILED_GLSL

uint bitfieldExtract(uint src, uint off, uint bits) 
{
    uint mask = (1 << bits) - 1;
    return (src >> off) & mask;
}

uint bitfieldInsert(uint src, uint ins, uint bits) 
{
    uint mask = (1 << bits) - 1;
    return (ins & mask) | (src & (~mask));
}

#endif