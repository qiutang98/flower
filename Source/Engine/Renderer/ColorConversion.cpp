// AMD Cauldron code
// 
// Copyright(c) 2020 Advanced Micro Devices, Inc.All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include "Pch.h"

#include "ColorConversion.h"
float ColorSpacePrimaries[4][4][2] = 
{
    //Rec709
    {
        0.3127f,0.3290f, // White point
        0.64f,0.33f, // Red point
        0.30f,0.60f, // Green point
        0.15f,0.06f, // Blue point
    },
    //P3
    {
        0.3127f,0.3290f, // White point
        0.680f,0.320f, // Red point
        0.265f,0.690f, // Green point
        0.150f,0.060f, // Blue point
    },
    //Rec2020
    {
        0.3127f,0.3290f, // White point
        0.708f,0.292f, // Red point
        0.170f,0.797f, // Green point
        0.131f,0.046f, // Blue point
    },
    //Display Specific zeroed out now Please fill them in once you query them and want to use them for matrix calculations
    {
        0.0f,0.0f, // White point
        0.0f,0.0f, // Red point
        0.0f,0.0f, // Green point
        0.0f,0.0f // Blue point
    }
};

glm::mat4 CalculateRGBToXYZMatrix(float xw, float yw, float xr, float yr, float xg, float yg, float xb, float yb, bool scaleLumaFlag)
{
    float Xw = xw / yw;
    float Yw = 1;
    float Zw = (1 - xw - yw) / yw;

    float Xr = xr / yr;
    float Yr = 1;
    float Zr = (1 - xr - yr) / yr;

    float Xg = xg / yg;
    float Yg = 1;
    float Zg = (1 - xg - yg) / yg;

    float Xb = xb / yb;
    float Yb = 1;
    float Zb = (1 - xb - yb) / yb;

    glm::mat4 XRGB = glm::mat4(
        Xr, Xg, Xb, 0.0f,
        Yr, Yg, Yb, 0.0f,
        Zr, Zg, Zb, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 XRGBInverse = glm::inverse(XRGB);

    glm::vec4 referenceWhite = glm::vec4(Xw, Yw, Zw, 0.0f);
    glm::vec4 SRGB = glm::transpose(XRGBInverse) * referenceWhite;

    glm::mat4 RegularResult = glm::mat4(
        SRGB.x * Xr, SRGB.y * Xg, SRGB.z * Xb, 0.0f,
        SRGB.x * Yr, SRGB.y * Yg, SRGB.z * Yb, 0.0f,
        SRGB.x * Zr, SRGB.y * Zg, SRGB.z * Zb, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);

    if (!scaleLumaFlag)
        return RegularResult;

    glm::mat4 Scale = glm::scale(glm::mat4(1.0f),glm::vec3(100.0f, 100.0f, 100.0f));
    glm::mat4 Result = RegularResult * Scale;
    return Result;
}

glm::mat4 CalculateXYZToRGBMatrix(float xw, float yw, float xr, float yr, float xg, float yg, float xb, float yb, bool scaleLumaFlag)
{
    auto RGBToXYZ = CalculateRGBToXYZMatrix(xw, yw, xr, yr, xg, yg, xb, yb, scaleLumaFlag);
    return glm::inverse(RGBToXYZ);
}

void FillDisplaySpecificPrimaries(float xw, float yw, float xr, float yr, float xg, float yg, float xb, float yb)
{
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_WHITE][ColorPrimariesCoordinates_X] = xw;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_WHITE][ColorPrimariesCoordinates_Y] = yw;

    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_RED][ColorPrimariesCoordinates_X] = xr;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_RED][ColorPrimariesCoordinates_Y] = yr;

    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_GREEN][ColorPrimariesCoordinates_X] = xg;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_GREEN][ColorPrimariesCoordinates_Y] = yg;

    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_BLUE][ColorPrimariesCoordinates_X] = xb;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_BLUE][ColorPrimariesCoordinates_Y] = yb;
}

void SetupGamutMapperMatrices(ColorSpace gamutIn, ColorSpace gamutOut, glm::mat4* inputToOutputRecMatrix)
{
    glm::mat4 intputGamut_To_XYZ = CalculateRGBToXYZMatrix(
        ColorSpacePrimaries[gamutIn][ColorPrimaries_WHITE][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutIn][ColorPrimaries_WHITE][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_RED][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutIn][ColorPrimaries_RED][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_GREEN][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutIn][ColorPrimaries_GREEN][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_BLUE][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutIn][ColorPrimaries_BLUE][ColorPrimariesCoordinates_Y],
        false);

    glm::mat4 XYZ_To_OutputGamut = CalculateXYZToRGBMatrix(
        ColorSpacePrimaries[gamutOut][ColorPrimaries_WHITE][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutOut][ColorPrimaries_WHITE][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_RED][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutOut][ColorPrimaries_RED][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_GREEN][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutOut][ColorPrimaries_GREEN][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_BLUE][ColorPrimariesCoordinates_X], ColorSpacePrimaries[gamutOut][ColorPrimaries_BLUE][ColorPrimariesCoordinates_Y],
        false);

    glm::mat4 intputGamut_To_OutputGamut = intputGamut_To_XYZ * XYZ_To_OutputGamut;
    *inputToOutputRecMatrix = glm::transpose(intputGamut_To_OutputGamut);
}