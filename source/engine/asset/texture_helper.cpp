#include "texture_helper.h"
#include "asset_texture.h"
#include <stb/stb_dxt.h>
#include <execution>
#include "../engine.h"
#include "nameof/nameof.hpp"
namespace engine
{
	static const int kFindBestAlphaCount = 50;
	static float getAlphaCoverageRGBA8(const unsigned char* data, uint32_t width, uint32_t height, float scale, int cutoff)
	{
		// float value may no enough for multi add.
		double value = 0.0;

		// 4 char to 1 uint32_t
		uint32_t* pImg = (uint32_t*)data;

		// Loop all texture to get coverage alpha data.
		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				// pImg++ to next pixel 4 char = 1 uint32_t
				uint8_t* pPixel = (uint8_t*)pImg++;

				// alpha in pixel[3]
				int alpha = (int)(scale * (float)pPixel[3]);

				if (alpha > 255)     { alpha = 255; }
				if (alpha <= cutoff) { continue;    }

				value += alpha;
			}
		}
		return (float)(value / (height * width * 255));
	}

	static void scaleAlphaRGBA8(unsigned char* data, uint32_t width, uint32_t height, float scale)
	{
		uint32_t* pImg = (uint32_t*)data;

		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				uint8_t* pPixel = (uint8_t*)pImg++;
				int alpha = (int)(scale * (float)pPixel[3]);

				pPixel[3] = uint8_t(math::clamp(alpha, 0, 255));
			}
		}
	}

	void engine::buildMipmapData8Bit(
		uint32_t channelCount,
		uint32_t pixelOffsetPerSample,
		unsigned char* srcPixels, 
		AssetTextureBin& outBinData, 
		bool bSRGB, 
		uint32_t mipmapCount, 
		uint32_t inWidth, 
		uint32_t inHeight,
		float alphaMipmapCutoff)
	{
		CHECK(channelCount == 1 || channelCount == 2 || channelCount == 4);
		CHECK(channelCount + pixelOffsetPerSample <= 4);

		const uint32_t kPixelSize = channelCount;

		float alphaCoverage = 1.0f;
		const bool bKeepAlphaCoverage = (alphaMipmapCutoff < 1.0f) && (channelCount == 4);

		outBinData.mipmapDatas.resize(mipmapCount);
		for (size_t mip = 0; mip < outBinData.mipmapDatas.size(); mip++)
		{
			auto& destMipData = outBinData.mipmapDatas[mip];

			uint32_t destWidth  = math::max<uint32_t>(inWidth  >> mip, 1);
			uint32_t destHeight = math::max<uint32_t>(inHeight >> mip, 1);

			if (mip == 0)
			{
				// Copy raw data to mip 0.
				destMipData.resize(inWidth * inHeight * kPixelSize);

				if (channelCount == 4)
				{
					CHECK(pixelOffsetPerSample == 0);
					memcpy(destMipData.data(), srcPixels, destMipData.size());
				}
				else
				{
					// De-interval from stb load.
					for (size_t index = 0; index < inWidth * inHeight; index ++)
					{
						for (size_t i = 0; i < channelCount; i++)
						{
							destMipData[index * kPixelSize + i] = srcPixels[index * 4 + i + pixelOffsetPerSample];
						}
					}
				}

				if (bKeepAlphaCoverage)
				{
					alphaCoverage = getAlphaCoverageRGBA8(destMipData.data(), destWidth, destHeight, 1.0f, (int)(alphaMipmapCutoff * 255));
				}
			}
			else // Other mip process.
			{
				const size_t srcMip = mip - 1;
				const auto& srcMipData = outBinData.mipmapDatas[srcMip];

				destMipData.resize(destWidth * destHeight * kPixelSize);
				for (size_t y = 0; y < destHeight; y++)
				{
					for (size_t x = 0; x < destWidth; x++)
					{
						// Get src data.
						uint32_t srcWidth  = std::max<uint32_t>(inWidth  >> srcMip, 1);
						uint32_t srcHeight = std::max<uint32_t>(inHeight >> srcMip, 1);

						// Clamp src data fetech edge.
						size_t srcX0 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 0), srcWidth  - 1);
						size_t srcX1 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 1), srcWidth  - 1);
						size_t srcY0 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 0), srcHeight - 1);
						size_t srcY1 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 1), srcHeight - 1);

						// Prepare src pixel start pos.
						size_t srcPixelStart[] =
						{
							(srcY0 * srcWidth + srcX0) * channelCount, // X0Y0
							(srcY0 * srcWidth + srcX1) * channelCount, // X1Y0
							(srcY1 * srcWidth + srcX0) * channelCount, // X0Y1
							(srcY1 * srcWidth + srcX1) * channelCount, // X1Y1
						};

						// All color operation must done in linear space.
						// https://paroj.github.io/gltut/Texturing/Tut16%20Mipmaps%20and%20Linearity.html
						uint32_t R = 0, G = 0, B = 0, A = 0;
						for (size_t i = 0; i < 4; i++)
						{
							                        R += bSRGB ? srgbToLinear(srcMipData[srcPixelStart[i] + 0]) : srcMipData[srcPixelStart[i] + 0];
							if(channelCount >= 2) { G += bSRGB ? srgbToLinear(srcMipData[srcPixelStart[i] + 1]) : srcMipData[srcPixelStart[i] + 1]; }
							if(channelCount >= 3) { B += bSRGB ? srgbToLinear(srcMipData[srcPixelStart[i] + 2]) : srcMipData[srcPixelStart[i] + 2]; }
							if(channelCount >= 4) { A += srcMipData[srcPixelStart[i] + 3]; }
						}

						R /= 4; R = bSRGB ? linearToSrgb(R) : R;
						G /= 4; G = bSRGB ? linearToSrgb(G) : G;
						B /= 4; B = bSRGB ? linearToSrgb(B) : B;
						A /= 4;

						// Store to dest.
						size_t destPixelPosStart = (y * destWidth + x) * channelCount;

						                         destMipData[destPixelPosStart + 0] = R;
						if (channelCount >= 2) { destMipData[destPixelPosStart + 1] = G; } 
						if (channelCount >= 3) { destMipData[destPixelPosStart + 2] = B; }
						if (channelCount >= 4) { destMipData[destPixelPosStart + 3] = A; }
					}
				}

				// Find best alpha coverage for mip-map.
				if (alphaCoverage < 1.0f)
				{
					float ini =  0.0f;
					float fin = 10.0f;

					float mid;
					float alphaPercentage;

					for (int iter = 0; iter < kFindBestAlphaCount; iter++)
					{
						mid = (ini + fin) / 2;
						alphaPercentage = getAlphaCoverageRGBA8(destMipData.data(), destWidth, destHeight, mid, (int)(alphaMipmapCutoff * 255));

						if (glm::abs(alphaPercentage - alphaCoverage) < .001) { break; }
						if (alphaPercentage > alphaCoverage) { fin = mid; }
						if (alphaPercentage < alphaCoverage) { ini = mid; }
					}
					scaleAlphaRGBA8(destMipData.data(), destWidth, destHeight, mid);
				}
			}

			CHECK(destWidth * destHeight * channelCount == destMipData.size());
		}
	}

	// block compression functions.

	uint16_t uint8PackTo565(uint8_t* c)
	{
		uint16_t result = 0;
		result |= ((uint16_t)math::floor(31.0f * c[2] / 255.0f) <<  0); 
		result |= ((uint16_t)math::floor(63.0f * c[1] / 255.0f) <<  5); 
		result |= ((uint16_t)math::floor(31.0f * c[0] / 255.0f) << 11);
		return result;
	}

	constexpr uint32_t kBCBlockDim  = 4U; 
	constexpr uint32_t kBCBlockSize = kBCBlockDim * kBCBlockDim;

	static void compressBC3(uint8_t* dest, const uint8_t* src)
	{
		// Alpha pack.
		{
			uint8_t minAlpha = 255;
			uint8_t maxAlpha = 0;

			for (int i = 0; i < 16; i++)
			{
				uint8_t alpha = *(src + i * 4 + 3);
				minAlpha = math::min(minAlpha, alpha);
				maxAlpha = math::max(maxAlpha, alpha);
			}

			// Use six lerp point.
			dest[0] = maxAlpha; // alpha 0
			dest[1] = minAlpha; // alpha 1

			if (maxAlpha > minAlpha)
			{
				uint64_t packAlphaBit = 0;

				const float minAlphaFloat = float(minAlpha);
				const float maxAlphaFloat = float(maxAlpha);

				const float alphaRange = maxAlphaFloat - minAlphaFloat;
				const float alphaMult = 7.0f / alphaRange;

				for (int i = 0; i < 16; i++)
				{
					const uint8_t alpha = *(src + i * 4 + 3);
					uint64_t index = uint64_t(math::round(float(alpha) - minAlphaFloat) * alphaMult);

					if (index == 7) { index = 0; }
					           else { index  ++; }

					packAlphaBit |= (index << (i * 3));
				}

				for (int i = 2; i < 8; i++)
				{
					dest[i] = (packAlphaBit >> ((i - 2) * 8)) & 0xff;
				}
			}
			else
			{
				for (int i = 2; i < 8; i++)
				{
					dest[i] = 0; // All alpha value same.
				}
			}
		}

		// Color pack.
		{
			uint8_t minColor[3] = { 255, 255, 255 };
			uint8_t maxColor[3] = { 0,   0,   0 };

			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					uint8_t c = *(src + i * 4 + j);
					minColor[j] = math::min(minColor[j], c);
					maxColor[j] = math::max(maxColor[j], c);
				}
			}

			uint16_t minColor565 = uint8PackTo565(minColor);
			uint16_t maxColor565 = uint8PackTo565(maxColor);

			// Fill max color 565 as color 0.
			dest[8] = uint8_t((maxColor565 >> 0) & 0xff);
			dest[9] = uint8_t((maxColor565 >> 8) & 0xff);

			// Fill min color 565 as color 1.
			dest[10] = uint8_t((minColor565 >> 0) & 0xff);
			dest[11] = uint8_t((minColor565 >> 8) & 0xff);


			if (maxColor565 > minColor565)
			{
				uint32_t packColorIndex = 0;

				// TODO: Color error diffusion avoid too much block artifact.
				vec3 minColorVec = math::vec3(minColor[0], minColor[1], minColor[2]);
				vec3 maxColorVec = math::vec3(maxColor[0], maxColor[1], maxColor[2]);

				// Color vector max -> min.
				vec3 maxMinVec = maxColorVec - minColorVec;

				// Color vector direction and length.
				float lenInvert = 1.0f / math::length(maxMinVec);
				vec3 colorDirVec = maxMinVec * lenInvert;

				for (int i = 0; i < 16; i++)
				{
					vec3 computeVec = 
						math::vec3(*(src + i * 4 + 0), *(src + i * 4 + 1), *(src + i * 4 + 2)) - minColorVec;

					// Project current color into color direction vector and scale to [0, 3]
					uint32_t index = uint32_t(math::round(dot(computeVec, colorDirVec) * 3.0f * lenInvert));

					if (index == 3) { index = 0; }
					          else  { index  ++; }

					packColorIndex |= (index << (i * 2));
				}

				for (int i = 12; i < 16; i++)
				{
					dest[i] = (packColorIndex >> ((i - 12) * 8)) & 0xff;
				}
			}
			else
			{
				for (int i = 12; i < 16; i++)
				{
					dest[i] = 0; // All color value same.
				}
			}
		}
	}

	template<size_t kComponentCount, size_t kCompressionRatio, size_t kPerBlockCompressedSize>
	static void executeTaskForBC(
		uint32_t mipWidth,
		uint32_t mipHeight,
		std::vector<uint8_t>& compressMipData,
		const std::vector<uint8_t>& srcMipData,
		std::function<void(unsigned char* dest, const unsigned char* src)>&& functor)
	{
		compressMipData.resize(math::max(kPerBlockCompressedSize, srcMipData.size() / kCompressionRatio));

		const auto buildBC = [&](const size_t loopStart, const size_t loopEnd)
		{
			for (size_t taskIndex = loopStart; taskIndex < loopEnd; ++taskIndex)
			{
				const uint32_t pixelPosX = (taskIndex * kBCBlockDim) % mipWidth;
				const uint32_t pixelPosY = kBCBlockDim * ((taskIndex * kBCBlockDim) / mipWidth);
				const uint32_t bufferOffset = taskIndex * kPerBlockCompressedSize;

				std::array<uint8_t, kBCBlockSize * kComponentCount> block{ };
				uint32_t blockLocation = 0;

				// Fill block.
				for (uint32_t j = 0; j < kBCBlockDim; j++)
				{
					for (uint32_t i = 0; i < kBCBlockDim; i++)
					{
						const uint32_t dimX = pixelPosX + i;
						const uint32_t dimY = pixelPosY + j;
						const uint32_t pixelLocation = (dimX + dimY * mipWidth) * kComponentCount;

						if (pixelLocation < srcMipData.size())
						{
							const uint8_t* src = srcMipData.data() + pixelLocation;
							uint8_t* dest = block.data() + blockLocation;

							memcpy(dest, src, kComponentCount);
							blockLocation += kComponentCount;
						}
					}
				}

				functor(&compressMipData[bufferOffset], block.data());
			}
		};
		Engine::get()->getThreadPool()->parallelizeLoop(0, math::max(1U, mipWidth * mipHeight / kBCBlockSize), buildBC).wait();
	}

	void mipmapCompressBC3(
		std::vector<uint8_t>& compressMipData,
		const std::vector<uint8_t>& srcMipData,
		const AssetTexture& meta, 
		uint32_t mipWidth,
		uint32_t mipHeight)
	{
		constexpr size_t kCompressionRatio = 4;
		constexpr size_t kPerBlockCompressedSize = 16; // 128 bit
		constexpr size_t kComponentCount   = 4;

		// Allocate size.

		executeTaskForBC<kComponentCount, kCompressionRatio, kPerBlockCompressedSize>(
			mipWidth, mipHeight, compressMipData, srcMipData, [](unsigned char* dest, const unsigned char* src)
		{
			stb_compress_dxt_block(dest, src, 1, STB_DXT_HIGHQUAL);
		});
	}

	void mipmapCompressBC4(
		std::vector<uint8_t>& compressMipData,
		const std::vector<uint8_t>& srcMipData,
		const AssetTexture& meta,
		uint32_t mipWidth,
		uint32_t mipHeight)
	{
		constexpr size_t kCompressionRatio = 2;
		constexpr size_t kPerBlockCompressedSize = 8; // 64 bit
		constexpr size_t kComponentCount = 1;

		// Allocate size.

		executeTaskForBC<kComponentCount, kCompressionRatio, kPerBlockCompressedSize>(
			mipWidth, mipHeight, compressMipData, srcMipData, [](unsigned char* dest, const unsigned char* src)
			{
				stb_compress_bc4_block(dest, src);
			});
	}

	void mipmapCompressBC1(
		std::vector<uint8_t>& compressMipData,
		const std::vector<uint8_t>& srcMipData,
		const AssetTexture& meta,
		uint32_t mipWidth,
		uint32_t mipHeight)
	{
		constexpr size_t kCompressionRatio =  8;
		constexpr size_t kPerBlockCompressedSize =  8; // 64 bit
		constexpr size_t kComponentCount   =  4;

		// Allocate size.
		executeTaskForBC<kComponentCount, kCompressionRatio, kPerBlockCompressedSize>(
			mipWidth, mipHeight, compressMipData, srcMipData, [](unsigned char* dest, const unsigned char* src)
		{
			stb_compress_dxt_block(dest, src, 0, STB_DXT_HIGHQUAL);
		});
	}

	void mipmapCompressBC5(
		std::vector<uint8_t>& compressMipData,
		const std::vector<uint8_t>& srcMipData,
		const AssetTexture& meta,
		uint32_t mipWidth,
		uint32_t mipHeight)
	{
		constexpr size_t kCompressionRatio = 2;
		constexpr size_t kPerBlockCompressedSize = 16; // 128 bit
		constexpr size_t kComponentCount   = 2;

		executeTaskForBC<kComponentCount, kCompressionRatio, kPerBlockCompressedSize>(
			mipWidth, mipHeight, compressMipData, srcMipData, [](unsigned char* dest, const unsigned char* src)
		{
			stb_compress_bc5_block(dest, src);
		});
	}

	void engine::mipmapCompressBC(AssetTextureBin& inOutBin, const AssetTexture& meta)
	{
		std::vector<std::vector<uint8_t>> compressedMipdatas;
		compressedMipdatas.resize(inOutBin.mipmapDatas.size());

		for (size_t mipIndex = 0; mipIndex < compressedMipdatas.size(); mipIndex++)
		{
			auto& compressMipData = compressedMipdatas[mipIndex];
			const auto& srcMipData = inOutBin.mipmapDatas[mipIndex];

			uint32_t mipWidth = math::max<uint32_t>(meta.getDimension().x >> mipIndex, 1);
			uint32_t mipHeight = math::max<uint32_t>(meta.getDimension().y >> mipIndex, 1);

			if (meta.getFormat() == VK_FORMAT_BC3_SRGB_BLOCK || meta.getFormat() == VK_FORMAT_BC3_UNORM_BLOCK)
			{
				mipmapCompressBC3(compressMipData, srcMipData, meta, mipWidth, mipHeight);
			}
			else if(meta.getFormat() == VK_FORMAT_BC5_UNORM_BLOCK)
			{
				mipmapCompressBC5(compressMipData, srcMipData, meta, mipWidth, mipHeight);
			}
			else if (meta.getFormat() == VK_FORMAT_BC1_RGB_UNORM_BLOCK || meta.getFormat() == VK_FORMAT_BC1_RGB_SRGB_BLOCK)
			{
				mipmapCompressBC1(compressMipData, srcMipData, meta, mipWidth, mipHeight);
			}
			else if (meta.getFormat() == VK_FORMAT_BC4_UNORM_BLOCK)
			{
				mipmapCompressBC4(compressMipData, srcMipData, meta, mipWidth, mipHeight);
			}
			else
			{
				LOG_FATAL("Format {} still no process, need developer fix.", nameof::nameof_enum(meta.getFormat()));
			}
		}

		inOutBin.mipmapDatas = std::move(compressedMipdatas);
	}
}
