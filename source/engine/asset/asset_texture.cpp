#include "asset_texture.h"
#include "asset_system.h"

#include <execution>
#include <stb/stb_dxt.h>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

namespace engine
{
	AssetTexture::AssetTexture(const std::string& assetNameUtf8, const std::string& assetRelativeRootProjectPathUtf8)
		: AssetInterface(assetNameUtf8, assetRelativeRootProjectPathUtf8)
	{

	}

	float getAlphaCoverageRGBA8(const unsigned char* data, uint32_t width, uint32_t height, float scale, int cutoff)
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

				if (alpha > 255) { alpha = 255; }
				if (alpha <= cutoff) { continue; }

				value += alpha;
			}
		}
		return (float)(value / (height * width * 255));
	}

	void scaleAlpha(unsigned char* data, uint32_t width, uint32_t height, float scale)
	{
		uint32_t* pImg = (uint32_t*)data;

		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				uint8_t* pPixel = (uint8_t*)pImg++;

				int alpha = (int)(scale * (float)pPixel[3]);
				if (alpha > 255) { alpha = 255; }

				pPixel[3] = alpha;
			}
		}
	}

	template<typename T> inline float getQuantifySize() { CHECK(false); return 0.0f; }
	template<> inline float getQuantifySize<uint8_t>() { return float(1 << 8) - 1.0f; }
	template<> inline float getQuantifySize<uint16_t>() { return float(1 << 16) - 1.0f; }
	template<> inline float getQuantifySize<float>() { return 1.0f; }

	template<typename T>
	void buildMipmapData(T* srcPixels, const AssetTexture& meta, AssetTextureBin& outBinData, uint32_t channelCount, uint32_t channelOffset)
	{
		const float kQuantitySize = getQuantifySize<T>();

		CHECK(meta.getAlphaCutoff() >= 1.0f);
		CHECK(!meta.isSrgb());

		outBinData.mipmapDatas.resize(meta.getMipmapCount());
		const auto kStripSize = sizeof(T) * channelCount;
		for (size_t mip = 0; mip < outBinData.mipmapDatas.size(); mip++)
		{
			auto& destMipData = outBinData.mipmapDatas[mip];
			uint32_t destWidth = math::max<uint32_t>(meta.getWidth() >> mip, 1);
			uint32_t destHeight = math::max<uint32_t>(meta.getHeight() >> mip, 1);
			destMipData.resize(destWidth * destHeight * kStripSize);

			T* pDestData = (T*)destMipData.data();

			if (mip == 0)
			{
				for (size_t i = 0; i < destWidth * destHeight; i++)
				{
					for (size_t j = 0; j < channelCount; j++)
					{
						pDestData[i * channelCount + j] = srcPixels[i * 4 + j + channelOffset];
					}
				}
			}
			else
			{
				const size_t srcMip = mip - 1;
				const auto& srcMipData = outBinData.mipmapDatas[srcMip];
				for (size_t y = 0; y < destHeight; y++)
				{
					for (size_t x = 0; x < destWidth; x++)
					{
						// Get src data.
						uint32_t srcWidth = std::max<uint32_t>(meta.getWidth() >> srcMip, 1);
						uint32_t srcHeight = std::max<uint32_t>(meta.getHeight() >> srcMip, 1);

						// Clamp src data fetech edge.
						size_t srcX0 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 0), srcWidth - 1);
						size_t srcX1 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 1), srcWidth - 1);
						size_t srcY0 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 0), srcHeight - 1);
						size_t srcY1 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 1), srcHeight - 1);
						size_t srcPixelStart[] =
						{
							(srcY0 * srcWidth + srcX0) * channelCount, // X0Y0
							(srcY0 * srcWidth + srcX1) * channelCount, // X1Y0
							(srcY1 * srcWidth + srcX0) * channelCount, // X0Y1
							(srcY1 * srcWidth + srcX1) * channelCount, // X1Y1
						};

						const T* pSrcData = (const T*)srcMipData.data();
						size_t destPixelPosStart = (y * destWidth + x) * channelCount;

						for (size_t channelId = 0; channelId < channelCount; channelId++)
						{
							float sumValue = 0.0f;
							for (size_t srcPixelId = 0; srcPixelId < 4; srcPixelId++)
							{
								const T& valueLoad = pSrcData[srcPixelStart[srcPixelId] + channelId];
								const float v = float(valueLoad) / kQuantitySize;

								sumValue += v;
							}
							sumValue *= 0.25f;

							pDestData[destPixelPosStart + channelId] = T(sumValue * kQuantitySize);
						}
					}
				}
			}
			CHECK(destHeight * destWidth * kStripSize == destMipData.size());
		}
	}

	void buildMipmapDataRGBA8(
		stbi_uc* srcPixels, 
		AssetTextureBin& outBinData,
		float alphaCutOff,
		uint32_t mipmapCount, 
		bool bSRGB,
		uint32_t inWidth,
		uint32_t inHeight)
	{
		const float cutOff = alphaCutOff;
		float alphaCoverageMip0 = 1.0f;
		outBinData.mipmapDatas.resize(mipmapCount);

		// Look no good when convert to linear space do mipmap.
		const bool bSrgb = bSRGB;

		for (size_t mip = 0; mip < outBinData.mipmapDatas.size(); mip++)
		{
			auto& destMipData = outBinData.mipmapDatas[mip];
			uint32_t destWidth  = math::max<uint32_t>(inWidth >> mip, 1);
			uint32_t destHeight = math::max<uint32_t>(inHeight >> mip, 1);

			if (mip == 0)
			{
				// Copy raw data to mip 0.
				destMipData.resize(inWidth * inHeight * 4);
				memcpy(destMipData.data(), srcPixels, destMipData.size());

				alphaCoverageMip0 = cutOff < 0.9999f
					? getAlphaCoverageRGBA8(destMipData.data(), destWidth, destHeight, 1.0f, (int)(cutOff * 255))
					: 1.0f;
			}
			else // Other mip process.
			{
				const size_t srcMip = mip - 1;
				const auto& srcMipData = outBinData.mipmapDatas[srcMip];

				destMipData.resize(destWidth * destHeight * 4);
				for (size_t y = 0; y < destHeight; y++)
				{
					for (size_t x = 0; x < destWidth; x++)
					{
						// Get src data.
						uint32_t srcWidth  = std::max<uint32_t>(inWidth >> srcMip,  1);
						uint32_t srcHeight = std::max<uint32_t>(inHeight >> srcMip, 1);

						// Clamp src data fetech edge.
						size_t srcX0 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 0), srcWidth  - 1);
						size_t srcX1 = (size_t)std::min<uint32_t>(uint32_t(x * 2 + 1), srcWidth  - 1);
						size_t srcY0 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 0), srcHeight - 1);
						size_t srcY1 = (size_t)std::min<uint32_t>(uint32_t(y * 2 + 1), srcHeight - 1);

						// Prepare src pixel start pos. * 4 is because per pixel has RGBA four components.
						size_t srcPixelStart[] =
						{
							(srcY0 * srcWidth + srcX0) * 4, // X0Y0
							(srcY0 * srcWidth + srcX1) * 4, // X1Y0
							(srcY1 * srcWidth + srcX0) * 4, // X0Y1
							(srcY1 * srcWidth + srcX1) * 4, // X1Y1
						};

						// Perpixel own 4 uint8_t pack to one uint32_t value.
						const uint32_t* pSrcData = (const uint32_t*)srcMipData.data();

						// All color operation must done in linear space.
						// https://paroj.github.io/gltut/Texturing/Tut16%20Mipmaps%20and%20Linearity.html
						uint32_t R = 0, G = 0, B = 0, A = 0;
						for (size_t i = 0; i < 4; i++)
						{
							R += bSrgb ? srgbToLinear(srcMipData[srcPixelStart[i] + 0]) : srcMipData[srcPixelStart[i] + 0];
							G += bSrgb ? srgbToLinear(srcMipData[srcPixelStart[i] + 1]) : srcMipData[srcPixelStart[i] + 1];
							B += bSrgb ? srgbToLinear(srcMipData[srcPixelStart[i] + 2]) : srcMipData[srcPixelStart[i] + 2];

							A += srcMipData[srcPixelStart[i] + 3];
						}
						R /= 4; R = bSrgb ? linearToSrgb(R) : R;
						G /= 4; G = bSrgb ? linearToSrgb(G) : G;
						B /= 4; B = bSrgb ? linearToSrgb(B) : B;
						A /= 4;

						// Store to dest.
						size_t destPixelPosStart = (y * destWidth + x) * 4;
						destMipData[destPixelPosStart + 0] = R;
						destMipData[destPixelPosStart + 1] = G;
						destMipData[destPixelPosStart + 2] = B;
						destMipData[destPixelPosStart + 3] = A;
					}
				}

				if (alphaCoverageMip0 < 1.0f)
				{
					float ini = 0;
					float fin = 10;

					float mid;
					float alphaPercentage;

					// find best alpha coverage for mip-map.
					int iter = 0;
					for (; iter < 50; iter++)
					{
						mid = (ini + fin) / 2;
						alphaPercentage = getAlphaCoverageRGBA8(destMipData.data(), destWidth, destHeight, mid, (int)(cutOff * 255));

						if (glm::abs(alphaPercentage - alphaCoverageMip0) < .001) { break; }
						if (alphaPercentage > alphaCoverageMip0) { fin = mid; }
						if (alphaPercentage < alphaCoverageMip0) { ini = mid; }
					}

					scaleAlpha(destMipData.data(), destWidth, destHeight, mid);
				}
			}

			CHECK(destWidth * destHeight * 4 == destMipData.size());
		}
	}

	static inline void getChannelCountOffset(uint32_t& channelCount, uint32_t& pixelSampleOffset, const AssetTexture::ImportConfig& config)
	{
		     if (config.channel == AssetTexture::ImportConfig::EChannel::RGBA) { channelCount = 4; pixelSampleOffset = 0; }
		else if (config.channel == AssetTexture::ImportConfig::EChannel::RGB)  { channelCount = 3; pixelSampleOffset = 0; }
		else if (config.channel == AssetTexture::ImportConfig::EChannel::R)    { channelCount = 1; pixelSampleOffset = 0; }
		else if (config.channel == AssetTexture::ImportConfig::EChannel::G)    { channelCount = 1; pixelSampleOffset = 1; }
		else if (config.channel == AssetTexture::ImportConfig::EChannel::B)    { channelCount = 1; pixelSampleOffset = 2; }
		else if (config.channel == AssetTexture::ImportConfig::EChannel::A)    { channelCount = 1; pixelSampleOffset = 3; }
		else { CHECK_ENTRY(); }
	}

	static inline void mipmapCompressBC3(AssetTextureBin& inOutBin, const AssetTexture& meta)
	{
		std::vector<std::vector<uint8_t>> compressedMipdatas;
		compressedMipdatas.resize(inOutBin.mipmapDatas.size());

		for(size_t mipIndex = 0; mipIndex < compressedMipdatas.size(); mipIndex ++)
		{
			auto& compressMipData = compressedMipdatas[mipIndex];
			auto& srcMipData = inOutBin.mipmapDatas[mipIndex];

			uint32_t mipWidth = math::max<uint32_t>(meta.getWidth() >> mipIndex, 1);
			uint32_t mipHeight = math::max<uint32_t>(meta.getHeight() >> mipIndex, 1);

			if (mipWidth >= 4 && mipHeight >= 4)
			{
				uint32_t compressSize = mipWidth * mipHeight;
				compressMipData.resize(compressSize);
				uint8_t* outBuffer = compressMipData.data();

				struct BlockTask
				{
					uint32_t pixelPosX;
					uint32_t pixelPosY;
					uint32_t bufferOffset;
				};

				std::vector<BlockTask> compressTasks(mipWidth * mipHeight / 16);
				for (uint32_t i = 0; i < compressTasks.size(); i++)
				{
					compressTasks[i].pixelPosX = (i * 4) % mipWidth;
					compressTasks[i].pixelPosY = 4 * ((i * 4) / mipWidth);

					compressTasks[i].bufferOffset = i * 16;
				}

				std::for_each(std::execution::par, compressTasks.begin(), compressTasks.end(), [&](const BlockTask& item)
				{
					std::array<uint8_t, 64> block{ };
					uint32_t blockLocation = 0;

					for (uint32_t j = 0; j < 4; j++)
					{
						for (uint32_t i = 0; i < 4; i++)
						{
							const uint32_t dimX = item.pixelPosX + i;
							const uint32_t dimY = item.pixelPosY + j;
							const uint32_t pixelLocation = (dimX + dimY * mipWidth) * 4;

							const uint8_t* dataStart = srcMipData.data() + pixelLocation;
							for (uint32_t k = 0; k < 4; k++)
							{
								block[blockLocation] = *dataStart;
								blockLocation++;
								dataStart++;
							}
						}
					}

					stb_compress_dxt_block(&compressMipData[item.bufferOffset], block.data(), 1, STB_DXT_HIGHQUAL);
				});
			}
			else
			{
				uint32_t bufferOffset = 0;
				compressMipData.resize(16 * math::max<uint32_t>(1, mipWidth / 4) * math::max<uint32_t>(1, mipHeight / 4));

				std::array<uint8_t, 64> block{ };
				for (uint32_t blockX = 0; blockX < mipWidth; blockX += 4)
				{
					for (uint32_t blockY = 0; blockY < mipHeight; blockY += 4)
					{
						// Idea: We can't just use 4x4 level result, which make alpha value error. we upscale mipmap data to 4x4 and do one compression.
						// Eg: 1x1 -> copy upscale to 4x4, then compress to one block.
						//     2x2 -> copy upscale to 4x4, then compress to one block.
						//   a    a a a a  -> compress a`  a b   a a b b   -> compress a`
						//        a a a a                  c d   a a b b 
						//		  a a a a                        c c d d 
						//		  a a a a                        c c d d
						const uint32_t kScaleX = math::max<uint32_t>(1, 4 / mipWidth);
						const uint32_t kScaleY = math::max<uint32_t>(1, 4 / mipHeight);

						uint32_t blockLocation = 0;
						for (uint32_t dimX = 0; dimX < 4; dimX++)
						{
							for (uint32_t dimY = 0; dimY < 4; dimY++)
							{
								uint32_t posX = blockX + dimX / kScaleX;
								uint32_t posY = blockY + dimY / kScaleY;

								const uint32_t pixelLocation = (posX + posY * mipWidth) * 4;
								const uint8_t* dataStart = srcMipData.data() + pixelLocation;
								for (uint32_t k = 0; k < 4; k++)
								{
									block[blockLocation] = *dataStart;
									blockLocation++;
									dataStart++;
								}
							}
						}
						stb_compress_dxt_block(&compressMipData[bufferOffset], block.data(), 1, STB_DXT_HIGHQUAL);
						bufferOffset += 16;
					}
				}
			}
		}

		inOutBin.mipmapDatas = std::move(compressedMipdatas);
	}

	bool AssetTexture::buildFromConfigs(
		const ImportConfig& config, 
		const std::filesystem::path& projectRootPath,
		const std::filesystem::path& savePath, 
		const std::filesystem::path& srcPath,
		AssetTexture& outMeta,
		const UUID& overriderUUID)
	{
		std::string assetNameUtf8 = utf8::utf16to8(savePath.filename().u16string());
		AssetTexture meta(assetNameUtf8, buildRelativePathUtf8(projectRootPath, savePath));

		if (!overriderUUID.empty())
		{
			meta.setUUID(overriderUUID);
		}

		auto getFormat = [&](const AssetTexture& meta, const ImportConfig& config)
		{
			if (meta.m_bHdr)
			{
				if (config.channel == ImportConfig::EChannel::RGBA)
				{
					return VK_FORMAT_R32G32B32A32_SFLOAT;
				}
				else if(config.channel == ImportConfig::EChannel::RGB)
				{
					return VK_FORMAT_R32G32B32_SFLOAT;
				}
				else if (
					config.channel == ImportConfig::EChannel::R ||
					config.channel == ImportConfig::EChannel::G || 
					config.channel == ImportConfig::EChannel::B || 
					config.channel == ImportConfig::EChannel::A)
				{
					return VK_FORMAT_R32_SFLOAT;
				}
			}
			else
			{
				if (config.bHalfFixed)
				{
					if (config.channel == ImportConfig::EChannel::RGBA)
					{
						return VK_FORMAT_R16G16B16A16_UNORM;
					}
					else if (config.channel == ImportConfig::EChannel::RGB)
					{
						return VK_FORMAT_R16G16B16_UNORM;
					}
					else if (
						config.channel == ImportConfig::EChannel::R ||
						config.channel == ImportConfig::EChannel::G ||
						config.channel == ImportConfig::EChannel::B ||
						config.channel == ImportConfig::EChannel::A)
					{
						return VK_FORMAT_R16_UNORM;
					}
				}
				else
				{
					// LDR.
					if (meta.m_bSRGB)
					{
						return meta.m_bCompressed ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_R8G8B8A8_SRGB;  // SRGB 4 Channel.
					}
					else
					{
						return meta.m_bCompressed ? VK_FORMAT_BC3_UNORM_BLOCK : VK_FORMAT_R8G8B8A8_UNORM; // UNORM 4 Channel.
					}
				}

			}

			CHECK_ENTRY();
			return VK_FORMAT_R8_UNORM;
		};

		auto importExr = [&]() -> bool
		{
			std::string input = srcPath.string();
			float* out; // width * height * RGBA

			int width, height;
			const char* err = nullptr;

			int ret = LoadEXR(&out, &width, &height, input.c_str(), &err);

			if (ret != TINYEXR_SUCCESS) 
			{
				if (err) 
				{
					LOG_ERROR("Err import exr: {}.", err);
					FreeEXRErrorMessage(err); 
				}

				LOG_ERROR("Fail import exr file!");
				return false;
			}
			else
			{
				const bool bPOT = isPOT(width) && isPOT(height);

				meta.m_bSRGB = false;
				meta.m_bCompressed = false;
				meta.m_bMipmap = bPOT ? config.bGenerateMipmap : false;
				meta.m_bHdr = true;
				meta.m_width = width;
				meta.m_height = height;
				meta.m_depth = 1;

				if (meta.m_width < 4 || meta.m_height < 4)
				{
					meta.m_bCompressed = false;
				}

				if (meta.m_bMipmap)
				{
					uint32_t maxDim = math::max(meta.m_width, meta.m_height);
					meta.m_mipmapCount = std::bit_width(maxDim);
				}
				else
				{
					meta.m_mipmapCount = 1;
				}
				meta.m_alphaCutoff = 1.0f;
				meta.m_format = (uint32_t)getFormat(meta, config);

				uint32_t channelCount;
				uint32_t pixelSampleOffset;
				getChannelCountOffset(channelCount, pixelSampleOffset, config);

				AssetTextureBin bin{};
				buildMipmapData<float>(out, meta, bin, channelCount, pixelSampleOffset);

				saveAsset<AssetTextureBin>(bin, savePath, ".imagebin");

				// Build snapshot.
				{
					uint32_t widthSnapShot;
					uint32_t heightSnapShot;
					quantifySnapshotDim(widthSnapShot, heightSnapShot, width, height);

					std::vector<float> snapshotData;
					snapshotData.resize(widthSnapShot * heightSnapShot * 4);

					// Do srgb convert for all texture, so they will looks same with browser editor.
					stbir_resize_float(
						out,
						width,
						height,
						0,
						snapshotData.data(),
						widthSnapShot,
						heightSnapShot,
						0,
						4
					);

					std::vector<uint8_t> ldrDatas;
					ldrDatas.resize(snapshotData.size());

					for (size_t i = 0; i < ldrDatas.size(); i++)
					{
						ldrDatas[i] = uint8_t(snapshotData[i] / (1.0f + snapshotData[i]) * 255);
					}

					meta.buildSnapshot(widthSnapShot, heightSnapShot, ldrDatas.data());
				}

				free(out);
				return true;
			}

			return true;
		};

		auto imporHalfFixed = [&]() -> bool
		{
			int32_t texWidth, texHeight, texChannels;
			stbi_us* pixels = stbi_load_16(srcPath.string().c_str(), &texWidth, &texHeight, &texChannels, 4);
			if (!pixels)
			{
				return false;
			}

			// Texture dim is power of two?
			const bool bPOT = isPOT(texWidth) && isPOT(texHeight);
			meta.m_bSRGB = false;
			meta.m_bCompressed = false;
			meta.m_bMipmap = bPOT ? config.bGenerateMipmap : false;
			meta.m_bHdr = false;
			meta.m_width = texWidth;
			meta.m_height = texHeight;
			meta.m_depth = 1;
			if (meta.m_bMipmap)
			{
				uint32_t maxDim = math::max(meta.m_width, meta.m_height);
				meta.m_mipmapCount = std::bit_width(maxDim);
			}
			else
			{
				meta.m_mipmapCount = 1;
			}
			meta.m_alphaCutoff = 1.0f;
			meta.m_format = (uint32_t)getFormat(meta, config);
			{
				uint32_t channelCount;
				uint32_t pixelSampleOffset;
				getChannelCountOffset(channelCount, pixelSampleOffset, config);

				AssetTextureBin bin{};
				buildMipmapData<uint16_t>(pixels, meta, bin, channelCount, pixelSampleOffset);
				saveAsset<AssetTextureBin>(bin, savePath, ".imagebin");
			}
			// Build snapshot.
			{
				uint32_t widthSnapShot;
				uint32_t heightSnapShot;
				quantifySnapshotDim(widthSnapShot, heightSnapShot, texWidth, texHeight);

				std::vector<uint16_t> snapshotData;
				snapshotData.resize(widthSnapShot* heightSnapShot * 4);

				// Do srgb convert for all texture, so they will looks same with browser editor.
				stbir_resize_uint16_generic(
					pixels,
					texWidth,
					texHeight,
					0,
					snapshotData.data(),
					widthSnapShot,
					heightSnapShot,
					0,
					4,
					3,
					0,
					STBIR_EDGE_CLAMP,
					STBIR_FILTER_DEFAULT,
					STBIR_COLORSPACE_LINEAR,
					nullptr
				);

				std::vector<uint8_t> ldrDatas;
				ldrDatas.resize(snapshotData.size());

				for (size_t i = 0; i < ldrDatas.size(); i++)
				{
					ldrDatas[i] = uint8_t(float(snapshotData[i]) / 65535.0f * 255.0f);
				}

				meta.buildSnapshot(widthSnapShot, heightSnapShot, ldrDatas.data());
			}
			return true;
		};

		auto importLdr = [&]() -> bool
		{
			int32_t texWidth, texHeight, texChannels;
			stbi_uc* pixels = stbi_load(srcPath.string().c_str(), &texWidth, &texHeight, &texChannels, 4);

			if (!pixels)
			{
				return false;
			}

			// Texture dim is power of two?
			const bool bPOT = isPOT(texWidth) && isPOT(texHeight);
			const int rawDataSize = texWidth * texHeight * 4;

			// Save raw data to project path.
			if (!saveAssetBinaryWithCompression(pixels, rawDataSize, savePath, ".imageraw"))
			{
				LOG_ERROR("Fail to save raw asset, the image import fail!");
				return false;
			}

			// Save meta info.
			{
				// Prepare config.
				meta.m_bSRGB = config.bSRGB;
				meta.m_bCompressed = bPOT ? config.bCompressed : false;
				meta.m_bMipmap = bPOT ? config.bGenerateMipmap : false;
				meta.m_bHdr = false;

				// Store dimension.
				meta.m_width = texWidth;
				meta.m_height = texHeight;
				meta.m_depth = 1;

				if (meta.m_width < 4 || meta.m_height < 4)
				{
					meta.m_bCompressed = false;
				}

				if (meta.m_bMipmap)
				{
					uint32_t maxDim = math::max(meta.m_width, meta.m_height);
					meta.m_mipmapCount = std::bit_width(maxDim);
				}
				else
				{
					meta.m_mipmapCount = 1;
				}
				meta.m_alphaCutoff = config.cutoffAlpha;
				meta.m_format = (uint32_t)getFormat(meta, config);
				{
					AssetTextureBin bin{};
					buildMipmapDataRGBA8(pixels, bin,
						meta.getAlphaCutoff(),
						meta.getMipmapCount(),
						meta.isSrgb(), 
						meta.getWidth(), 
						meta.getHeight());

					if (meta.m_bCompressed)
					{
						mipmapCompressBC3(bin, meta);
					}

					saveAsset<AssetTextureBin>(bin, savePath, ".imagebin");
				}

				// Build snapshot.
				{
					uint32_t widthSnapShot;
					uint32_t heightSnapShot;
					quantifySnapshotDim(widthSnapShot, heightSnapShot, texWidth, texHeight);

					std::vector<uint8_t> snapshotData;
					snapshotData.resize(widthSnapShot * heightSnapShot * 4);

					// Do srgb convert for all texture, so they will looks same with browser editor.
					stbir_resize_uint8_srgb_edgemode(
						pixels,
						texWidth,
						texHeight,
						0,
						snapshotData.data(),
						widthSnapShot,
						heightSnapShot,
						0,
						4,
						3,
						STBIR_FLAG_ALPHA_PREMULTIPLIED,
						STBIR_EDGE_CLAMP
					);

					meta.buildSnapshot(widthSnapShot, heightSnapShot, snapshotData.data());
				}
			}
			return true;
		};

		if (config.bExr)
		{
			if (!importExr()) return false;
		}
		else
		{
			if (config.bHalfFixed)
			{
				if (!imporHalfFixed()) return false;
			}
			else
			{
				if (!importLdr()) return false;
			}

		}

		// Save meta info.
		if (!saveAssetMeta<AssetTexture>(meta, savePath, ".image"))
		{
			LOG_ERROR("Fail to save meta asset, the image import fail!");
			return false;
		}

		// Copy result to meta.
		outMeta = meta;

		return true;
	}

	VkFormat AssetTexture::getFormat() const
	{
		return (VkFormat)m_format;
	}

	void AssetTextureCacheLoadTask::uploadFunction(
		uint32_t stageBufferOffset, void* bufferPtrStart, RHICommandBufferBase& commandBuffer, VulkanBuffer& stageBuffer)
	{
		auto savePath = getAssetSystem()->getProjectRootPath();
		auto filePath = "\\." + cacheAsset->getRelativePathUtf8() + ".imagebin";
		savePath += filePath;

		AssetTextureBin textureBin{};
		loadAsset(textureBin, savePath);

		VkImageSubresourceRange rangeAllMips = buildBasicImageSubresource();
		rangeAllMips.levelCount = cacheAsset->getMipmapCount();

		imageAssetGPU->prepareToUpload(commandBuffer, rangeAllMips);

		uint32_t bufferOffset = 0;
		uint32_t bufferSize = 0;

		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		std::vector<VkBufferImageCopy> copyRegions{};

		const auto& mipmapDatas = textureBin.mipmapDatas;
		for (uint32_t level = 0; level < cacheAsset->getMipmapCount(); level++)
		{
			const auto& currentMip = mipmapDatas.at(level);
			const uint32_t currentMipSize = (uint32_t)currentMip.size();

			uint32_t mipWidth  = std::max<uint32_t>(cacheAsset->getWidth()  >> level, 1);
			uint32_t mipHeight = std::max<uint32_t>(cacheAsset->getHeight() >> level, 1);

			memcpy((void*)((char*)bufferPtrStart + bufferOffset), currentMip.data(), currentMipSize);

			region.bufferOffset = stageBufferOffset + bufferOffset;
			region.imageSubresource.mipLevel = level;
			region.imageExtent = { mipWidth, mipHeight, 1 };

			copyRegions.push_back(region);

			bufferOffset += currentMipSize;
			bufferSize += currentMipSize;
		}
		ASSERT(uploadSize() >= bufferSize, "Upload size must bigger than buffer size!");

		vkCmdCopyBufferToImage(commandBuffer.cmd, stageBuffer, imageAssetGPU->getImage().getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)copyRegions.size(), copyRegions.data());

		imageAssetGPU->finishUpload(commandBuffer, rangeAllMips);
	}

	std::shared_ptr<AssetTextureCacheLoadTask> AssetTextureCacheLoadTask::build(
		VulkanContext* context, std::shared_ptr<AssetTexture> asset)
	{
		auto* fallbackWhite = context->getEngineTextureWhite().get();

		auto newAsset = std::make_shared<GPUImageAsset>(
			context,
			fallbackWhite,
			asset->getFormat(),
			asset->getNameUtf8(),
			asset->getMipmapCount(),
			asset->getWidth(),
			asset->getHeight(),
			asset->getDepth()
		);

		context->insertGPUAsset(asset->getUUID(), newAsset);

		auto newTask = std::make_shared<AssetTextureCacheLoadTask>(asset);
		newTask->imageAssetGPU = newAsset;

		return newTask;
	}

}