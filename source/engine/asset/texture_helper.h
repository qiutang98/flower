#pragma once

namespace engine
{
	struct AssetTextureBin;
	class AssetTexture;

	enum class ETextureFormat
	{
		R8G8B8A8,
		BC3,

		// Always mean RGB channel in GPU, don't care alpha.
		// But still load 4 channel from file.
		BC1, 

		BC5,
		R8G8, 

		// Load file with greyscale compute, and store in R8.
		Greyscale,
		R8, // Select R component.
		G8, // Select G component.
		B8, // Select B component.
		A8, // Select A component.

		BC4Greyscale, // Greyscale from file store within BC4.
		BC4R8, // Select R component.
		BC4G8, // Select G component.
		BC4B8, // Select B component.
		BC4A8, // Select A component.

		RGBA16Unorm,
		R16Unorm,

		Max,
	};

	extern void buildMipmapData8Bit(
		uint32_t channelCount,
		uint32_t pixelOffsetPerSample,
		unsigned char* srcPixels,
		AssetTextureBin& outBinData,
		bool bSRGB,
		uint32_t mipmapCount,
		uint32_t inWidth,
		uint32_t inHeight,
		float alphaMipmapCutoff = 1.0f);

	extern void mipmapCompressBC(
		AssetTextureBin& inOutBin,
		const AssetTexture& meta);




}