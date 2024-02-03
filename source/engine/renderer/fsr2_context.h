#pragma once

#include "ffx-fsr2-api/ffx_fsr2.h"
#include "ffx-fsr2-api/vk/ffx_fsr2_vk.h"
#include "render_functions.h"

namespace engine
{
	enum class EReactiveMaskMode
	{
		Off = 0,  // Nothing written to the reactive mask
		On,       // Particles written to the reactive mask

		Count
	};

	struct FfxConfig
	{
		// Basic config from engine.
		float jitterX;
		float jitterY;

		uint32_t renderWidth;
		uint32_t renderHeight;

		float nearPlane;
		float farPlane;
		float fovV;
		float deltaTime;

		//////////////////////////////////////////////////////

		// 
		EReactiveMaskMode reactiveMaskMode = EReactiveMaskMode::On;
		float fFsr2AutoReactiveScale = 1.f;
		float fFsr2AutoReactiveThreshold = 0.2f;
		float fFsr2AutoReactiveBinaryValue = 0.9f;
		bool  bFsr2AutoReactiveTonemap = true;
		bool  bFsr2AutoReactiveInverseTonemap = false;
		bool  bFsr2AutoReactiveThreshold = true;
		bool  bFsr2AutoReactiveUseMax = true;

		// Translucency composition mask.
		bool bCompositionMask = true;

		// Camera reset or camera cut state on current frame?
		bool bCameraReset;

		// RCAS
		bool bUseRcas = true;
		float sharpening = 0.8f;

		float lodTextureBasicBias = 0.0f;
	};

	struct FfxUpscaleSetup
	{
		PoolImageSharedRef unresolvedColorResource = nullptr;                 // input0
		PoolImageSharedRef motionvectorResource = nullptr;                    // input1
		PoolImageSharedRef depthbufferResource = nullptr;                     // input2
		PoolImageSharedRef reactiveMapResource = nullptr;                     // input3
		PoolImageSharedRef transparencyAndCompositionResource = nullptr;      // input4
		PoolImageSharedRef opaqueOnlyColorResource = nullptr;                 // input5
		PoolImageSharedRef resolvedColorResource = nullptr;                   // output

		VkImageView unresolvedColorResourceView = VK_NULL_HANDLE;            // input0
		VkImageView motionvectorResourceView = VK_NULL_HANDLE;               // input1
		VkImageView depthbufferResourceView = VK_NULL_HANDLE;                // input2
		VkImageView reactiveMapResourceView = VK_NULL_HANDLE;                // input3
		VkImageView transparencyAndCompositionResourceView = VK_NULL_HANDLE; // input4
		VkImageView opaqueOnlyColorResourceView = VK_NULL_HANDLE;            // input5
		VkImageView resolvedColorResourceView = VK_NULL_HANDLE;              // output
	};


	// FSR2 Context for renderer.
	class FSR2Context : NonCopyable
	{
	public:
		void onCreateWindowSizeDependentResources(
			uint32_t renderWidth,
			uint32_t renderHeight,
			uint32_t displayWidth,
			uint32_t displayHeight);

		void onDestroyWindowSizeDependentResources();

		void generateReactiveMask(VkCommandBuffer pCommandList, const FfxUpscaleSetup& cameraSetup);
		void draw(VkCommandBuffer commandBuffer, const FfxUpscaleSetup& cameraSetup);

		// Call release when destroy.
		~FSR2Context()
		{
			onDestroyWindowSizeDependentResources();
		}

		FfxConfig config;

	private:
		FfxFsr2ContextDescription m_initializationParameters = {};
		FfxFsr2Context m_context;
		float m_memoryUsageInMegabytes = 0.0f;
	};
}