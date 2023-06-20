#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <util/util.h>
#include <util/lru.h>

#include "sampler_cache.h"
#include "shader_cache.h"
#include "descriptor.h"
#include "swapchain.h"
#include "async_upload.h"
#include "gpu_asset.h"
#include "render_texture_pool.h"
#include "pass.h"
#include "dynamic_uniform_buffer.h"
#include "ssbo_buffers.h"

namespace engine
{


	enum class EBuiltinEngineAsset
	{
		Texture_Min,
		Texture_White, // 255, 255, 255, 255
		Texture_Grey,  // 128, 128, 128, 255
		Texture_Black, // 0, 0, 0, 255
		Texture_Translucent, // 0, 0, 0, 0
		Texture_Normal, // 125, 130, 255, 0
		Texture_Specular, // 255, 255, 0, 0 .r AO, .g roughness, .b metal
		Texture_CloudWeather,
		Texture_CurlNoise,
		Texture_Noise,
		Texture_Sky3d,
		Texture_Max,

		StaticMesh_Min,
		StaticMesh_Box, // 
		StaticMesh_Sphere, // 
		StaticMesh_Max,
	};

	class VulkanContext final : public IRuntimeModule
	{
	public:
		VulkanContext(Engine* engine) : IRuntimeModule(engine) { }
		~VulkanContext() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual void release() override;

		struct GPUCommandPool
		{
			VkQueue queue = VK_NULL_HANDLE;
			VkCommandPool pool = VK_NULL_HANDLE;
		};

		void setResourceName(VkObjectType objectType, uint64_t handle, const char* name) const;
		void setPerfMarkerBegin(VkCommandBuffer cmdBuf, const char* name, const math::vec4& color) const;
		void setPerfMarkerEnd(VkCommandBuffer cmdBuf) const;

		const VkPhysicalDeviceMemoryProperties& getPhysicalDeviceMemoryProperties() const { return m_memoryProperties; }
		const VkPhysicalDeviceDescriptorIndexingPropertiesEXT& getPhysicalDeviceDescriptorIndexingProperties() const { return m_descriptorIndexingProperties; }
		const VkPhysicalDeviceProperties& getPhysicalDeviceProperties() const { return m_deviceProperties; }

		VkDevice getDevice() const { return m_device; }
		VkPhysicalDevice getGPU() const { return m_gpu; }
		VmaAllocator getVMA() const { return m_vma; }
		VkInstance getInstance() const { return m_instance; }
		VkSurfaceKHR getSurface() const { return m_surface; }

		// Major graphics queue with 1.0 priority.
		VkQueue getMajorGraphicsQueue() const { return m_majorGraphicsPool.queue; }

		// Major graphics command pool with 1.0 priority.
		VkCommandPool getMajorGraphicsCommandPool() const { return m_majorGraphicsPool.pool; }

		// Create command buffer from major graphics queue and command pool.
		[[nodiscard]] VkCommandBuffer createMajorGraphicsCommandBuffer();
		void freeMajorGraphicsCommandBuffer(VkCommandBuffer cmd);

		// Second major graphics queue with 0.8 priority.
		VkQueue getSecondMajorGraphicsQueue() const { return m_secondMajorGraphicsPool.queue; }

		// Second major graphics command pool with 0.8 priority.
		VkCommandPool getSecondMajorGraphicsCommandPool() const { return m_secondMajorGraphicsPool.pool; }

		// Major compute queue with priority 0.8.
		VkQueue getMajorComputeQueue() const { return m_majorComputePool.queue; }

		// Major compute command pool with priority 0.8.
		VkCommandPool getMajorComputeCommandPool() const { return m_majorComputePool.pool; }

		// Normal copy command pools with 0.5 priority.
		const auto& getNormalCopyCommandPools() const { return m_copyPools; }

		// Normal compute command pools with 0.5 priority.
		const auto& getNormalComputeCommandPools() const { return m_computePools; }

		// Normal graphics command pools with 0.5 priority.
		const auto& getNormalGraphicsCommandPools() const { return m_graphicsPools; }

		const auto& getGPUQueuesInfo() const { return m_queues; }
		uint32_t getGraphiscFamily() const { return m_queues.graphicsFamily; }
		uint32_t getComputeFamily() const { return m_queues.computeFamily; }
		uint32_t getCopyFamily() const { return m_queues.copyFamily; }

		void executeImmediately(VkCommandPool commandPool, VkQueue queue, std::function<void(VkCommandBuffer cb)>&& func) const;
		void executeImmediatelyMajorGraphics(std::function<void(VkCommandBuffer cb)>&& func) const;

		SamplerCache& getSamplerCache() { return m_samplerCache; }
		ShaderCache& getShaderCache() { return m_shaderCache; }

		// Context owned bindless sampler.
		BindlessSampler& getBindlessSampler() { return m_bindlessSampler; }
		const BindlessSampler& getBindlessSampler() const { return m_bindlessSampler; }
		VkDescriptorSet getBindlessSamplerSet() const { return m_bindlessSampler.getSet(); }
		VkDescriptorSetLayout getBindlessSamplerSetLayout() const { return m_bindlessSampler.getSetLayout(); }

		// Context owned bindless texture.
		BindlessTexture& getBindlessTexture() { return m_bindlessTexture; }
		const BindlessTexture& getBindlessTexture() const { return m_bindlessTexture; }
		VkDescriptorSet getBindlessTextureSet() const { return m_bindlessTexture.getSet(); }
		VkDescriptorSetLayout getBindlessTextureSetLayout() const { return m_bindlessTexture.getSetLayout(); }


		// Context owned bindless texture.
		BindlessStorageBuffer& getBindlessSSBOs() { return m_bindlessStorageBuffer; }
		const BindlessStorageBuffer& getBindlessSSBOs() const { return m_bindlessStorageBuffer; }
		VkDescriptorSet getBindlessSSBOSet() const { return m_bindlessStorageBuffer.getSet(); }
		VkDescriptorSetLayout getBindlessSSBOSetLayout() const { return m_bindlessStorageBuffer.getSetLayout(); }

		const auto& getSwapchain() const { return m_swapchain; }

		GLFWwindow* getWindow() const { return m_window; }

		VkFormat getSupportedBestDepthOnlyFormat() const { return m_cacheSupportDepthOnlyFormat; }
		VkFormat getSupportedBestDepthStencilFormat() const { return m_cacheSupportDepthStencilFormat; }

		// Swapchain using back buffer format type.
		const EBackBufferFormat& getBackbufferFormatType() const { return m_backbufferFormat; }
		uint32_t getBackBufferCount() const { return m_swapchain.getBackbufferCount(); }

		DescriptorLayoutCache& getDescriptorLayoutCache() { return m_descriptorLayoutCache; }
		const DescriptorLayoutCache& getDescriptorLayoutCache() const { return m_descriptorLayoutCache; }

		DescriptorFactory descriptorFactoryBegin();
		VkPipelineLayout createPipelineLayout(const VkPipelineLayoutCreateInfo& info);

		void recreateSwapChain();

		MulticastDelegate<> onBeforeSwapchainRecreate;
		MulticastDelegate<> onAfterSwapchainRecreate;

		uint32_t acquireNextPresentImage();
		void present();

		// Submit to major graphics queue with reset sync fence.
		void submit(uint32_t count, VkSubmitInfo* infos);

		// Submit to major graphics queue with special fence.
		void submit(uint32_t count, VkSubmitInfo* infos, VkFence fence);

		// Submit to major graphics queue with special fence.
		void submit(uint32_t count, const RHISubmitInfo* infos, VkFence fence);

		// Submit to major graphics queue without reset sync fence.
		void submitNoFence(uint32_t count, VkSubmitInfo* infos);

		// Just reset major graphics queue sync fence.
		void resetFence();

		VkSemaphore getCurrentFrameWaitSemaphore() const { return m_presentContext.semaphoresImageAvailable[m_presentContext.currentFrame]; }
		VkSemaphore getCurrentFrameFinishSemaphore() const { return m_presentContext.semaphoresRenderFinished[m_presentContext.currentFrame]; }

		void waitDeviceIdle() const;

		AsyncUploaderManager& getAsyncUploader() { return *m_uploader; }

		

		static UUID getBuiltEngineAssetUUID(EBuiltinEngineAsset type);
		bool isEngineAssetExist(const UUID& uuid) { return m_engineAssets.contains(uuid); }
		bool isLRUAssetExist(const UUID& uuid) { return m_lru->contain(uuid); }
		void insertEngineAsset(const UUID& uuid, std::shared_ptr<LRUAssetInterface> asset);
		void insertLRUAsset(const UUID& uuid, std::shared_ptr<LRUAssetInterface> asset);
		enum class EContextState
		{
			init,
			ticking,
			release,
		};

		// Context start to kill all? Will set true after call release.
		EContextState getState() const { return m_state; }
		bool isReleaseing() const { return m_state == EContextState::release; }

		std::shared_ptr<GPUImageAsset> getEngineTextureWhite() const { return std::dynamic_pointer_cast<GPUImageAsset>(getEngineAsset(EBuiltinEngineAsset::Texture_White)); }
		std::shared_ptr<GPUImageAsset> getEngineTextureNormal() const { return std::dynamic_pointer_cast<GPUImageAsset>(getEngineAsset(EBuiltinEngineAsset::Texture_Normal)); }
		std::shared_ptr<GPUImageAsset> getEngineTextureSpecular() const { return std::dynamic_pointer_cast<GPUImageAsset>(getEngineAsset(EBuiltinEngineAsset::Texture_Specular)); }
		std::shared_ptr<GPUImageAsset> getEngineTextureTranslucent() const { return std::dynamic_pointer_cast<GPUImageAsset>(getEngineAsset(EBuiltinEngineAsset::Texture_Translucent)); }
		std::shared_ptr<GPUStaticMeshAsset> getEngineStaticMeshBox() const { return std::dynamic_pointer_cast<GPUStaticMeshAsset>(getEngineAsset(EBuiltinEngineAsset::StaticMesh_Box)); }
		std::shared_ptr<LRUAssetInterface> getEngineAsset(EBuiltinEngineAsset asset) const;
		std::shared_ptr<LRUAssetInterface> getEngineAsset(const UUID& uuid) const;
		void insertGPUAsset(const UUID& uuid, std::shared_ptr<LRUAssetInterface> asset){ m_lru->insert(uuid, asset); }

		const StaticMeshRenderBounds& getEngineMeshRenderBounds(const UUID& uuid) const { return m_engineMeshBounds.at(uuid); }


		std::shared_ptr<GPUStaticMeshAsset> getOrCreateStaticMeshAsset(const UUID& uuid);
		std::shared_ptr<GPUImageAsset> getOrCreateTextureAsset(const UUID& uuid);
		const auto& getLRU() const { return m_lru; }

		const auto& getPasses() const { return *m_passCollector; }
		auto& getPasses() { return *m_passCollector; }

		const auto& getRenderTargetPools() const { return *m_rtPool; }
		auto& getRenderTargetPools() { return *m_rtPool; }

		const auto& getGraphicsCardState() const { return m_graphicsSupportStates; }

		const auto& getBufferParameters() const { return *m_bufferParameters; }
		auto& getBufferParameters() { return *m_bufferParameters; }

		const auto& getDynamicUniformBuffers() const { return *m_dynamicUniformBuffer; }
		auto& getDynamicUniformBuffers() { return *m_dynamicUniformBuffer; }

		const auto& getPhysicalDeviceAccelerationStructurePropertiesKHR() const { return m_accelerationStructureProperties; }

		void pushDescriptorSet(
			VkCommandBuffer commandBuffer, 
			VkPipelineBindPoint pipelineBindPoint, 
			VkPipelineLayout layout, 
			uint32_t set, 
			uint32_t descriptorWriteCount, 
			const VkWriteDescriptorSet* pDescriptorWrites);

		void pushGpuResourceAsPendingKill(std::shared_ptr<GpuResource> asset);
	private:
		void initInstance();
		void destroyInstance();

		void selectGPU();
		void queryGPUInfo();

		void initDeviceAndQueue();
		void destroyDevice();

		void initVMA();
		void destroyVMA();

		void initCommandPools();
		void destroyCommandPools();

		void initPresentContext();
		void destroyPresentContext();

		void quertDepthFormatSupportState();

		void initEngineAssets();

	private:
		EContextState m_state = EContextState::init;

		// Instance of vulkan.
		VkInstance m_instance = VK_NULL_HANDLE;

		// Vulkan device.
		VkDevice m_device = VK_NULL_HANDLE;

		// Using gpu.
		VkPhysicalDevice m_gpu = VK_NULL_HANDLE;

		// AMD's vulkan memory allocator.
		VmaAllocator m_vma = VK_NULL_HANDLE;

		VkDebugUtilsMessengerEXT m_debugUtilsHandle = VK_NULL_HANDLE;
		
		// Windows handle and surface handle, it can be nullptr when application run with console.
		GLFWwindow*  m_window = nullptr;
		VkSurfaceKHR m_surface = VK_NULL_HANDLE;

		// Cache device infos.
		VkPhysicalDeviceMemoryProperties   m_memoryProperties;
		VkPhysicalDeviceProperties         m_deviceProperties;
		VkPhysicalDeviceProperties2        m_deviceProperties2;
		VkPhysicalDeviceSubgroupProperties m_subgroupProperties;
		VkPhysicalDeviceDescriptorIndexingPropertiesEXT m_descriptorIndexingProperties;
		VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelerationStructureProperties;

		struct DeviceSupportStates
		{
			bool bSupportHDR = true;

			bool bSupportRaytrace = true;
		} m_graphicsSupportStates;

		struct GPUQueuesInfo
		{
			uint32_t graphicsFamily = ~0;
			uint32_t copyFamily = ~0;
			uint32_t computeFamily = ~0;

			std::vector<VkQueue> computeQueues;  // Priority: #0 0.8f, #1...#n 0.5f
			std::vector<VkQueue> copyQueues;     // Priority: #0...#n 0.5f
			std::vector<VkQueue> graphcisQueues; // Priority: #0 1.0f, #1 0.8f, #2...#n 0.5f
		} m_queues;

		// Shader cache.
		ShaderCache m_shaderCache;

		// Sampler cache.
		SamplerCache m_samplerCache;

		// Descriptor allocator.
		DescriptorAllocator m_descriptorAllocator;
		
		// Descriptor layout cache.
		DescriptorLayoutCache m_descriptorLayoutCache;

		// Windows swapchain.
		Swapchain m_swapchain;

		// Surface present used backbuffer format.
		EBackBufferFormat m_backbufferFormat = EBackBufferFormat::SRGB_NonLinear;

		// Sampler bindless.
		BindlessSampler m_bindlessSampler;

		// Texture bindless.
		BindlessTexture m_bindlessTexture;

		// Storage buffer bindless.
		BindlessStorageBuffer m_bindlessStorageBuffer;

		// Major graphics queue with priority 1.0f.
		GPUCommandPool m_majorGraphicsPool;

		// Major compute queue with priority 0.8f. Use for AsyncScheduler.
		GPUCommandPool m_majorComputePool;

		// Second major queue with priority 0.8f. Use fir Async Scheduler.
		GPUCommandPool m_secondMajorGraphicsPool;

		// Other command pool with priority 0.5f.
		std::vector<GPUCommandPool> m_graphicsPools;
		std::vector<GPUCommandPool> m_computePools;

		// Copy pool used for async uploader.
		std::vector<GPUCommandPool> m_copyPools;

		struct PresentContext
		{
			bool bSwapchainChange = false;
			uint32_t imageIndex;
			uint32_t currentFrame = 0;
			std::vector<VkSemaphore> semaphoresImageAvailable;
			std::vector<VkSemaphore> semaphoresRenderFinished;
			std::vector<VkFence> inFlightFences;
			std::vector<VkFence> imagesInFlight;
		} m_presentContext;

		struct SwapchainRebuildContext
		{
			int currentWidth;
			int currentHeight;
			int lastWidth = ~0;
			int lastHeight = ~0;
		} m_swapchainRebuildContext;

		// Cache support depth formats.
		VkFormat m_cacheSupportDepthStencilFormat = VK_FORMAT_UNDEFINED;
		VkFormat m_cacheSupportDepthOnlyFormat = VK_FORMAT_UNDEFINED;

		std::unique_ptr<AsyncUploaderManager> m_uploader;

		std::unique_ptr<LRUAssetCache> m_lru;
		std::unordered_map<UUID, std::shared_ptr<LRUAssetInterface>> m_engineAssets;
		std::unordered_map<UUID, StaticMeshRenderBounds> m_engineMeshBounds;

		// Render texture pool.
		std::unique_ptr<RenderTexturePool> m_rtPool;

		std::unique_ptr<PassCollector> m_passCollector;

		std::unique_ptr<DynamicUniformBuffer> m_dynamicUniformBuffer;

		std::unique_ptr<BufferParameterPool> m_bufferParameters;

		std::vector<std::vector<std::shared_ptr<GpuResource>>> m_gpuResourcePending;
	};

	extern VulkanContext* getContext();
}