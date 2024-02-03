#pragma once

#include "base.h"
#include "bindless.h"
#include "descriptor.h"
#include "sampler.h"
#include "uploader.h"
#include "uniform.h"
#include "swapchain.h"
#include "pool.h"
#include "shader.h"
#include "query.h"

#include <vma/vk_mem_alloc.h>
#include "gpu_asset.h"
#include "pass.h"
#include <profile/profile.h>
namespace engine
{
	enum class EBuiltinTextures
	{
		min = 0,

		white,         // 255, 255, 255, 255
		grey,          // 128, 128, 128, 255
		black,         //   0,   0,   0, 255
		translucent,   //   0,   0,   0,   0
		normal,        // 128, 128, 255, 255
		metalRoughness,// 255, 255,   0,   0

		cloudWeather,
		cloudNoise,
		curlNoise,

		sceneIcon,
		materialIcon,

		max
	};

	enum class EBuiltinStaticMeshes
	{
		min = 0,

		box,      // 1x1x1 box.
		sphere,   // Radius 0.5 Sphere.
		plane,    // 10x10 plane grid.

		max
	};

	extern UUID getBuiltinTexturesUUID(EBuiltinTextures value);
	extern UUID getBuiltinStaticMeshUUID(EBuiltinStaticMeshes value);

	class VulkanContext : public IRuntimeModule
	{
	public:
		struct SupportStates
		{
			bool bSupportHDR      = true;
			bool bSupportRaytrace = true;
		};

		struct GPUCommandPools
		{
			// Major graphics queue with priority 1.0f.
			GPUCommandPool majorGraphics;

			// 0.8f priority queue.
			GPUCommandPool majorCompute;
			GPUCommandPool secondMajorGraphics;

			// Other command pool with priority 0.5f.
			std::vector<GPUCommandPool> graphics;
			std::vector<GPUCommandPool> computes;
			std::vector<GPUCommandPool> copies;
		};

		// Cache GPU queue infos.
		struct GPUQueuesInfo
		{
			uint32_t graphicsFamily = ~0;
			uint32_t copyFamily = ~0;
			uint32_t computeFamily = ~0;

			// Exist three type priority: 1.0f, 0.8f, 0.5f.
			//
			std::vector<VkQueue> computeQueues;  // Priority: #0 is 0.8f, #1...#n are 0.5f.
			std::vector<VkQueue> copyQueues;     // Priority: #0...#n are 0.5f.
			std::vector<VkQueue> graphcisQueues; // Priority: #0 is 1.0f, #1 is 0.8f, #2...#n are 0.5f.
		};

		explicit VulkanContext(Engine* engine) : IRuntimeModule(engine) { }
		virtual ~VulkanContext() = default;

		virtual void registerCheck(Engine* engine) override;
		virtual bool init() override;
		virtual bool tick(const RuntimeModuleTickData& tickData) override;
		virtual bool beforeRelease() override;
		virtual bool release() override;

	public:
		VkSurfaceKHR getSurface() const { return m_surface; }
		GLFWwindow* getWindow() const { return m_window; }
		const auto& getSwapchain() const { return m_swapchain; }
		uint32_t getBackBufferCount() const { return m_swapchain.getBackbufferCount(); }
		const EBackBufferFormat& getBackbufferFormatType() const { return m_backbufferFormat; }
		void recreateSwapChain();
		MulticastDelegate<> onBeforeSwapchainRecreate;
		MulticastDelegate<> onAfterSwapchainRecreate;

		int32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
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


	public:
		VkDevice         getDevice()   const { return m_device; }
		VkPhysicalDevice getGPU()      const { return m_gpu; }

		// Global VMA, common use for no temporal(frame frequency destroy) asset.
		VmaAllocator     getVMABuffer()      const { return m_vmaBuffer; }
		VmaAllocator     getVMAImage()      const { return m_vmaImage; }
		VmaAllocator     getVMAFrequencyBuffer()      const { return m_vmaFrequencyDestroyBuffer; }
		VmaAllocator     getVMAFrequencyImage() const { return m_vmaFrequencyDestroyImage; }
		VkInstance       getInstance() const { return m_instance; }

		void waitDeviceIdle() const;

		// Get queue familys.
		const GPUQueuesInfo& getGPUQueuesInfo() const { return m_queues; }

		uint32_t getGraphiscFamily() const { return m_queues.graphicsFamily; }
		uint32_t getComputeFamily()  const { return m_queues.computeFamily; }
		uint32_t getCopyFamily()     const { return m_queues.copyFamily; }

		// Major graphics queue or command pool with 1.0f priority.
		VkQueue getMajorGraphicsQueue() const { return m_commandPools.majorGraphics.queue; }
		VkCommandPool getMajorGraphicsCommandPool() const { return m_commandPools.majorGraphics.pool; }

		// Create or free command buffer from major graphics queue and command pool.
		[[nodiscard]] VkCommandBuffer createMajorGraphicsCommandBuffer();
		void freeMajorGraphicsCommandBuffer(VkCommandBuffer cmd);

		// Second major graphics queue or command pool with 0.8f priority.
		VkQueue getSecondMajorGraphicsQueue() const { return m_commandPools.secondMajorGraphics.queue; }
		VkCommandPool getSecondMajorGraphicsCommandPool() const { return m_commandPools.secondMajorGraphics.pool; }

		// Major compute queue or command pool with priority 0.8f.
		VkQueue getMajorComputeQueue() const { return m_commandPools.majorCompute.queue; }
		VkCommandPool getMajorComputeCommandPool() const { return m_commandPools.majorCompute.pool; }

		// Normal command pools with 0.5 priority.
		const auto& getNormalCopyCommandPools() const { return m_commandPools.copies; }
		const auto& getNormalComputeCommandPools() const { return m_commandPools.computes; }
		const auto& getNormalGraphicsCommandPools() const { return m_commandPools.graphics; }

		void executeImmediately(VkCommandPool commandPool, VkQueue queue, std::function<void(VkCommandBuffer cb)>&& func) const;
		void executeImmediatelyMajorGraphics(std::function<void(VkCommandBuffer cb)>&& func) const;

		void setResourceName(VkObjectType objectType, uint64_t handle, const char* name) const;
		void setPerfMarkerBegin(VkCommandBuffer cmdBuf, const char* name, const math::vec4& color) const;
		void setPerfMarkerEnd(VkCommandBuffer cmdBuf) const;

		const VkPhysicalDeviceMemoryProperties& getPhysicalDeviceMemoryProperties() const { return m_memoryProperties; }
		const VkPhysicalDeviceDescriptorIndexingPropertiesEXT& getPhysicalDeviceDescriptorIndexingProperties() const { return m_descriptorIndexingProperties; }
		const VkPhysicalDeviceProperties& getPhysicalDeviceProperties() const { return m_deviceProperties; }

		const auto& getASProperties() const{ return m_accelerationStructureProperties; }

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

		DescriptorFactory descriptorFactoryBegin();
		DescriptorLayoutCache& getDescriptorLayoutCache() { return m_descriptorLayoutCache; }
		const DescriptorLayoutCache& getDescriptorLayoutCache() const { return m_descriptorLayoutCache; }

		SamplerCache& getSamplerCache() { return m_samplerCache; }
		AsyncUploaderManager& getAsyncUploader() { return *m_uploader; }
		const auto& getDynamicUniformBuffers() const { return *m_dynamicUniformBuffer; }
		auto& getDynamicUniformBuffers() { return *m_dynamicUniformBuffer; }
		const auto& getRenderTargetPools() const { return *m_rtPool; }
		auto& getRenderTargetPools() { return *m_rtPool; }
		const auto& getBufferParameters() const { return *m_bufferParameters; }
		auto& getBufferParameters() { return *m_bufferParameters; }
		const auto& getShaderCache() const { return *m_shaderCache; }
		auto& getShaderCache() { return *m_shaderCache; }
		const auto& getPasses() const { return *m_passCollector; }
		auto& getPasses() { return *m_passCollector; }

		void pushDescriptorSet(
			VkCommandBuffer commandBuffer,
			VkPipelineBindPoint pipelineBindPoint,
			VkPipelineLayout layout,
			uint32_t set,
			uint32_t descriptorWriteCount,
			const VkWriteDescriptorSet* pDescriptorWrites);

		const auto& getLRU() const { return m_lru; }
		bool isLRUAssetExist(const UUID& uuid) { return m_lru->contain(uuid); }
		void insertLRUAsset(const UUID& uuid, std::shared_ptr<StorageInterface> asset) { m_lru->insert(uuid, asset); }


		bool isBuiltinAssetExist(const UUID& uuid) const { return m_builtinAssets.contains(uuid); }
		void insertBuiltinAsset(const UUID& uuid, std::shared_ptr<UploadAssetInterface> asset);

		std::shared_ptr<UploadAssetInterface> getBuiltinAsset(const UUID& uuid) const;

		auto getBuiltinTexture(EBuiltinTextures asset) const
		{
			return std::dynamic_pointer_cast<GPUImageAsset>(
				getBuiltinAsset(getBuiltinTexturesUUID(asset)));
		}

		auto getBuiltinTexture(const UUID& id) const
		{
			return std::dynamic_pointer_cast<GPUImageAsset>(getBuiltinAsset(id));
		}

		auto getBuiltinStaticMesh(EBuiltinStaticMeshes asset) const
		{
			return std::dynamic_pointer_cast<GPUStaticMeshAsset>(
				getBuiltinAsset(getBuiltinStaticMeshUUID(asset)));
		}

		auto getBuiltinStaticMesh(const UUID& id) const
		{
			return std::dynamic_pointer_cast<GPUStaticMeshAsset>(getBuiltinAsset(id));
		}

		auto getBuiltinTextureWhite() const { return getBuiltinTexture(EBuiltinTextures::white); }
		auto getBuiltinTextureNormal() const { return getBuiltinTexture(EBuiltinTextures::normal); }
		auto getBuiltinTextureMetalRoughness() const { return getBuiltinTexture(EBuiltinTextures::metalRoughness); }
		auto getBuiltinTextureTranslucent() const { return getBuiltinTexture(EBuiltinTextures::translucent); }

		auto getBuiltinStaticMeshBox() const { return getBuiltinStaticMesh(EBuiltinStaticMeshes::box); }

		const auto& getGraphicsState() const { return m_graphicsSupportStates; }

		VkPipelineLayout createPipelineLayout(const VkPipelineLayoutCreateInfo& info);

	private:
		void initInstance();
		void destroyInstance();

		void selectGpuAndQueryGpuInfos();

		void initDeviceAndQueue();
		void destroyDevice();

		void initVMA();
		void destroyVMA();

		void initCommandPools();
		void destroyCommandPools();

		void initPresentContext();
		void destroyPresentContext();

		void initBuiltinAssets();
		void destroyBuiltinAsset();

	protected:
		// Vulkan instance.
		VkInstance m_instance = VK_NULL_HANDLE;

		// Vulkan debug utils handle.
		VkDebugUtilsMessengerEXT m_debugUtilsHandle = VK_NULL_HANDLE;

		// Vulkan device.
		VkDevice m_device = VK_NULL_HANDLE;

		// Using gpu.
		VkPhysicalDevice m_gpu = VK_NULL_HANDLE;

		// AMD's vulkan memory allocator.
		VmaAllocator m_vmaBuffer = VK_NULL_HANDLE;
		VmaAllocator m_vmaImage = VK_NULL_HANDLE;
		VmaAllocator m_vmaFrequencyDestroyBuffer = VK_NULL_HANDLE;
		VmaAllocator m_vmaFrequencyDestroyImage = VK_NULL_HANDLE;

		// Cache device infos.
		VkPhysicalDeviceMemoryProperties   m_memoryProperties;
		VkPhysicalDeviceProperties         m_deviceProperties;
		VkPhysicalDeviceProperties2        m_deviceProperties2;
		VkPhysicalDeviceSubgroupProperties m_subgroupProperties;
		VkPhysicalDeviceDescriptorIndexingPropertiesEXT m_descriptorIndexingProperties;
		VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelerationStructureProperties;

		// Cache device support state.
		SupportStates m_graphicsSupportStates;

		// Queue and command pools.
		GPUQueuesInfo m_queues;
		GPUCommandPools m_commandPools;

		// Descriptor allocator and layout cache.
		DescriptorAllocator m_descriptorAllocator;
		DescriptorLayoutCache m_descriptorLayoutCache;

		// Bindless set.
		BindlessSampler m_bindlessSampler;
		BindlessTexture m_bindlessTexture;
		BindlessStorageBuffer m_bindlessStorageBuffer;

		SamplerCache                          m_samplerCache;
		std::unique_ptr<AsyncUploaderManager> m_uploader;
		std::unique_ptr<DynamicUniformBuffer> m_dynamicUniformBuffer;
		std::unique_ptr<RenderTexturePool>    m_rtPool;
		std::unique_ptr<BufferParameterPool>  m_bufferParameters;
		std::unique_ptr<ShaderCache>          m_shaderCache;
		std::unique_ptr<LRUAssetCache>        m_lru;
		std::unique_ptr<PassCollector>        m_passCollector;

		// Engine builtin assets.
		std::unordered_map<UUID, std::shared_ptr<UploadAssetInterface>> m_builtinAssets;

	protected:
		// Windows handle and surface handle, it can be nullptr when application run with console.
		GLFWwindow*       m_window  = nullptr;
		VkSurfaceKHR      m_surface = VK_NULL_HANDLE;
		Swapchain         m_swapchain;
		EBackBufferFormat m_backbufferFormat = EBackBufferFormat::SRGB_NonLinear;

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
	};

	extern VulkanContext* getContext();
	extern VkDevice getDevice();
}