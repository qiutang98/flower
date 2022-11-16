#pragma once

#include "RHICommon.h"
#include "SwapChain.h"
#include "Sampler.h"
#include "Descriptor.h"
#include "Shader.h"
#include "Resource.h"
#include "Bindless.h"
#include "Query.h"
#include "CommandBuffer.h"
#include "AccelerateStructure.h"

namespace Flower
{
	class VulkanContext : NonCopyable
	{
	private:
		void pickupSuitableGpu(const std::vector<const char*>& requestExtens);
		bool isPhysicalDeviceSuitable(const std::vector<const char*>& requestExtens);
		void createLogicDevice(VkPhysicalDeviceFeatures features, void* nextChain, const std::vector<const char*>& requestExtens);

	public:
		SwapchainSupportDetails querySwapchainSupportDetail();
		VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
		int32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	private:
		GLFWwindow* m_window;
		VkSurfaceKHR m_surface = VK_NULL_HANDLE;

		VkInstance m_instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT m_debugUtilsHandle = VK_NULL_HANDLE;
		VkDebugReportCallbackEXT m_debugReportHandle = VK_NULL_HANDLE;
		
		VkDevice m_device = VK_NULL_HANDLE;
		VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
		VkPhysicalDeviceProperties m_physicalDeviceProperties{};
		VkPhysicalDeviceMemoryProperties m_physicalDeviceMemoryProperties{};
		GPUQueuesInfo m_queues;

		VkFormat m_cacheSupportDepthStencilFormat;
		VkFormat m_cacheSupportDepthOnlyFormat;

		VkPhysicalDeviceAccelerationStructureFeaturesKHR m_accelerationStructure;
		VkPhysicalDeviceDescriptorIndexingPropertiesEXT m_descriptorIndexingProperties{};

		Swapchain m_swapchain;
		SamplerCache m_samplerCache;
		VmaAllocator m_vmaAllocator = {};

		DescriptorAllocator m_descriptorAllocator = {};
		DescriptorLayoutCache m_descriptorLayoutCache = {};

		struct PresentContext
		{
			bool bSwapchainChange = false;
			uint32_t imageIndex;
			uint32_t currentFrame = 0;
			std::vector<VkSemaphore> semaphoresImageAvailable;
			std::vector<VkSemaphore> semaphoresRenderFinished;
			std::vector<VkFence> inFlightFences;
			std::vector<VkFence> imagesInFlight;

			void init();
			void release();
		} m_presentContext;

		ShaderCache m_shaderCache;

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

	private:
		void initInstance(const std::vector<const char*>& requiredExtensions, const std::vector<const char*>& requiredLayers);
		void releaseInstance();

		void initDevice(VkPhysicalDeviceFeatures features, const std::vector<const char*>& requestExtens, void* nextChain);
		void releaseDevice();

		void initVMA();
		void releaseVMA();

		void initCommandPool();
		void releaseCommandPool();

	public:
		GLFWwindow* getWindow() { return m_window; }
		VkSurfaceKHR getSurface() const { return m_surface; };

		VkFormat getSupportDepthStencilFormat() const { return m_cacheSupportDepthStencilFormat; }
		VkFormat getSupportDepthOnlyFormat() const { return m_cacheSupportDepthOnlyFormat; }

		uint32_t getMaxMemoryAllocationCount() const { return m_physicalDeviceProperties.limits.maxMemoryAllocationCount; }
	public:
		void init(GLFWwindow* window);
		void release();
		void recreateSwapChain();
	private:
		int currentWidth;
		int currentHeight;
		int lastWidth  = ~0;
		int lastHeight = ~0;

		bool swapchainRebuild();
		
		
	public:
		uint32_t acquireNextPresentImage();
		void present();
		void submit(uint32_t count, VkSubmitInfo* infos);
		void submitNoFence(uint32_t count, VkSubmitInfo* infos);
		void resetFence();

		MulticastDelegate<> onBeforeSwapchainRecreate;
		MulticastDelegate<> onAfterSwapchainRecreate;

	public: 
		// Major graphics queue used for present and ui render. priority 1.0.
		VkQueue getMajorGraphicsQueue() const { return m_majorGraphicsPool.queue; }
		VkCommandPool getMajorGraphicsCommandPool() const { return m_majorGraphicsPool.pool; }
		VkCommandBuffer createMajorGraphicsCommandBuffer();

		// Major compute queue. priority 0.8.
		VkQueue getMajorComputeQueue() const { return m_majorComputePool.queue; }
		VkCommandPool getMajorComputeCommandPool() const { return m_majorComputePool.pool; }

		VkQueue getSecondMajorGraphicsQueue() const { return m_secondMajorGraphicsPool.queue; }
		VkCommandPool getSecondMajorGraphicsCommandPool() const { return m_secondMajorGraphicsPool.pool; }

		// Other queues, priority 0.5.
		const auto& getAsyncCopyCommandPools() const { return m_copyPools; }
		const auto& getAsyncComputeCommandPools() const { return m_computePools; }
		const auto& getAsyncGraphicsCommandPools() const { return m_graphicsPools; }

		VkInstance getInstance() const { return m_instance; }
		VkPhysicalDeviceDescriptorIndexingPropertiesEXT getPhysicalDeviceDescriptorIndexingProperties() const { return m_descriptorIndexingProperties; }

		const GPUQueuesInfo& getGPUQueuesInfo() const { return m_queues; }
		uint32_t getGraphiscFamily() const { return m_queues.graphicsFamily; }
		uint32_t getComputeFamily() const { return m_queues.computeFamily; }
		uint32_t getCopyFamily() const { return m_queues.copyFamily; }

		DescriptorLayoutCache& getDescriptorLayoutCache() { return m_descriptorLayoutCache; }

		DescriptorFactory descriptorFactoryBegin();
		VkPipelineLayout createPipelineLayout(const VkPipelineLayoutCreateInfo& info);

		const uint32_t getCurrentFrameIndex() const { return m_presentContext.currentFrame; }

		Swapchain& getSwapchain() { return m_swapchain; }
		std::vector<VkImageView>& getSwapchainImageViews() { return m_swapchain.getImageViews(); }
		std::vector<VkImage>& getSwapchainImages() { return m_swapchain.getImages(); }

		VkFormat getSwapchainFormat() const { return m_swapchain.getImageFormat(); }
		VkExtent2D getSwapchainExtent() const { return m_swapchain.getExtent(); }

		VkPhysicalDeviceProperties getPhysicalDeviceProperties() const { return m_physicalDeviceProperties; }

		
		VkSemaphore getCurrentFrameWaitSemaphore() const { return m_presentContext.semaphoresImageAvailable[m_presentContext.currentFrame]; }
		VkSemaphore getCurrentFrameFinishSemaphore() const { return m_presentContext.semaphoresRenderFinished[m_presentContext.currentFrame]; }
	};

	namespace RHI
	{
		extern size_t GMaxSwapchainCount;

		extern VkPhysicalDevice GPU;
		extern VkDevice Device;
		extern SamplerCache* SamplerManager;
		extern VmaAllocator VMA;
		extern ShaderCache* ShaderManager;

		// Hdr 10.
		enum DisplayMode
		{
			DISPLAYMODE_SDR,
			DISPLAYMODE_HDR10_2084,
			DISPLAYMODE_HDR10_SCRGB
		};
		extern DisplayMode eDisplayMode;
		extern bool bSupportHDR;
		extern bool bSupportHDR10_2084;
		extern bool bSupportHDR10_SCRGB;
		extern VkHdrMetadataEXT HdrMetadataEXT;

		extern bool bSupportRayTrace;

		inline constexpr auto get = []() { return Singleton<VulkanContext>::get(); };

		extern void setResourceName(VkObjectType objectType, uint64_t handle, const char* name);

		

		extern void setPerfMarkerBegin(VkCommandBuffer cmd_buf, const char* name, const glm::vec4& color);
		extern void setPerfMarkerEnd(VkCommandBuffer cmd_buf);

		struct ScopePerframeMarker
		{
			VkCommandBuffer cmd;
			ScopePerframeMarker(VkCommandBuffer cmdBuf, const char* name, const glm::vec4& color)
				: cmd(cmdBuf)
			{
				setPerfMarkerBegin(cmdBuf, name, color);
			}

			~ScopePerframeMarker()
			{
				setPerfMarkerEnd(cmd);
			}
		};

		extern void executeImmediately(VkCommandPool commandPool, VkQueue queue, std::function<void(VkCommandBuffer cb)>&& func);
		extern void executeImmediatelyMajorGraphics(std::function<void(VkCommandBuffer cb)>&& func);
		
		// Used to compute RHI resource used.
		extern void addGpuResourceMemoryUsed(size_t in);
		extern void minusGpuResourceMemoryUsed(size_t in);
		extern size_t getGpuResourceMemoryUsed();

		extern PFN_vkCmdPushDescriptorSetKHR PushDescriptorSetKHR;
		extern PFN_vkCmdPushDescriptorSetWithTemplateKHR PushDescriptorSetWithTemplateKHR;

		extern PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructure;
		extern PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructure;
		extern PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructures;
		extern PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddress;
		extern PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizes;

		


		// Functions for regular HDR ex: HDR10
		extern PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR GetPhysicalDeviceSurfaceCapabilities2KHR;
		extern PFN_vkGetPhysicalDeviceSurfaceFormats2KHR      GetPhysicalDeviceSurfaceFormats2KHR;
		extern PFN_vkSetHdrMetadataEXT                        SetHdrMetadataEXT;

	};
}