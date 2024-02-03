#pragma once

#include "base.h"
#include "context.h"
#include "log.h"
#include "resource.h"
#include "descriptor.h"

namespace engine
{
	extern void cmdSetPolygonFillMode(VkCommandBuffer commandBuffer, VkPolygonMode polygonMode);

	inline uint32_t getGroupCount(uint32_t threadCount, uint32_t localSize)
	{
		return (threadCount + localSize - 1) / localSize;
	}

	inline void contextSafeRelease(VkPipeline& pipeline)
	{
		if (pipeline != VK_NULL_HANDLE)
		{
			vkDestroyPipeline(getContext()->getDevice(), pipeline, nullptr);
			pipeline = VK_NULL_HANDLE;
		}
	}

	inline void contextSafeRelease(VkPipelineLayout& pipelineLayout)
	{
		if (pipelineLayout != VK_NULL_HANDLE)
		{
			vkDestroyPipelineLayout(getContext()->getDevice(), pipelineLayout, nullptr);
			pipelineLayout = VK_NULL_HANDLE;
		}
	}

	inline VkDescriptorSetLayout getLayoutStatic(VkDescriptorType type, uint32_t descriptorCount = 1)
	{
		DescriptorLayoutCache& cache = getContext()->getDescriptorLayoutCache();

		VkDescriptorSetLayoutBinding binding{};
		binding.binding = 0;
		binding.descriptorType = type;
		binding.descriptorCount = descriptorCount;
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.bindingCount = 1;
		info.pBindings = &binding;

		return cache.createDescriptorLayout(&info);
	}

	struct ScopePerframeMarker
	{
		std::string    name;
		VkCommandBuffer cmd;
		GPUTimestamps* timer = nullptr;

		ScopePerframeMarker(VkCommandBuffer cmdBuf, const std::string& name, const glm::vec4& color, GPUTimestamps* timer)
			: cmd(cmdBuf), name(name), timer(timer)
		{
			getContext()->setPerfMarkerBegin(cmdBuf, name.c_str(), color);
		}

		~ScopePerframeMarker()
		{
			getContext()->setPerfMarkerEnd(cmd);

			if (timer)
			{
				timer->getTimeStamp(cmd, name.c_str());
			}
		}
	};

	struct ColorAttachmentsBuilder
	{
		std::vector<VkRenderingAttachmentInfo> result;

		ColorAttachmentsBuilder& add(VulkanImage& image, VkAttachmentLoadOp loadAction = VK_ATTACHMENT_LOAD_OP_CLEAR, VkAttachmentStoreOp storeAction = VK_ATTACHMENT_STORE_OP_STORE)
		{
			result.push_back(RHIRenderingAttachmentInfo(image.getOrCreateView().view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, loadAction, storeAction, VkClearValue{ .color = {0.0f, 0.0f, 0.0f, 0.0f} }));
			return *this;
		}

		ColorAttachmentsBuilder& add(PoolImageSharedRef image, VkAttachmentLoadOp loadAction = VK_ATTACHMENT_LOAD_OP_CLEAR, VkAttachmentStoreOp storeAction = VK_ATTACHMENT_STORE_OP_STORE)
		{
			return add(image->getImage(), loadAction, storeAction);
		}
	};

	inline auto getDepthAttachment(VulkanImage& image, VkAttachmentLoadOp loadAction = VK_ATTACHMENT_LOAD_OP_CLEAR, VkAttachmentStoreOp storeAction = VK_ATTACHMENT_STORE_OP_STORE, float clearZ = 0.0f)
	{
		return RHIRenderingAttachmentInfo(
			image.getOrCreateView(RHIDefaultImageSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)).view,
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			loadAction,
			storeAction,
			VkClearValue{ .depthStencil = { clearZ, 1} }
		);
	}

	inline auto getDepthAttachment(PoolImageSharedRef image, VkAttachmentLoadOp loadAction = VK_ATTACHMENT_LOAD_OP_CLEAR, VkAttachmentStoreOp storeAction = VK_ATTACHMENT_STORE_OP_STORE, float clearZ = 0.0f)
	{
		return getDepthAttachment(image->getImage(), loadAction, storeAction, clearZ);
	}

	class ScopeRenderCmdObject : NonCopyable
	{
	private:
		std::unique_ptr<ScopePerframeMarker> frameMarker;

	public:
		VkCommandBuffer cmd;
		VkRect2D scissor;
		VkViewport viewport;

		ScopeRenderCmdObject(VkCommandBuffer inCmd, GPUTimestamps* timer, const std::string& name, VulkanImage& rt, const std::vector<VkRenderingAttachmentInfo>& colorAttachments, const VkRenderingAttachmentInfo& depthAttachment)
			: cmd(inCmd)
		{
			frameMarker = std::make_unique<ScopePerframeMarker>(inCmd, name, math::vec4{ 0.6f, 0.2f, 0.4f, 0.8f }, timer);

			const uint32_t renderWidth = rt.getExtent().width;
			const uint32_t renderHeight = rt.getExtent().height;


			const bool bValidDepth = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO == depthAttachment.sType;
			const VkRenderingInfo renderInfo
			{
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
				.renderArea = VkRect2D{.offset {0,0}, .extent { renderWidth, renderHeight}},
				.layerCount = 1,
				.colorAttachmentCount = uint32_t(colorAttachments.size()),
				.pColorAttachments = colorAttachments.size() > 0 ? colorAttachments.data() : nullptr,
				.pDepthAttachment = bValidDepth ? &depthAttachment : nullptr,
			};

			scissor = VkRect2D{ .offset{ 0,0 }, .extent {renderWidth, renderHeight} };
			viewport = VkViewport
			{
				.x = 0.0f,
				.y = (float)renderHeight,
				.width = (float)renderWidth,
				.height = -(float)renderHeight,
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};

			vkCmdBeginRendering(cmd, &renderInfo);
			vkCmdSetScissor(cmd, 0, 1, &scissor);
			vkCmdSetViewport(cmd, 0, 1, &viewport);
			vkCmdSetDepthBias(cmd, 0, 0, 0);
			cmdSetPolygonFillMode(cmd, VK_POLYGON_MODE_FILL);
		}

		~ScopeRenderCmdObject()
		{
			vkCmdEndRendering(cmd);
			frameMarker.reset();
		}
	};

	extern VkResult createAccelerationStructure(
		const VkAccelerationStructureCreateInfoKHR* pCreateInfo, 
		const VkAllocationCallbacks* pAllocator, 
		VkAccelerationStructureKHR* pAccelerationStructure);

	extern void destroyAccelerationStructure(
		VkAccelerationStructureKHR accelerationStructure, 
		const VkAllocationCallbacks* pAllocator);

	extern void cmdBuildAccelerationStructures(
		VkCommandBuffer commandBuffer, 
		uint32_t infoCount, 
		const VkAccelerationStructureBuildGeometryInfoKHR* pInfos, 
		const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos);

	extern VkDeviceAddress getAccelerationStructureDeviceAddress(
		const VkAccelerationStructureDeviceAddressInfoKHR* pInfo);

	extern void getAccelerationStructureBuildSizesKHR(
		VkAccelerationStructureBuildTypeKHR buildType, 
		const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo, 
		const uint32_t* pMaxPrimitiveCounts, 
		VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo);

	extern VkResult buildAccelerationStructuresKHR(
		VkDeferredOperationKHR deferredOperation, 
		uint32_t infoCount, 
		const VkAccelerationStructureBuildGeometryInfoKHR* pInfos, 
		const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos);

	extern void cmdTraceRaysKHR(
		VkCommandBuffer commandBuffer, 
		const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable, 
		const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable, 
		const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable, 
		const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable, 
		uint32_t width, 
		uint32_t height, 
		uint32_t depth);

	extern VkResult createRayTracingPipelinesKHR(
		VkDeferredOperationKHR deferredOperation, 
		VkPipelineCache pipelineCache, 
		uint32_t createInfoCount, 
		const VkRayTracingPipelineCreateInfoKHR* pCreateInfos, 
		const VkAllocationCallbacks* pAllocator, 
		VkPipeline* pPipelines);

	extern VkResult getRayTracingShaderGroupHandlesKHR(
		VkPipeline pipeline, 
		uint32_t firstGroup, 
		uint32_t groupCount, 
		size_t dataSize, 
		void* pData);

	extern void cmdCopyAccelerationStructure(
		VkCommandBuffer commandBuffer, 
		const VkCopyAccelerationStructureInfoKHR* pInfo);

	extern void cmdWriteAccelerationStructuresProperties(
		VkCommandBuffer commandBuffer, 
		uint32_t accelerationStructureCount, 
		const VkAccelerationStructureKHR* pAccelerationStructures, 
		VkQueryType queryType, 
		VkQueryPool queryPool, 
		uint32_t firstQuery);


}