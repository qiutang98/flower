#include "Pch.h"
#include "ImGuiPass.h"

namespace Flower
{
	void setupVulkanInitInfo(ImGui_ImplVulkan_InitInfo* inout, VkDescriptorPool pool)
	{
		inout->Instance = RHI::get()->getInstance();
		inout->PhysicalDevice = RHI::GPU;
		inout->Device = RHI::Device;
		inout->QueueFamily = RHI::get()->getGraphiscFamily();
		inout->Queue = RHI::get()->getMajorGraphicsQueue();
		inout->PipelineCache = VK_NULL_HANDLE;
		inout->DescriptorPool = pool;
		inout->Allocator = nullptr;
		inout->MinImageCount = (uint32_t)RHI::get()->getSwapchainImageViews().size();
		inout->ImageCount = inout->MinImageCount;
		inout->MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		inout->CheckVkResultFn = RHICheck;
		inout->Subpass = 0;
	}

	//


	void ImguiPass::renderpassBuild()
	{
		vkDeviceWaitIdle(RHI::Device);

		// Create the Render Pass
		if(this->m_renderResource.renderPass == VK_NULL_HANDLE)
		{
			VkAttachmentDescription attachment = {};
			attachment.format = m_drawUIFormat;
			attachment.samples = VK_SAMPLE_COUNT_1_BIT;
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // Blit to back buffer.

			VkAttachmentReference color_attachment = {};
			color_attachment.attachment = 0;
			color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass = {};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &color_attachment;

			VkSubpassDependency dependency = {};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			VkRenderPassCreateInfo info = {};

			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			info.attachmentCount = 1;
			info.pAttachments = &attachment;
			info.subpassCount = 1;
			info.pSubpasses = &subpass;
			info.dependencyCount = 1;
			info.pDependencies = &dependency;
			RHICheck(vkCreateRenderPass(RHI::Device, &info, nullptr, &this->m_renderResource.renderPass));
		}

		{
			VkImageCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			info.flags = 0;
			info.imageType = VK_IMAGE_TYPE_2D;
			info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
			info.extent.width = RHI::get()->getSwapchainExtent().width;
			info.extent.height = RHI::get()->getSwapchainExtent().height;
			info.extent.depth = 1;
			info.mipLevels = 1;
			info.samples = VK_SAMPLE_COUNT_1_BIT;
			info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			info.arrayLayers = 1;
			info.tiling = VK_IMAGE_TILING_OPTIMAL;
			info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			info.queueFamilyIndexCount = 0;
			info.pQueueFamilyIndices = nullptr;
			info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			m_drawUIImages = VulkanImage::create("DrawUIImage", info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		// Create Framebuffer & CommandBuffer
		{
			VkImageView attachment[1];
			VkFramebufferCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			info.renderPass = this->m_renderResource.renderPass;
			info.attachmentCount = 1;
			info.pAttachments = attachment;
			info.width = RHI::get()->getSwapchainExtent().width;
			info.height = RHI::get()->getSwapchainExtent().height;
			info.layers = 1;

			auto backBufferSize = RHI::get()->getSwapchainImageViews().size();
			m_renderResource.framebuffers.resize(backBufferSize);
			m_renderResource.commandPools.resize(backBufferSize);
			m_renderResource.commandBuffers.resize(backBufferSize);

			for (uint32_t i = 0; i < backBufferSize; i++)
			{
				attachment[0] = m_drawUIImages->getView(buildBasicImageSubresource());
				RHICheck(vkCreateFramebuffer(RHI::Device, &info, nullptr, &m_renderResource.framebuffers[i]));
			}

			for (uint32_t i = 0; i < backBufferSize; i++)
			{
				// Command pool
				{
					VkCommandPoolCreateInfo info = {};
					info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
					info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
					info.queueFamilyIndex = RHI::get()->getGraphiscFamily();
					RHICheck(vkCreateCommandPool(RHI::Device, &info, nullptr, &m_renderResource.commandPools[i]));
				}

				// Command buffer
				{
					VkCommandBufferAllocateInfo info = {};
					info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
					info.commandPool = m_renderResource.commandPools[i];
					info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
					info.commandBufferCount = 1;
					RHICheck(vkAllocateCommandBuffers(RHI::Device, &info, &m_renderResource.commandBuffers[i]));
				}
			}
		}
	}

	void ImguiPass::renderpassRelease(bool bFullRelease)
	{
		vkDeviceWaitIdle(RHI::Device);

		if (bFullRelease)
		{
			vkDestroyRenderPass(RHI::Device, m_renderResource.renderPass, nullptr);
		}

		
		auto backBufferSize = m_renderResource.framebuffers.size();
		for (uint32_t i = 0; i < backBufferSize; i++)
		{
			vkFreeCommandBuffers(RHI::Device, m_renderResource.commandPools[i], 1, &m_renderResource.commandBuffers[i]);
			vkDestroyCommandPool(RHI::Device, m_renderResource.commandPools[i], nullptr);
			vkDestroyFramebuffer(RHI::Device, m_renderResource.framebuffers[i], nullptr);
		}

		m_renderResource.framebuffers.resize(0);
		m_renderResource.commandPools.resize(0);
		m_renderResource.commandBuffers.resize(0);
	}

	void ImguiPass::init()
	{
		// Descriptor pool prepare.
		{
			VkDescriptorPoolSize pool_sizes[] =
			{
				{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
				{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
				{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
				{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
				{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
			};

			VkDescriptorPoolCreateInfo pool_info = {};
			pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
			pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
			pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
			pool_info.pPoolSizes = pool_sizes;
			RHICheck(vkCreateDescriptorPool(RHI::Device, &pool_info, nullptr, &m_renderResource.descriptorPool));
		}

		renderpassBuild();

		// register swapchain rebuild functions.
		m_beforeSwapChainRebuildHandle = RHI::get()->onBeforeSwapchainRecreate.addLambda([&]() { renderpassRelease(false); });
		m_afterSwapChainRebuildHandle = RHI::get()->onAfterSwapchainRecreate.addLambda([&]() { renderpassBuild(); });

		// init vulkan resource.
		ImGui_ImplVulkan_InitInfo vkInitInfo{ };
		setupVulkanInitInfo(&vkInitInfo, m_renderResource.descriptorPool);

		// init vulkan here.
		ImGui_ImplVulkan_Init(&vkInitInfo, m_renderResource.renderPass);

		// upload font texture to gpu.
		{
			VkCommandPool command_pool = m_renderResource.commandPools[0];
			VkCommandBuffer command_buffer = m_renderResource.commandBuffers[0];
			RHICheck(vkResetCommandPool(RHI::Device, command_pool, 0));
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			RHICheck(vkBeginCommandBuffer(command_buffer, &begin_info));
			ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
			VkSubmitInfo end_info = {};
			end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			end_info.commandBufferCount = 1;
			end_info.pCommandBuffers = &command_buffer;
			RHICheck(vkEndCommandBuffer(command_buffer));
			RHICheck(vkQueueSubmit(vkInitInfo.Queue, 1, &end_info, VK_NULL_HANDLE));
			RHICheck(vkDeviceWaitIdle(vkInitInfo.Device));
			ImGui_ImplVulkan_DestroyFontUploadObjects();
		}
	}

	void ImguiPass::renderFrame(uint32_t backBufferIndex)
	{
		ImDrawData* main_draw_data = ImGui::GetDrawData();
		{
			RHICheck(vkResetCommandPool(RHI::Device, m_renderResource.commandPools[backBufferIndex], 0));
			VkCommandBufferBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			RHICheck(vkBeginCommandBuffer(m_renderResource.commandBuffers[backBufferIndex], &info));
			RHI::setPerfMarkerBegin(m_renderResource.commandBuffers[backBufferIndex], "ImGUI", { 1.0f, 1.0f, 0.0f, 1.0f });
		}
		{
			VkRenderPassBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = m_renderResource.renderPass;
			info.framebuffer = m_renderResource.framebuffers[backBufferIndex];
			info.renderArea.extent.width = RHI::get()->getSwapchainExtent().width;
			info.renderArea.extent.height = RHI::get()->getSwapchainExtent().height;
			info.clearValueCount = 1;

			VkClearValue clearColor{ };
			clearColor.color.float32[0] = m_clearColor.x * m_clearColor.w;
			clearColor.color.float32[1] = m_clearColor.y * m_clearColor.w;
			clearColor.color.float32[2] = m_clearColor.z * m_clearColor.w;
			clearColor.color.float32[3] = m_clearColor.w;
			info.pClearValues = &clearColor;
			vkCmdBeginRenderPass(m_renderResource.commandBuffers[backBufferIndex], &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		ImGui_ImplVulkan_RenderDrawData(main_draw_data, m_renderResource.commandBuffers[backBufferIndex]);

		vkCmdEndRenderPass(m_renderResource.commandBuffers[backBufferIndex]);
		RHI::setPerfMarkerEnd(m_renderResource.commandBuffers[backBufferIndex]);

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = RHI::get()->getSwapchainImages().at(backBufferIndex);
		barrier.subresourceRange = buildBasicImageSubresource();
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(
			m_renderResource.commandBuffers[backBufferIndex],
			VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			{},
			0,
			nullptr,
			0,
			nullptr,
			1,
			&barrier
		);

		VkImageSubresourceLayers copyLayer
		{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};

		VkImageBlit copyRegion
		{
			.srcSubresource = copyLayer,
			.dstSubresource = copyLayer,
		};

		copyRegion.srcOffsets[0] = { 0, 0, 0};
		copyRegion.dstOffsets[0] = copyRegion.srcOffsets[0];

		copyRegion.srcOffsets[1] = { (int)m_drawUIImages.get()->getExtent().width, (int)m_drawUIImages.get()->getExtent().height, 1};
		copyRegion.dstOffsets[1] = copyRegion.srcOffsets[1];

		vkCmdBlitImage(m_renderResource.commandBuffers[backBufferIndex], 
			m_drawUIImages->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			RHI::get()->getSwapchainImages().at(backBufferIndex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion, VK_FILTER_NEAREST);


		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; 

		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(
			m_renderResource.commandBuffers[backBufferIndex],
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			{},
			0,
			nullptr,
			0,
			nullptr,
			1,
			&barrier
		);

		// VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL

		// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR

		RHICheck(vkEndCommandBuffer(m_renderResource.commandBuffers[backBufferIndex]));
	}

	void ImguiPass::release()
	{
		// unregister swapchain rebuild functions.
		bool res0 = RHI::get()->onBeforeSwapchainRecreate.remove(m_beforeSwapChainRebuildHandle);
		bool res1 = RHI::get()->onAfterSwapchainRecreate.remove(m_afterSwapChainRebuildHandle);
		CHECK(res0 && res1 && "fail to unregister swapchain rebuild callbacks.");

		// shut down vulkan here.
		ImGui_ImplVulkan_Shutdown();

		renderpassRelease(true);
		vkDestroyDescriptorPool(RHI::Device, m_renderResource.descriptorPool, nullptr);
	}
}
