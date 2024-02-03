#include "imgui_manager.h"
#include "imgui.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <filesystem>

#include "ImGuizmo.h"
#include "../../engine.h"

namespace engine
{
	static AutoCVarFloat cVarUIFontSize("r.ImGui.FontSize", "ImGui font size.", "ImGui", 20.0f, CVarFlags::ReadOnly);
	static AutoCVarFloat cVarUIIconSize("r.ImGui.IconSize", "ImGui icon size.", "ImGui", 16.0f, CVarFlags::ReadOnly);

	static AutoCVarString cVarFontFilePath("r.ImGui.FontEnglishFilePath", "ImGui font english file path.", "ImGui", "font/deng.ttf", CVarFlags::ReadOnly);
	static AutoCVarString cVarFontAwesomeFilePath("r.ImGui.FontAwesomeFilePath", "ImGui awesome font file path.", "ImGui", "font/fa-solid-900.ttf", CVarFlags::ReadOnly);
	static AutoCVarString cVarFontAwesomeBandsFilePath("r.ImGui.FontAwesomeBandsFilePath", "ImGui awesome font brands file path.", "ImGui", "font/fa-brands-400.ttf", CVarFlags::ReadOnly);

	void styleProfessionalDark()
	{
		ImGui::StyleColorsDark();

		ImGuiStyle& style = ImGui::GetStyle();
		ImVec4* colors = ImGui::GetStyle().Colors;

		colors[ImGuiCol_BorderShadow] = ImVec4(0.1f, 0.1f, 0.0f, 0.39f);
		{
			style.Colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
			style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
			style.Colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
			style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
			style.Colors[ImGuiCol_FrameBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
			style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
			style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
			style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
			style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
			style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
			style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
			style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
			style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
			style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
			style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
			style.Colors[ImGuiCol_CheckMark] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
			style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.11f, 0.64f, 0.92f, 0.40f);
			style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
			style.Colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
			style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
			style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
			style.Colors[ImGuiCol_Header] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
			style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
			style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
			style.Colors[ImGuiCol_Separator] = style.Colors[ImGuiCol_Border];
			style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.41f, 0.42f, 0.44f, 1.00f);
			style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
			style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
			style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.29f, 0.30f, 0.31f, 0.67f);
			style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
			style.Colors[ImGuiCol_Tab] = ImVec4(0.08f, 0.08f, 0.09f, 0.83f);
			style.Colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.34f, 0.36f, 0.83f);
			style.Colors[ImGuiCol_TabActive] = ImVec4(0.23f, 0.23f, 0.24f, 1.00f);
			style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
			style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
			style.Colors[ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
			style.Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
			style.Colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
			style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
			style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
			style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
			style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
			style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
			style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
			style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
			style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
			style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
			style.Colors[ImGuiCol_CheckMark] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
			style.Colors[ImGuiCol_SliderGrab] = ImVec4(1.0f, 1.0f, 1.0f, 0.3f);
			style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(1.0f, 1.0f, 1.0f, 1.00f);
		}


		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, -1.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(2.0f, 3.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowTitleAlign, ImVec2(0.5f, 0.5f));
		ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 0.0f);

		colors[ImGuiCol_BorderShadow] = ImVec4(0.1f, 0.1f, 0.0f, 0.39f);
		style.WindowBorderSize = 1;
		style.ChildBorderSize = 1;
		style.PopupBorderSize = 1;
		style.FrameBorderSize = 1;
		style.TabBorderSize = 1;
		style.WindowRounding = 0;
		style.ChildRounding = 0;
		style.FrameRounding = 3;
		style.PopupRounding = 0;
		style.ScrollbarRounding = 0;
		style.GrabRounding = 0;
		style.LogSliderDeadzone = 0;
		style.TabRounding = 0;

		ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
		ImGui::GetIO().ConfigWindowsResizeFromEdges = true;

		style.AntiAliasedLines = true;
		style.WindowMenuButtonPosition = ImGuiDir_Right;
		style.PopupRounding = 3;

		style.WindowPadding = ImVec2(4, 4);
		style.FramePadding = ImVec2(6, 4);
		style.ItemSpacing = ImVec2(6, 2);

		style.ScrollbarSize = 18;

		style.WindowBorderSize = 1;
		style.ChildBorderSize = 1;
		style.PopupBorderSize = 1;
		style.FrameBorderSize = 1;

		style.WindowRounding = 3;
		style.ChildRounding = 3;
		style.FrameRounding = 0;
		style.ScrollbarRounding = 2;
		style.GrabRounding = 0;

		style.TabBorderSize = 0;
		style.TabRounding = 3;
		style.WindowRounding = 0.0f;


		ImGuizmo::Style& styleGizmo = ImGuizmo::GetStyle();
		styleGizmo.TranslationLineThickness = 3.0f;
		styleGizmo.TranslationLineArrowSize = 6.0f;
		styleGizmo.RotationLineThickness = 4.0f;
		styleGizmo.RotationOuterLineThickness = 4.0f;
		styleGizmo.ScaleLineThickness = 2.5f;
		styleGizmo.ScaleLineCircleSize = 5.0f;
		styleGizmo.HatchedAxisLineThickness = 6.0f;
		styleGizmo.CenterCircleSize = 2.5f;

		styleGizmo.Colors[ImGuizmo::DIRECTION_X] = ImVec4(1.0f, 0.21f, 0.23f, 0.9f);
		styleGizmo.Colors[ImGuizmo::DIRECTION_Y] = ImVec4(0.60f, 0.9f, 0.067f, 0.9f);
		styleGizmo.Colors[ImGuizmo::DIRECTION_Z] = ImVec4(0.184f, 0.218f, 0.98f, 0.9f);
		styleGizmo.Colors[ImGuizmo::PLANE_X] = ImVec4(0.99f, 0.2f, 0.23f, 0.6f);
		styleGizmo.Colors[ImGuizmo::PLANE_Y] = ImVec4(0.60f, 0.9f, 0.067f, 0.6f);
		styleGizmo.Colors[ImGuizmo::PLANE_Z] = ImVec4(0.184f, 0.218f, 0.98f, 0.6f);

		styleGizmo.Colors[ImGuizmo::SELECTION] = ImVec4(1.000f, 0.500f, 0.062f, 0.541f);
		styleGizmo.Colors[ImGuizmo::INACTIVE] = ImVec4(0.600f, 0.600f, 0.600f, 0.600f);
		styleGizmo.Colors[ImGuizmo::TRANSLATION_LINE] = ImVec4(0.666f, 0.666f, 0.666f, 0.666f);
		styleGizmo.Colors[ImGuizmo::SCALE_LINE] = ImVec4(0.250f, 0.250f, 0.250f, 1.000f);
		styleGizmo.Colors[ImGuizmo::ROTATION_USING_BORDER] = ImVec4(1.000f, 0.500f, 0.062f, 1.000f);
		styleGizmo.Colors[ImGuizmo::ROTATION_USING_FILL] = ImVec4(1.000f, 0.500f, 0.062f, 0.500f);
		styleGizmo.Colors[ImGuizmo::HATCHED_AXIS_LINES] = ImVec4(0.000f, 0.000f, 0.000f, 0.500f);
		styleGizmo.Colors[ImGuizmo::TEXT] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
		styleGizmo.Colors[ImGuizmo::TEXT_SHADOW] = ImVec4(0.000f, 0.000f, 0.000f, 1.000f);

		ImGuizmo::AllowAxisFlip(false);

		colors[ImGuiCol_BorderShadow] = ImVec4(0.1f, 0.1f, 0.0f, 0.39f);
		style.WindowBorderSize = 1;
		style.ChildBorderSize = 1;
		style.PopupBorderSize = 1;
		style.FrameBorderSize = 1;
		style.TabBorderSize = 1;
		style.WindowRounding = 0;
		style.ChildRounding = 0;
		style.FrameRounding = 1;
		style.PopupRounding = 0;
		style.ScrollbarRounding = 0;
		style.GrabRounding =2;
		style.GrabMinSize = 8;
		style.LogSliderDeadzone = 0;
		style.TabRounding = 0;

		style.SliderThickness = 0.3f;
		style.SliderContrast = 1.0f;
	}

	void styleFontSetup()
	{
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		// load font data to memory.
		io.Fonts->AddFontFromFileTTF(cVarFontFilePath.get().c_str(), cVarUIFontSize.get(), NULL, io.Fonts->GetGlyphRangesChineseFull());

		// load font data to memory.
		{
			// merge in icons from Font Awesome
			static const ImWchar iconsRanges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
			ImFontConfig iconsConfig;
			iconsConfig.MergeMode = true;
			iconsConfig.PixelSnapH = true;

			io.Fonts->AddFontFromFileTTF(cVarFontAwesomeFilePath.get().c_str(), cVarUIIconSize.get(), &iconsConfig, iconsRanges);
		}
		{
			// merge in icons from Font Awesome bands.
			static const ImWchar iconsRanges[] = { ICON_MIN_FAB, ICON_MAX_FAB, 0 };
			ImFontConfig iconsConfig;
			iconsConfig.MergeMode = true;
			iconsConfig.PixelSnapH = true;

			io.Fonts->AddFontFromFileTTF(cVarFontAwesomeBandsFilePath.get().c_str(), cVarUIIconSize.get(), &iconsConfig, iconsRanges);
		}
	}

	void setupVulkanInitInfo(ImGui_ImplVulkan_InitInfo* inout, VkDescriptorPool pool, const VulkanContext* context)
	{
		inout->Instance = context->getInstance();
		inout->PhysicalDevice = context->getGPU();
		inout->Device = context->getDevice();
		inout->QueueFamily = context->getGraphiscFamily();
		inout->Queue = context->getMajorGraphicsQueue();
		inout->PipelineCache = VK_NULL_HANDLE;
		inout->DescriptorPool = pool;
		inout->Allocator = nullptr;
		inout->MinImageCount = (uint32_t)context->getBackBufferCount();
		inout->ImageCount = inout->MinImageCount;
		inout->MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		inout->CheckVkResultFn = RHICheck;
		inout->Subpass = 0;
	}

	void ImguiManager::init()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO(); (void)io;

		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigViewportsNoDecoration = false;

		std::stringstream iniFilePath;
		iniFilePath << "config/" << Engine::get()->getGLFWWindows()->getName() << "-imgui.ini";

		static std::string iniFilePathStr;
		iniFilePathStr = std::filesystem::absolute(iniFilePath.str()).string();

		io.IniFilename = iniFilePathStr.c_str();

		styleProfessionalDark();
		ImGuiStyle& style = ImGui::GetStyle();

		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			style.WindowRounding = 0.0f;
			style.Colors[ImGuiCol_WindowBg].w = 1.0f;
		}

		ImGui_ImplGlfw_InitForVulkan(Engine::get()->getGLFWWindows()->getGLFWWindowHandle(), true);

		styleFontSetup();

		// Descriptor pool prepare.
		{
			VkDescriptorPoolSize poolSizes[] =
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

			VkDescriptorPoolCreateInfo poolInfo = {};
			poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
			poolInfo.maxSets = 1000 * IM_ARRAYSIZE(poolSizes);
			poolInfo.poolSizeCount = (uint32_t)IM_ARRAYSIZE(poolSizes);
			poolInfo.pPoolSizes = poolSizes;
			RHICheck(vkCreateDescriptorPool(getDevice(), &poolInfo, nullptr, &m_resources.descriptorPool));
		}

		renderpassBuild();

		// register swapchain rebuild functions.
		m_beforeSwapChainRebuildHandle = getContext()->onBeforeSwapchainRecreate.addLambda([&]() { renderpassRelease(false); });
		m_afterSwapChainRebuildHandle  = getContext()->onAfterSwapchainRecreate.addLambda([&]() { renderpassBuild(); });

		// init vulkan resource.
		ImGui_ImplVulkan_InitInfo vkInitInfo{ };
		setupVulkanInitInfo(&vkInitInfo, m_resources.descriptorPool, getContext());

		// init vulkan here.
		ImGui_ImplVulkan_Init(&vkInitInfo, m_resources.renderPass);

		// upload font texture to gpu.
		{
			VkCommandPool commandPool = m_resources.commandPools[0];
			VkCommandBuffer commandBuffer = m_resources.commandBuffers[0];
			RHICheck(vkResetCommandPool(getDevice(), commandPool, 0));

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			RHICheck(vkBeginCommandBuffer(commandBuffer, &beginInfo));
			ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);

			VkSubmitInfo endInfo = {};
			endInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			endInfo.commandBufferCount = 1;
			endInfo.pCommandBuffers = &commandBuffer;
			RHICheck(vkEndCommandBuffer(commandBuffer));
			RHICheck(vkQueueSubmit(vkInitInfo.Queue, 1, &endInfo, VK_NULL_HANDLE));
			RHICheck(vkDeviceWaitIdle(vkInitInfo.Device));
			ImGui_ImplVulkan_DestroyFontUploadObjects();
		}
	}


	void ImguiManager::renderpassBuild()
	{
		vkDeviceWaitIdle(getDevice());

		// Pick one suitable backbuffer format for ui draw, we always need at least 8 bit to mix alpha.
		const bool bBackbufferSupportAlpha = Swapchain::isBackBufferSupportAlphaBlend(getContext()->getBackbufferFormatType());
		m_drawUIFormat = bBackbufferSupportAlpha
			? getContext()->getSwapchain().getImageFormat()
			: VK_FORMAT_R16G16B16A16_SFLOAT; // When no support alpha blend, it only set alpha to 0 bit or 2 bit to save more bit for rgb.
											 // Then can present more color bit in the screen, often do this trick in hdr10 mode.

		// Create the Render Pass
		if (this->m_resources.renderPass == VK_NULL_HANDLE)
		{
			VkAttachmentDescription attachment = {};
			attachment.format = m_drawUIFormat;
			attachment.samples = VK_SAMPLE_COUNT_1_BIT;
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			// NOTE: if ui buffer is different from back buffer format, we must add one blit pass.
			attachment.finalLayout = shouldBlitAfterRenderUI() ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

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
			RHICheck(vkCreateRenderPass(getDevice(), &info, nullptr, &this->m_resources.renderPass));

			getContext()->setResourceName(VK_OBJECT_TYPE_RENDER_PASS, (uint64_t)this->m_resources.renderPass, "ImguiRenderpass");
		}

		if (shouldBlitAfterRenderUI())
		{
			VkImageCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			info.flags = 0;
			info.imageType = VK_IMAGE_TYPE_2D;
			info.format = m_drawUIFormat;
			info.extent.width = getContext()->getSwapchain().getExtent().width;
			info.extent.height = getContext()->getSwapchain().getExtent().height;
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

			m_drawUIImages = std::make_unique<VulkanImage>(getContext()->getVMAImage(), "DrawUIImage", info);
			LOG_TRACE("Backbuffer format not support alpha blend, so we create new image to draw and blit ui.");
		}
		else
		{
			LOG_TRACE("Backbuffer format support alpha blend, we just draw ui in back buffer.");
			m_drawUIImages = nullptr;
		}

		// Create Framebuffer & CommandBuffer
		{
			VkImageView attachment[1];
			VkFramebufferCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			info.renderPass = this->m_resources.renderPass;
			info.attachmentCount = 1;
			info.pAttachments = attachment;
			info.width = getContext()->getSwapchain().getExtent().width;
			info.height = getContext()->getSwapchain().getExtent().height;
			info.layers = 1;

			auto backBufferSize = getContext()->getBackBufferCount();
			m_resources.framebuffers.resize(backBufferSize);
			m_resources.commandPools.resize(backBufferSize);
			m_resources.commandBuffers.resize(backBufferSize);

			for (uint32_t i = 0; i < backBufferSize; i++)
			{
				if (shouldBlitAfterRenderUI())
				{
					// Use temp create image view.
					attachment[0] = m_drawUIImages->getOrCreateView(buildBasicImageSubresource()).view;
				}
				else
				{
					// Reuse back buffer image view.
					attachment[0] = getContext()->getSwapchain().getImageViews().at(i);
				}

				RHICheck(vkCreateFramebuffer(getDevice(), &info, nullptr, &m_resources.framebuffers[i]));
			}

			for (uint32_t i = 0; i < backBufferSize; i++)
			{
				// Command pool
				{
					VkCommandPoolCreateInfo info = {};
					info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
					info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
					info.queueFamilyIndex = getContext()->getGraphiscFamily();
					RHICheck(vkCreateCommandPool(getDevice(), &info, nullptr, &m_resources.commandPools[i]));
				}

				// Command buffer
				{
					VkCommandBufferAllocateInfo info = {};
					info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
					info.commandPool = m_resources.commandPools[i];
					info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
					info.commandBufferCount = 1;
					RHICheck(vkAllocateCommandBuffers(getDevice(), &info, &m_resources.commandBuffers[i]));
				}
			}
		}
	}

	void ImguiManager::renderpassRelease(bool bFullRelease)
	{
		vkDeviceWaitIdle(getDevice());

		if (bFullRelease)
		{
			vkDestroyRenderPass(getDevice(), m_resources.renderPass, nullptr);
		}


		auto backBufferSize = m_resources.framebuffers.size();
		for (uint32_t i = 0; i < backBufferSize; i++)
		{
			vkFreeCommandBuffers(getDevice(), m_resources.commandPools[i], 1, &m_resources.commandBuffers[i]);
			vkDestroyCommandPool(getDevice(), m_resources.commandPools[i], nullptr);
			vkDestroyFramebuffer(getDevice(), m_resources.framebuffers[i], nullptr);
		}

		m_resources.framebuffers.resize(0);
		m_resources.commandPools.resize(0);
		m_resources.commandBuffers.resize(0);
	}

	bool ImguiManager::shouldBlitAfterRenderUI() const
	{
		return m_drawUIFormat != getContext()->getSwapchain().getImageFormat();
	}

	void ImguiManager::render()
	{
		ImGui::Render();
	}

	void ImguiManager::newFrame()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();
	}

	void ImguiManager::updateAfterSubmit()
	{
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		// Update and Render additional Platform Windows
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}
	}

	bool ImguiManager::isMainMinimized()
	{
		ImDrawData* mainDrawData = ImGui::GetDrawData();
		return (mainDrawData->DisplaySize.x <= 0.0f || mainDrawData->DisplaySize.y <= 0.0f);
	}

	void ImguiManager::drawImGuiDemo()
	{
		static bool bEnable = true;
		ImGui::ShowDemoWindow(&bEnable);
	}

	void ImguiManager::renderFrame(uint32_t backBufferIndex)
	{
		ImDrawData* main_draw_data = ImGui::GetDrawData();
		{
			RHICheck(vkResetCommandPool(getDevice(), m_resources.commandPools[backBufferIndex], 0));
			VkCommandBufferBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			RHICheck(vkBeginCommandBuffer(m_resources.commandBuffers[backBufferIndex], &info));
			getContext()->setPerfMarkerBegin(m_resources.commandBuffers[backBufferIndex], "ImGUI", { 1.0f, 1.0f, 0.0f, 1.0f });
		}
		{
			VkRenderPassBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = m_resources.renderPass;
			info.framebuffer = m_resources.framebuffers[backBufferIndex];
			info.renderArea.extent.width  = getContext()->getSwapchain().getExtent().width;
			info.renderArea.extent.height = getContext()->getSwapchain().getExtent().height;
			info.clearValueCount = 1;

			VkClearValue clearColor{ };
			clearColor.color.float32[0] = m_clearColor.x * m_clearColor.w;
			clearColor.color.float32[1] = m_clearColor.y * m_clearColor.w;
			clearColor.color.float32[2] = m_clearColor.z * m_clearColor.w;
			clearColor.color.float32[3] = m_clearColor.w;
			info.pClearValues = &clearColor;
			vkCmdBeginRenderPass(m_resources.commandBuffers[backBufferIndex], &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		ImGui_ImplVulkan_RenderDrawData(main_draw_data, m_resources.commandBuffers[backBufferIndex]);

		vkCmdEndRenderPass(m_resources.commandBuffers[backBufferIndex]);
		getContext()->setPerfMarkerEnd(m_resources.commandBuffers[backBufferIndex]);

		if (shouldBlitAfterRenderUI())
		{
			LOG_ERROR_ONCE("TODO: Add custom color convert here to fix hdr present problem.");
			CHECK(m_drawUIImages);

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = getContext()->getSwapchain().getImages().at(backBufferIndex);
			barrier.subresourceRange = buildBasicImageSubresource();
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vkCmdPipelineBarrier(
				m_resources.commandBuffers[backBufferIndex],
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

			copyRegion.srcOffsets[0] = { 0, 0, 0 };
			copyRegion.dstOffsets[0] = copyRegion.srcOffsets[0];

			copyRegion.srcOffsets[1] = { (int)m_drawUIImages.get()->getExtent().width, (int)m_drawUIImages.get()->getExtent().height, 1 };
			copyRegion.dstOffsets[1] = copyRegion.srcOffsets[1];

			vkCmdBlitImage(m_resources.commandBuffers[backBufferIndex],
				m_drawUIImages->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				getContext()->getSwapchain().getImages().at(backBufferIndex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion, VK_FILTER_NEAREST);


			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(
				m_resources.commandBuffers[backBufferIndex],
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
		}
		else
		{
			ASSERT(m_drawUIImages == nullptr, "When backbuffer format can use alpha blend, you should no create this image!");
		}

		RHICheck(vkEndCommandBuffer(m_resources.commandBuffers[backBufferIndex]));
	}

	void ImguiManager::release()
	{
		// unregister swapchain rebuild functions.
		bool res0 = getContext()->onBeforeSwapchainRecreate.remove(m_beforeSwapChainRebuildHandle);
		bool res1 = getContext()->onAfterSwapchainRecreate.remove(m_afterSwapChainRebuildHandle);
		CHECK(res0 && res1 && "fail to unregister swapchain rebuild callbacks.");

		// shut down vulkan here.
		ImGui_ImplVulkan_Shutdown();

		renderpassRelease(true);
		vkDestroyDescriptorPool(getDevice(), m_resources.descriptorPool, nullptr);

		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
}

