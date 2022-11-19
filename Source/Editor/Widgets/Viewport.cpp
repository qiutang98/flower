#include "Pch.h"
#include "Viewport.h"

using namespace Flower;
using namespace Flower::UI;

const std::string VIERPORT_GViewportTileIcon = ICON_FA_CLOVER;

static AutoCVarInt32 cVarEnableStatUnit(
	"stat.unit",
	"Enable stat unit frame.",
	"stat",
	1, 
	CVarFlags::ReadAndWrite
);

static AutoCVarInt32 cVarEnableStatFrameGraph(
	"stat.frameGraph",
	"Enable stat frame graph.",
	"stat",
	1,
	CVarFlags::ReadAndWrite
);

WidgetViewport::WidgetViewport()
	: Widget("  " + VIERPORT_GViewportTileIcon + "  Viewport")
{
	
}

WidgetViewport::~WidgetViewport() noexcept
{

}

void WidgetViewport::onInit()
{
	m_renderer = GEngine->getRuntimeModule<Renderer>();
	m_viewportImageSampler = RHI::SamplerManager->createSampler(SamplerFactory::pointClampBorder0000());

	m_camera = std::make_unique<ViewportCamera>(this);
	m_viewportRenderer = std::make_unique<DeferredRenderer>("ViewportRenderer", m_camera.get());
	m_viewportRenderer->init();

	m_rendererDelegate = m_renderer->rendererTickHooks.addLambda([this](const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd) 
	{
		m_viewportRenderer->tick(tickData, graphicsCmd);
	});
}

void WidgetViewport::onRelease()
{
	m_renderer->rendererTickHooks.remove(m_rendererDelegate);
	m_viewportRenderer->release();
	m_viewportRenderer.reset();
}

void WidgetViewport::beforeTick()
{
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
}
void WidgetViewport::afterTick()
{
	ImGui::PopStyleVar(1);
}

void WidgetViewport::onTick(const RuntimeModuleTickData& tickData)
{
	
}

void WidgetViewport::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	float width = glm::ceil(ImGui::GetContentRegionAvail().x);
	float height = glm::ceil(ImGui::GetContentRegionAvail().y);

	if (m_cacheWidth != width || m_cacheHeight != height)
	{
		if (!ImGui::IsMouseDragging(0))
		{
			m_cacheWidth = width;
			m_cacheHeight = height;
			m_viewportRenderer->updateRenderSize(uint32_t(width), uint32_t(height), 1.0f, 1.0f);

			tryReleaseDescriptorSet(tickData.tickCount);

			m_descriptorSet = ImGui_ImplVulkan_AddTexture(
				m_viewportImageSampler,
				m_viewportRenderer->getDisplayOutput().getView(buildBasicImageSubresource()),
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			);
		}
	}

	ImVec2 startPos = ImGui::GetCursorPos();
	ImGui::Image(m_descriptorSet, ImVec2(width, height));
	m_bMouseInViewport = ImGui::IsItemHovered();
	m_camera->tick(tickData);


	ImGui::SetCursorPos(startPos);

	ImGui::Spacing(); ImGui::Spacing();
	ImGui::Spacing(); ImGui::Spacing();
	ImGui::Indent(2.0f);
	if(cVarEnableStatUnit.get() > 0)
	{
		const std::vector<TimeStamp>& timeStamps = m_viewportRenderer->getTimingValues();
		const bool bTimeStampsAvailable = timeStamps.size() > 0;
		if (bTimeStampsAvailable)
		{
			m_profileViewer.RECENT_HIGHEST_FRAME_TIME = 0;

			m_profileViewer.FRAME_TIME_ARRAY[m_profileViewer.NUM_FRAMES - 1] = timeStamps.back().microseconds;
			for (uint32_t i = 0; i < m_profileViewer.NUM_FRAMES - 1; i++)
			{
				m_profileViewer.FRAME_TIME_ARRAY[i] = m_profileViewer.FRAME_TIME_ARRAY[i + 1];
			}
			m_profileViewer.RECENT_HIGHEST_FRAME_TIME =
				std::max(m_profileViewer.RECENT_HIGHEST_FRAME_TIME, m_profileViewer.FRAME_TIME_ARRAY[m_profileViewer.NUM_FRAMES - 1]);
		}
		const float& frameTime_us = m_profileViewer.FRAME_TIME_ARRAY[m_profileViewer.NUM_FRAMES - 1];
		const float  frameTime_ms = frameTime_us * 0.001f;
		const int fps = bTimeStampsAvailable ? static_cast<int>(1000000.0f / frameTime_us) : 0;
		static const char* textFormat = "%s : %.2f %s";

		auto profileUI = [&]()
		{
			ImGui::BeginGroupPanel("Profiler");
			{
				ImGui::Text("Resolution : %ix%i", (int32_t)width, (int32_t)height);
				ImGui::Text("FPS : %d (%.2f ms)", fps, frameTime_ms);

				for (uint32_t i = 0; i < timeStamps.size(); i++)
				{
					float value = m_profileViewer.bShowMilliseconds ? timeStamps[i].microseconds / 1000.0f : timeStamps[i].microseconds;
					const char* pStrUnit = m_profileViewer.bShowMilliseconds ? "ms" : "us";
					ImGui::Text(textFormat, timeStamps[i].label.c_str(), value, pStrUnit);
				}
			}
			ImGui::Spacing();
			ImGui::EndGroupPanel();
		};

		const auto srcPos = ImGui::GetCursorPos();
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
		ImGui::BeginDisabled();
		profileUI();
		ImGui::EndDisabled();
		ImGui::PopStyleVar();
		ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 0, 0, 139), 2.0f);
		
		ImGui::SetCursorPos(srcPos);
		profileUI();
		
	}

	if (cVarEnableStatFrameGraph.get() > 0)
	{
		size_t iFrameTimeGraphMaxValue = 0;
		size_t iFrameTimeGraphMinValue = 0;
		for (int i = 0; i < m_profileViewer.countNum; ++i)
		{
			if (m_profileViewer.RECENT_HIGHEST_FRAME_TIME < m_profileViewer.FRAME_TIME_GRAPH_MAX_VALUES[i])
			{
				iFrameTimeGraphMaxValue = std::min(int(m_profileViewer.countNum - 1), i + 1);
				break;
			}
		}

		auto frameGraphView = [&]()
		{
			ImGui::BeginGroupPanel("GPU frame time (us)");
			{
				ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
				ImGui::PushStyleColor(ImGuiCol_FrameBg, { 0,0,0,0 });
				ImGui::PlotLines("",
					m_profileViewer.FRAME_TIME_ARRAY,
					m_profileViewer.NUM_FRAMES,
					0,
					0,
					0.0f,
					m_profileViewer.FRAME_TIME_GRAPH_MAX_VALUES[iFrameTimeGraphMaxValue],
					ImVec2(200, 80));
				ImGui::PopStyleColor();
				ImGui::PopStyleVar();
			}
			ImGui::EndGroupPanel();
		};

		
		const auto srcPos = ImGui::GetCursorPos();
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
		ImGui::BeginDisabled();
		frameGraphView();
		ImGui::EndDisabled();
		ImGui::PopStyleVar();
		ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 0, 0, 139), 2.0f);
		ImGui::SetCursorPos(srcPos);
		frameGraphView();
	}
	
	ImGui::Unindent();
}

void WidgetViewport::tryReleaseDescriptorSet(uint64_t tickTime)
{
	if (m_descriptorSet != VK_NULL_HANDLE)
	{
		m_lazyDestroy.push_back({ tickTime, m_descriptorSet });
		m_descriptorSet = VK_NULL_HANDLE;
	}

	auto it = m_lazyDestroy.begin();
	while (it != m_lazyDestroy.end())
	{
		if (it->tickTime + RHI::GMaxSwapchainCount < tickTime)
		{
			ImGui_ImplVulkan_FreeTextureSet(&it->set);
			it = m_lazyDestroy.erase(it);
		}
		else 
		{
			++it;
		}
	}
}