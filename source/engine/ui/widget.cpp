#include "widget.h"
#include "../engine.h"

namespace engine
{
	WidgetBase::WidgetBase(const char* widgetName, const char* name)
		: m_name(name)
		, m_widgetName(widgetName)
		, m_bShow(true)
		, m_bPrevShow(false)
		, m_engine(Engine::get())
		, m_context(getContext())
		, m_renderer(getRenderer())
	{
		m_runtimeUUID = buildRuntimeUUID64u();
	}

	void WidgetBase::tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
	{
		// Sync visible state.
		if (m_bPrevShow != m_bShow)
		{
			m_bPrevShow = m_bShow;
			if (m_bShow)
			{
				// last time is hide, current show.
				onShow(tickData);
			}
			else
			{
				// last time show, current hide.
				onHide(tickData);
			}
		}

		beforeTick(tickData);

		onTick(tickData, context);

		if (m_bShow)
		{
			// Default window size set to 400x300
			ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_FirstUseEver);
			if (ImGui::Begin(m_name.c_str(), &m_bShow, m_flags))
			{
				ImGui::PushID(m_runtimeUUID);
				onVisibleTick(tickData);
				ImGui::PopID();
			}

			ImGui::End();
		}

		afterTick(tickData);
	}

	void WidgetBase::tickWithCmd(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, VulkanContext* context)
	{
		onTickCmd(tickData, cmd, context);

		if (m_bShow)
		{
			onVisibleTickCmd(tickData, cmd);
		}
	}
}

