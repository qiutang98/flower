#include "widget.h"
#include "editor.h"
#include <imgui/imgui.h>

using namespace engine;
using namespace engine::ui;

Widget::Widget(Editor* editor, const std::string& name)
	: m_name(name)
	, m_bShow(true)
	, m_bPrevShow(false)
	, m_editor(editor)
	, m_engine(editor->getEngine())
	, m_context(editor->getContext())
	, m_renderer(editor->getRenderer())
	, m_sceneManager(editor->getSceneManager())
	, m_assetSystem(editor->getAssetSystem())
	, m_undo(&editor->getUndo())
{

}

void Widget::tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
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
			ImGui::PushID(m_name.c_str());
			onVisibleTick(tickData);
			ImGui::PopID();
		}

		ImGui::End();
	}

	afterTick(tickData);
}

void Widget::tickWithCmd(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, VulkanContext* context)
{
	onTickCmd(tickData, cmd, context);

	if (m_bShow)
	{
		onVisibleTickCmd(tickData, cmd);
	}
}