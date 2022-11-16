#include "Pch.h"
#include "Widget.h"
#include "../Editor.h"

using namespace Flower;
using namespace Flower::UI;

Widget::Widget(std::string tile) :
	m_title(tile), m_bShow(true), m_bLastShow(true)
{
	m_renderer = GEngine->getRuntimeModule<Renderer>();
	ASSERT(m_renderer, "You must register one renderer module when require widget ui.");
}

void Widget::init()
{
	// register tick function on imgui pass..
	m_tickFunctionHandle = m_renderer->imguiTickFunctions.addRaw(this, &Widget::tick);

	onInit();
}

void Widget::tick(const RuntimeModuleTickData& tickData)
{
	// Sync visible state.
	if (m_bLastShow != m_bShow)
	{
		m_bLastShow = m_bShow;
		if (m_bShow)
		{
			// last time is hide, current show.
			onShow();
		}
		else
		{
			// last time show, current hide.
			onHide();
		}
	}

	beforeTick();

	onTick(tickData);

	// When project select, disable other widget visible draw disabled.
	if (m_bShow && !GEditor->getProjectSelect()->getVisible())
	{
		ImGui::SetNextWindowSize(ImVec2(520, 600), ImGuiCond_FirstUseEver);
		if (ImGui::Begin(getTile().c_str(), &m_bShow))
		{
			ImGui::PushID(getTile().c_str());
			onVisibleTick(tickData);
			ImGui::PopID();
		}

		ImGui::End();
	}

	afterTick();
}

void Widget::release()
{
	// unregister tick function.
	CHECK(m_renderer->imguiTickFunctions.remove(m_tickFunctionHandle));

	onRelease();
}