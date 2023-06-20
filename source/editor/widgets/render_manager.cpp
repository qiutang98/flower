#include "render_manager.h"
#include "scene_outliner.h"
#include "imgui/ui.h"
#include "imgui/region_string.h"
#include "../editor.h"
#include <regex>

using namespace engine;
using namespace engine::ui;

RegionStringInit RenderManager_Title("RenderManager_Title", "Render", "Render");
const static std::string ICON_RENDERMANAGER = ICON_FA_GEARS;

RenderManagerWidget::RenderManagerWidget(Editor* editor)
	: Widget(editor, "RenderManager")
{
}

void RenderManagerWidget::onInit()
{
	m_name = combineIcon(RenderManager_Title, ICON_RENDERMANAGER);
}

void RenderManagerWidget::onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context)
{
}

void RenderManagerWidget::onRelease()
{

}


void RenderManagerWidget::onVisibleTick(const engine::RuntimeModuleTickData& tickData)
{
	auto* viewportRenderer = m_editor->getViewportWidget()->getRenderer();
	auto* viewportCamera = m_editor->getViewportWidget()->getCamera();

	ImGui::Spacing();

	ui::beginGroupPanel("Viewport Camera");
	{
		ImGui::PushItemWidth(100.0f);
		float renderPercentage = viewportRenderer->getRenderPercentage();
		ImGui::SliderFloat("Render Percentage", &renderPercentage, 0.25f, 1.0f);
		if (renderPercentage != viewportRenderer->getRenderPercentage())
		{
			m_editor->getViewportWidget()->markShouldResize();
			viewportRenderer->setRenderPercentage(renderPercentage);
		}

		float fovy = math::degrees(viewportCamera->getFovY());
		ImGui::SliderFloat("Fovy", &fovy, 10.0f, 90.0f);
		fovy = math::radians(fovy);
		if (fovy != viewportCamera->getFovY())
		{
			viewportCamera->setFovY(fovy);
		}
		ImGui::PopItemWidth();
	}
	ui::endGroupPanel();
}