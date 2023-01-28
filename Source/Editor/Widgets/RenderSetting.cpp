#include "Pch.h"
#include "RenderSetting.h"
#include "../Editor.h"
#include "DrawComponent/DrawComponent.h"
#include "ViewportCamera.h"

using namespace Flower;
using namespace Flower::UI;

static const std::string DETAIL_RenderSettinglIcon = ICON_FA_GEAR;

WidgetRenderSetting::WidgetRenderSetting()
	: Widget("  " + DETAIL_RenderSettinglIcon + "  RenderSetting")
{

}

WidgetRenderSetting::~WidgetRenderSetting() noexcept
{

}

void WidgetRenderSetting::onInit()
{

}

void WidgetRenderSetting::onRelease()
{

}

void WidgetRenderSetting::onTick(const RuntimeModuleTickData& tickData)
{
	m_viewportCamera = GEditor->getWidgetViewport()->getCamera();
}

void WidgetRenderSetting::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ImGui::Spacing();

	ImGui::TextDisabled("Global render setting for flower engine.");

	ImGui::Separator();
	ImGui::Spacing();

	bool bHDR = RenderSettingManager::get()->displayMode == RHI::DISPLAYMODE_HDR10_2084;
	ImGui::Checkbox("Hdr10_2084", &bHDR);
	RenderSettingManager::get()->displayMode = bHDR ? RHI::DISPLAYMODE_HDR10_2084 : RHI::DISPLAYMODE_SDR;


	if (ImGui::CollapsingHeader("Viewport Camera Control"))
	{
		ImGui::Spacing();
		ImGui::Indent();
		ImGui::PushItemWidth(100.0f);
		ImGui::DragFloat("Camera speed min", &m_viewportCamera->m_minMouseMoveSpeed, 1.0f, 1.0f, m_viewportCamera->m_maxMouseMoveSpeed);
		ImGui::DragFloat("Camera speed max", &m_viewportCamera->m_maxMouseMoveSpeed, 1.0f, m_viewportCamera->m_minMouseMoveSpeed, 1000.0f);

		ImGui::DragFloat("Camera speed", &m_viewportCamera->m_moveSpeed, 1.0f, m_viewportCamera->m_minMouseMoveSpeed, m_viewportCamera->m_maxMouseMoveSpeed);

		ImGui::Text("Camera atmosphere config");
		ImGui::Spacing();
		{
			ImGui::DragFloat("Camera atmosphere height offset(km)", &m_viewportCamera->atmosphereHeightOffset, 0.1f, 0.0f);
			ImGui::DragFloat("Camera atmosphere position scale(m)", &m_viewportCamera->atmosphereMoveScale, 1.0f, 1.0f);
		}

		ImGui::DragFloat("aperture", &m_viewportCamera->aperture, 1.0f, 1.0f, 100.0f);
		ImGui::DragFloat("ISO", &m_viewportCamera->iso, 1.0f, 50.0f, 1000.0f);
		ImGui::DragFloat("shutter speed", &m_viewportCamera->shutterSpeed, 0.01f, 1.0f / 360.0f, 20.0f);
		ImGui::DragFloat("exposure compensation", &m_viewportCamera->exposureCompensation, 1.0f, -100.0f, 100.0f);

		ImGui::PopItemWidth();
		ImGui::Unindent();
		ImGui::Spacing();
	}


}