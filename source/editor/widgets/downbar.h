#pragma once
#include "../editor.h"

class DownbarWidget : public engine::WidgetBase
{
public:
	DownbarWidget();

	static void draw(
		bool bNewWindow, 
		const engine::RuntimeModuleTickData& tickData, 
		engine::VulkanContext* context, 
		const std::string& name, 
		ImGuiID ID);

	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;
};