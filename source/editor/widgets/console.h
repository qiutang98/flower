#pragma once
#include "../editor.h"

class Console;

class WidgetConsole : public engine::WidgetBase
{
public:
	WidgetConsole();

	static const std::string& getShowName();

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

private:
	std::unique_ptr<Console> m_console;
};