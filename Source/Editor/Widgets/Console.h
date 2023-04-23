#pragma once
#include "../widget.h"

class Console;

class WidgetConsole : public Widget
{
public:
	WidgetConsole(Editor* editor);

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