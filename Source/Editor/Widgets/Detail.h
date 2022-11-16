#pragma once
#include "Pch.h"
#include "Widget.h"

class WidgetDetail : public Widget
{
public:
	WidgetDetail();
	virtual ~WidgetDetail() noexcept;

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData&) override;

	void drawComponent(std::shared_ptr<Flower::SceneNode> node);

private:
	class WidgetSceneOutliner* m_outliner;

	ImGuiTextFilter m_filter;
};