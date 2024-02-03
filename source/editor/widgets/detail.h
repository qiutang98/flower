#pragma once


#include "../editor.h"
#include "scene_outliner.h"

class WidgetDetail : public engine::WidgetBase
{
public:
	WidgetDetail(size_t index);
	virtual ~WidgetDetail() noexcept;

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

	void drawComponent(std::shared_ptr<engine::SceneNode> node);

private:
	void onOutlinerSelectionChange(Selection<SceneNodeSelctor>& selector);

private:
	ImGuiTextFilter m_filter;


	Selection<SceneNodeSelctor>* m_selector = nullptr;

	engine::DelegateHandle m_onSelectorChange;
};

