#pragma once
#include "Pch.h"
#include "Widget.h"

class WidgetAssetInspector : public Widget
{
public:
	WidgetAssetInspector();
	virtual ~WidgetAssetInspector() noexcept;

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData&) override;
};