#pragma once
#include "Pch.h"
#include "Widget.h"

class WidgetDownbar : public Widget
{
public:
	WidgetDownbar();
	~WidgetDownbar() = default;

	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;
};