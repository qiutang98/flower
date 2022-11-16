#pragma once
#include "Pch.h"
#include "MainMenu.h"
#include "Widget.h"

class DockSpace : public Widget
{
public:
	DockSpace();
	~DockSpace() = default;

	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

private:
	MainMenu m_mainMenu;
};