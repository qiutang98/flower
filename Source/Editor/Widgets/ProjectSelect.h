#pragma once
#include "Pch.h"
#include "Widget.h"

class WidgetProjectSelect : public Widget
{
public:
	WidgetProjectSelect();
	virtual ~WidgetProjectSelect() noexcept {}

	void setVisible(bool bState) { m_bActive = bState; }
	bool getVisible() const { return m_bActive; }

protected:
	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	bool loadProject();
	void createProject(const std::filesystem::path& path);

private:
	// When application start, open popup and let user select project.
	bool m_bActive = true;

	bool m_bActiveCreateDetail = false;
	static const auto GCreateProjectPathSize = 256;
	char m_createProjectPath[GCreateProjectPathSize] = "";
	char m_createProjectName[GCreateProjectPathSize] = "";
};