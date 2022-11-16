#pragma once
#include "Pch.h"

class Widget
{
private:
	std::string m_title;
	bool m_bLastShow;
	Flower::DelegateHandle m_tickFunctionHandle;

protected:
	bool m_bShow;
	Flower::Renderer* m_renderer;

public:
	void setVisible(bool bVisible)
	{
		m_bShow = bVisible;
	}

	bool getVisible() const
	{
		return m_bShow;
	}

	std::string getTile() const
	{
		return m_title;
	}

public:
	Widget(std::string tile);
	virtual ~Widget() {  }

	void init();
	void release();

private:
	void tick(const Flower::RuntimeModuleTickData& tickData);

protected:
	// event init.
	virtual void onInit() { }

	virtual void beforeTick() {}
	

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) {  }

	virtual void afterTick() { }

	// event release.
	virtual void onRelease() {  }

	// event on widget visible state change. sync on tick function first.
	virtual void onHide() {  }
	virtual void onShow() {  }

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData& tickData) {  }
};