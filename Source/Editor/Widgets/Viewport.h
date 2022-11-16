#pragma once
#include "Pch.h"
#include "Widget.h"
#include "ViewportCamera.h"

struct ProfilerViewer
{
	bool bShowProfilerWindow = true;
	bool bShowMilliseconds = true;

	static const size_t NUM_FRAMES = 128;
	float FRAME_TIME_ARRAY[NUM_FRAMES] = { 0 };

	float RECENT_HIGHEST_FRAME_TIME = 0.0f;

	const static size_t countNum = 14;
	const int FRAME_TIME_GRAPH_MAX_FPS[countNum] = { 800, 240, 120, 90, 60, 45, 30, 15, 10, 5, 4, 3, 2, 1 };
	float FRAME_TIME_GRAPH_MAX_VALUES[countNum]  = { 0 };

	ProfilerViewer()
	{
		for (int i = 0; i < countNum; ++i)
		{ 
			FRAME_TIME_GRAPH_MAX_VALUES[i] = 1000000.f / FRAME_TIME_GRAPH_MAX_FPS[i]; 
		}
	}
};

class WidgetViewport : public Widget
{
public:
	WidgetViewport();
	virtual ~WidgetViewport() noexcept;

	float getRenderWidth() const
	{
		return m_cacheWidth;
	}

	float getRenderHeight() const
	{
		return m_cacheHeight;
	}

	bool isMouseInViewport() const
	{
		return m_bMouseInViewport;
	}

	ViewportCamera* getCamera() const
	{
		return m_camera.get();
	}

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	virtual void beforeTick() override;
	virtual void afterTick() override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData&) override;

private:
	std::unique_ptr<Flower::DeferredRenderer> m_viewportRenderer; 
	Flower::DelegateHandle m_rendererDelegate;

	Flower::Renderer* m_renderer = nullptr;
	std::unique_ptr<ViewportCamera> m_camera;
	float m_cacheWidth = 0.0f;
	float m_cacheHeight = 0.0f;

	bool m_bMouseInViewport = false;

	VkSampler m_viewportImageSampler;
	VkDescriptorSet m_descriptorSet  = VK_NULL_HANDLE;

	struct LazyDestroyDescriptorSet
	{
		uint64_t tickTime;
		VkDescriptorSet set;
	};
	std::vector<LazyDestroyDescriptorSet> m_lazyDestroy;

	void tryReleaseDescriptorSet(uint64_t tickTime);


	ProfilerViewer m_profileViewer;




};