#pragma once

#include "../editor.h"
#include <utils/camera_interface.h>
#include <renderer/deferred_renderer.h>
#include "transform_handle.h"


class ViewportCamera : public engine::CameraInterface
{
public:
	enum class EMoveType
	{
		Forward,
		Backward,
		Left,
		Right,
	};

public:
	// Cache viewport.
	class ViewportWidget* m_viewport = nullptr;

	// worldspace up.
	mutable engine::math::vec3 m_worldUp = { 0.0f, 1.0f, 0.0f };

	// yaw and pitch. in degree.
	float m_yaw = -90.0f;
	float m_pitch = 0.0f;

	// mouse speed.
	float m_moveSpeed = 10.0f;
	float m_mouseSensitivity = 0.1f;
	float m_maxMouseMoveSpeed = 400.0f;
	float m_minMouseMoveSpeed = 1.0f;

	// first time 
	bool  m_bFirstMouse = true;

	// mouse position of prev frame.
	float m_lastX = 0.0f;
	float m_lastY = 0.0f;

	// Cache matrix.
	engine::math::mat4 m_viewMatrix { 1.0f };
	engine::math::mat4 m_projectMatrix { 1.0f };
	engine::math::mat4 m_imgizmoProjection { 1.0f };
	bool isControlingCamera() const { return m_bActiveViewport; }

private:
	bool m_bActiveViewport = false;
	bool m_bActiveViewportLastframe = false;
	bool m_bHideMouseCursor = false;

private:
	void updateCameraVectors();
	void updateMatrixMisc();
	void processKeyboard(EMoveType direction, float deltaTime);
	void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
	void processMouseScroll(float yoffset);


public:
	// return camera view matrix.
	virtual engine::math::mat4 getViewMatrix() const override
	{
		return m_viewMatrix;
	}

	// return camera project matrix.
	virtual engine::math::mat4 getProjectMatrix() const override
	{
		return m_projectMatrix;
	}

	const engine::math::mat4& getImGizmoProjectMatrix() const { return m_imgizmoProjection; }

	ViewportCamera(class ViewportWidget* inViewport);

	void tick(const engine::RuntimeModuleTickData& tickData);
};

struct ProfilerViewer
{
	bool bShowProfilerWindow = true;
	bool bShowMilliseconds = true;

	static const size_t kNumFrames = 128;
	float frameTimeArray[kNumFrames] = { 0.0f };

	float recentHighestFrameTime = 0.0f;

	const static size_t kCountNum = 14;
	const int frameTimeGraphMaxFps[kCountNum] = { 800, 240, 120, 90, 60, 45, 30, 15, 10, 5, 4, 3, 2, 1 };
	float frameTimeGraphMaxValues[kCountNum]  = { 0.0f };

	ProfilerViewer()
	{
		for (int i = 0; i < kCountNum; ++i)
		{
			frameTimeGraphMaxValues[i] = 1000000.f / frameTimeGraphMaxFps[i];
		}
	}
};

class ViewportWidget : public engine::WidgetBase
{
public:
	ViewportWidget(size_t index);

	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) override;

	virtual void beforeTick(const engine::RuntimeModuleTickData& tickData) override;
	virtual void afterTick(const engine::RuntimeModuleTickData& tickData) override;

	// Get renderer dimension.
	float getRenderWidth() const { return m_cacheWidth; }
	float getRenderHeight() const { return m_cacheHeight; }

	// Mouse in viewport.
	bool isMouseInViewport() const { return m_bMouseInViewport; }

	engine::DeferredRenderer* getDeferredRenderer() const { return m_deferredRenderer.get(); }

private:
	void drawProfileViewer(uint32_t width, uint32_t height);

private:
	// Viewport deferred renderer.
	std::unique_ptr<engine::DeferredRenderer> m_deferredRenderer;
	engine::DelegateHandle m_deferredRendererDelegate;

	// Cache viewport size.
	float m_cacheWidth = 0.0f;
	float m_cacheHeight = 0.0f;

	// State to know mouse in viewport. Warning: On glfw3.3 may cause some error state when set cursor to disabled.
	bool m_bMouseInViewport = false;

	std::unique_ptr<ViewportCamera> m_camera;
	ProfilerViewer m_profileViewer;
	TransformHandle m_transformHandler;
};