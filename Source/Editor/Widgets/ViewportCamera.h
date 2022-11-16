#pragma once
#include "Pch.h"

class ViewportCamera : public Flower::CameraInterface
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
	class WidgetViewport* m_viewport = nullptr;

	// worldspace up.
	mutable glm::vec3 m_worldUp = {0.0f, 1.0f, 0.0f};

	// yaw and pitch. in degree.
	float m_yaw = -90.0f;
	float m_pitch = 0.0f;

	// mouse speed.
	float m_moveSpeed = 10.0f;
	float m_mouseSensitivity = 0.1f;
	float m_maxMouseMoveSpeed = 20.0f;
	float m_minMouseMoveSpeed = 1.0f;

	// first time 
	bool  m_bFirstMouse = true;

	float m_lastX = 0.0f;
	float m_lastY = 0.0f;

	glm::mat4 m_viewMatrix { 1.0f };
	glm::mat4 m_projectMatrix { 1.0f };

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
	virtual glm::mat4 getViewMatrix() const override
	{
		return m_viewMatrix;
	}

	// return camera project matrix.
	virtual glm::mat4 getProjectMatrix() const override
	{
		return m_projectMatrix;
	}

	ViewportCamera(WidgetViewport* inViewport);

	void tick(const Flower::RuntimeModuleTickData& tickData);
};