#include "Pch.h"
#include "ViewportCamera.h"
#include "Viewport.h"

using namespace Flower;
using namespace Flower::UI;

void ViewportCamera::updateCameraVectors()
{
	// Get front vector from yaw and pitch angel.
	glm::vec3 front;
	front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
	front.y = sin(glm::radians(m_pitch));
	front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
	m_front = glm::normalize(front);

	// Double cross to get camera true up and right vector.
	m_right = glm::normalize(glm::cross(m_front, m_worldUp));
	m_up = glm::normalize(glm::cross(m_right, m_front));
}

void ViewportCamera::processKeyboard(EMoveType direction, float deltaTime)
{
	float velocity = m_moveSpeed * deltaTime;

	if (direction == EMoveType::Forward)
	{
		m_position += m_front * velocity;
	}
	else if (direction == EMoveType::Backward)
	{
		m_position -= m_front * velocity;
	}
	else if (direction == EMoveType::Left)
	{
		m_position -= m_right * velocity;
	}
	else if (direction == EMoveType::Right)
	{
		m_position += m_right * velocity;
	}
	else
	{
		CHECK(false && "Non entry implement.");
	}
}

void ViewportCamera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
	xoffset *= m_mouseSensitivity;
	yoffset *= m_mouseSensitivity;

	m_yaw += xoffset;
	m_pitch += yoffset;

	if (constrainPitch)
	{
		if (m_pitch > 89.0f)
		{
			m_pitch = 89.0f;
		}
			
		if (m_pitch < -89.0f)
		{
			m_pitch = -89.0f;
		}
	}

	updateCameraVectors();
}

void ViewportCamera::processMouseScroll(float yoffset)
{
	m_moveSpeed += (float)yoffset;
	m_moveSpeed = glm::clamp(m_moveSpeed, m_minMouseMoveSpeed, m_maxMouseMoveSpeed);
}

ViewportCamera::ViewportCamera(WidgetViewport* inViewport)
	: m_viewport(inViewport)
{
	updateCameraVectors();
}

void ViewportCamera::tick(const Flower::RuntimeModuleTickData& tickData)
{
	size_t renderWidth  = size_t(m_viewport->getRenderWidth());
	size_t renderHeight = size_t(m_viewport->getRenderHeight());

	auto* input = GLFWWindowData::get();
	float dt = tickData.deltaTime;

	// prepare view size.
	if (m_width != renderWidth) m_width = std::max(GMinRenderDim, renderWidth);
	if (m_height != renderHeight) m_height = std::max(GMinRenderDim, renderHeight);

	// handle first input.
	if (m_bFirstMouse)
	{
		m_lastX = input->getMouseX();
		m_lastY = input->getMouseY();

		m_bFirstMouse = false;
	}

	// handle active viewport state.
	m_bActiveViewport = false;
	if (m_viewport->isMouseInViewport())
	{
		if (input->isMouseButtonPressed(Mouse::ButtonRight))
		{
			m_bActiveViewport = true;
		}
	}

	// active viewport. disable cursor.
	if (m_bActiveViewport && !m_bHideMouseCursor)
	{
		m_bHideMouseCursor = true;
		glfwSetInputMode(input->getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	// un-active viewport. enable cursor.
	if (!m_bActiveViewport && m_bHideMouseCursor)
	{
		m_bHideMouseCursor = false;
		glfwSetInputMode(input->getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// first time un-active viewport.
	if (m_bActiveViewportLastframe && !m_bActiveViewport)
	{
		m_lastX = input->getMouseX();
		m_lastY = input->getMouseY();
	}

	// continue active viewport.
	float xoffset = 0.0f;
	float yoffset = 0.0f;
	if (m_bActiveViewportLastframe && m_bActiveViewport)
	{
		xoffset = input->getMouseX() - m_lastX;
		yoffset = m_lastY - input->getMouseY();
	}

	// update state.
	m_bActiveViewportLastframe = m_bActiveViewport;
	if (m_bActiveViewport)
	{
		m_lastX = input->getMouseX();
		m_lastY = input->getMouseY();

		processMouseMovement(xoffset, yoffset);
		processMouseScroll(input->getScrollOffset().y);

		if (input->isKeyPressed(Key::W))
		{
			processKeyboard(EMoveType::Forward, dt);
		}
		if (input->isKeyPressed(Key::S))
		{
			processKeyboard(EMoveType::Backward, dt);
		}
		if (input->isKeyPressed(Key::A))
		{
			processKeyboard(EMoveType::Left, dt);
		}
		if (input->isKeyPressed(Key::D))
		{
			processKeyboard(EMoveType::Right, dt);
		}
	}

	updateMatrixMisc();
}

// Update camera view matrix and project matrix.
// We use reverse z projection.
void ViewportCamera::updateMatrixMisc()
{
	// update view matrix.
	m_viewMatrix = glm::lookAt(m_position, m_position + m_front, m_up);

	// reverse z.
	m_projectMatrix = glm::perspective(m_fovy, getAspect(), m_zFar, m_zNear);
}