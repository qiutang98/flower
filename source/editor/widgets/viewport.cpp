#include "viewport.h"
#include "scene_outliner.h"
#include "../selection.h"
#include <scene/scene_manager.h>
#include <scene/scene.h>
#include "../builtin_resources.h"

using namespace engine;
using namespace engine::ui;

const static std::string kIconViewport = ICON_FA_EARTH_ASIA;

static AutoCVarInt32 cVarEnableStatUnit("stat.unit", "Enable stat unit frame.", "stat", 1, CVarFlags::ReadAndWrite);
static AutoCVarInt32 cVarEnableStatFrameGraph("stat.frameGraph", "Enable stat frame graph.", "stat", 1, CVarFlags::ReadAndWrite);



static AutoCVarFloat cVarScreenPercentage(
	"r.viewport.screenpercentage",
	"set all deferred renderer screen percentage.",
	"Render",
	1.0f,
	CVarFlags::ReadAndWrite);
static float sCacheScreenPercentage = cVarScreenPercentage.get();

ViewportWidget::ViewportWidget(size_t index)
	: WidgetBase(
		combineIcon("Viewport", kIconViewport).c_str(),
		combineIcon(combineIndex("Viewport", index), kIconViewport).c_str())
{

}

void ViewportWidget::onInit()
{
	// Camera prepare.
	m_camera = std::make_unique<ViewportCamera>(this);

	// Viewport renderer.
	m_deferredRenderer = std::make_unique<DeferredRenderer>();

	m_deferredRendererDelegate = m_renderer->tickCmdFunctions.addLambda([this](const RuntimeModuleTickData& tickData, VkCommandBuffer graphicsCmd, VulkanContext*)
	{
		if(m_bShow)
		{
			m_deferredRenderer->tick(tickData, graphicsCmd, m_camera.get());
		}
	});

	m_flags = ImGuiWindowFlags_NoScrollWithMouse;
}

void ViewportWidget::onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context)
{

}

void ViewportWidget::onRelease()
{
	// Clear renderer.
	m_renderer->tickCmdFunctions.remove(m_deferredRendererDelegate);
	m_deferredRenderer.reset();
}

void ViewportWidget::beforeTick(const engine::RuntimeModuleTickData& tickData)
{
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
}

void ViewportWidget::afterTick(const engine::RuntimeModuleTickData& tickData)
{
	ImGui::PopStyleVar(1);
}

void ViewportWidget::drawProfileViewer(uint32_t width, uint32_t height)
{
	ImGui::Indent(2.0f);
	if (cVarEnableStatUnit.get() > 0)
	{
		const auto& timeStamps = m_deferredRenderer->getTimingValues();
		const bool bTimeStampsAvailable = timeStamps.size() > 0;
		if (bTimeStampsAvailable)
		{
			m_profileViewer.recentHighestFrameTime = 0;

			m_profileViewer.frameTimeArray[m_profileViewer.kNumFrames - 1] = timeStamps.back().microseconds;
			for (uint32_t i = 0; i < m_profileViewer.kNumFrames - 1; i++)
			{
				m_profileViewer.frameTimeArray[i] = m_profileViewer.frameTimeArray[i + 1];
			}
			m_profileViewer.recentHighestFrameTime =
				std::max(m_profileViewer.recentHighestFrameTime, m_profileViewer.frameTimeArray[m_profileViewer.kNumFrames - 1]);
		}
		const float& frameTime_us = m_profileViewer.frameTimeArray[m_profileViewer.kNumFrames - 1];
		const float  frameTime_ms = frameTime_us * 0.001f;
		const int fps = bTimeStampsAvailable ? static_cast<int>(1000000.0f / frameTime_us) : 0;
		static const char* textFormat = "%s : %.2f %s";

		auto profileUI = [&]()
		{
			ui::beginGroupPanel("Profiler");
			{
				ImGui::Text("Resolution : %ix%i", (int32_t)width, (int32_t)height);
				ImGui::Text("FPS : %d (%.2f ms)", fps, frameTime_ms);

				for (uint32_t i = 0; i < timeStamps.size(); i++)
				{
					float value = m_profileViewer.bShowMilliseconds ? timeStamps[i].microseconds / 1000.0f : timeStamps[i].microseconds;
					const char* pStrUnit = m_profileViewer.bShowMilliseconds ? "ms" : "us";
					ImGui::Text(textFormat, timeStamps[i].label.c_str(), value, pStrUnit);
				}
			}
			ImGui::Spacing();
			ui::endGroupPanel();
		};

		const auto srcPos = ImGui::GetCursorPos();
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
		ImGui::BeginDisabled();
		profileUI();
		ImGui::EndDisabled();
		ImGui::PopStyleVar();
		ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 0, 0, 139), 2.0f);

		ImGui::SetCursorPos(srcPos);
		profileUI();
	}

	if (cVarEnableStatFrameGraph.get() > 0)
	{
		size_t iFrameTimeGraphMaxValue = 0;
		size_t iFrameTimeGraphMinValue = 0;
		for (int i = 0; i < m_profileViewer.kCountNum; ++i)
		{
			if (m_profileViewer.recentHighestFrameTime < m_profileViewer.frameTimeGraphMaxValues[i])
			{
				iFrameTimeGraphMaxValue = std::min(int(m_profileViewer.kCountNum - 1), i + 1);
				break;
			}
		}

		auto frameGraphView = [&]()
		{
			ui::beginGroupPanel("GPU frame time (us)");
			{
				ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
				ImGui::PushStyleColor(ImGuiCol_FrameBg, { 0,0,0,0 });
				ImGui::PlotLines("",
					m_profileViewer.frameTimeArray,
					m_profileViewer.kNumFrames,
					0,
					0,
					0.0f,
					m_profileViewer.frameTimeGraphMaxValues[iFrameTimeGraphMaxValue],
					ImVec2(200, 80));
				ImGui::PopStyleColor();
				ImGui::PopStyleVar();
			}
			ui::endGroupPanel();
		};

		const auto srcPos = ImGui::GetCursorPos();
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
		ImGui::BeginDisabled();
		frameGraphView();
		ImGui::EndDisabled();
		ImGui::PopStyleVar();
		ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 0, 0, 139), 2.0f);
		ImGui::SetCursorPos(srcPos);
		frameGraphView();
	}
	ImGui::Unindent();
}

void ViewportWidget::onVisibleTick(const engine::RuntimeModuleTickData& tickData)
{
	ZoneScoped;

	float width = math::ceil(ImGui::GetContentRegionAvail().x);
	float height = math::ceil(ImGui::GetContentRegionAvail().y);
	ImGui::BeginChild("ViewportChild", { width, height }, false, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoDecoration);
	{
		ImVec2 startPos = ImGui::GetCursorPos();

		{
			auto viewImageSet = Editor::get()->getClampToTransparentBorderImGuiTexture(&m_deferredRenderer->getOutputVulkanImage());
			ImGui::Image(viewImageSet, ImVec2(width, height));
		}

		bool bClickViewport = ImGui::IsItemClicked();
		const auto minPos = ImGui::GetItemRectMin();
		const auto maxPos = ImGui::GetItemRectMax();
		const auto mousePos = ImGui::GetMousePos();

		m_bMouseInViewport = ImGui::IsItemHovered();

		auto prevCamPos = m_camera->getPosition();
		m_camera->tick(tickData);
		if (m_camera->getPosition() != prevCamPos)
		{
			Editor::get()->setActiveViewportCameraPos(m_camera->getPosition());
		}

		ImGui::SetCursorPos(startPos);
		ImGui::NewLine();

		// Draw profile viewer.
		drawProfileViewer(width, height);

		// Change viewport size, need notify renderer change render size.
		if (m_cacheWidth != width || m_cacheHeight != height || sCacheScreenPercentage != cVarScreenPercentage.get())
		{
			if (!ImGui::IsMouseDragging(0))
			{
				m_cacheWidth = width;
				m_cacheHeight = height;

	

				sCacheScreenPercentage = cVarScreenPercentage.get();
				sCacheScreenPercentage = math::clamp(sCacheScreenPercentage, 1.0f, 3.0f);
				cVarScreenPercentage.set(sCacheScreenPercentage);
				m_deferredRenderer->updateDimension(uint32_t(width), uint32_t(height), sCacheScreenPercentage, 1.0f);
			}
		}

		// Draw icons for special components.
		bool bClickIcon = false;
		{
			const auto& projection = m_camera->getProjectMatrix();
			const auto& view = m_camera->getViewMatrix();
			const auto camViewProj = projection * view;

			const float kIconDim = 60.0f;
			const auto* builtinResources = Editor::get()->getBuiltinResources();

			auto scene = getSceneManager()->getActiveScene();

			ImTextureID image = Editor::get()->getClampToTransparentBorderImGuiTexture(
				builtinResources->pawnImage.get());
			scene->loopNodeTopToDown([&](std::shared_ptr<SceneNode> node)
			{
				if (node->isRoot())
				{
					return;
				}

				auto drawImage = image;
				if (node->getName().starts_with("Postprocess"))
				{
					drawImage = Editor::get()->getClampToTransparentBorderImGuiTexture(
						builtinResources->postImage.get());
				}
				else if (node->getName().starts_with("Sky"))
				{
					drawImage = Editor::get()->getClampToTransparentBorderImGuiTexture(
						builtinResources->sunImage.get());
				}

				const math::vec4 worldPos = node->getTransform()->getWorldMatrix() * math::vec4(0.0f, 0.0f, 0.0f, 1.0f);
				math::vec4 projPos = camViewProj * worldPos;

				projPos.x = projPos.x / projPos.w;
				projPos.y = projPos.y / projPos.w;
				projPos.z = projPos.z / projPos.w;
				projPos.x = projPos.x * 0.5f + 0.5f;
				projPos.y = projPos.y * 0.5f + 0.5f;

				if (projPos.z > 0.0f && projPos.z < 1.0f)
				{
					int32_t screenPosX = int32_t(projPos.x * width + startPos.x);
					int32_t screenPosY = int32_t((1.0 - projPos.y) * height + startPos.y);

					ImGui::SetCursorPosX(screenPosX - kIconDim * 0.5);
					ImGui::SetCursorPosY(screenPosY - kIconDim * 0.5);

					ImGui::BeginGroup();
					ImGui::Image(drawImage, { kIconDim, kIconDim });
					ImGui::EndGroup();
					if (ImGui::IsItemClicked())
					{
						Editor::get()->getSceneOutlineWidegt()->getSelection().clear();
						Editor::get()->getSceneOutlineWidegt()->getSelection().add(SceneNodeSelctor(node));
						bClickIcon = true;
					}
				}
			}, scene->getRootNode());
		}

		if (bClickViewport && (!bClickIcon) && (!ImGuizmo::IsUsing()) && (!ImGuizmo::IsOver()))
		{
			math::ivec2 samplePos = { mousePos.x - minPos.x, mousePos.y - minPos.y };
			samplePos.x = math::clamp(samplePos.x, 0, int32_t(width) - 1);
			samplePos.y = math::clamp(samplePos.y, 0, int32_t(height) - 1);

			m_deferredRenderer->markCurrentFramePick(samplePos, [&](uint32_t pickId)
			{
				Editor::get()->getSceneOutlineWidegt()->clearSelection();

				if (pickId != 0)
				{
					auto node = getSceneManager()->getActiveScene()->getNode(pickId);
					if (node)
					{
						Editor::get()->getSceneOutlineWidegt()->getSelection().add(SceneNodeSelctor(node));
					}
				}
			});
		}

		// Draw transform handle when scene node selected.
		if (Editor::get()->getSceneOutlineWidegt()->getSelection().getNum() == 1)
		{
			// Mode switch.
			if (!m_camera->isControlingCamera())
			{
				if (ImGui::IsKeyPressed(ImGuiKey_W))
				{
					m_transformHandler.operation = ImGuizmo::TRANSLATE;
				}
				else if (ImGui::IsKeyPressed(ImGuiKey_E))
				{
					m_transformHandler.operation = ImGuizmo::ROTATE;
				}
				else if (ImGui::IsKeyPressed(ImGuiKey_R))
				{
					m_transformHandler.operation = ImGuizmo::SCALE;
				}
			}

			m_transformHandler.mode = ImGuizmo::WORLD;

			ImGuizmo::SetOrthographic(m_transformHandler.bOrthographic);
			ImGuizmo::SetDrawlist();
			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, width, height);

			// Get some data
			const auto& projection = m_camera->getImGizmoProjectMatrix();
			const auto& view = m_camera->getViewMatrix();

			const bool bShiftDown = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
			const float snapValue = m_transformHandler.getSnap();
			const float snapValues[3] = { snapValue, snapValue, snapValue };

			if (auto node = Editor::get()->getSceneOutlineWidegt()->getSelection().getElem(0).node.lock())
			{
				math::mat4 transform = node->getTransform()->getWorldMatrix();

				ImGuizmo::Manipulate(glm::value_ptr(view),
					glm::value_ptr(projection),
					m_transformHandler.operation,
					m_transformHandler.mode,
					glm::value_ptr(transform),
					nullptr,
					bShiftDown ? snapValues : nullptr);

				if (ImGuizmo::IsUsing())
				{
					node->getTransform()->setMatrix(transform);
				}
			}
		}


	}
	ImGui::EndChild();
}


//////////////////////////////////////////////////////////////////////////////////
// Viewport camera.
void ViewportCamera::updateCameraVectors()
{
	// Get front vector from yaw and pitch angel.
	math::vec3 front;
	front.x = cos(math::radians(m_yaw)) * cos(math::radians(m_pitch));
	front.y = sin(math::radians(m_pitch));
	front.z = sin(math::radians(m_yaw)) * cos(math::radians(m_pitch));
	m_front = math::normalize(front);

	// Double cross to get camera true up and right vector.
	m_right = math::normalize(math::cross(m_front, m_worldUp));
	m_up = math::normalize(math::cross(m_right, m_front));
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
	m_moveSpeed = math::clamp(m_moveSpeed, m_minMouseMoveSpeed, m_maxMouseMoveSpeed);
}

ViewportCamera::ViewportCamera(ViewportWidget* inViewport)
	: m_viewport(inViewport)
{
	updateCameraVectors();
}

void ViewportCamera::tick(const RuntimeModuleTickData& tickData)
{
	size_t renderWidth = size_t(m_viewport->getRenderWidth());
	size_t renderHeight = size_t(m_viewport->getRenderHeight());

	const auto& windowData = Engine::get()->getGLFWWindows()->getData();
	float dt = tickData.deltaTime;

	// prepare view size.
	if (m_width != renderWidth)
	{
		m_width = std::max(kMinRenderDim, renderWidth);
	}

	if (m_height != renderHeight)
	{
		m_height = std::max(kMinRenderDim, renderHeight);
	}

	// handle first input.
	if (m_bFirstMouse)
	{
		m_lastX = windowData.getMouseX();
		m_lastY = windowData.getMouseY();

		m_bFirstMouse = false;
	}

	// handle active viewport state.
	m_bActiveViewport = false;
	if (windowData.isMouseButtonPressed(Mouse::ButtonRight))
	{
		if (m_viewport->isMouseInViewport() || m_bHideMouseCursor)
		{
			m_bActiveViewport = true;
		}
	}

	// active viewport. disable cursor.
	if (m_bActiveViewport && !m_bHideMouseCursor)
	{
		m_bHideMouseCursor = true;
		glfwSetInputMode(windowData.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	// un-active viewport. enable cursor.
	if (!m_bActiveViewport && m_bHideMouseCursor)
	{
		m_bHideMouseCursor = false;
		glfwSetInputMode(windowData.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// first time un-active viewport.
	if (m_bActiveViewportLastframe && !m_bActiveViewport)
	{
		m_lastX = windowData.getMouseX();
		m_lastY = windowData.getMouseY();
	}

	// continue active viewport.
	float xoffset = 0.0f;
	float yoffset = 0.0f;
	if (m_bActiveViewportLastframe && m_bActiveViewport)
	{
		xoffset = windowData.getMouseX() - m_lastX;
		yoffset = m_lastY - windowData.getMouseY();
	}

	// update state.
	m_bActiveViewportLastframe = m_bActiveViewport;
	if (m_bActiveViewport)
	{
		m_lastX = windowData.getMouseX();
		m_lastY = windowData.getMouseY();

		processMouseMovement(xoffset, yoffset);
		processMouseScroll(ImGui::GetIO().MouseWheel);

		if (windowData.isKeyPressed(Key::W))
		{
			processKeyboard(EMoveType::Forward, dt);
		}
		if (windowData.isKeyPressed(Key::S))
		{
			processKeyboard(EMoveType::Backward, dt);
		}
		if (windowData.isKeyPressed(Key::A))
		{
			processKeyboard(EMoveType::Left, dt);
		}
		if (windowData.isKeyPressed(Key::D))
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
	m_viewMatrix = math::lookAt(m_position, m_position + m_front, m_up);

	// reverse z.
	m_projectMatrix = math::perspective(m_fovy, getAspect(), m_zFar, m_zNear);

	// Prepare project matrix for imgui gizmo.
	m_imgizmoProjection = math::perspectiveRH_NO(
		m_fovy,
		getAspect(),
		m_zNear,
		m_zFar);
}