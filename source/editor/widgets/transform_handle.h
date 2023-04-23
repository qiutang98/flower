#pragma once

#include "../widget.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>
#include <util/camera_interface.h>
#include <renderer/deferred_renderer.h>
#include <imgui/gizmo/ImGuizmo.h>

struct TransformHandle
{
	ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
	ImGuizmo::MODE mode = ImGuizmo::LOCAL;

	bool bOrthographic = false;

	float snapTranslate = 0.5f;
	float snapRotate = 45.0f;
	float snapScale = 0.5f;

	float getSnap() const
	{
		if (operation == ImGuizmo::TRANSLATE)
		{
			return snapTranslate;
		}
		else if (operation == ImGuizmo::ROTATE)
		{
			return snapRotate;
		}
		else if (operation == ImGuizmo::SCALE)
		{
			return snapScale;
		}
		else
		{
			CHECK_ENTRY();
		}

		return 0.0f;
	}
};