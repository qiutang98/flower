#pragma once

#include <ui/imgui/ImGuizmo.h>

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