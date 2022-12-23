#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"

using namespace Flower;
using namespace Flower::UI;

void ComponentDrawer::drawPMX(std::shared_ptr<SceneNode> node)
{
	std::shared_ptr<PMXComponent> comp = node->getComponent<PMXComponent>();

	nfdchar_t* readPath = NULL;

	ImGui::Separator();
	{
		ImGui::TextWrapped("PMX: %s, vmd: %s, wave: %s, camera: %s.",
			comp->getPmxPath().c_str(),
			comp->getVmdPath().c_str(),
			comp->getWavPath().c_str(),
			comp->getCameraPath().c_str());



		if (ImGui::Button("  PMX  "))
		{
			nfdresult_t result = NFD_OpenDialog("pmx", NULL, &readPath);
			if (result == NFD_OKAY)
			{
				comp->setPmxPath(readPath);

				free(readPath);
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("  VMD  "))
		{
			nfdresult_t result = NFD_OpenDialog("vmd", NULL, &readPath);
			if (result == NFD_OKAY)
			{
				comp->setVmdPath(readPath);

				free(readPath);
			}
		}

		ImGui::Spacing();
	}
	{
		if (ImGui::Button("   WAVE  "))
		{
			nfdresult_t result = NFD_OpenDialog("wav", NULL, &readPath);
			if (result == NFD_OKAY)
			{
				comp->setWavPath(readPath);

				free(readPath);
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("  CAMERA  "))
		{
			nfdresult_t result = NFD_OpenDialog("vmd", NULL, &readPath);
			if (result == NFD_OKAY)
			{
				comp->setCameraPath(readPath);

				free(readPath);
			}
		}

		ImGui::Spacing();

	}
	{
		ImGui::Spacing();
	}
	ImGui::Separator();
	{
		if (ImGui::Button("Reset Animation"))
		{
			comp->resetAnimation();
		}

		bool bState = comp->getPlayAnimationState();
		ImGui::Checkbox("Paly Animation", &bState);

		comp->setPlayAnimationState(bState);
	}

	ImGui::Separator();
}