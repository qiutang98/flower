#include "Pch.h"
#include "../Detail.h"
#include "DrawComponent.h"
#include "EditorAsset.h"

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

	// Replace to character.
	if (ImGui::Button("Mark as character"))
	{
		for (size_t i = 0; i < comp->m_materials.size(); i++)
		{
			auto& mat = comp->m_materials.at(i);
			if (!IsPMXCharacter((EPMXShadingModel)mat.pmxShadingModel))
			{
				mat.pmxShadingModel = int(EPMXShadingModel::CharacterBasic);
			}
		}
	}

	// Draw Materials.
	for (size_t i = 0; i < comp->m_materials.size(); i++)
	{
		ImGui::PushID(i);
		auto& mat = comp->m_materials.at(i);
		if (ImGui::TreeNode(mat.material.m_name.c_str()))
		{
			ImGui::TextDisabled("English Name: %s.", mat.material.m_enName.c_str());

			VkDescriptorSet set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID).get());

			ImGui::TextDisabled("Basic texture: %s.", mat.material.m_texture.c_str());
			if (!mat.material.m_texture.empty())
			{
				set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(mat.material.m_texture).get());
			}
			ImGui::Image(set, { 80 , 80 });
			
			ImGui::TextDisabled("Toon texture: %s.", mat.material.m_toonTexture.c_str());
			if (mat.mmdToonTex != ~0)
			{
				set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(mat.material.m_toonTexture).get());
				ImGui::Image(set, { 80 , 80 });
			}

			ImGui::TextDisabled("Sp texture: %s.", mat.material.m_spTexture.c_str());
			if (mat.mmdSphereTex != ~0)
			{
				set = EditorAssetSystem::get()->getSetByAssetAsSnapShot(TextureManager::get()->getImage(mat.material.m_spTexture).get());
				ImGui::Image(set, { 80 , 80 });
			}

			ImGui::Checkbox("Translucent", &mat.bTranslucent);
			ImGui::Checkbox("Hide", &mat.bHide);

			ImGui::DragFloat("Pixel depth offset", &mat.pixelDepthOffset);

			ImGui::TreePop();
			ImGui::Separator();
		}
		ImGui::PopID();
	}


}