#include "staticmesh_component.h"

#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include <editor/widgets/content.h>
#include <editor/editor.h>
#include <engine/asset/asset_manager.h>
#include <engine/asset/asset_staticmesh.h>
#include <asset/asset_material.h>


namespace engine
{
	constexpr size_t kMinSubMeshNumStartParallel = 250;

	bool StaticMeshComponent::uiDrawComponent()
	{
		static const auto& meta = StaticMeshComponent::uiComponentReflection();

		auto drawStaticMeshSelect = [&]()
		{
			static const auto suffix = AssetStaticMesh::getCDO()->getSuffix();
			ImGui::Spacing();
			const auto& set = getAssetManager()->getAssetTypeMap(suffix);
			for (const auto& meshId : set)
			{
				auto asset = getAssetManager()->getAsset(meshId);
				if (ImGui::MenuItem((std::string("  ") + ICON_FA_CHESS_PAWN"   " + asset->getSaveInfo().getName()).c_str()))
				{
					setAssetUUID(meshId);
				}

				ui::hoverTip(asset->getSaveInfo().getStorePathU8().c_str());
			}
		};

		bool bChangedValue = false;
		{
			ImGui::Unindent();
			ui::beginGroupPanel("Basic");

			if (getAssetUUID().empty())
			{
				ImGui::TextDisabled("Non-mesh set on the mesh component.");
				ImGui::TextDisabled("Please select one mesh asset for this component.");
			}
			else
			{
				ImGui::TextDisabled("Staticmesh asset setting ready.");
				ImGui::TextDisabled("Asset uuid: %s.", getAssetUUID().c_str());
			}

			ImGui::Spacing();

			ImTextureID set = Editor::get()->getClampToTransparentBorderImGuiTexture(getContext()->getBuiltinTextureTranslucent()->getReadyImage());
			const float kItemDim = ImGui::GetTextLineHeightWithSpacing() * 5.0f;

			ImGui::Image(set, { kItemDim , kItemDim });
			ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 255, 255, 80));

			if (ImGui::BeginDragDropTarget())
			{
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(Editor::getDragDropAssetsName()))
				{
					const auto& dragAssets = Editor::get()->getDragDropAssets();
					if (dragAssets.selectAssets.size() == 1)
					{
						const std::filesystem::path& assetPath = *dragAssets.selectAssets.begin();
						if (AssetStaticMesh::isStaticMesh(assetPath.extension().string().c_str()))
						{
							auto meshAsset = getAssetManager()->getOrLoadAsset<AssetStaticMesh>(assetPath).lock();

							this->setAssetUUID(meshAsset->getSaveInfo().getUUID());
						}
					}
				}
				ImGui::EndDragDropTarget();
			}

			ImGui::SameLine();
			ImGui::BeginGroup();

			static const std::string selectButtonName = meta.iconCreated + " Chose ";
			if (ImGui::Button(selectButtonName.c_str()))
				ImGui::OpenPopup("StaticMeshSelectPopUp");
			if (ImGui::BeginPopup("StaticMeshSelectPopUp"))
			{
				ImGui::TextDisabled("Select StaticMesh...");
				ImGui::Spacing();
				drawStaticMeshSelect();
				ImGui::EndPopup();
			}

			ImGui::TextDisabled("Submesh  count: %d.", getSubmeshCount());
			ImGui::TextDisabled("Vertices count: %d.", getVerticesCount());
			ImGui::TextDisabled("Indices  count: %d.", getIndicesCount());

			ImGui::EndGroup();

			ui::endGroupPanel();
			ImGui::Indent();

		}
		return bChangedValue;
	}

	const UIComponentReflectionDetailed& StaticMeshComponent::uiComponentReflection()
	{
		static const UIComponentReflectionDetailed reflection =
		{
			.bOptionalCreated = true,
			.iconCreated = ICON_FA_BUILDING + std::string("   StaticMesh"),
		};
		return reflection;
	}

	void StaticMeshComponent::tick(const RuntimeModuleTickData& tickData)
	{
		if (!m_assetUUID.empty() && m_meshCache.empty())
		{
			buildCacheSync();
		}
	}

	bool StaticMeshComponent::setAssetUUID(const UUID& in)
	{
		if (m_assetUUID != in)
		{
			clearCache();
			m_assetUUID = in;

			buildCacheSync();

			return true;
		}
		return false;
	}

	uint32_t StaticMeshComponent::getSubmeshCount() const
	{
		if (auto mesh = m_meshCache.assetWeakPtr.lock())
		{
			return (uint32_t)mesh->getSubMeshes().size();
		}
		return 0;
	}

	uint32_t StaticMeshComponent::getVerticesCount() const
	{
		if (auto mesh = m_meshCache.assetWeakPtr.lock())
		{
			return mesh->getVerticesCount();
		}
		return 0;
	}

	uint32_t StaticMeshComponent::getIndicesCount() const
	{
		if (auto mesh = m_meshCache.assetWeakPtr.lock())
		{
			return mesh->getIndicesCount();
		}
		return 0;
	}

	static inline void fillVkAccelerationStructureInstance(VkAccelerationStructureInstanceKHR& as, uint64_t address)
	{
		as.accelerationStructureReference = address;
		as.mask = 0xFF;

		// NOTE: VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR // Faster.
		//       VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR // Two side.
		as.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		as.instanceShaderBindingTableRecordOffset = 0;
	}

	// TODO: Add some cache.
	void StaticMeshComponent::collectRenderObject(RenderScene& renderScene)
	{
		if (m_meshCache.cacheMeshGPU && m_meshCache.cacheMeshGPU->isAssetReady())
		{

		}
		else
		{
			return;
		}

		math::mat4 modelMatrix = getNode()->getTransform()->getWorldMatrix();
		math::mat4 modelMatrixPrev = getNode()->getTransform()->getPrevWorldMatrix();



		const bool bSelected = getNode()->editorSelected();
		const auto sceneNodeId = getNode()->getId();
		if (auto asset = m_meshCache.assetWeakPtr.lock())
		{
			// Update aabb.
			{
				AABBBounds newAABB{};
				newAABB.min = asset->getMinPosition();
				newAABB.max = asset->getMaxPosition();

				newAABB.transform(modelMatrix);

				auto& aabb = renderScene.getAABBBounds();
				aabb.min = math::min(aabb.min, newAABB.min);
				aabb.max = math::max(aabb.max, newAABB.max);
			}

			auto& collector = renderScene.getObjectCollector();
			const size_t objectOffsetId = collector.size();

			auto& meshInfos = m_meshCache.cachePerObjectData;
			auto& rtInfos = m_meshCache.cachePerObjectAs;

			VkAccelerationStructureInstanceKHR instanceTamplate{};
			{
				math::mat4 temp = math::transpose(modelMatrix);
				memcpy(&instanceTamplate.transform, &temp, sizeof(VkTransformMatrixKHR));
			}

			if (m_meshCache.bNewlyCreated)
			{
				m_meshCache.bNewlyCreated = false;
				renderScene.unvalidTLAS();
			}

			auto updateObject = [&](size_t index)
			{
				auto& object = meshInfos[index];
				object.modelMatrix     = modelMatrix;
				object.modelMatrixPrev = modelMatrixPrev;
				object.sceneNodeId = sceneNodeId;
				object.bSelected = bSelected ? 1U : 0U;

				auto& rtObject = rtInfos[index];
				rtObject.transform = instanceTamplate.transform;

				// NOTE: instance custom index used to index object info.
				rtObject.instanceCustomIndex = objectOffsetId + index;
			};

			if (meshInfos.size() > kMinSubMeshNumStartParallel)
			{
				const auto loop = [&, this](const size_t loopStart, const size_t loopEnd)
				{
					for (size_t i = loopStart; i < loopEnd; ++i)
					{
						updateObject(i);
					}
				};
				Engine::get()->getThreadPool()->parallelizeLoop(0, meshInfos.size(), loop).wait();
			}
			else
			{
				for (size_t i = 0; i < meshInfos.size(); i++)
				{
					updateObject(i);
				}
			}

			{
				collector.insert(collector.end(),
					m_meshCache.cachePerObjectData.begin(),
					m_meshCache.cachePerObjectData.end());
			}


			if (getContext()->getGraphicsState().bSupportRaytrace)
			{
				auto& asInstances = renderScene.getBLASObjectCollector();
				asInstances.insert(asInstances.end(), m_meshCache.cachePerObjectAs.begin(), m_meshCache.cachePerObjectAs.end());
			}
		}
	}

	void StaticMeshComponent::clearCache()
	{
		m_meshCache.clear();
	}

	void StaticMeshComponent::buildCacheSync()
	{
		// Pre-return if asset uuid is empty.
		if (m_assetUUID.empty())
		{
			return;
		}

		// Update cache.
		if (!m_meshCache.assetWeakPtr.lock())
		{
			CHECK(m_meshCache.empty());
			CHECK(m_meshCache.cachePerObjectMaterials.empty());

			m_meshCache.assetWeakPtr = std::dynamic_pointer_cast<AssetStaticMesh>(
				getAssetManager()->getAsset(m_assetUUID));

			auto meshAsset = m_meshCache.assetWeakPtr.lock();
			m_meshCache.cacheMeshGPU = meshAsset->getGPUAsset();

			// Fill per-object cache and material.
			const auto& submeshes = meshAsset->getSubMeshes();
			m_meshCache.resize(submeshes.size());
			for (size_t i = 0; i < submeshes.size(); i++)
			{
				const auto& submesh = submeshes[i];
				auto& cacheObject = m_meshCache.cachePerObjectData.at(i);

				// Update mesh info data.
				{
					cacheObject.meshInfoData.meshType = EMeshType_StaticMesh;
					cacheObject.meshInfoData.indicesCount = submesh.indicesCount;
					cacheObject.meshInfoData.indexStartPosition = submesh.indicesStart;
					cacheObject.meshInfoData.indicesArrayId = m_meshCache.cacheMeshGPU->getIndices().bindless;

					cacheObject.meshInfoData.normalsArrayId = m_meshCache.cacheMeshGPU->getNormals().bindless;
					cacheObject.meshInfoData.tangentsArrayId = m_meshCache.cacheMeshGPU->getTangents().bindless;
					cacheObject.meshInfoData.positionsArrayId = m_meshCache.cacheMeshGPU->getPositions().bindless;
					cacheObject.meshInfoData.uv0sArrayId = m_meshCache.cacheMeshGPU->getUV0s().bindless;

					cacheObject.meshInfoData.sphereBounds = math::vec4(submesh.bounds.origin, submesh.bounds.radius);
					cacheObject.meshInfoData.extents = submesh.bounds.extents;
					cacheObject.meshInfoData.submeshIndex = i;
				}

				m_meshCache.cacheMaterialId[i] = submesh.material;
				if (submesh.material.empty())
				{
					cacheObject.materialInfoData = buildDefaultBSDFMaterialInfo();
				}
				else
				{
					// Insert material if no exist cache.
					auto& cacheMaterialPair = m_meshCache.cachePerObjectMaterials[submesh.material];

					auto material = std::static_pointer_cast<AssetMaterial>(
						getAssetManager()->getAsset(submesh.material));

					cacheMaterialPair.handle = material->buildCache();
					cacheMaterialPair.asset  = material;

					cacheObject.materialInfoData = material->getGPUOnly();
				}
			}
		}

		// Sync until all process finish.
		getContext()->waitDeviceIdle();

		// Update material after sync.
		updateMaterials();

		if (getContext()->getGraphicsState().bSupportRaytrace)
		{
			auto& rtInfos = m_meshCache.cachePerObjectAs;

			// Try build blas when loading ready.
			auto& blas = m_meshCache.cacheMeshGPU->getOrBuilddBLAS();
			for (size_t i = 0; i < rtInfos.size(); i++)
			{
				auto& cacheAs = rtInfos.at(i);
				fillVkAccelerationStructureInstance(cacheAs, blas.getBlasDeviceAddress(i));
			}
		}
	}

	void StaticMeshComponent::updateMaterials()
	{
		if (!m_meshCache.assetWeakPtr.lock())
		{
			return;
		}
		CHECK(m_meshCache.cacheMeshGPU);

		for (auto& materialPair : m_meshCache.cachePerObjectMaterials)
		{
			auto& material = materialPair.second.asset;
			material->getAndTryBuildGPU();
		}

		if (m_meshCache.cachePerObjectData.size() > kMinSubMeshNumStartParallel)
		{
			const auto loop = [this](const size_t loopStart, const size_t loopEnd)
			{
				for (size_t i = loopStart; i < loopEnd; ++i)
				{
					auto& object = m_meshCache.cachePerObjectData[i];

					const auto& id = m_meshCache.cacheMaterialId[i];
					const auto& material = m_meshCache.cachePerObjectMaterials.at(id).asset;

					object.materialInfoData = material->getGPUOnly();
				}
			};
			Engine::get()->getThreadPool()->parallelizeLoop(0, m_meshCache.cachePerObjectData.size(), loop).wait();
		}
		else
		{
			for (size_t i = 0; i < m_meshCache.cachePerObjectData.size(); i++)
			{
				auto& object = m_meshCache.cachePerObjectData[i];
				const auto& id = m_meshCache.cacheMaterialId[i];

				if (!id.empty())
				{
					const auto& material = m_meshCache.cachePerObjectMaterials.at(id).asset;
					object.materialInfoData = material->getGPUOnly();
				}
			}
		}
	}
}