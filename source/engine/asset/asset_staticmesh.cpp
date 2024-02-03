#include "asset_staticmesh.h"
#include "../ui/ui.h"

#include <rttr/registration.h>
#include "assimp_import.h"
#include "asset_manager.h"
#include "../serialization/serialization.h"
#include "graphics/context.h"
#include <renderer/render_scene.h>
#include <profile/profile.h>
namespace engine
{
	AssetStaticMesh::AssetStaticMesh(const AssetSaveInfo& saveInfo)
		: AssetInterface(saveInfo)
	{

	}

	void AssetStaticMesh::onPostAssetConstruct()
	{

	}

	const AssetStaticMesh* AssetStaticMesh::getCDO()
	{
		static AssetStaticMesh mesh{ };
		return &mesh;
	}

	static void drawStaticMeshImportConfig(
		std::shared_ptr<AssetImportConfigInterface> ptr)
	{
		auto gltfConfig = std::static_pointer_cast<AssetStaticMeshImportConfig>(ptr);
		ImGui::Spacing();
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));


		ImGui::PushID(std::hash<std::string>{}(gltfConfig->path.second.string()));
		ImGui::Indent();
		{
			std::string utf8Name = utf8::utf16to8(gltfConfig->path.first.u16string());
			std::string saveUtf8 = utf8::utf16to8(gltfConfig->path.second.u16string());

			ImGui::TextDisabled(std::format("Load gltf from: {}", utf8Name).c_str());
			ImGui::TextDisabled(std::format("Save gltf to: {}", saveUtf8).c_str());
			ImGui::Spacing();

		}
		ImGui::Unindent();
		ImGui::PopStyleVar();
		ImGui::PopID();

		ImGui::Spacing();
		ImGui::Spacing();
		ImGui::Separator();
	}

	static bool importStaticMeshFromConfigThreadSafe(
		std::shared_ptr<AssetImportConfigInterface> inPtr)
	{
		auto ptr = std::static_pointer_cast<AssetStaticMeshImportConfig>(inPtr);

		const std::filesystem::path& srcPath = ptr->path.first;
		const std::filesystem::path& savePath = ptr->path.second;

		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(srcPath.string(),
			aiProcessPreset_TargetRealtime_Quality |
			aiProcess_FlipUVs | 
			aiProcess_GenBoundingBoxes);

		if (scene == nullptr)
		{
			LOG_ERROR("Mesh {} import fail.", utf8::utf16to8(srcPath.u16string()));
			return false;
		}

		std::string assetNameUtf8 = utf8::utf16to8(savePath.filename().u16string());

		if (std::filesystem::exists(savePath))
		{
			LOG_ERROR("Path {0} already exist, asset {1} import fail!", utf8::utf16to8(savePath.u16string()), assetNameUtf8);
			return false;
		}

		if (!std::filesystem::create_directory(savePath))
		{
			LOG_ERROR("Folder {0} create failed, asset {1} import fail!", utf8::utf16to8(savePath.u16string()), assetNameUtf8);
			return false;
		}

		const auto textureFolderPath = savePath / "textures";
		const auto materialFolderPath = savePath / "materials";
		const auto rawAssetFolderPath = savePath / "raw";

		std::filesystem::create_directory(textureFolderPath);
		std::filesystem::create_directory(materialFolderPath);
		std::filesystem::create_directory(rawAssetFolderPath);

		const auto name = savePath.filename().u16string() + utf8::utf8to16(AssetStaticMesh::getCDO()->getSuffix());
		const auto relativePathUtf8 = buildRelativePathUtf8(getAssetManager()->getProjectConfig().assetPath, savePath);

		auto saveInfo = AssetSaveInfo(utf8::utf16to8(name), relativePathUtf8);
		auto meshPtr = getAssetManager()->createAsset<AssetStaticMesh>(saveInfo).lock();
		meshPtr->markDirty();

		// Copy raw asset.
		{
			auto copyDest = rawAssetFolderPath / srcPath.filename();
			std::filesystem::copy(srcPath, copyDest);

			// Copy material path.
			{
				auto fileName = srcPath.filename();
				auto mtlFileName = fileName.stem();
				mtlFileName += ".mtl";

				std::filesystem::path mtlPath = srcPath.parent_path();
				mtlPath /= mtlFileName;

				if (std::filesystem::exists(mtlPath))
				{
					auto copyDestMtl = rawAssetFolderPath / mtlPath.filename();
					std::filesystem::copy(mtlPath, copyDestMtl);
				}
			}

			meshPtr->m_rawAssetPath = buildRelativePathUtf8(getAssetManager()->getProjectConfig().assetPath, rawAssetFolderPath);
		}

		AssimpStaticMeshImporter processor(srcPath, materialFolderPath, textureFolderPath);
		processor.processNode(scene->mRootNode, scene);

		// Save asset meta.
		processor.fillMeshAssetMeta(*meshPtr);

		// Save static mesh binary file.
		{
			StaticMeshBin meshBin{};
			meshBin.indices = processor.moveIndices();
			meshBin.tangents = processor.moveTangents();
			meshBin.normals = processor.moveNormals();
			meshBin.uv0s = processor.moveUv0s();
			meshBin.positions = processor.movePositions();

			saveAsset(meshBin, meshPtr->getBinPath(), false);
		}

		return meshPtr->save();
	}

	const AssetReflectionInfo& AssetStaticMesh::uiGetAssetReflectionInfo()
	{
		const static AssetReflectionInfo kInfo =
		{
			.name = "StaticMesh",
			.icon = ICON_FA_BUILDING,
			.decoratedName = std::string("  ") + ICON_FA_BUILDING + std::string("     StaticMesh"),
			.importConfig =
			{
				.bImportable = true,
				.importRawAssetExtension = "obj",
				.buildAssetImportConfig = []() 
				{ 
					return std::make_shared<AssetStaticMeshImportConfig>(); 
				},
				.drawAssetImportConfig = [](AssetReflectionInfo::ImportConfigPtr ptr) 
				{ 
					drawStaticMeshImportConfig(ptr); 
				},
				.importAssetFromConfigThreadSafe = [](AssetReflectionInfo::ImportConfigPtr ptr) 
				{ 
					return importStaticMeshFromConfigThreadSafe(ptr); 
				},
			}
		};
		return kInfo;
	}

	bool AssetStaticMesh::saveImpl()
	{
		std::shared_ptr<AssetInterface> asset = getptr<AssetStaticMesh>();
		return saveAsset(asset, getSavePath(), false);
	}

	void AssetStaticMesh::unloadImpl()
	{

	}


	void AssetStaticMeshLoadFromCacheTask::uploadFunction(
		uint32_t stageBufferOffset, 
		void* bufferPtrStart, 
		RHICommandBufferBase& commandBuffer, 
		VulkanBuffer& stageBuffer)
	{
		StaticMeshBin meshBin{};
		if (!std::filesystem::exists(cachePtr->getBinPath()))
		{
			UN_IMPLEMENT();
		}
		else
		{
			LOG_TRACE("Found bin for asset {} cache in disk so just load.",
				utf8::utf16to8(cachePtr->getSaveInfo().getStorePath()));
			loadAsset(meshBin, cachePtr->getBinPath());
		}

		uint32_t sizeAccumulate = 0;

		auto copyBuffer = [&](const GPUStaticMeshAsset::ComponentBuffer& comp, const void* data)
		{
			VkBufferCopy regionCopy{ };

			regionCopy.size      = comp.stripeSize * comp.num;
			regionCopy.srcOffset = stageBufferOffset + sizeAccumulate;
			regionCopy.dstOffset = 0;

			memcpy((void*)((char*)bufferPtrStart + sizeAccumulate), data, regionCopy.size);

			vkCmdCopyBuffer(commandBuffer.cmd, stageBuffer, comp.buffer->getVkBuffer(), 1, &regionCopy);

			sizeAccumulate += regionCopy.size;
		};

		copyBuffer(meshAssetGPU->getIndices(),   meshBin.indices.data());
		copyBuffer(meshAssetGPU->getPositions(), meshBin.positions.data());
		copyBuffer(meshAssetGPU->getNormals(),   meshBin.normals.data());
		copyBuffer(meshAssetGPU->getUV0s(),      meshBin.uv0s.data());
		copyBuffer(meshAssetGPU->getTangents(),  meshBin.tangents.data());

		ASSERT(uploadSize() == sizeAccumulate, "Static mesh size un-match!");
	}

	std::shared_ptr<AssetStaticMeshLoadFromCacheTask> 
		AssetStaticMeshLoadFromCacheTask::build(std::shared_ptr<AssetStaticMesh> meta)
	{
		auto newAsset = std::make_shared<GPUStaticMeshAsset>(
			meta,
			getContext()->getBuiltinStaticMeshBox().get(),
			meta->getSaveInfo().getName(),
			meta->getVerticesCount(),
			meta->getIndicesCount()
		);

		getContext()->insertLRUAsset(meta->getBinUUID(), newAsset);

		auto newTask  = std::make_shared<AssetStaticMeshLoadFromCacheTask>();
		newTask->meshAssetGPU = newAsset;
		newTask->cachePtr = meta;

		return newTask;
	}

	std::shared_ptr<GPUStaticMeshAsset> AssetStaticMesh::getGPUAsset()
	{
		ZoneScoped;
		if (!m_gpuWeakPtr.lock())
		{
			if (getSaveInfo().isBuiltin())
			{
				m_gpuWeakPtr = getContext()->getBuiltinStaticMesh(getSaveInfo().getName());
			}
			else 
			{
				if (!getContext()->isLRUAssetExist(getBinUUID()))
				{
					auto newTask = AssetStaticMeshLoadFromCacheTask::build(getptr<AssetStaticMesh>());
					getContext()->getAsyncUploader().addTask(newTask);
				}

				m_gpuWeakPtr =
					std::dynamic_pointer_cast<GPUStaticMeshAsset>(getContext()->getLRU()->tryGet(getBinUUID()));
			}
		}

		return m_gpuWeakPtr.lock();
	}
}