#include "Pch.h"
#include "AssetSystem.h"
#include "AssetRegistry.h"
#include "TextureManager.h"
#include "MeshManager.h"
#include "AsyncUploader.h"
#include "MeshManager.h"
#include "LRUCache.h"

namespace Flower
{
	void AssetSystem::addUnusedAsset(std::shared_ptr<LRUAssetInterface> asset)
	{
		m_unusedAseets.push_back(asset);
	}

	void AssetSystem::addUploadTask(std::shared_ptr<AssetLoadTask> inTask)
	{
		m_uploadTasks.push_back(inTask);
	}

	void AssetSystem::flushUploadTask()
	{
		submitAllUploadTask();
		GpuUploader::get()->flushTask();
	}

	AssetSystem::AssetSystem(ModuleManager* in, std::string name)
		: IRuntimeModule(in, name)
	{

	}

	bool AssetSystem::init()
	{
		TextureManager::get()->init();
		MeshManager::get()->init();
		engineAssetInit();

		return true;
	}

	void AssetSystem::engineAssetInit()
	{
		auto GEngineTextureWhiteLoad = RawAssetTextureLoadTask::buildFlatTexture(
			"EngineWhite_4x4x1", 
			EngineTextures::GWhiteTextureUUID, 
			{255, 255, 255, 255});
		GpuUploader::get()->addTask(GEngineTextureWhiteLoad);

		auto GEngineTextureGrayLoad = RawAssetTextureLoadTask::buildFlatTexture(
			"EngineGray_4x4x1",
			EngineTextures::GGreyTextureUUID,
			{ 128, 128, 128, 255 });
		GpuUploader::get()->addTask(GEngineTextureGrayLoad);

		auto GEngineTextureBlackLoad = RawAssetTextureLoadTask::buildFlatTexture(
			"EngineBlack_4x4x1", 
			EngineTextures::GBlackTextureUUID, 
			{ 0,0,0,255 });
		GpuUploader::get()->addTask(GEngineTextureBlackLoad);

		auto GEngineTextureTranslucentLoad = RawAssetTextureLoadTask::buildFlatTexture(
			"EngineTranslucent_4x4x1", 
			EngineTextures::GTranslucentTextureUUID, 
			{ 0,0,0, 0 });
		GpuUploader::get()->addTask(GEngineTextureTranslucentLoad);

		auto GEngineTextureNormalLoad = RawAssetTextureLoadTask::buildFlatTexture(
			"EngineNormal_4x4x1", 
			EngineTextures::GNormalTextureUUID, 
			{125,130,255,0});
		GpuUploader::get()->addTask(GEngineTextureNormalLoad);

		auto GEngineTexturePBRLoad = RawAssetTextureLoadTask::buildFlatTexture(
			"EngineMetalRoughness_4x4x1",
			EngineTextures::GDefaultSpecularUUID,
			{ 255, 255, 0, 0 });
		GpuUploader::get()->addTask(GEngineTexturePBRLoad);

		auto GEngineTextureCloudWeather = RawAssetTextureLoadTask::build(
			"Image/WeatherTexture.tga",
			EngineTextures::GCloudWeatherUUID,
			VK_FORMAT_R8G8B8A8_UNORM);
		GpuUploader::get()->addTask(GEngineTextureCloudWeather);
		 
		auto GEngineTextureClouGradient = RawAssetTextureLoadTask::build(
			"Image/CloudGradient.png",
			EngineTextures::GCloudGradientUUID,
			VK_FORMAT_R8G8B8A8_UNORM);
		GpuUploader::get()->addTask(GEngineTextureClouGradient);

		// Mesh upload.
		auto GEngineMeshBoxLoad = StaticMeshRawDataLoadTask::buildFromPath(
			"EngineMeshBox",
			"./Mesh/Box.obj",
			EngineMeshes::GBoxUUID,
			true
		);
		GpuUploader::get()->addTask(GEngineMeshBoxLoad);

		auto GEngineMeshSphereLoad = StaticMeshRawDataLoadTask::buildFromPath(
			"EngineMeshBox",
			"./Mesh/Sphere.obj",
			EngineMeshes::GSphereUUID,
			true
		);
		GpuUploader::get()->addTask(GEngineMeshSphereLoad);

		GpuUploader::get()->flushTask();

		EngineTextures::GWhiteTextureId = TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID)->getBindlessIndex();

		EngineTextures::GGreyTextureId = TextureManager::get()->getImage(EngineTextures::GGreyTextureUUID)->getBindlessIndex();
		EngineTextures::GBlackTextureId = TextureManager::get()->getImage(EngineTextures::GBlackTextureUUID)->getBindlessIndex();
		EngineTextures::GTranslucentTextureId = TextureManager::get()->getImage(EngineTextures::GTranslucentTextureUUID)->getBindlessIndex();
		EngineTextures::GNormalTextureId = TextureManager::get()->getImage(EngineTextures::GNormalTextureUUID)->getBindlessIndex();
		EngineTextures::GDefaultSpecularId = TextureManager::get()->getImage(EngineTextures::GDefaultSpecularUUID)->getBindlessIndex();

		EngineMeshes::GBoxPtrRef = MeshManager::get()->getMesh(EngineMeshes::GBoxUUID);
		EngineMeshes::GSpherePtrRef = MeshManager::get()->getMesh(EngineMeshes::GSphereUUID);
		EngineTextures::GWhiteTexturePtr = TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID);
		
	}

	void AssetSystem::submitAllUploadTask()
	{
		if (m_uploadTasks.size() > 0)
		{
			for (auto& task : m_uploadTasks)
			{
				GpuUploader::get()->addTask(task);
			}
			m_uploadTasks.clear();
		}
	}

	void AssetSystem::setupProject(const std::filesystem::path& path)
	{
		if (m_projectPath == path)
		{
			return;
		}

		m_projectPath = path;
		m_projectAssetBinFolderPath = path / "Cache";
		m_projectAssetHeaderFolderPath = path / "Header";
		m_projectAssetSceneFolderPath = path / "Scene";
		m_projectAssetEntryPath = path / "Registry.tree";

		if (!std::filesystem::exists(m_projectAssetBinFolderPath))
		{
			std::filesystem::create_directory(m_projectAssetBinFolderPath);
		}

		if (!std::filesystem::exists(m_projectAssetSceneFolderPath))
		{
			std::filesystem::create_directory(m_projectAssetSceneFolderPath);
		}

		if (!std::filesystem::exists(m_projectAssetHeaderFolderPath))
		{
			std::filesystem::create_directory(m_projectAssetHeaderFolderPath);
		}

		AssetRegistryManager::get()->setupProject(
			m_projectAssetEntryPath, 
			m_projectAssetHeaderFolderPath, 
			m_projectAssetBinFolderPath
		);
		
		m_bProjectSetup = true;
	}

	void AssetSystem::tick(const RuntimeModuleTickData& tickData)
	{
		if (m_bCallGPULRUCacheShrink)
		{
			MeshManager::get()->shrinkLRU();
			TextureManager::get()->shrinkLRU();

			// TODO: Can optimize by delay 3 frame to delete avoid gpu wait idle.
			vkDeviceWaitIdle(RHI::Device);
			m_unusedAseets.clear();
			m_bCallGPULRUCacheShrink = false;
		}

		submitAllUploadTask();
	}

	void AssetSystem::release()
	{
		GpuUploader::get()->release();

		MeshManager::get()->release();
		TextureManager::get()->release();

		AssetRegistryManager::get()->release();
	}

	UUID AssetSystem::importAsset(
		const std::filesystem::path& inPath, 
		EAssetType type, 
		std::shared_ptr<RegistryEntry> entry,
		const ImportOptions& inOptions)
	{
		if (entry == nullptr)
		{
			entry = AssetRegistryManager::get()->getRoot();
		}

		UUID assetUUID = {};
		switch (type)
		{ 
		case EAssetType::Texture:
		{
			ImportTextureOptions texOptions{};

			if (inOptions.texOptions.has_value())
			{
				texOptions = inOptions.texOptions.value();
			}

			assetUUID = AssetRegistryManager::get()->importAssetTexture(
				inPath,
				entry,
				texOptions.bSrgb,
				texOptions.cutoff,
				texOptions.bBuildMipmap,
				texOptions.bHdr);
		}
		break;
		case EAssetType::StaticMesh:
		{
			assetUUID = AssetRegistryManager::get()->importStaticMesh(inPath, entry);
		}
		break;
		default:
		{
			CHECK(false && "Non-entry implement.");
		}
		break;
		}

		CHECK(!assetUUID.empty());
		CHECK(AssetRegistryManager::get()->getTypeAssetSetMap().at(size_t(type)).contains(assetUUID));
		CHECK(AssetRegistryManager::get()->getHeaderMap().contains(assetUUID));

		std::shared_ptr<RegistryEntry> newRegistry = std::make_shared<RegistryEntry>(
			assetUUID,
			inPath.stem().string());

		AssetRegistryManager::get()->addChild(entry, newRegistry, true);
		
		return assetUUID;
	}

}