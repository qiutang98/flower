#pragma once
#include "Pch.h"
#include "../RuntimeModule.h"
#include "../Engine.h"
#include "AssetCommon.h"

namespace Flower
{
	struct ImportTextureOptions
	{
		bool bSrgb = true;
		bool bBuildMipmap = true;
		float cutoff = 1.0f;
		bool bHdr = false;
	};

	struct ImportOptions
	{
		std::optional<ImportTextureOptions> texOptions { };
	};

	class RegistryEntry;
	class AssetSystem : public IRuntimeModule
	{
	protected:
		bool m_bProjectSetup = false;
		std::filesystem::path m_projectPath;

		std::filesystem::path m_projectAssetEntryPath;
		std::filesystem::path m_projectAssetHeaderFolderPath;
		std::filesystem::path m_projectAssetBinFolderPath;

	private:
		void engineAssetInit();

	public:
		bool projectAlreadySet() const 
		{ 
			return m_bProjectSetup;
		}

		void setupProject(const std::filesystem::path& path);

		const auto& getProjectPath() const
		{
			return m_projectPath;
		}

		const auto& getProjectBinFolderPath() const
		{
			return m_projectAssetBinFolderPath;
		}

		const auto& getProjectHeaderFolderPath() const
		{
			return m_projectAssetHeaderFolderPath;
		}

		const auto& getProjectAssetEntryPath() const
		{
			return m_projectAssetEntryPath;
		}

		UUID importAsset(const std::filesystem::path& path, EAssetType type, std::shared_ptr<RegistryEntry> entry, const ImportOptions& inOption = {});

	public:
		AssetSystem(ModuleManager* in, std::string name = "AssetSystem");

		virtual bool init() override;
		virtual void release() override;
		virtual void tick(const RuntimeModuleTickData& tickData) override;
	};
}