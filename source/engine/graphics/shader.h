#pragma once
#include "base.h"
#include "log.h"

namespace engine
{
	class PassInterface;
	struct ShaderStageInfoForCompile
	{
		std::wstring name;
		std::wstring basicDefined;
	};

	enum EShaderStage
	{
		eVertexShader = 0,
		ePixelShader,
		eComputeShader,

		eMax,
	};
	constexpr size_t kMaxShaderStage = size_t(EShaderStage::eMax);
	extern const std::array<ShaderStageInfoForCompile, kMaxShaderStage> kShaderCompileInfos;

	class ShaderVariant
	{
		friend class ShaderCache;
	public:
		explicit ShaderVariant(std::filesystem::path path = {}, EShaderStage stage = EShaderStage::eMax)
			: m_path(path), m_shaderStage(stage)
		{

		}

		ShaderVariant& setStage(EShaderStage type)
		{
			m_shaderStage = type;
			return *this;
		}

		ShaderVariant& setMacro(const std::wstring& name)
		{
			m_macroSet.insert(name);
			return *this;
		}

		ShaderVariant& setInt(const std::wstring& name, int32_t value)
		{
			m_keyMap[name] = value;
			return *this;
		}

		ShaderVariant& setBool(const std::wstring& name, bool bValue)
		{
			m_keyMap[name] = int32_t(bValue);
			return *this;
		}

		// Check this shader variant is valid.
		bool isValid() const
		{
			return (!m_path.empty()) && (m_shaderStage != EShaderStage::eMax);
		}

	private:
		size_t getHash() const;

	private:
		std::filesystem::path m_path;
		EShaderStage m_shaderStage;

		std::unordered_map<std::wstring, int32_t> m_keyMap;
		std::unordered_set<std::wstring> m_macroSet;
	};

	class ShaderCache final : NonCopyable
	{
	public:
		VkShaderModule getShader(const ShaderVariant& variant, bool reload = false, bool recompile = false);

		~ShaderCache()
		{
			release(false);
		}

		void release(bool bDeleteFiles);

	private:
		std::unordered_map<size_t, VkShaderModule> m_moduleCache;

		void releaseModule(VkShaderModule shader);

		// Create shader module which may blocking application.
		[[nodiscard]] VkShaderModule createShaderModule(const ShaderVariant& variant, bool bRecompile = false) const;
	};
}