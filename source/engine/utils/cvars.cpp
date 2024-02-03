#include "cvars.h"

#include <fstream>
#include <filesystem>
#include <regex>
#include <inipp/inipp.h>

namespace engine
{
	CVarSystem* CVarSystem::get()
	{
		static CVarSystem cVarSystem;
		return &cVarSystem;
	}

	// Export all config to path file.
	void CVarSystem::exportAllConfig(const std::string& path)
	{
		inipp::Ini<char> ini;

		std::unordered_map<std::string, std::vector<const CVarParameter*>> cVarCategories;
		for (const auto& cVarPair : m_cacheCVars)
		{
			const auto& cVar = cVarPair.second;
			cVarCategories[cVar.category].push_back(&cVar);
		}

		for (const auto& cVarCategoryPair : cVarCategories)
		{
			const auto& category = cVarCategoryPair.first;
			auto& cVarSection = ini.sections[category];

			for (const auto* cVar : cVarCategoryPair.second)
			{
				if (cVar->type == CVarType::Int32)
				{
					cVarSection[cVar->name] = std::format("{} # {}", *getCVar<int32_t>(cVar->name), cVar->description);
				}
				else if (cVar->type == CVarType::Bool)
				{
					cVarSection[cVar->name] = std::format("{} # {}", *getCVar<bool>(cVar->name), cVar->description);
				}
				else if (cVar->type == CVarType::String)
				{
					cVarSection[cVar->name] = std::format("{} # {}", *getCVar<std::string>(cVar->name), cVar->description);
				}
				else if (cVar->type == CVarType::Float)
				{
					cVarSection[cVar->name] = std::format("{} # {}", *getCVar<float>(cVar->name), cVar->description);
				}
				else
				{
					assert(false && "Unimplement cvar type.");
				}
			}

			ini.interpolate();
		}

		std::ofstream of(path);
		ini.generate(of);
	}

	bool CVarSystem::importConfig(const std::string& path)
	{
		if (!std::filesystem::exists(path))
		{
			return false;
		}

		inipp::Ini<char> ini;
		std::ifstream is(path);

		ini.parse(is);

		for (const auto& section : ini.sections)
		{
			const auto& keymap = section.second;
			for (const auto& cVarPair : keymap)
			{
				const auto& name = cVarPair.first;

				// Remove comment.
				auto valueStrNoComment = cVarPair.second.substr(0, cVarPair.second.find("#"));

				// Remove whitespace.
				auto valueStr = valueStrNoComment.substr(0, valueStrNoComment.find_last_not_of(" \r\n\t\v\f") + 1);

				const auto& cVar = getCVarParameter(name.c_str());
				if (cVar->type == CVarType::Int32)
				{
					setCVar<int32_t>(name.c_str(), (int32_t)std::stoi(valueStr));
				}
				else if (cVar->type == CVarType::Bool)
				{
					if (valueStr.starts_with("false") || valueStr.starts_with("FALSE") || valueStr.starts_with("0"))
					{
						setCVar<bool>(name.c_str(), false);
					}
					else
					{
						setCVar<bool>(name.c_str(), true);
					}
				}
				else if (cVar->type == CVarType::String)
				{
					setCVar<std::string>(name.c_str(), valueStr);
				}
				else if (cVar->type == CVarType::Float)
				{
					setCVar<float>(name.c_str(), std::stof(valueStr));
				}
				else
				{
					assert(false && "Unimplement cvar type.");
				}
			}
		}

		return true;
	}
}