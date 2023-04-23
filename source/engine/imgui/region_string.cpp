#include "region_string.h"

#include <utf8/cpp17.h>
#include <utf8.h>
#include <inipp/inipp/inipp.h>

namespace engine::ui
{
	RegionStringManager::RegionStringManager()
	{
		kDefaultEnglishRegion = "English";

		// 0 default set to english region.
		m_regionId[kDefaultEnglishRegion] = RegionDetail
		{
			.localName = "English",
			.index = 0,
		};

		// Update region and region id.
		m_activeRegion = kDefaultEnglishRegion;
		m_activeRegionId = m_regionId.at(m_activeRegion).index;
	}

	RegionStringManager* RegionStringManager::get()
	{
		static RegionStringManager manager;
		return &manager;
	}

	bool RegionStringManager::setActiveRegion(const std::string& region)
	{
		if (m_regionId.contains(region))
		{
			m_activeRegion = region;
			m_activeRegionId = m_regionId.at(m_activeRegion).index;
			return true;
		}

		return false;
	}

	const std::string& RegionStringManager::getActiveRegionValue(const std::string& name) const
	{
		return m_strings.at(name).at(m_activeRegionId).getValue();
	}

	void RegionStringManager::exportEnglishRegionConfigs(const std::string& outPath)
	{
		const auto& regionDetail = m_regionId.at(kDefaultEnglishRegion);

		inipp::Ini<char> ini;
		auto& baseSection = ini.sections["Base"];
		baseSection["localName"] = std::format("\"{}\"", regionDetail.localName);

		for (auto& groupValuesPair : m_englishGroupValues)
		{
			auto& groupSection = ini.sections[groupValuesPair.first];

			for (auto* str : groupValuesPair.second)
			{
				groupSection[str->getName()] = std::format("\"{}\"", str->getValue());
			}

			ini.interpolate();
		}

		std::ofstream of(outPath);
		ini.generate(of);
	}

	void RegionStringManager::insertEnglishString(const RegionStringInit& inString)
	{
		// Insert string only should only call once!
		auto& regionValues = m_strings[inString.getName()];

		// 0 is default is english region.
		CHECK(regionValues.size() == 0);
		regionValues.push_back(inString);

		// Prepare group values.
		m_englishGroupValues[inString.getGroup()].push_back(&regionValues[0]);
	}

	const std::string& RegionStringInit::getActiveRegionValue() const
	{
		return RegionStringManager::get()->getActiveRegionValue(m_name);
	}

	RegionStringInit::RegionStringInit(
		const std::string& name, 
		const std::string& value, 
		const std::string& group)
		: m_name(name), m_value(value), m_group(group)
	{
		RegionStringManager::get()->insertEnglishString(*this);
	}

}

