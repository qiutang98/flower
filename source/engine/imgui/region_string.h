#pragma once

#include <util/util.h>

namespace engine::ui
{
	// Region string convert utf16 or utf32 string to utf8, which is support char in IMGUI.
	class RegionStringInit
	{
	private:
		std::string m_group;
		std::string m_name;
		std::string m_value;

	public:
		const std::string& getName() const { return m_name; }
		const std::string& getValue() const { return m_value; }
		const std::string& getGroup() const { return m_group; }

		// Get current active region value.
		const std::string& getActiveRegionValue() const;

		// Get cstr for imgui.
		const char* imgui() const { return getActiveRegionValue().c_str(); }

		// Init as english name.
		RegionStringInit(const std::string& name, const std::string& value, const std::string& group);
	};

	class RegionStringManager
	{
	public:
		friend RegionStringInit;

		static RegionStringManager* get();

		const std::string& getActiveRegion() const { return m_activeRegion; }
		
		bool setActiveRegion(const std::string& region);

		const auto& getRegionIdMap() const { return m_regionId; }

		const std::string& getActiveRegionValue(const std::string& name) const;

		// bool scanRegionConfigs(const std::string& folder);
		void exportEnglishRegionConfigs(const std::string& outPath);

	private:
		RegionStringManager();

		void insertEnglishString(const RegionStringInit& inString);

	private:
		std::string kDefaultEnglishRegion;

		std::map<std::string, std::vector<RegionStringInit>> m_strings;

		// English group values.
		std::map<std::string, std::vector<RegionStringInit*>> m_englishGroupValues;

		struct RegionDetail
		{
			std::string localName;
			size_t index;
		};
		std::map<std::string, RegionDetail> m_regionId;
		
		std::string m_activeRegion;
		size_t m_activeRegionId;
	};
}