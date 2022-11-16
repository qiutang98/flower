#pragma once
#include "Core/Core.h"

namespace Flower
{
	// Flower project.
	class Project
	{
	private:
		bool m_bValid = false;

		std::string m_name;
		std::string m_nameWithSuffix;

		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(m_name);
			archive(m_nameWithSuffix);
			archive(m_bValid);
		}


	public:
		Project() = default;
		
		Project(const std::string& name)
			: m_name(name), m_bValid(true)
		{
			m_nameWithSuffix = m_name + ".flower";
		}

		const auto& getName() const { return m_name; }
		const auto& getNameWithSuffix() const { return m_nameWithSuffix; }

		bool isValid() const
		{
			return m_bValid;
		}
	};

	struct ProjectInfoMisc
	{
		// Active project.
		Project project;

		// Active project path. like C://A/A.flower
		// path is C://A/
		std::filesystem::path path;

		bool isValid() const
		{
			return project.isValid();
		}
	};

	using ProjectContext = Singleton<ProjectInfoMisc>;
}