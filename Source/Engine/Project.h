#pragma once
#include "Core/Core.h"

namespace Flower
{
	// Flower project.
	class Project
	{
		ARCHIVE_DECLARE;

	//
	private:
		bool m_bValid = false;
		std::set<std::string> m_scenes;
		std::string m_name;
		std::string m_nameWithSuffix;


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

		bool existScene(const std::string& in) const
		{
			return m_scenes.contains(in);
		}

		void addScene(const std::string& in)
		{
			CHECK(!existScene(in));
			m_scenes.insert(in);
		}

		const auto& getScenes() const
		{
			return m_scenes;
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

	inline void saveActiveProject(const std::filesystem::path& path)
	{
		std::ofstream os(path);
		cereal::JSONOutputArchive archive(os);
		archive(ProjectContext::get()->project);
	}
}

#include "Version.h"

template<class Archive>
void Flower::Project::serialize(Archive& archive, uint32_t version)
{
	archive(m_name);
	archive(m_nameWithSuffix);
	archive(m_bValid);
	archive(m_scenes);
}

CEREAL_CLASS_VERSION(Flower::Project, PROJECT_VERSION_CONTROL)