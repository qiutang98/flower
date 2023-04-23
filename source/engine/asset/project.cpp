#include "project.h"

namespace engine
{
	void Project::init(const std::filesystem::path& projectFilePath)
	{
		m_projectFilePath = std::filesystem::absolute(projectFilePath);

		// Get project name.
		m_projectName = m_projectFilePath.stem();

		// Get root path.
		m_projectRootPath = m_projectFilePath.parent_path();
	}
}
