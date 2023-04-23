#pragma once

#include <util/util.h>
#include <rhi/rhi.h>

namespace engine
{
    // Project interface class for engine framework.
    class Project
    {
    public:
        void init(const std::filesystem::path& projectFilePath);

        // Project file absolute path.
        const std::filesystem::path& getProjectFilePath() const { return m_projectFilePath; }

        // Project root absolute path.
        const std::filesystem::path& getProjectRootPath() const { return m_projectRootPath; }

        // Project name, no included period.
        const std::filesystem::path& getProjectName() const { return m_projectName; }

    protected:
        // Project stem name, not include period.
        std::filesystem::path m_projectName;

        // Project file absolute path file in this computer, this is runtime generate value.
        std::filesystem::path m_projectFilePath;

        // Project root path where project file exist, this is runtime generate value.
        std::filesystem::path m_projectRootPath;

          
    };
}
