#include "Pch.h"
#include "RuntimeModule.h"
#include "Engine.h"

namespace Flower
{
	bool ModuleManager::init()
	{
		if (m_runtimeModules.size() <= 0)
		{
			return true;
		}

		bool result = true;

		for (size_t moduleIndex = 0; moduleIndex < m_runtimeModules.size(); moduleIndex++)
		{
			CHECK(m_runtimeModules[moduleIndex]->canInitCorrectly(moduleIndex) && "Some module init error, crash...");
			if (!m_runtimeModules[moduleIndex]->init())
			{
				LOG_ERROR("Runtime module {0} failed to init.", typeid(*m_runtimeModules[moduleIndex]).name());
				result = false;
			}
		}
		return result;
	}

	void ModuleManager::tick(const RuntimeModuleTickData& tickData)
	{
		if (m_runtimeModules.size() <= 0)
		{
			return;
		}

		for (const auto& runtimeModule : m_runtimeModules)
		{
			runtimeModule->tick(tickData);
		}
	}

	void ModuleManager::release()
	{
		if (m_runtimeModules.size() <= 0)
		{
			return;
		}

		for (size_t i = m_runtimeModules.size(); i > 0; i--)
		{
			m_runtimeModules[i - 1]->release();
		}

		// destruct from end to start.
		for (size_t i = m_runtimeModules.size() - 1; i > 0; i--)
		{
			m_runtimeModules[i].reset();
		}

		m_runtimeModules.clear();
	}
}