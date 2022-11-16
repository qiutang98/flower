#pragma once
#include "RendererCommon.h"

namespace Flower
{
	constexpr auto GCommonShaderStage = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	class PassInterface : NonCopyable
	{
		friend class PassCollector;

	protected:
		virtual void init() {}
		virtual void release() {}
	};

	class PassCollector : NonCopyable
	{
	private:
		std::unordered_map<const char*, std::unique_ptr<PassInterface>> m_passMap;

	public:
		template<typename PassType>
		PassType* getPass()
		{
			static_assert(std::is_base_of_v<PassInterface, PassType>);

			const char* passName = typeid(PassType).name();
			if (!m_passMap.contains(passName))
			{
				m_passMap[passName] = std::make_unique<PassType>();
				m_passMap[passName]->init();
			}

			return dynamic_cast<PassType*>(m_passMap[passName].get());
		}

		void updateAllPasses()
		{
			vkDeviceWaitIdle(RHI::Device);
			for (auto& pair : m_passMap)
			{
				pair.second->release();
				pair.second->init();
			}
		}

		~PassCollector()
		{
			vkDeviceWaitIdle(RHI::Device);
			for (auto& pair : m_passMap)
			{
				pair.second->release();
			}
		}
	};
}