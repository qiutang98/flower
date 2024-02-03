#pragma once

#include "../renderer/renderer.h"

namespace engine
{
	inline std::string combineIcon(const std::string& name, const std::string& icon)
	{
		return std::format("  {}  {}", icon, name);
	}

	inline std::string combineIndex(const std::string& name, size_t index)
	{
		return std::format("{} #{}", name, index);
	}

	class WidgetBase
	{
	public:
		explicit WidgetBase(const char* widgetName, const char* name);
		virtual ~WidgetBase() = default;

		// Get widget name.
		const std::string& getName() const { return m_name; }
		const std::string& getWidgetName() const { return m_widgetName; }


		const UUID64u getRuntimeUUID() const { return m_runtimeUUID; }

		// Visible state set and get.
		void setVisible(bool bVisible) { m_bShow = bVisible; }
		bool getVisible() const { return m_bShow; }

		void init() { onInit(); }
		void release() { onRelease(); }

		void tick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context);
		void tickWithCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context);

	protected:
		// event init.
		virtual void onInit() { }

		// event on widget visible state change. sync on tick function first.
		virtual void onHide(const engine::RuntimeModuleTickData& tickData) {  }
		virtual void onShow(const engine::RuntimeModuleTickData& tickData) {  }

		// evetn before tick.
		virtual void beforeTick(const engine::RuntimeModuleTickData& tickData) {}

		// event always tick.
		virtual void onTick(const engine::RuntimeModuleTickData& tickData, engine::VulkanContext* context) {  }

		// event when widget visible tick, draw imgui logic here.
		virtual void onVisibleTick(const engine::RuntimeModuleTickData& tickData) {  }

		virtual void afterTick(const engine::RuntimeModuleTickData& tickData) { }

		// Tick with graphics command.
		virtual void onTickCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd, engine::VulkanContext* context) {  }
		virtual void onVisibleTickCmd(const engine::RuntimeModuleTickData& tickData, VkCommandBuffer cmd) {  }

		// event release.
		virtual void onRelease() {  }

	protected:
		UUID64u m_runtimeUUID;

		std::string m_widgetName;
		std::string m_name;

		// Cache renderer module.
		class RendererManager* m_renderer;

		// Cache engine handle.
		class Engine* m_engine;

		// Cache vulkan context.
		VulkanContext* m_context;

		// Widget show state.
		bool m_bShow;

		// Widget prev frame show state.
		bool m_bPrevShow;

		// Window show flags.
		ImGuiWindowFlags m_flags = 0;
	};

	class WidgetManager : NonCopyable
	{
	public:
		WidgetManager() = default;

		template<typename T, typename... Args>
		[[nodiscard]] T* addWidget(Args... args)
		{
			static_assert(std::is_base_of_v<WidgetBase, T>, "T must derived from WidgetBase.");
			static_assert(std::is_constructible_v<T, Args...>, "T must constructable with default constructor.");

			auto newT = std::make_unique<T>(args...);
			T* result = newT.get();
			m_widgets.push_back(std::move(newT));

			result->init();
			return result;
		}

		template<typename T>
		[[nodiscard]] bool removeWidget(T* handle)
		{
			size_t i = 0;
			for (auto& widget : m_widgets)
			{
				if (widget.get() == handle)
				{
					break;
				}
				i++;
			}

			if (i >= m_widgets.size())
			{
				return false;
			}

			m_widgets[i]->release();

			m_widgets[i] = std::move(m_widgets.back());
			m_widgets.pop_back();
			return true;
		}

		void clearAllWidgets()
		{
			for (auto& widget : m_widgets)
			{
				widget->release();
			}

			m_widgets.clear();
		}

		void release()
		{
			clearAllWidgets();
		}

		void tick(const RuntimeModuleTickData& tickData, VulkanContext* context)
		{
			for (auto& widget : m_widgets)
			{
				widget->tick(tickData, context);
			}
		}

		void tickWithCmd(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd, VulkanContext* context)
		{
			for (auto& widget : m_widgets)
			{
				widget->tickWithCmd(tickData, cmd, context);
			}
		}

	protected:
		std::vector<std::unique_ptr<WidgetBase>> m_widgets;
	};
}

