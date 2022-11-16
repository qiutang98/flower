#pragma once
#include "Pch.h"
#include "Widget.h"

class Console;

class WidgetConsole : public Widget
{
public:
	WidgetConsole();
	virtual ~WidgetConsole() noexcept;

protected:
	// event init.
	virtual void onInit() override;

	// event always tick.
	virtual void onTick(const Flower::RuntimeModuleTickData& tickData) override;

	// event release.
	virtual void onRelease() override;

	// event when widget visible tick.
	virtual void onVisibleTick(const Flower::RuntimeModuleTickData& tickData) override;

private:
	std::unique_ptr<Console> m_console;
};

class Console
{
private:
	Flower::DelegateHandle m_logCacheHandle;

	// command input buffer.
	char m_inputBuffer[256];

	// precache commands on cvar system.
	ImVector<const char*> m_commands;
	ImVector<char*> m_historyCommands;
	std::vector<const char*> m_activeCommands;

	// -1 is new line, [0, m_historyCommands.size() - 1] is browsing history.
	int32_t m_historyPos;

	// filter for log items.
	ImGuiTextFilter m_filter;

	// whether auto scroll if log items full.
	bool m_bAutoScroll = true;
	bool m_bScrollToBottom;

	bool m_bCommandSelectPop = false;
	int m_selectedCommandIndex = -1;

	// log items deque.
	std::deque<std::pair<std::string, Flower::ELogType>> m_logItems;
	static const uint32_t MAX_LOG_ITEMS_COUNT = 200;
	int32_t m_hoverItem = -1;
	bool m_bLogItemMenuPopup = false;

	bool m_logVisible[(size_t)Flower::ELogType::Max] = {
		true, // Trace
		true, // Info
		true, // Warn
		true, // Error
		true  // Other
	};
	uint32_t m_logTypeCount[(size_t)Flower::ELogType::Max] = {
		0, // Trace
		0, // Info
		0, // Warn
		0, // Error
		0  // Other
	};

private:
	void clearLog();
	void addLog(std::string info, Flower::ELogType type);
	void addLog(const char* fmt, ...);

	void execCommand(const char* command);

	void inputCallbackOnComplete(ImGuiInputTextCallbackData* data);
	void inputCallbackOnEdit(ImGuiInputTextCallbackData* data);
	void inputCallbackOnHistory(ImGuiInputTextCallbackData* data);

public:
	Console() = default;

	void init();
	void draw();
	void release();

	int32_t textEditCallback(ImGuiInputTextCallbackData* data);
};