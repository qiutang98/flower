#include "console.h"

using namespace engine;
using namespace engine::ui;

#pragma warning(disable: 4996)

#include <regex>
#include "dockspace.h"


const static std::string kIconConsoleClone  = ICON_FA_COPY;
const static std::string kIconConsoleClear  = ICON_NONE;
const static std::string kIconConsoleTitle  = ICON_FA_COMMENT;
const static std::string kIconConsoleFilter = ICON_FA_MAGNIFYING_GLASS;

class Console
{
private:
	DelegateHandle m_logCacheHandle;

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
	std::deque<std::pair<std::string, ELogType>> m_logItems;
	static const uint32_t kMaxLogsItemNum = 200;
	int32_t m_hoverItem = -1;
	bool m_bLogItemMenuPopup = false;

	bool m_logVisible[(size_t)ELogType::Max] = 
	{
		true, // Trace
		true, // Info
		true, // Warn
		true, // Error
		true, // Fatal
		true  // Other
	};

	uint32_t m_logTypeCount[(size_t)ELogType::Max] =
	{
		0, // Trace
		0, // Info
		0, // Warn
		0, // Error
		0, // Fatal
		0  // Other
	};

private:
	static int myStricmp(const char* s1, const char* s2)
	{
		int d;
		while ((d = toupper(*s2) - toupper(*s1)) == 0 && *s1)
		{
			s1++;
			s2++;
		}
		return d;
	}

	static int myStrnicmp(const char* s1, const char* s2, int n)
	{
		int d = 0;
		while (n > 0 && (d = toupper(*s2) - toupper(*s1)) == 0 && *s1)
		{
			s1++;
			s2++;
			n--;
		}
		return d;
	}

	static char* myStrdup(const char* s)
	{
		IM_ASSERT(s);
		size_t len = strlen(s) + 1;
		void* buf = malloc(len);
		IM_ASSERT(buf);
		return (char*)memcpy(buf, (const void*)s, len);
	}

	static void myStrtrim(char* s)
	{
		char* str_end = s + strlen(s);
		while (str_end > s && str_end[-1] == ' ')
		{
			str_end--;
		}
			
		*str_end = 0;
	}

	static int textEditCallbackStub(ImGuiInputTextCallbackData* data)
	{
		Console* console = (Console*)data->UserData;
		return console->textEditCallback(data);
	}

	void clearLog()
	{
		m_logItems.clear();
		for (auto& i : m_logTypeCount)
		{
			i = 0;
		}
	}

	void addLog(std::string info, ELogType type)
	{
		if (static_cast<uint32_t>(m_logItems.size()) == kMaxLogsItemNum - 1)
		{
			m_logItems.pop_front();
		}

		m_logTypeCount[size_t(type)] ++;
		m_logItems.push_back({ info, type });


	}

	void addLog(const char* fmt, ...)
	{
		char buf[1024];
		va_list args;
		va_start(args, fmt);
		vsnprintf(buf, IM_ARRAYSIZE(buf), fmt, args);
		buf[IM_ARRAYSIZE(buf) - 1] = 0;
		va_end(args);

		addLog(std::string(myStrdup(buf)), ELogType::Other);
	}

	void execCommand(const char* command)
	{
		addLog("# %s\n", command);

		// Insert into history. First find match and delete it so it can be pushed to the back.
		// This isn't trying to be smart or optimal.
		m_historyPos = -1;
		for (int i = m_historyCommands.Size - 1; i >= 0; i--)
		{
			if (myStricmp(m_historyCommands[i], command) == 0)
			{
				free(m_historyCommands[i]);
				m_historyCommands.erase(m_historyCommands.begin() + i);
				break;
			}
		}

		m_historyCommands.push_back(myStrdup(command));

		std::string cmd_info = command;
		std::regex ws_re("\\s+"); // whitespace

		std::vector<std::string> tokens(
			std::sregex_token_iterator(cmd_info.begin(), cmd_info.end(), ws_re, -1),
			std::sregex_token_iterator()
		);

		auto cVar = CVarSystem::get()->getCVarParameter(tokens[0].c_str());
		if (cVar != nullptr)
		{
			// Try to get command value.
			if (tokens.size() == 1)
			{
				static const std::string cmdbeginStr = "cmd.";

				int32_t arrayIndex = cVar->arrayIndex;
				if (tokens[0].rfind(cmdbeginStr, 0) == 0)
				{
					CVarSystem::get()->getArray<bool>().setCurrent(true, arrayIndex);
				}
				else
				{
					std::string val;

					// Load value.
					if (cVar->type == CVarType::Int32)
					{
						val = std::to_string(CVarSystem::get()->getArray<int32_t>().getCurrent(arrayIndex));
					}
					else if (cVar->type == CVarType::Bool)
					{
						val = std::to_string(CVarSystem::get()->getArray<bool>().getCurrent(arrayIndex));
					}
					else if (cVar->type == CVarType::Float)
					{
						val = std::format("{}",CVarSystem::get()->getArray<float>().getCurrent(arrayIndex));
					}
					else if (cVar->type == CVarType::String)
					{
						val = CVarSystem::get()->getArray<std::string>().getCurrent(arrayIndex);
					}

					// Print value.
					{
						std::string outHelp = "Help: " + std::string(cVar->description);
						addLog(outHelp.c_str());
						addLog(("Current value: " + tokens[0] + " " + val).c_str());
					}
				}

			}
			// Try to set command value.
			else if (tokens.size() == 2)
			{
				int32_t arrayIndex = cVar->arrayIndex;
				uint32_t flag = cVar->flag;

				// Readonly flag check.
				const bool bReadOnly = flag & ReadOnly;
				if (bReadOnly)
				{
					addLog("%s is a readonly value, can't change on console!\n", tokens[0].c_str());
				}
				else if (cVar->type == CVarType::Int32)
				{
					CVarSystem::get()->getArray<int32_t>().setCurrent((int32_t)std::stoi(tokens[1]), cVar->arrayIndex);
				}
				else if (cVar->type == CVarType::Bool)
				{
					const bool bNewValue = stringToBoolApprox(tokens[1]);
					CVarSystem::get()->getArray<bool>().setCurrent(bNewValue, cVar->arrayIndex);
				}
				else if (cVar->type == CVarType::Float)
				{
					CVarSystem::get()->getArray<float>().setCurrent((float)std::stof(tokens[1]), cVar->arrayIndex);
				}
				else if (cVar->type == CVarType::String)
				{
					CVarSystem::get()->getArray<std::string>().setCurrent(tokens[1], cVar->arrayIndex);
				}
			}
			else
			{
				addLog("Error command parameters, all CVar only keep one parameter, but '%d' paramters input this time.\n", tokens.size() - 1);
			}
		}
		else
		{
			addLog("Unkonwn command: '%s'!\n", command);
		}

		// On command input, we scroll to bottom even if AutoScroll==false
		m_bScrollToBottom = true;
	}

	void inputCallbackOnComplete(ImGuiInputTextCallbackData* data)
	{
		// locate beginning of current word
		const char* word_end = data->Buf + data->CursorPos;
		const char* word_start = word_end;
		while (word_start > data->Buf)
		{
			const char c = word_start[-1];
			if (c == ' ' || c == '\t' || c == ',' || c == ';')
				break;
			word_start--;
		}

		// build a list of candidates
		ImVector<const char*> candidates;
		for (int i = 0; i < m_commands.Size; i++)
		{
			if (myStrnicmp(m_commands[i], word_start, (int)(word_end - word_start)) == 0)
			{
				candidates.push_back(m_commands[i]);
			}
			else
			{
				std::string lowerCommand = m_commands[i];
				std::transform(lowerCommand.begin(), lowerCommand.end(), lowerCommand.begin(), ::tolower);

				if (myStrnicmp(lowerCommand.c_str(), word_start, (int)(word_end - word_start)) == 0)
				{
					candidates.push_back(m_commands[i]);
				}
			}
		}

		if (candidates.Size == 0)
		{
			// No match
		}
		else if (candidates.Size == 1)
		{
			// Single match. Delete the beginning of the word and replace it entirely so we've got nice casing.
			data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
			data->InsertChars(data->CursorPos, candidates[0]);
			data->InsertChars(data->CursorPos, " ");
		}
		else
		{
			// Multiple matches. Complete as much as we can..
			// So inputing "C"+Tab will complete to "CL" then display "CLEAR" and "CLASSIFY" as matches.
			int match_len = (int)(word_end - word_start);
			for (;;)
			{
				int c = 0;
				bool all_candidates_matches = true;
				for (int i = 0; i < candidates.Size && all_candidates_matches; i++)
					if (i == 0)
						c = toupper(candidates[i][match_len]);
					else if (c == 0 || c != toupper(candidates[i][match_len]))
						all_candidates_matches = false;
				if (!all_candidates_matches)
					break;
				match_len++;
			}

			if (match_len > 0)
			{
				data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
				data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
			}
		}
	}

	void inputCallbackOnEdit(ImGuiInputTextCallbackData* data)
	{
		m_activeCommands.clear();

		// locate beginning of current word
		const char* word_end = data->Buf + data->CursorPos;
		const char* word_start = word_end;
		while (word_start > data->Buf)
		{
			const char c = word_start[-1];
			if (c == ' ' || c == '\t' || c == ',' || c == ';')
				break;
			word_start--;
		}

		// build a list of candidates
		ImVector<const char*> candidates;
		for (int i = 0; i < m_commands.Size; i++)
		{
			int size = (int)(word_end - word_start);

			if (size > 0 && myStrnicmp(m_commands[i], word_start, size) == 0)
			{
				candidates.push_back(m_commands[i]);
			}
			else
			{
				std::string lowerCommand = m_commands[i];
				std::transform(lowerCommand.begin(), lowerCommand.end(), lowerCommand.begin(), ::tolower);

				if (size > 0 && myStrnicmp(lowerCommand.c_str(), word_start, size) == 0)
				{
					candidates.push_back(m_commands[i]);
				}
			}
		}

		if (candidates.Size >= 1)
		{
			for (int i = 0; i < candidates.Size; i++)
			{
				m_activeCommands.push_back(candidates[i]);
			}
		}
	}

	void inputCallbackOnHistory(ImGuiInputTextCallbackData* data)
	{
		if (m_bCommandSelectPop)
		{
			if (data->EventKey == ImGuiKey_UpArrow)
			{
				if (m_selectedCommandIndex <= 0)
				{
					m_selectedCommandIndex = int(m_activeCommands.size()) - 1;
				}
				else
				{
					m_selectedCommandIndex--;
				}
			}
			else if (data->EventKey == ImGuiKey_DownArrow)
			{
				if (++m_selectedCommandIndex >= m_activeCommands.size())
					m_selectedCommandIndex = 0;

			}

			if (m_bCommandSelectPop && m_selectedCommandIndex >= 0)
			{
				data->DeleteChars(0, data->BufTextLen);
				data->InsertChars(0, m_activeCommands[m_selectedCommandIndex]);
			}

			return;
		}

		const int prev_history_pos = m_historyPos;
		if (data->EventKey == ImGuiKey_UpArrow)
		{
			if (m_historyPos == -1)
				m_historyPos = m_historyCommands.Size - 1;
			else if (m_historyPos > 0)
				m_historyPos--;
		}
		else if (data->EventKey == ImGuiKey_DownArrow)
		{
			if (m_historyPos != -1)
				if (++m_historyPos >= m_historyCommands.Size)
					m_historyPos = -1;
		}

		// A better implementation would preserve the data on the current input line along with cursor position.
		if (prev_history_pos != m_historyPos)
		{
			const char* history_str = (m_historyPos >= 0) ? m_historyCommands[m_historyPos] : "";
			data->DeleteChars(0, data->BufTextLen);
			data->InsertChars(0, history_str);
		}
	}

public:
	Console() = default;

	void init()
	{
		auto& int32Array = CVarSystem::get()->getArray<int32_t>();
		auto& floatArray = CVarSystem::get()->getArray<float>();
		auto& boolArray = CVarSystem::get()->getArray<bool>();
		auto& stringArray = CVarSystem::get()->getArray<std::string>();

		auto parseArray = [&](auto& cVarsArray)
		{
			auto typeBit = cVarsArray.lastCVar;
			while (typeBit > 0)
			{
				typeBit--;
				auto storageValue = cVarsArray.getCurrentStorage(typeBit);
				m_commands.push_back(storageValue->parameter->name);
			}
		};

		clearLog();
		memset(m_inputBuffer, 0, sizeof(m_inputBuffer));
		m_historyPos = -1;

		parseArray(int32Array);
		parseArray(floatArray);
		parseArray(boolArray);
		parseArray(stringArray);

		m_bAutoScroll = true;
		m_bScrollToBottom = false;

		m_logCacheHandle = LoggerSystem::get()->pushCallback([&](const std::string& info, ELogType type)
		{
			this->addLog(info, type);
		});
	}

	std::string Console_Title = "Console";

	void draw()
	{
		ImGui::Separator();
		// Log type visibility toggle.
		{
			const auto buttonLogTypeVisibilityToggle = [this](ELogType index, std::string Name)
			{
				bool& visibility = m_logVisible[(size_t)index];

				std::string numCount = (m_logTypeCount[(size_t)index] <= 99) ? std::format("{}", m_logTypeCount[(size_t)index]) : "99+";

				ImGui::Checkbox(std::format(" {} [{}] ", Name, numCount).c_str(), &visibility);
			};

			const std::string sFilterName = combineIcon("Filter", kIconConsoleFilter);
			
			m_filter.Draw(sFilterName.c_str(), 180);
			ImGui::SameLine();

			buttonLogTypeVisibilityToggle(ELogType::Trace, "Trace");
			ImGui::SameLine();

			buttonLogTypeVisibilityToggle(ELogType::Info,  "Info");
			ImGui::SameLine();

			buttonLogTypeVisibilityToggle(ELogType::Warn, "Warn");
			ImGui::SameLine();

			buttonLogTypeVisibilityToggle(ELogType::Error, "Error");
			ImGui::SameLine();

			buttonLogTypeVisibilityToggle(ELogType::Other, "Other");
		}

		ImGui::Separator();

		// Reserve enough left-over height for 1 separator + 1 input text
		const float footerHeightToReserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
		ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footerHeightToReserve), false, ImGuiWindowFlags_HorizontalScrollbar);

		if (!ImGui::IsItemHovered() && !m_bLogItemMenuPopup)
		{
			m_hoverItem = -1;
		}

		// Tighten spacing
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

		// Print log items.
		for (int i = 0; i < m_logItems.size(); i++)
		{
			const char* item = m_logItems[i].first.c_str();
			if (item == "" || !m_filter.PassFilter(item))
				continue;

			ImVec4 color;
			bool bHasColor = false;
			bool bOutSeparate = false;
			if (m_logItems[i].second == ELogType::Error)
			{
				if (!m_logVisible[size_t(ELogType::Error)])
					continue;

				color = ImVec4(1.0f, 0.08f, 0.08f, 1.0f);
				bHasColor = true;
			}
			else if (m_logItems[i].second == ELogType::Warn)
			{
				if (!m_logVisible[size_t(ELogType::Warn)])
					continue;

				color = ImVec4(1.0f, 1.0f, 0.1f, 1.0f);
				bHasColor = true;
			}
			else if (m_logItems[i].second == ELogType::Trace)
			{
				if (!m_logVisible[size_t(ELogType::Trace)])
					continue;
			}
			else if (m_logItems[i].second == ELogType::Info)
			{
				if (!m_logVisible[size_t(ELogType::Info)])
					continue;
			}
			else if (strncmp(item, "# ", 2) == 0)
			{
				if (!m_logVisible[size_t(ELogType::Other)])
					continue;
				bOutSeparate = true;
				color = ImVec4(1.0f, 0.8f, 0.6f, 1.0f);
				bHasColor = true;
			}
			else if (strncmp(item, "Help: ", 5) == 0)
			{
				if (!m_logVisible[size_t(ELogType::Other)])
					continue;

				color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);
				bHasColor = true;
			}
			else
			{
				if (!m_logVisible[size_t(ELogType::Other)])
					continue;
			}


			if (bOutSeparate)
			{
				ImGui::Separator();
			}

			if (bHasColor)
				ImGui::PushStyleColor(ImGuiCol_Text, color);
			ImGui::Selectable(item, m_hoverItem == i);
			if (bHasColor)
				ImGui::PopStyleColor();

			if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly) && !m_bLogItemMenuPopup)
			{
				m_hoverItem = i;
			}
		}

		if (ImGui::BeginPopupContextWindow())
		{
			m_bLogItemMenuPopup = true;

			std::string sCopyName  = combineIcon(" Copy  Item ", kIconConsoleClone);
			std::string sClearName = combineIcon("Clear  All ", kIconConsoleClear);
			if (ImGui::Selectable(sClearName.c_str()))
			{
				clearLog();
			}
			ImGui::Spacing();

			if (m_hoverItem >= 0 && m_hoverItem < kMaxLogsItemNum)
			{
				if (ImGui::Selectable(sCopyName.c_str()))
				{
					ImGui::SetClipboardText(m_logItems[m_hoverItem].first.c_str());
				}
			}
			else
			{
				ImGui::TextDisabled(sCopyName.c_str());
			}

			ImGui::EndPopup();
		}
		else
		{
			m_bLogItemMenuPopup = false;
		}

		if (m_bScrollToBottom || (m_bAutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()))
		{
			ImGui::SetScrollHereY(1.0f);
		}

		m_bScrollToBottom = false;

		ImGui::PopStyleVar();
		ImGui::EndChild();

		ImGui::Separator();

		// Command-line
		bool reclaim_focus = false;
		ImGuiInputTextFlags input_text_flags =
			ImGuiInputTextFlags_EnterReturnsTrue |
			ImGuiInputTextFlags_CallbackEdit |
			ImGuiInputTextFlags_CallbackCompletion |
			ImGuiInputTextFlags_CallbackHistory;

		auto TipPos = ImGui::GetWindowPos();

		TipPos.x += ImGui::GetStyle().WindowPadding.x;
		const float WindowHeight = ImGui::GetWindowHeight();
		const auto ItemSize = ImGui::GetTextLineHeightWithSpacing();
		TipPos.y = TipPos.y + WindowHeight - (m_activeCommands.size() + 2.5f) * ItemSize;

		// CVar Input.
		if (ImGui::InputText("CVar Input", m_inputBuffer, IM_ARRAYSIZE(m_inputBuffer), input_text_flags, &textEditCallbackStub, (void*)this))
		{
			char* s = m_inputBuffer;
			myStrtrim(s);
			if (s[0])
			{
				execCommand(s);
				m_activeCommands.clear();
			}

			strcpy(s, "");
			reclaim_focus = true;
		}

		if (m_activeCommands.size() > 0 && ImGui::IsWindowFocused())
		{
			ImGui::SetNextWindowPos(TipPos);
			ImGui::BeginTooltip();
			for (size_t i = 0; i < m_activeCommands.size(); i++)
			{
				ImGui::Selectable(m_activeCommands[i], m_selectedCommandIndex == i);
			}
			ImGui::EndTooltip();
			m_bCommandSelectPop = true;
		}
		else
		{
			m_bCommandSelectPop = false;
			m_selectedCommandIndex = -1;
		}

		// Auto-focus on window apparition
		ImGui::SetItemDefaultFocus();
		if (reclaim_focus)
		{
			ImGui::SetKeyboardFocusHere(-1); // Auto focus previous widget
		}
	}

	void release()
	{
		LoggerSystem::get()->popCallback(m_logCacheHandle);

		clearLog();
		for (int i = 0; i < m_historyCommands.Size; i++)
		{
			free(m_historyCommands[i]);
		}
	}

	int32_t textEditCallback(ImGuiInputTextCallbackData* data)
	{
		// do event callback.
		switch (data->EventFlag)
		{
		case ImGuiInputTextFlags_CallbackCompletion:
		{
			inputCallbackOnComplete(data);
			break;
		}
		case ImGuiInputTextFlags_CallbackEdit:
		{
			inputCallbackOnEdit(data);
			break;
		}
		case ImGuiInputTextFlags_CallbackHistory:
		{
			inputCallbackOnHistory(data);
			break;
		}
		}
		return 0;
	}
};


// Widget console.

WidgetConsole::WidgetConsole() : WidgetBase(combineIcon("Console", kIconConsoleTitle).c_str(), combineIcon("Console", kIconConsoleTitle).c_str())
{
	m_console = std::make_unique<Console>();
}

const std::string& WidgetConsole::getShowName()
{
	static const auto name = combineIcon("Console", kIconConsoleTitle);
	return name;
}

void WidgetConsole::onInit()
{
	m_console->init();
}

void WidgetConsole::onRelease()
{
	m_console->release();
}

void WidgetConsole::onTick(const RuntimeModuleTickData& tickData, VulkanContext* context) 
{
	m_name = getShowName();
}


void WidgetConsole::onVisibleTick(const RuntimeModuleTickData& tickData)
{
	ZoneScoped;
	m_console->draw();
}