#pragma once

#include "widget.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <asset/asset.h>
#include <utf8/cpp17.h>

#include <scene/scene.h>

template<typename T>
class Selection
{
public:
	const auto& getSelections() const { return m_selections; }
	auto& getSelections() { return m_selections; }

	void clearSelections() 
	{ 
		m_selections.clear(); 
	}

	void addSelected(const T& o)
	{ 
		m_selections.push_back(o); 
		sortSelection();
	}

	void sortSelection() 
	{ 
		std::sort(m_selections.begin(), m_selections.end());
	}

	bool isSelected(const T& t)
	{
		return std::binary_search(m_selections.begin(), m_selections.end(), t);
	}

	size_t getNum() const { return m_selections.size(); }

	void removeSelect(const T& t) 
	{
		std::erase_if(m_selections,[&t](const T& v)
		{ 
			return t == v;
		});

		// Can save?
		sortSelection();
	}

private:
	std::vector<T> m_selections;
};

struct SceneNodeSelctor
{
	std::weak_ptr<engine::SceneNode> node;
	size_t nodeId = ~0;

	SceneNodeSelctor(std::shared_ptr<engine::SceneNode> inNode)
		: node(inNode)
	{
		if (inNode)
		{
			nodeId = inNode->getId();
		}
	}

	bool operator==(const SceneNodeSelctor& rhs) const 
	{ 
		return nodeId == rhs.nodeId;
	}

	bool operator!=(const SceneNodeSelctor& rhs) const { return !(*this == rhs); }

	bool operator<(const SceneNodeSelctor& rhs) const { return nodeId < rhs.nodeId; }

	operator bool() const { return isValid(); }
	bool isValid() const { return node.lock() != nullptr; }
};