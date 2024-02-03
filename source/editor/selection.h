#pragma once

#include <utils/utils.h>

template<typename T>
class Selection
{
public:
	void setChangeCallback(std::function<void(Selection<T>*)>&& f)
	{
		m_onChangeCallback = std::move(f);
	}

	const auto& getSelections() const 
	{ 
		return m_selections; 
	}

	void clear()
	{
		m_selections.clear();

		if (m_onChangeCallback)
		{
			m_onChangeCallback(this);
		}
	}

	void add(const T& in)
	{
		m_selections.push_back(in);
		sortSelection();

		if (m_onChangeCallback)
		{
			m_onChangeCallback(this);
		}
	}

	void sortSelection()
	{
		std::sort(m_selections.begin(), m_selections.end());
	}

	bool isSelected(const T& t) const
	{
		return std::binary_search(m_selections.begin(), m_selections.end(), t);
	}

	size_t getNum() const 
	{ 
		return m_selections.size(); 
	}

	bool existElement() const
	{
		return !m_selections.empty();
	}

	bool empty() const
	{
		return m_selections.empty();
	}

	void remove(const T& t)
	{
		std::erase_if(m_selections, [&t](const T& v)
		{
			return t == v;
		});

		sortSelection();

		if (m_onChangeCallback)
		{
			m_onChangeCallback(this);
		}
	}

	const T& getElem(size_t i) const { return m_selections.at(i); }

private:
	std::vector<T> m_selections;
	std::function<void(Selection<T>*)> m_onChangeCallback = nullptr;
};


struct DragAndDropAssets
{
	void clear()
	{
		selectAssets.clear();
	}

	std::unordered_set<std::filesystem::path> selectAssets;
};