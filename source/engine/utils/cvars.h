#pragma once

#include "noncopyable.h"
 
#include <unordered_map>
#include <cassert>
#include <shared_mutex>
#include <functional>
#include <string>
#include <algorithm>

// Simple cVar system.

namespace engine
{
	// CVar static element array max size.
	constexpr auto kCVarMaxInt32Num  = 5000;
	constexpr auto kCVarMaxFloatNum  = 5000;
	constexpr auto kCVarMaxBoolNum   = 5000;
	constexpr auto kCVarMaxStringNum = 2500;

	enum CVarFlags
	{
		None,

		// Read only in editor or other console input. can change by code or config files.
		ReadOnly     = 0x00000001 << 0,

		// Read and write, no limit.
		ReadAndWrite = 0x00000001 << 1,

		Max,
	};
	static_assert(CVarFlags::Max < 0x10000001);

	enum class CVarType : uint8_t
	{
		None = 0x00,

		Int32,
		Bool,
		Float,
		String,

		Max,
	};

	struct CVarParameter
	{
		int32_t  arrayIndex;
		CVarType type;
		uint32_t flag;

		const char* category;
		const char* name;
		const char* description;
	};

	template<typename T>
	struct CVarStorage
	{
		T initVal;
		T currentVal;
		CVarParameter* parameter = nullptr;
	};

	template<typename T>
	constexpr void checkRangeValid(int32_t index);

	template<>
	constexpr void checkRangeValid<float>(int32_t index)
	{
		assert(index < kCVarMaxFloatNum && "Cvar float count overflow, please add more capacity.");
	}

	template<>
	constexpr void checkRangeValid<int32_t>(int32_t index)
	{
		assert(index < kCVarMaxInt32Num && "Cvar int32 count overflow, please add more capacity.");
	}

	template<>
	constexpr void checkRangeValid<bool>(int32_t index)
	{
		assert(index < kCVarMaxBoolNum && "Cvar bool count overflow, please add more capacity.");
	}

	template<>
	constexpr void checkRangeValid<std::string>(int32_t index)
	{
		assert(index < kCVarMaxStringNum && "Cvar string count overflow, please add more capacity.");
	}

	template<typename T>
	struct CVarArray : private NonCopyable
	{
		CVarStorage<T>* cvars;
		int32_t lastCVar = 0;
		int32_t capacity;

		CVarArray(size_t size)
		{
			cvars = new CVarStorage<T>[size]();
			capacity = (uint32_t)size;
		}

		~CVarArray()
		{
			delete[] cvars;
		}

		inline CVarStorage<T>* getCurrentStorage(int32_t index)
		{
			return &cvars[index];
		}

		inline T* getCurrentPtr(int32_t index)
		{
			return &cvars[index].currentVal;
		}

		inline T getCurrent(int32_t index)
		{
			return cvars[index].currentVal;
		};

		inline void setCurrent(const T& val, int32_t index)
		{
			cvars[index].currentVal = val;
		}

		inline int32_t add(const T& value, CVarParameter* param)
		{
			int32_t index = lastCVar;

			cvars[index].currentVal = value;
			cvars[index].initVal = value;
			cvars[index].parameter = param;
			param->arrayIndex = index;
			lastCVar++;

			checkRangeValid<T>(lastCVar);

			return index;
		}

		inline int32_t add(const T& initialValue, const T& currentValue, CVarParameter* param)
		{
			int32_t index = lastCVar;

			cvars[index].currentVal = currentValue;
			cvars[index].initVal = initialValue;
			cvars[index].parameter = param;
			param->arrayIndex = index;
			lastCVar++;

			checkRangeValid<T>(lastCVar);

			return index;
		}
	};

	class CVarSystem : private NonCopyable
	{
	public:
		static CVarSystem* get();

		// Getter functions.
		template<typename T> CVarArray<T>& getArray();

		template<> CVarArray<std::string>& getArray() { return m_stringCVars; }
		template<> CVarArray<int32_t>& getArray() { return m_int32CVars;  }
		template<> CVarArray<float>& getArray() { return m_floatCVars;  }
		template<> CVarArray<bool>& getArray() { return m_boolCVars; }

		void exportAllConfig(const std::string& path);
		bool importConfig(const std::string& path);

	private:
		// Simple hash by cVar name.
		inline size_t hash(std::string str)
		{
			std::transform(str.begin(), str.end(), str.begin(), ::tolower);
			size_t result = 0;
			for (auto it = str.cbegin(); it != str.cend(); ++it)
			{
				result = (result * 131) + *it;
			}

			return result;
		}

		template <typename T>
		inline std::string toString(T v)
		{
			return std::to_string(v);
		}

		template <>
		inline std::string toString(std::string c)
		{
			std::string ret = "\""; ret += c; ret += "\"";
			return ret;
		}

	private:
		CVarArray<int32_t> m_int32CVars{ kCVarMaxInt32Num };
		CVarArray<float> m_floatCVars{ kCVarMaxFloatNum };
		CVarArray<bool>  m_boolCVars{ kCVarMaxBoolNum };
		CVarArray<std::string> m_stringCVars{ kCVarMaxStringNum };

		std::shared_mutex m_lockMutex;
		std::unordered_map<size_t, CVarParameter> m_cacheCVars;

		inline CVarParameter* initCVar(const char* name, const char* description)
		{
			size_t hashId = hash(name);
			m_cacheCVars[hashId] = CVarParameter{ };
			auto& newParm = m_cacheCVars[hashId];
			newParm.name = name;
			newParm.description = description;
			return &newParm;
		}

		template<typename T> CVarArray<T>* getCVarArray();
		template<> CVarArray<int32_t>* getCVarArray() { return &m_int32CVars; }
		template<> CVarArray<float>* getCVarArray() { return &m_floatCVars; }
		template<> CVarArray<bool>* getCVarArray() { return &m_boolCVars; }
		template<> CVarArray<std::string>* getCVarArray() { return &m_stringCVars; }

		template<typename T>
		inline T* getCVarCurrent(const char* name)
		{
			CVarParameter* par = getCVarParameter(name);
			if (!par)
			{
				return nullptr;
			}
			else
			{
				return getCVarArray<T>()->getCurrentPtr(par->arrayIndex);
			}
		}

		template<typename T>
		inline void setCVarCurrent(const char* name, const T& value)
		{
			CVarParameter* cvar = getCVarParameter(name);

			if (cvar)
			{
				getCVarArray<T>()->setCurrent(value, cvar->arrayIndex);
			}
		}

	public:
		CVarParameter* getCVarParameter(const char* name)
		{
			std::shared_lock<std::shared_mutex> lock(m_lockMutex);
			size_t hashKey = hash(name);
			auto it = m_cacheCVars.find(hashKey);
			if (it != m_cacheCVars.end())
			{
				return &(*it).second;
			}

			return nullptr;
		}

		template<typename T>
		T* getCVar(const char* name)
		{
			return getCVarCurrent<T>(name);
		}

		template<typename T>
		void setCVar(const char* name, T value)
		{
			setCVarCurrent<T>(name, value);
		}

	private:
		template<typename BaseType>
		inline CVarParameter* addCVarTypeParam(
			CVarType type,
			const char* name,
			const char* description,
			const char* category,
			BaseType defaultValue,
			BaseType currentValue)
		{
			std::unique_lock<std::shared_mutex> lock(m_lockMutex);
			CVarParameter* param = initCVar(name, description);
			if (!param) 
			{
				return nullptr;
			}

			param->type = type;
			param->category = category;
			getCVarArray<BaseType>()->add(defaultValue, currentValue, param);
			return param;
		}

		inline CVarParameter* addFloatCVar(
			const char* name,
			const char* description,
			const char* category,
			float defaultValue,
			float currentValue)
		{
			return addCVarTypeParam<float>(
				CVarType::Float,
				name,
				description,
				category,
				defaultValue,
				currentValue);
		}

		inline CVarParameter* addBoolCVar(
			const char* name,
			const char* description,
			const char* category,
			bool defaultValue,
			bool currentValue)
		{
			return addCVarTypeParam<bool>(
				CVarType::Bool,
				name,
				description,
				category,
				defaultValue,
				currentValue);
		}

		inline CVarParameter* addInt32CVar(
			const char* name,
			const char* description,
			const char* category,
			int32_t defaultValue,
			int32_t currentValue)
		{
			return addCVarTypeParam<int32_t>(
				CVarType::Int32,
				name,
				description,
				category,
				defaultValue,
				currentValue);
		}

		inline CVarParameter* addStringCVar(
			const char* name,
			const char* description,
			const char* category,
			const std::string& defaultValue,
			const std::string& currentValue)
		{
			return addCVarTypeParam<std::string>(
				CVarType::String,
				name,
				description,
				category,
				defaultValue,
				currentValue);
		}

		friend struct AutoCVarFloat;
		friend struct AutoCVarBool;
		friend struct AutoCVarInt32;
		friend struct AutoCVarString;

		template<typename Ty> friend Ty   getCVarCurrentByIndex(int32_t);
		template<typename Ty> friend Ty*  ptrGetCVarCurrentByIndex(int32_t);
		template<typename Ty> friend void setCVarCurrentByIndex(int32_t, const Ty&);
	};

	template<typename T>
	struct AutoCVar
	{
	protected:
		int32_t index;
		using CVarType = T;
	};

	struct AutoCVarFloat : AutoCVar<float>
	{
		AutoCVarFloat(
			const char* name,
			const char* description,
			const char* category,
			float defaultValue,
			uint32_t flags = CVarFlags::None)
		{
			CVarParameter* cvar = CVarSystem::get()->addFloatCVar(
				name,
				description,
				category,
				defaultValue,
				defaultValue
			);

			cvar->flag = flags;
			index = cvar->arrayIndex;
		}

		inline float  get();
		inline float* getPtr();
		inline void   set(float val);
	};

	struct AutoCVarBool : AutoCVar<bool>
	{
		AutoCVarBool(
			const char* name,
			const char* description,
			const char* category,
			bool defaultValue,
			uint32_t flags = CVarFlags::None)
		{
			CVarParameter* cvar = CVarSystem::get()->addBoolCVar(
				name,
				description,
				category,
				defaultValue,
				defaultValue
			);

			cvar->flag = flags;
			index = cvar->arrayIndex;
		}

		// Auto cmd init function.
		AutoCVarBool(const char* name, const char* description)
		{
			constexpr const char* category = "Cmd";
			constexpr bool defaultValue = false;

			CVarParameter* cvar = CVarSystem::get()->addBoolCVar(
				name,
				description,
				category,
				defaultValue,
				defaultValue
			);

			// Command cvar should read and write.
			cvar->flag = uint32_t(CVarFlags::ReadAndWrite);
			index = cvar->arrayIndex;
		}

		inline bool  get();
		inline bool* getPtr();
		inline void  set(bool val);
	};

	// CVar cmd is set once value.
	using AutoCVarCmd = AutoCVarBool;

	// Handle cvar cmd.
	inline void CVarCmdHandle(AutoCVarCmd& in, std::function<void()>&& func)
	{
		if (in.get())
		{
			in.set(false);
			func();
		}
	}

	struct AutoCVarInt32 : AutoCVar<int32_t>
	{
		AutoCVarInt32(
			const char* name,
			const char* description,
			const char* category,
			int32_t defaultValue,
			uint32_t flags = CVarFlags::None)
		{
			CVarParameter* cvar = CVarSystem::get()->addInt32CVar(
				name,
				description,
				category,
				defaultValue,
				defaultValue
			);

			cvar->flag = flags;
			index = cvar->arrayIndex;
		}

		inline int32_t  get();
		inline int32_t* getPtr();
		inline void     set(int32_t val);
	};



	struct AutoCVarString : AutoCVar<std::string>
	{
		AutoCVarString(
			const char* name,
			const char* description,
			const char* category,
			const std::string& defaultValue,
			uint32_t flags = CVarFlags::None)
		{
			CVarParameter* cvar = CVarSystem::get()->addStringCVar(
				name,
				description,
				category,
				defaultValue,
				defaultValue
			);

			cvar->flag = flags;
			index = cvar->arrayIndex;
		}

		inline std::string get();
		inline std::string* getPtr();
		inline void        set(const std::string& val);
	};

	template<typename T>
	inline T getCVarCurrentByIndex(int32_t index)
	{
		return CVarSystem::get()->getCVarArray<T>()->getCurrent(index);
	}

	template<typename T>
	inline T* ptrGetCVarCurrentByIndex(int32_t index)
	{
		return CVarSystem::get()->getCVarArray<T>()->getCurrentPtr(index);
	}

	template<typename T>
	inline void setCVarCurrentByIndex(int32_t index, const T& data)
	{
		CVarSystem::get()->getCVarArray<T>()->setCurrent(data, index);
	}

	inline float AutoCVarFloat::get()
	{
		return getCVarCurrentByIndex<CVarType>(index);
	}

	inline bool AutoCVarBool::get()
	{
		return getCVarCurrentByIndex<CVarType>(index);
	}

	inline int32_t AutoCVarInt32::get()
	{
		return getCVarCurrentByIndex<CVarType>(index);
	}

	inline std::string AutoCVarString::get()
	{
		return getCVarCurrentByIndex<CVarType>(index);
	}

	inline void AutoCVarFloat::set(float f)
	{
		setCVarCurrentByIndex<CVarType>(index, f);
	}

	inline void AutoCVarBool::set(bool f)
	{
		setCVarCurrentByIndex<CVarType>(index, f);
	}

	inline void AutoCVarInt32::set(int32_t f)
	{
		setCVarCurrentByIndex<CVarType>(index, f);
	}

	inline void AutoCVarString::set(const std::string& f)
	{
		setCVarCurrentByIndex<CVarType>(index, f);
	}

	inline float* AutoCVarFloat::getPtr()
	{
		return ptrGetCVarCurrentByIndex<CVarType>(index);
	}

	inline bool* AutoCVarBool::getPtr()
	{
		return ptrGetCVarCurrentByIndex<CVarType>(index);
	}

	inline int32_t* AutoCVarInt32::getPtr()
	{
		return ptrGetCVarCurrentByIndex<CVarType>(index);
	}

	inline std::string* AutoCVarString::getPtr()
	{
		return ptrGetCVarCurrentByIndex<CVarType>(index);
	}
}