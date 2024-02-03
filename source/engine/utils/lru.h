#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <list>
#include <unordered_map>
#include <mutex>
#include <memory>

#include "noncopyable.h"
#include "uuid.h"

namespace engine
{
	// LRU storage interface.
	class StorageInterface : NonCopyable
	{
	public:
		// This storage memory size.
		virtual uint32_t getSize() const = 0;
	};

	// All asset used in this application will not release immediate when it don't need.
	// It will cache in LRU map when memory is enough, but if memory is not enough for new asset,
	// it will prune the cache ones.
	class LRUAssetCache : NonCopyable
	{
	public:
		using ValueType = StorageInterface;
		using KeyType = UUID;

		// Init lru asset cache with capacity and elasticity in MB unit.
		explicit LRUAssetCache(size_t capacity, size_t elasticity)
			: m_capacity(capacity * 1024 * 1024)
			, m_elasticity(elasticity * 1024 * 1024)
		{

		}

		~LRUAssetCache()
		{
			clear();
		}

		size_t getCapacity() const { return m_capacity; }
		size_t getElasticity() const { return m_elasticity; }
		size_t getMaxAllowedSize() const { return m_capacity + m_elasticity; }

		// Current LRU owner shared_ptr map cache used size, no included asset owner by other actor.
		size_t getOwnerUsedSize() const { return m_usedSize.load(); }

		// Is contain current asset key.
		bool contain(const KeyType& key)
		{
			// Need lock when search from weak ptr map.
			std::lock_guard<std::mutex> lockGuard(m_lock);

			// First search from lru owner map.
			if (m_lruMap.contains(key))
			{
				return true;
			}

			// Then if lru owner map no exist, search from lru weak map.
			if (m_cacheWeakPtrMap.contains(key))
			{
				if (auto lockPtr = m_cacheWeakPtrMap[key].lock())
				{
					return true;
				}
				else
				{
					// Erase unvalid waek ptr.
					m_cacheWeakPtrMap.erase(key);

					// Non valid pointer.
					return false;
				}
			}

			return false;
		}

		// Clear all lru cache.
		void clear()
		{
			std::lock_guard<std::mutex> lockGuard(m_lock);
			
			// Make used size as zero.
			m_usedSize = 0;

			// LRU cache clear.
			m_lruMap.clear();
			m_lruList.clear();
			m_cacheWeakPtrMap.clear();
		}

		// Insert one 
		void insert(const KeyType& key, std::shared_ptr<ValueType> value)
		{
			std::lock_guard<std::mutex> lockGuard(m_lock);

			// Cache weak ptr value.
			m_cacheWeakPtrMap[key] = value;
			m_usedSize += value->getSize();

			// LRU asset can insert repeatly, it loop whole map to find exist or not.
			const auto iter = m_lruMap.find(key);
			if (iter != m_lruMap.end())
			{
				// Key exist, update map value, and update list.
				m_usedSize -= iter->second->second->getSize();
				iter->second->second = value;

				m_lruList.splice(m_lruList.begin(), m_lruList, iter->second);
				return;
			}

			// Key no exist, emplace to list front, and update map key-value.
			m_lruList.emplace_front(key, std::move(value));
			m_lruMap[key] = m_lruList.begin();

			// May oversize, need reduce.
			prune();
		}

		// Try to get value, will return nullptr if no exist.
		std::shared_ptr<ValueType> tryGet(const KeyType& inKey)
		{
			std::lock_guard<std::mutex> lockGuard(m_lock);

			// Else found in lru map.
			const auto iter = m_lruMap.find(inKey);
			if (iter == m_lruMap.end())
			{
				// May still valid in weak ptr cache, try get.
				std::weak_ptr<ValueType> weakPtr = m_cacheWeakPtrMap[inKey];
				if (auto sharePtr = weakPtr.lock())
				{
					return sharePtr;
				}
				else
				{
					// Unvalid weak ptr, remove.
					m_cacheWeakPtrMap.erase(inKey);
				}

				// No valid instance, return nullptr and need reload.
				return nullptr;
			}

			// Still exist in lru map, set as first guy.
			m_lruList.splice(m_lruList.begin(), m_lruList, iter->second);
			return iter->second->second;
		}

		// Prune lru map.
		size_t prune(std::function<void(std::shared_ptr<ValueType>)>&& reduceFunction = nullptr)
		{
			size_t maxAllowed = m_capacity + m_elasticity;
			if (m_capacity == 0 || m_usedSize < maxAllowed)
			{
				return 0;
			}

			// Loop until release enough resource.
			size_t reduceSize = 0;
			while (m_usedSize > m_capacity)
			{
				size_t eleSize = m_lruList.back().second->getSize();

				// Do a callback if valid.
				if (reduceFunction)
				{
					reduceFunction(m_lruList.back().second);
				}

				// Erase last one.
				m_lruMap.erase(m_lruList.back().first);
				m_lruList.pop_back();

				// Update size.
				m_usedSize -= eleSize;
				reduceSize += eleSize;
			}

			return reduceSize;
		}

	protected:
		static_assert(std::is_base_of_v<StorageInterface, ValueType>, 
			"Value type must derived from StorageInterface");

		using LRUList = std::list<std::pair<KeyType, std::shared_ptr<ValueType>>>;
		using LRUListNode = LRUList::iterator;

		// LRU data struct.
		LRUList m_lruList;
		std::unordered_map<KeyType, LRUListNode> m_lruMap;

		// Some asset may prune when space is not enough, but may cache in other system, in this case, it still can used.
		std::unordered_map<KeyType, std::weak_ptr<ValueType>> m_cacheWeakPtrMap;

		// LRU cache desire capacity.
		size_t m_capacity;

		// Some elasticity space to enable LRU cache no always prune.
		size_t m_elasticity;

		// Shared_ptr(m_lruMap) used size.
		std::atomic<size_t> m_usedSize = 0;

		// Lock mutex for map edit.
		std::mutex m_lock;
	};
}