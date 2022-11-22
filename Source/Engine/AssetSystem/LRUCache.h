#pragma once
#include "../Core/Core.h"
#include "../Core/UUID.h"

namespace Flower
{
	class LRUAssetInterface : NonCopyable
	{
	protected:
		bool m_bPersistent;

		UUID m_uuid;

		std::atomic<bool> m_bAsyncLoading = true;
		LRUAssetInterface* m_fallback = nullptr;

	public:
		explicit LRUAssetInterface(LRUAssetInterface* fallback, bool bPersistent)
			: m_fallback(fallback), m_uuid(buildUUID()), m_bPersistent(bPersistent)
		{
			if (m_fallback)
			{
				CHECK(m_fallback->isAssetReady() && "Fallback asset must already load.");
			}
		}

		bool isAssetLoading() const
		{
			return m_bAsyncLoading;
		}

		bool isAssetReady() const
		{
			return !m_bAsyncLoading;
		}

		bool isPersistent() const
		{
			return m_bPersistent;
		}

		virtual size_t getSize() const = 0;

		void setAsyncLoadState(bool bState) 
		{ 
			m_bAsyncLoading = bState; 
		}

		UUID getUUID() const 
		{ 
			return m_uuid; 
		}
	};

	template<typename ValueType = LRUAssetInterface, typename KeyType = UUID>
	class LRUAssetCache : NonCopyable
	{
	protected:
		static_assert(std::is_base_of_v<LRUAssetInterface, ValueType>, "Value type must derived from LRUAssetInterface");

		using LRUList = std::list<std::pair<KeyType, std::shared_ptr<ValueType>>>;
		using LRUListNode = LRUList::iterator;

		// LRU data struct.
		LRUList m_lruList;
		std::unordered_map<KeyType, LRUListNode> m_lruMap;
		std::unordered_map<KeyType, std::weak_ptr<ValueType>> m_cacheWeakPtrMap;

		// Cache persistent resource.
		std::atomic<size_t> m_persistentAssetSize = 0;
		std::unordered_map<KeyType, std::shared_ptr<ValueType>> m_cachePersistentMap;

		// LRU cache desire capacity.
		size_t m_capacity;

		// Some elasticity space to enable LRU cache no always prune.
		size_t m_elasticity;

		// Shared_ptr use size.
		std::atomic<size_t> m_usedSize = 0;

		// Lock mutex for map edit.
		std::mutex m_lock;

	public:
		explicit LRUAssetCache(size_t capacity, size_t elasticity)
			: m_capacity(capacity * 1024 * 1024), m_elasticity(elasticity * 1024 * 1024)
		{

		}

		virtual ~LRUAssetCache() = default;

		size_t getCapacity() const 
		{ 
			return m_capacity; 
		}

		size_t getElasticity() const 
		{ 
			return m_elasticity; 
		}

		size_t getMaxAllowedSize() const 
		{ 
			return m_capacity + m_elasticity; 
		}

		size_t getPersistentAssetSize() const
		{
			return m_persistentAssetSize.load();
		}

		size_t getLRUAssetUsedSize() const
		{
			return m_usedSize.load();
		}

		bool contain(const KeyType& key)
		{
			// Need lock when search from weak ptr map.
			std::lock_guard<std::mutex> lockGuard(m_lock);

			// First search from persistent map.
			if (m_cachePersistentMap.contains(key))
			{
				return true;
			}

			// Then search from lru owner map.
			if (m_lruMap.contains(key))
			{
				return true;
			}

			// Final if lru owner map no exist, search from lru weak map.
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
				}
			}

			return false;
		}

		void clear()
		{
			std::lock_guard<std::mutex> lockGuard(m_lock);
			m_usedSize = 0;

			m_lruMap.clear();
			m_lruList.clear();
			m_cacheWeakPtrMap.clear();

			// Never clear persistent map.
		}

		void insert(const KeyType& key, std::shared_ptr<ValueType> value)
		{
			std::lock_guard<std::mutex> lockGuard(m_lock);

			// Persistent asset
			if (value->isPersistent())
			{
				if (!m_cachePersistentMap.contains(key))
				{
					m_cachePersistentMap[key] = value;
					m_persistentAssetSize += value->getSize();
				}
				else
				{
					LOG_ERROR("Try insert persistent asset repeatly, this is an illegal operation.");
				}

				// Pre return.
				return;
			}

			// LRU asset.

			// Cache weak ptr value.
			m_cacheWeakPtrMap[key] = value;
			m_usedSize += value->getSize();

			// LRU asset can insert repeatly.
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

		std::shared_ptr<ValueType> tryGet(const KeyType& inKey)
		{
			std::lock_guard<std::mutex> lockGuard(m_lock);

			// Try found in persistent map first.
			if (m_cachePersistentMap.contains(inKey))
			{
				return m_cachePersistentMap[inKey];
			}

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

			size_t reduceSize = 0;
			while (m_usedSize > m_capacity)
			{
				size_t eleSize = m_lruList.back().second->getSize();

				if (reduceFunction)
				{
					reduceFunction(m_lruList.back().second);
				}

				m_lruMap.erase(m_lruList.back().first);
				m_lruList.pop_back();

				m_usedSize -= eleSize;
				reduceSize += eleSize;
			}
			return reduceSize;
		}
	};
}