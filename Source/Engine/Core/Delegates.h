#pragma once

#include "Core.h"
#include "Misc.h"

namespace Flower
{
	// Cpp delegatess.
	// See https://simoncoenen.com/blog/programming/CPP_Delegates to find more implement details.

	// There exist four types delegate can use.
	// a. StaticDelegate
	// b. RawDelegate
	// c. SPDelegate
	// d. MulticastDelegate

	namespace DelegatesInteral
	{
		constexpr size_t cDelegateInlineAllocationSize = cCPUCacheLineSize;

		template<bool bConst, typename Object, typename RetVal, typename ...Args>
		struct MemberFunction;

		template<typename Object, typename RetVal, typename ...Args>
		struct MemberFunction<true, Object, RetVal, Args...>
		{
			using Type = RetVal(Object::*)(Args...) const;
		};

		template<typename Object, typename RetVal, typename ...Args>
		struct MemberFunction<false, Object, RetVal, Args...>
		{
			using Type = RetVal(Object::*)(Args...);
		};
	}

	class IDelegateBase
	{
	public:
		IDelegateBase() = default;
		virtual ~IDelegateBase() noexcept = default;

		virtual const void* getOwner() const 
		{ 
			return nullptr; 
		}

		virtual void clone(void* pDestination) = 0;
	};

	template<typename RetVal, typename... Args>
	class IDelegate : public IDelegateBase
	{
	public:
		virtual RetVal execute(Args&&... args) = 0;
	};

	template<typename RetVal, typename... Args2>
	class StaticDelegate;

	template<typename RetVal, typename... Args, typename... Args2>
	class StaticDelegate<RetVal(Args...), Args2...> : public IDelegate<RetVal, Args...>
	{
	public:
		using DelegateFunction = RetVal(*)(Args..., Args2...);

	private:
		template<std::size_t... Is>
		RetVal executeInternal(Args&&... args, std::index_sequence<Is...>)
		{
			return m_function(std::forward<Args>(args)..., std::get<Is>(m_payload)...);
		}

		DelegateFunction m_function;
		std::tuple<Args2...> m_payload;

	public:
		StaticDelegate(DelegateFunction function, Args2&&... payload)
			: m_function(function), m_payload(std::forward<Args2>(payload)...)
		{}

		StaticDelegate(DelegateFunction function, const std::tuple<Args2...>& payload)
			: m_function(function), m_payload(payload)
		{}

		virtual RetVal execute(Args&&... args) override
		{
			return executeInternal(std::forward<Args>(args)..., std::index_sequence_for<Args2...>());
		}

		virtual void clone(void* pDestination) override
		{
			new (pDestination) StaticDelegate(m_function, m_payload);
		}
	};

	template<bool bConst, typename T, typename RetVal, typename... Args2>
	class RawDelegate;

	template<bool bConst, typename T, typename RetVal, typename... Args, typename... Args2>
	class RawDelegate<bConst, T, RetVal(Args...), Args2...> : public IDelegate<RetVal, Args...>
	{
	public:
		using DelegateFunction = typename DelegatesInteral::MemberFunction<bConst, T, RetVal, Args..., Args2...>::Type;

	private:
		template<std::size_t... Is>
		RetVal executeInternal(Args&&... args, std::index_sequence<Is...>)
		{
			return (m_pObject->*m_function)(std::forward<Args>(args)..., std::get<Is>(m_payload)...);
		}

		T* m_pObject;
		DelegateFunction m_function;
		std::tuple<Args2...> m_payload;

	public:
		RawDelegate(T* pObject, DelegateFunction function, Args2&&... payload)
			: m_pObject(pObject), m_function(function), m_payload(std::forward<Args2>(payload)...)
		{}

		RawDelegate(T* pObject, DelegateFunction function, const std::tuple<Args2...>& payload)
			: m_pObject(pObject), m_function(function), m_payload(payload)
		{}

		virtual RetVal execute(Args&&... args) override
		{
			return executeInternal(std::forward<Args>(args)..., std::index_sequence_for<Args2...>());
		}
		virtual const void* getOwner() const override
		{
			return m_pObject;
		}

		virtual void clone(void* pDestination) override
		{
			new (pDestination) RawDelegate(m_pObject, m_function, m_payload);
		}
	};

	template<typename TLambda, typename RetVal, typename... Args>
	class LambdaDelegate;

	template<typename TLambda, typename RetVal, typename... Args, typename... Args2>
	class LambdaDelegate<TLambda, RetVal(Args...), Args2...> : public IDelegate<RetVal, Args...>
	{
	private:
		template<std::size_t... Is>
		RetVal executeInternal(Args&&... args, std::index_sequence<Is...>)
		{
			return (RetVal)((m_Lambda)(std::forward<Args>(args)..., std::get<Is>(m_payload)...));
		}

		TLambda m_Lambda;
		std::tuple<Args2...> m_payload;

	public:
		explicit LambdaDelegate(TLambda&& lambda, Args2&&... payload)
			: m_Lambda(std::forward<TLambda>(lambda)),
			m_payload(std::forward<Args2>(payload)...)
		{}

		explicit LambdaDelegate(const TLambda& lambda, const std::tuple<Args2...>& payload)
			: m_Lambda(lambda),
			m_payload(payload)
		{}

		RetVal execute(Args&&... args) override
		{
			return executeInternal(std::forward<Args>(args)..., std::index_sequence_for<Args2...>());
		}

		virtual void clone(void* pDestination) override
		{
			new (pDestination) LambdaDelegate(m_Lambda, m_payload);
		}
	};

	template<bool bConst, typename T, typename RetVal, typename... Args>
	class SPDelegate;

	template<bool bConst, typename RetVal, typename T, typename... Args, typename... Args2>
	class SPDelegate<bConst, T, RetVal(Args...), Args2...> : public IDelegate<RetVal, Args...>
	{
	public:
		using DelegateFunction = typename DelegatesInteral::MemberFunction<bConst, T, RetVal, Args..., Args2...>::Type;

	private:
		template<std::size_t... Is>
		RetVal executeInternal(Args&&... args, std::index_sequence<Is...>)
		{
			if (m_pObject.expired())
			{
				return RetVal();
			}
			else
			{
				std::shared_ptr<T> pPinned = m_pObject.lock();
				return (pPinned.get()->*m_pFunction)(std::forward<Args>(args)..., std::get<Is>(m_payload)...);
			}
		}

		std::weak_ptr<T> m_pObject;
		DelegateFunction m_pFunction;
		std::tuple<Args2...> m_payload;

	public:
		SPDelegate(std::shared_ptr<T> pObject, DelegateFunction pFunction, Args2&&... payload)
			: m_pObject(pObject),
			m_pFunction(pFunction),
			m_payload(std::forward<Args2>(payload)...)
		{}

		SPDelegate(std::weak_ptr<T> pObject, DelegateFunction pFunction, const std::tuple<Args2...>& payload)
			: m_pObject(pObject),
			m_pFunction(pFunction),
			m_payload(payload)
		{}

		virtual RetVal execute(Args&&... args) override
		{
			return executeInternal(std::forward<Args>(args)..., std::index_sequence_for<Args2...>());
		}

		virtual const void* getOwner() const override
		{
			return m_pObject.expired() ? nullptr : m_pObject.lock().get();
		}

		virtual void clone(void* pDestination) override
		{
			new (pDestination) SPDelegate(m_pObject, m_pFunction, m_payload);
		}
	};

	class DelegateHandle
	{
	private:
		unsigned int m_id;
		static unsigned int CURRENT_ID;

		static int getNewID()
		{
			unsigned int output = DelegateHandle::CURRENT_ID++;
			if (DelegateHandle::CURRENT_ID == INVALID_ID)
			{
				DelegateHandle::CURRENT_ID = 0;
			}
			return output;
		}

	public:
		constexpr static const unsigned int INVALID_ID = (unsigned int)~0;

		constexpr DelegateHandle() noexcept
			: m_id(INVALID_ID)
		{
		}

		explicit DelegateHandle(bool /*generateId*/) noexcept
			: m_id(getNewID())
		{
		}

		~DelegateHandle() noexcept = default;
		DelegateHandle(const DelegateHandle& other) = default;
		DelegateHandle& operator=(const DelegateHandle& other) = default;

		DelegateHandle(DelegateHandle&& other) noexcept
			: m_id(other.m_id)
		{
			other.reset();
		}

		DelegateHandle& operator=(DelegateHandle&& other) noexcept
		{
			m_id = other.m_id;
			other.reset();
			return *this;
		}

		operator bool() const noexcept
		{
			return isValid();
		}

		bool operator==(const DelegateHandle& other) const noexcept
		{
			return m_id == other.m_id;
		}

		bool operator<(const DelegateHandle& other) const noexcept
		{
			return m_id < other.m_id;
		}

		bool isValid() const noexcept
		{
			return m_id != INVALID_ID;
		}

		void reset() noexcept
		{
			m_id = INVALID_ID;
		}
	};

	template<size_t MaxStackSize>
	class InlineAllocator
	{
	private:
		union
		{
			char buffer[MaxStackSize];
			void* pPtr; // Heap memory, use when memory size bigger than MaxStackSize.
		};
		size_t m_size;

	public:
		inline size_t getSize() const { return m_size; }
		inline bool hasAllocation() const { return m_size > 0; }
		inline bool hasHeapAllocation() const { return m_size > MaxStackSize; }

		void* allocateInternal(const size_t size)
		{
			if (m_size != size)
			{
				freeInternal();
				m_size = size;
				if (size > MaxStackSize)
				{
					pPtr = malloc(size);
					return pPtr;
				}
			}
			return (void*)buffer;
		}

		void freeInternal()
		{
			if (m_size > MaxStackSize)
			{
				free(pPtr);
			}
			m_size = 0;
		}

		void* getAllocation() const
		{
			if (hasAllocation())
			{
				return hasHeapAllocation() ? pPtr : (void*)buffer;
			}
			else
			{
				return nullptr;
			}
		}

		InlineAllocator() noexcept : m_size(0)
		{
			static_assert(MaxStackSize > sizeof(void*), "MaxStackSize is smaller or equal to the size of a pointer.");
		}

		InlineAllocator(const InlineAllocator& other) : m_size(0)
		{
			if (other.hasAllocation())
			{
				memcpy(allocateInternal(other.m_size), other.getAllocation(), other.m_size);
			}
			m_size = other.m_size;
		}

		~InlineAllocator() noexcept
		{
			freeInternal();
		}

		InlineAllocator& operator=(const InlineAllocator& other)
		{
			if (other.hasAllocation())
			{
				memcpy(allocateInternal(other.m_size), other.getAllocation(), other.m_size);
			}
			m_size = other.m_size;

			return *this;
		}

		InlineAllocator(InlineAllocator&& other) noexcept : m_size(other.m_size)
		{
			other.m_size = 0;
			if (m_size > MaxStackSize)
			{
				std::swap(pPtr, other.pPtr);
			}
			else
			{
				memcpy(buffer, other.buffer, m_size);
			}
		}

		InlineAllocator& operator=(InlineAllocator&& other) noexcept
		{
			freeInternal();

			m_size = other.m_size;
			other.m_size = 0;

			if (m_size > MaxStackSize)
			{
				std::swap(pPtr, other.pPtr);
			}
			else
			{
				memcpy(buffer, other.buffer, m_size);
			}
			return *this;
		}
	};

	class DelegateBase
	{
	protected:
		InlineAllocator<DelegatesInteral::cDelegateInlineAllocationSize> m_allocator;

		IDelegateBase* getDelegate() const
		{
			return static_cast<IDelegateBase*>(m_allocator.getAllocation());
		}

		void release()
		{
			if (m_allocator.hasAllocation())
			{
				getDelegate()->~IDelegateBase();
				m_allocator.freeInternal();
			}
		}

	public:
		inline size_t getSize() const { return m_allocator.getSize(); }
		inline bool isBound() const { return m_allocator.hasAllocation(); }

		bool isBoundTo(void* pObject) const
		{
			if (pObject == nullptr || m_allocator.hasAllocation() == false)
			{
				return false;
			}
			return getDelegate()->getOwner() == pObject;
		}

		inline void clear()
		{
			release();
		}

		void clearIfBoundTo(void* pObject)
		{
			if (pObject != nullptr && isBoundTo(pObject))
			{
				release();
			}
		}

		const void* getOwner() const
		{
			if (m_allocator.hasAllocation())
			{
				return getDelegate()->getOwner();
			}
			return nullptr;
		}

	public:
		DelegateBase() noexcept
			: m_allocator()
		{}

		virtual ~DelegateBase() noexcept
		{
			release();
		}

		DelegateBase(const DelegateBase& other)
		{
			if (other.m_allocator.hasAllocation())
			{
				m_allocator.allocateInternal(other.m_allocator.getSize());
				other.getDelegate()->clone(m_allocator.getAllocation());
			}
		}

		DelegateBase& operator=(const DelegateBase& other)
		{
			release();
			if (other.m_allocator.hasAllocation())
			{
				m_allocator.allocateInternal(other.m_allocator.getSize());
				other.getDelegate()->clone(m_allocator.getAllocation());
			}
			return *this;
		}

		DelegateBase(DelegateBase&& other) noexcept
			: m_allocator(std::move(other.m_allocator))
		{}

		DelegateBase& operator=(DelegateBase&& other) noexcept
		{
			release();
			m_allocator = std::move(other.m_allocator);
			return *this;
		}
	};

	template<typename RetVal, typename... Args>
	class Delegate : public DelegateBase
	{
	private:
		template<typename T, typename... Args2>
		using ConstMemberFunction = typename DelegatesInteral::MemberFunction<true, T, RetVal, Args..., Args2...>::Type;

		template<typename T, typename... Args2>
		using NonConstMemberFunction = typename DelegatesInteral::MemberFunction<false, T, RetVal, Args..., Args2...>::Type;

	private:
		template<typename T, typename... Args3>
		void bind(Args3&&... args)
		{
			release();
			void* pAlloc = m_allocator.allocateInternal(sizeof(T));
			new (pAlloc) T(std::forward<Args3>(args)...);
		}

	public:
		using IDelegateT = IDelegate<RetVal, Args...>;

		template<typename T, typename... Args2>
		[[nodiscard]] static Delegate createRaw(T* pObj, NonConstMemberFunction<T, Args2...> pFunction, Args2... args)
		{
			Delegate handler;
			handler.bind<RawDelegate<false, T, RetVal(Args...), Args2...>>(pObj, pFunction, std::forward<Args2>(args)...);
			return handler;
		}

		template<typename T, typename... Args2>
		[[nodiscard]] static Delegate createRaw(T* pObj, ConstMemberFunction<T, Args2...> pFunction, Args2... args)
		{
			Delegate handler;
			handler.bind<RawDelegate<true, T, RetVal(Args...), Args2...>>(pObj, pFunction, std::forward<Args2>(args)...);
			return handler;
		}

		template<typename... Args2>
		[[nodiscard]] static Delegate createStatic(RetVal(*pFunction)(Args..., Args2...), Args2... args)
		{
			Delegate handler;
			handler.bind<StaticDelegate<RetVal(Args...), Args2...>>(pFunction, std::forward<Args2>(args)...);
			return handler;
		}

		template<typename T, typename... Args2>
		[[nodiscard]] static Delegate createSP(const std::shared_ptr<T>& pObject, NonConstMemberFunction<T, Args2...> pFunction, Args2... args)
		{
			Delegate handler;
			handler.bind<SPDelegate<false, T, RetVal(Args...), Args2...>>(pObject, pFunction, std::forward<Args2>(args)...);
			return handler;
		}

		template<typename T, typename... Args2>
		[[nodiscard]] static Delegate createSP(const std::shared_ptr<T>& pObject, ConstMemberFunction<T, Args2...> pFunction, Args2... args)
		{
			Delegate handler;
			handler.bind<SPDelegate<true, T, RetVal(Args...), Args2...>>(pObject, pFunction, std::forward<Args2>(args)...);
			return handler;
		}

		template<typename TLambda, typename... Args2>
		[[nodiscard]] static Delegate createLambda(TLambda&& lambda, Args2... args)
		{
			Delegate handler;
			handler.bind<LambdaDelegate<TLambda, RetVal(Args...), Args2...>>(std::forward<TLambda>(lambda), std::forward<Args2>(args)...);
			return handler;
		}

		template<typename T, typename... Args2>
		void bindRaw(T* pObject, NonConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			static_assert(!std::is_const<T>::value, "Cannot bind a non-const function on a const object");
			*this = createRaw<T, Args2... >(pObject, pFunction, std::forward<Args2>(args)...);
		}

		template<typename T, typename... Args2>
		void bindRaw(T* pObject, ConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			*this = createRaw<T, Args2... >(pObject, pFunction, std::forward<Args2>(args)...);
		}

		template<typename... Args2>
		void bindStatic(RetVal(*pFunction)(Args..., Args2...), Args2&&... args)
		{
			*this = createStatic<Args2... >(pFunction, std::forward<Args2>(args)...);
		}

		template<typename LambdaType, typename... Args2>
		void bindLambda(LambdaType&& lambda, Args2&&... args)
		{
			*this = createLambda<LambdaType, Args2... >(std::forward<LambdaType>(lambda), std::forward<Args2>(args)...);
		}

		template<typename T, typename... Args2>
		void bindSP(std::shared_ptr<T> pObject, NonConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			static_assert(!std::is_const<T>::value, "Cannot bind a non-const function on a const object");
			*this = createSP<T, Args2... >(pObject, pFunction, std::forward<Args2>(args)...);
		}

		template<typename T, typename... Args2>
		void bindSP(std::shared_ptr<T> pObject, ConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			*this = createSP<T, Args2... >(pObject, pFunction, std::forward<Args2>(args)...);
		}

		RetVal execute(Args... args) const
		{
			CHECK(m_allocator.hasAllocation() && "Delegate is not bound");
			return ((IDelegateT*)getDelegate())->execute(std::forward<Args>(args)...);
		}

		RetVal executeIfBound(Args... args) const
		{
			if (isBound())
			{
				return ((IDelegateT*)getDelegate())->execute(std::forward<Args>(args)...);
			}
			return RetVal();
		}
	};

	template<typename... Args>
	class MulticastDelegate : public DelegateBase
	{
	public:
		using DelegateT = Delegate<void, Args...>;

	private:
		struct DelegateHandlerPair
		{
			DelegateHandle handle;
			DelegateT callback;

			DelegateHandlerPair() : handle(false) {}
			DelegateHandlerPair(const DelegateHandle& handle, const DelegateT& callback) : handle(handle), callback(callback) {}
			DelegateHandlerPair(const DelegateHandle& handle, DelegateT&& callback) : handle(handle), callback(std::move(callback)) {}
		};

		template<typename T, typename... Args2>
		using ConstMemberFunction = typename DelegatesInteral::MemberFunction<true, T, void, Args..., Args2...>::Type;

		template<typename T, typename... Args2>
		using NonConstMemberFunction = typename DelegatesInteral::MemberFunction<false, T, void, Args..., Args2...>::Type;

	private:
		std::vector<DelegateHandlerPair> m_events;
		unsigned int m_lock;

		inline void lock() { ++m_lock; }
		inline void unlock(){ CHECK(m_lock > 0); --m_lock; }
		inline bool isLocked() const { return m_lock > 0; }

	public:
		inline size_t getSize() const { return m_events.size(); }

	public:
		constexpr MulticastDelegate() : m_lock(0) { }
		~MulticastDelegate() noexcept = default;
		MulticastDelegate(const MulticastDelegate& other) = default;
		MulticastDelegate& operator=(const MulticastDelegate& other) = default;

		MulticastDelegate(MulticastDelegate&& other) noexcept 
			: m_events(std::move(other.m_events)), m_lock(std::move(other.m_lock))
		{
		
		}

		MulticastDelegate& operator=(MulticastDelegate&& other) noexcept
		{
			m_events = std::move(other.m_events);
			m_lock = std::move(other.m_lock);
			return *this;
		}

		template<typename T>
		DelegateHandle operator+=(T&& l) { return add(DelegateT::createLambda(std::move(l))); }

		DelegateHandle operator+=(DelegateT&& handler) noexcept { return add(std::forward<DelegateT>(handler)); }
		bool operator-=(DelegateHandle& handle) { return remove(handle); }

		template<typename T, typename... Args2>
		DelegateHandle addRaw(T* pObject, NonConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			return add(DelegateT::createRaw(pObject, pFunction, std::forward<Args2>(args)...));
		}

		template<typename T, typename... Args2>
		DelegateHandle addRaw(T* pObject, ConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			return add(DelegateT::createRaw(pObject, pFunction, std::forward<Args2>(args)...));
		}

		template<typename... Args2>
		DelegateHandle addStatic(void(*pFunction)(Args..., Args2...), Args2&&... args)
		{
			return add(DelegateT::createStatic(pFunction, std::forward<Args2>(args)...));
		}

		template<typename LambdaType, typename... Args2>
		DelegateHandle addLambda(LambdaType&& lambda, Args2&&... args)
		{
			return add(DelegateT::createLambda(std::forward<LambdaType>(lambda), std::forward<Args2>(args)...));
		}

		template<typename T, typename... Args2>
		DelegateHandle addSP(std::shared_ptr<T> pObject, NonConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			return add(DelegateT::createSP(pObject, pFunction, std::forward<Args2>(args)...));
		}

		template<typename T, typename... Args2>
		DelegateHandle addSP(std::shared_ptr<T> pObject, ConstMemberFunction<T, Args2...> pFunction, Args2&&... args)
		{
			return add(DelegateT::createSP(pObject, pFunction, std::forward<Args2>(args)...));
		}

	public:
		DelegateHandle add(DelegateT&& handler) noexcept
		{
			for (size_t i = 0; i < m_events.size(); ++i)
			{
				if (m_events[i].handle.isValid() == false)
				{
					m_events[i] = DelegateHandlerPair(DelegateHandle(true), std::move(handler));
					return m_events[i].handle;
				}
			}
			m_events.emplace_back(DelegateHandle(true), std::move(handler));
			return m_events.back().handle;
		}

		void removeObject(void* pObject)
		{
			if (pObject != nullptr)
			{
				for (size_t i = 0; i < m_events.size(); ++i)
				{
					if (m_events[i].callback.getOwner() == pObject)
					{
						if (isLocked())
						{
							m_events[i].callback.clear();
						}
						else
						{
							std::swap(m_events[i], m_events[m_events.size() - 1]);
							m_events.pop_back();
						}
					}
				}
			}
		}

		bool remove(DelegateHandle& handle)
		{
			if (handle.isValid())
			{
				for (size_t i = 0; i < m_events.size(); ++i)
				{
					if (m_events[i].handle == handle)
					{
						if (isLocked())
						{
							m_events[i].callback.clear();
						}
						else
						{
							std::swap(m_events[i], m_events[m_events.size() - 1]);
							m_events.pop_back();
						}
						handle.reset();
						return true;
					}
				}
			}
			return false;
		}

		bool isBoundTo(const DelegateHandle& handle) const
		{
			if (handle.isValid())
			{
				for (size_t i = 0; i < m_events.size(); ++i)
				{
					if (m_events[i].handle == handle)
					{
						return true;
					}
				}
			}
			return false;
		}

		void removeAll()
		{
			if (isLocked())
			{
				for (DelegateHandlerPair& handler : m_events)
				{
					handler.callback.clear();
				}
			}
			else
			{
				m_events.clear();
			}
		}

		void shrink(const size_t maxSpace = 0)
		{
			if (isLocked() == false)
			{
				size_t toDelete = 0;
				for (size_t i = 0; i < m_events.size() - toDelete; ++i)
				{
					if (m_events[i].handle.isValid() == false)
					{
						std::swap(m_events[i], m_events[toDelete]);
						++toDelete;
					}
				}
				if (toDelete > maxSpace)
				{
					m_events.resize(m_events.size() - toDelete);
				}
			}
		}

		void broadcast(Args ...args)
		{
			lock();
			for (size_t i = 0; i < m_events.size(); ++i)
			{
				if (m_events[i].handle.isValid())
				{
					m_events[i].callback.execute(std::forward<Args>(args)...);
				}
			}
			unlock();
		}
	};
}