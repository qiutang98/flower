#pragma once
#include <memory>

namespace engine
{
	// Single object inline allocator. it keep one small stack memory.
	// When stack memory overflow, then use heap memory.
	template<size_t kMaxStackSize>
	class SingleInlineAllocator
	{
	private:
        static_assert(kMaxStackSize > sizeof(void*), "kMaxStackSize is smaller or equal to the size of a pointer.");
		union
		{
			char buffer[kMaxStackSize];
			void* pPtr; // Heap memory, use when memory size bigger than kMaxStackSize.
		};
		size_t m_size;

	public:
		size_t getSize() const { return m_size; }
		bool hasAllocation() const { return m_size > 0; }
		bool hasHeapAllocation() const { return m_size > kMaxStackSize; }

		void* allocateInternal(const size_t size)
		{
			if (m_size != size)
			{
				freeInternal();

				m_size = size;
				if (size > kMaxStackSize)
				{
					pPtr = malloc(size);
					return pPtr;
				}
			}
			return (void*)buffer;
		}

		void freeInternal()
		{
			if (m_size > kMaxStackSize)
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

		SingleInlineAllocator() noexcept : m_size(0)
		{
			
		}

		SingleInlineAllocator(const SingleInlineAllocator& other) : m_size(0)
		{
			if (other.hasAllocation())
			{
				memcpy(allocateInternal(other.m_size), other.getAllocation(), other.m_size);
			}
			m_size = other.m_size;
		}

		~SingleInlineAllocator() noexcept
		{
			freeInternal();
		}

		SingleInlineAllocator& operator=(const SingleInlineAllocator& other)
		{
			if (other.hasAllocation())
			{
				memcpy(allocateInternal(other.m_size), other.getAllocation(), other.m_size);
			}
			m_size = other.m_size;

			return *this;
		}

		// Move construct function.
		SingleInlineAllocator(SingleInlineAllocator&& other) noexcept : m_size(other.m_size)
		{
			other.m_size = 0;

			if (m_size > kMaxStackSize)
			{
				// Heap memory just swap.
				std::swap(pPtr, other.pPtr);
			}
			else
			{
				// Stack memory use copy.
				memcpy(buffer, other.buffer, m_size);
			}
		}

		// Move copy function.
		SingleInlineAllocator& operator=(SingleInlineAllocator&& other) noexcept
		{
			freeInternal();

			m_size = other.m_size;
			other.m_size = 0;

			if (m_size > kMaxStackSize)
			{
				// Heap memory just swap.
				std::swap(pPtr, other.pPtr);
			}
			else
			{
				// Stack memory use copy.
				memcpy(buffer, other.buffer, m_size);
			}
			return *this;
		}
	};
}