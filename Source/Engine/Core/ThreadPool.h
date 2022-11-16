#pragma once

#include "../Pch.h"
#include "NonCopyable.h"
#include "Singleton.h"

namespace Flower
{
	template<typename T>
	struct FutureCollection : NonCopyable
	{
		std::vector<std::future<T>> futures;

		FutureCollection(const size_t num = 0) 
			: futures(num) 
		{

		}

		[[nodiscard]] inline std::vector<T> get()
		{
			std::vector<T> results(futures.size());
			for (size_t i = 0; i < futures.size(); i++)
			{
				results[i] = futures[i].get();
			}
			return results;
		}

		inline void wait() const
		{
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
	};

	class ThreadPool : NonCopyable
	{
	public:
		std::atomic<bool> bPaused = false;

	private:
		std::atomic<bool> m_bRuning = false;
		std::atomic<bool> m_bWaiting = false;

		std::condition_variable m_cvTaskAvailable;
		std::condition_variable m_cvTaskDone;

		std::atomic<size_t> m_tasksQueueTotalNum = 0;
		mutable std::mutex m_taskQueueMutex;
		std::queue<std::function<void()>> m_tasksQueue;

		uint32_t m_threadCount = 0;
		std::unique_ptr<std::thread[]> m_threads = nullptr;

	private:
		void worker()
		{
			while (m_bRuning)
			{
				std::function<void()> task;

				std::unique_lock<std::mutex> lockTasksQueue(m_taskQueueMutex);
				m_cvTaskAvailable.wait(lockTasksQueue, [&] 
				{ 
					// When task queue no empty, keep moving.
					return !m_tasksQueue.empty() || !m_bRuning;
				});

				if (m_bRuning && !bPaused)
				{
					task = std::move(m_tasksQueue.front());
					m_tasksQueue.pop();

					// Unlock so other thread can find task from queue.
					lockTasksQueue.unlock();
					task();

					lockTasksQueue.lock();

					// When task finish, we need lock again to minus queue num.
					// Alway invoke new guy when waiting.
					--m_tasksQueueTotalNum;
					if (m_bWaiting)
					{
						m_cvTaskDone.notify_one();
					}
				}
			}
		}

		void createThread()
		{
			m_bRuning = true;
			for (uint32_t i = 0; i < m_threadCount; i++)
			{
				m_threads[i] = std::thread(&ThreadPool::worker, this);
			}
		}

		void destroyThreads()
		{
			m_bRuning = false;
			m_cvTaskAvailable.notify_all();
			for (uint32_t i = 0; i < m_threadCount; i++)
			{
				m_threads[i].join();
			}
		}

		inline uint32_t getThreadCount(bool bLeftOneFreeCore)
		{
			const static uint32_t gMaxCoreThreadNum = std::thread::hardware_concurrency();
			uint32_t result = 0;

			if (gMaxCoreThreadNum > 0)
			{
				result = bLeftOneFreeCore ? (gMaxCoreThreadNum - 1) : gMaxCoreThreadNum;
			}

			result = std::max(1u, result);
			return result;
		}

	public:
		[[nodiscard]] size_t getTasksQueuedNum() const
		{
			const std::scoped_lock tasksLock(m_taskQueueMutex);
			return m_tasksQueue.size();
		}

		[[nodiscard]] size_t getTasksRunningNum() const
		{
			const std::scoped_lock tasks_lock(m_taskQueueMutex);
			return m_tasksQueueTotalNum - m_tasksQueue.size();
		}

		[[nodiscard]] size_t getTasksTotal() const
		{
			return m_tasksQueueTotalNum;
		}

		[[nodiscard]] uint32_t getThreadCount() const
		{
			return m_threadCount;
		}

		// Wait for all task finish, if when pause, wait for all processing task finish.
		void waitForTasks()
		{
			m_bWaiting = true;

			std::unique_lock<std::mutex> tasksLock(m_taskQueueMutex);
			m_cvTaskDone.wait(tasksLock, [this] 
			{ 
				return (m_tasksQueueTotalNum == (bPaused ? m_tasksQueue.size() : 0)); 
			});
			m_bWaiting = false;
		}

		template <typename F, typename... A>
		void pushTask(const F& task, const A&... args)
		{
			{
				const std::scoped_lock tasksLock(m_taskQueueMutex);
				if constexpr (sizeof...(args) == 0)
				{
					m_tasksQueue.push(std::function<void()>(task));
				}
				else
				{
					m_tasksQueue.push(std::function<void()>([task, args...]{ task(args...); }));
				}
			}
			++m_tasksQueueTotalNum;
			m_cvTaskAvailable.notify_one();
		}

		template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
		[[nodiscard]] std::future<R> submit(const F& task, const A&... args)
		{
			std::shared_ptr<std::promise<R>> taskPromise = std::make_shared<std::promise<R>>();
			
			pushTask([task, args..., taskPromise]
			{
				if constexpr (std::is_void_v<R>)
				{
					task(args...);
					taskPromise->set_value();
				}
				else
				{
					taskPromise->set_value(task(args...));
				}
			});
			return taskPromise->get_future();
		}

		void reset(bool bLeftOneFreeCore = true)
		{
			const bool wasPaused = bPaused;

			bPaused = true;
			waitForTasks();
			destroyThreads();

			m_threadCount = getThreadCount(bLeftOneFreeCore);
			m_threads = std::make_unique<std::thread[]>(m_threadCount);

			bPaused = wasPaused;
			createThread();
		}

		template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
		[[nodiscard]] FutureCollection<R> parallelizeLoop(const T1& firstIndex, const T2& indexAfterLast, const F& loop, size_t numBlocks = 0)
		{
			T firstIndexT = static_cast<T>(firstIndex);
			T indexAfterLastT = static_cast<T>(indexAfterLast);

			if (firstIndexT == indexAfterLastT)
			{
				return FutureCollection<R>();
			}
				
			if (indexAfterLastT < firstIndexT)
			{
				std::swap(indexAfterLastT, firstIndexT);
			}
				
			if (numBlocks == 0)
			{
				numBlocks = m_threadCount;
			}
				
			const size_t totalSize = static_cast<size_t>(indexAfterLastT - firstIndexT);
			size_t blockSize = static_cast<size_t>(totalSize / numBlocks);
			if (blockSize == 0)
			{
				blockSize = 1;
				numBlocks = totalSize > 1 ? totalSize : 1;
			}

			FutureCollection<R> fc(numBlocks);
			for (size_t i = 0; i < numBlocks; ++i)
			{
				const T start = (static_cast<T>(i * blockSize) + firstIndexT);
				const T end = (i == numBlocks - 1) ? indexAfterLastT : (static_cast<T>((i + 1) * blockSize) + firstIndexT);
				fc.futures[i] = submit(loop, start, end);
			}
			return fc;
		}

	public:
		explicit ThreadPool(bool bLeftOneFreeCore = true)
		{
			m_threadCount = getThreadCount(bLeftOneFreeCore);
			m_threads = std::make_unique<std::thread[]>(m_threadCount);
			createThread();
		}

		~ThreadPool()
		{
			waitForTasks();
			destroyThreads();
		}
	};

	using GThreadPool = Singleton<ThreadPool>;
}