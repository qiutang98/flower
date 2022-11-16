#pragma once
#include "AsyncUploader.h"

namespace Flower
{
	enum class EAsyncTaskType
	{
		Raster, // Use graphics queue, race pixel warp with major grphics queue.
		Compute, // Use async compute, performance better.
	};

	struct AsyncGPUTask
	{
		virtual void finishCallback() = 0;
		virtual void taskFunction(VkCommandBuffer cmd) = 0;
	};

	class GpuTaskScheduler
	{
	private:
		std::string m_name;

		// This tasks finish fence.
		VkFence m_fence = VK_NULL_HANDLE;

		// Working queue.
		VkQueue m_queue = VK_NULL_HANDLE;

		// Working queue family.
		uint32_t m_queueFamily = VK_QUEUE_FAMILY_IGNORED;

		// Working pool.
		VkCommandPool m_pool = VK_NULL_HANDLE;

		// Uploader can still alive.
		std::atomic<bool> m_bRun = true;

		// Uploader is working.
		std::atomic<bool> m_bProcessing = false;

		// This thread end future.
		std::future<void> m_future;

	protected:
		virtual void threadFunction() = 0;

		void resetProcessState();
		void startRecord();
		void endRecordAndSubmit();

	public:
		GpuTaskScheduler(const std::string& name, AsyncUploaderManager& in, VkQueue inQueue, uint32_t inFamily);

		void setStop()
		{
			m_bRun.store(false);
		}

		void wait()
		{
			m_future.wait();
		}

		bool isProcessing() const
		{
			return m_bProcessing.load();
		}
	};

	// It just loop and push task into async compute queue.
	class AsyncQueueScheduler
	{
	
		struct 
		{
			std::condition_variable cv;

			// Mutex guard the task queue.
			std::mutex mutex;

			// Task queue.
			std::queue<std::shared_ptr<AsyncGPUTask>> tasks;
		} m_taskContext;

	public:
		explicit AsyncQueueScheduler();

		bool busy();

		void tick();

		void addTask(std::shared_ptr<AsyncGPUTask> inTask);

		void flushTask();

		void release();
		
	};

	// using GpuTaskScheduler = Singleton<AsyncQueueScheduler>;
}