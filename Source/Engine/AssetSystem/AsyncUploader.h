#pragma once
#include "AssetCommon.h"

namespace Flower
{
	// Static uploader: allocate static stage buffer and never release.
	// Dynamic uploader: Allocate dynamic stage buffer when need, and release when no task.

	constexpr size_t GStaticUploaderMaxSize = 64 * 1024 * 1024; // 64 (MB)
	constexpr size_t GDynamicUploaderMinSize = GStaticUploaderMaxSize / 2; // 32 (MB)

	struct AssetLoadTask
	{
		// When load task finish call.
		virtual void finishCallback() = 0;

		// Load task need stage buffer size.
		virtual uint32_t uploadSize() const = 0;

		// Upload main body function.
		virtual void uploadFunction(
			uint32_t stageBufferOffset, 
			void* bufferStart,
			RHICommandBufferBase& commandBuffer, 
			VulkanBuffer& stageBuffer) = 0;
	};

	class AsyncUploaderManager;

	class AsyncUploaderBase : NonCopyable
	{
	protected:
		std::string m_name;

		AsyncUploaderManager& m_manager;

		// This loader tasks finish fence.
		VkFence m_fence = VK_NULL_HANDLE;

		// Working queue.
		VkQueue m_queue = VK_NULL_HANDLE;

		// Working queue family.
		uint32_t m_queueFamily = VK_QUEUE_FAMILY_IGNORED;

		// Working pool.
		VkCommandPool m_pool = VK_NULL_HANDLE;

		// Working commmand buffer.
		VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;

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
		AsyncUploaderBase(const std::string& name, AsyncUploaderManager& in, VkQueue inQueue, uint32_t inFamily);

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


	class DynamicAsyncUploader : public AsyncUploaderBase
	{
	private:
		std::shared_ptr<AssetLoadTask> m_processingTask = nullptr;

		// Upload stage buffer.
		std::shared_ptr<VulkanBuffer> m_stageBuffer = nullptr;

	protected:
		void loadTick();
		void tryReleaseStageBuffer();
		virtual void threadFunction() override;

	public:
		DynamicAsyncUploader(const std::string& name, AsyncUploaderManager& in, VkQueue inQueue, uint32_t inFamily)
			: AsyncUploaderBase(name, in, inQueue, inFamily)
		{

		}
	};


	class StaticAsyncUploader : public AsyncUploaderBase
	{
	private:
		// Task upload use static stage buffer.
		std::vector<std::shared_ptr<AssetLoadTask>> m_processingTasks;

		std::shared_ptr<VulkanBuffer> m_stageBuffer = nullptr;

	private:
		void loadTick();
		virtual void threadFunction() override;

	public:
		StaticAsyncUploader(const std::string& name, AsyncUploaderManager& in, VkQueue inQueue, uint32_t inFamily)
			: AsyncUploaderBase(name, in, inQueue, inFamily)
		{

		}
	};

	class AsyncUploaderManager : NonCopyable
	{
	private:
		// Task need to load use static stage buffer.
		struct UploaderContext
		{
			std::condition_variable cv;

			// Mutex guard the task queue.
			std::mutex mutex;

			// Task queue.
			std::queue<std::shared_ptr<AssetLoadTask>> tasks;
		};
		
		UploaderContext m_staticContext;
		std::vector<std::unique_ptr<StaticAsyncUploader>> m_staticUploaders;

		UploaderContext m_dynamicContext;
		std::vector<std::unique_ptr<DynamicAsyncUploader>> m_dynamicUploaders;

		friend class AssetSystem;
		void addTask(std::shared_ptr<AssetLoadTask> inTask);
		void flushTask();
	public:
		explicit AsyncUploaderManager();

		std::condition_variable& getStaticCondition() 
		{ 
			return m_staticContext.cv;
		}

		std::mutex& getStaticMutex() 
		{ 
			return m_staticContext.mutex;
		}

		bool staticLoadAssetTaskEmpty()
		{
			std::lock_guard lock(m_staticContext.mutex);
			return m_staticContext.tasks.size() == 0;
		}

		void staticTasksAction(std::function<void(decltype(m_staticContext.tasks)&)>&& func)
		{
			std::lock_guard lock(m_staticContext.mutex);
			func(m_staticContext.tasks);
		}

		std::condition_variable& getDynamicCondition()
		{
			return m_dynamicContext.cv;
		}

		std::mutex& getDynamicMutex()
		{
			return m_dynamicContext.mutex;
		}

		bool dynamicLoadAssetTaskEmpty()
		{
			std::lock_guard lock(m_dynamicContext.mutex);
			return m_dynamicContext.tasks.size() == 0;
		}

		void dynamicTasksAction(std::function<void(decltype(m_dynamicContext.tasks)&)>&& func)
		{
			std::lock_guard lock(m_dynamicContext.mutex);
			func(m_dynamicContext.tasks);
		}

		bool busy();

		

		void release();
	};

	using GpuUploader = Singleton<AsyncUploaderManager>;
}