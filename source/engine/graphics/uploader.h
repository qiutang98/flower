#pragma once

#include "resource.h"

namespace engine
{
	// Static uploader: allocate static stage buffer and never release.
	// Dynamic uploader: allocate dynamic stage buffer when need, and release when no task.

	struct AssetLoadTask
	{
		// When load task finish call.
		virtual void finishCallback() = 0;

		// Load task need stage buffer size.
		virtual uint32_t uploadSize() const = 0;

		// Upload main body function.
		virtual void uploadFunction(
			uint32_t stageBufferOffset,
			void* bufferPtrStart,
			RHICommandBufferBase& commandBuffer,
			VulkanBuffer& stageBuffer) = 0;
	};

	class AsyncUploaderManager;

	class AsyncUploaderBase : NonCopyable
	{
	protected:
		std::string m_name;
		AsyncUploaderManager& m_manager;
		VkFence m_fence = VK_NULL_HANDLE;

		std::future<void> m_future;
		std::atomic<bool> m_bRun = true;
		std::atomic<bool> m_bWorking = false;

		// Pool and cmd buffer created and used in async thread.
		VkCommandPool m_poolAsync = VK_NULL_HANDLE;
		VkCommandBuffer m_commandBufferAsync = VK_NULL_HANDLE;

	protected:
		virtual void threadFunction() {}

		void startRecordAsync();
		void endRecordAsync();


		
	private:
		void prepareCommandBufferAsync();
		void destroyCommandBufferAsync();

	public:
		AsyncUploaderBase(const std::string& name, AsyncUploaderManager& in);

		void resetFence();
		const VkFence& getFence() const { return m_fence; }
		const VkCommandBuffer& getCommandBuffer() const { return m_commandBufferAsync; }

		void wait() { m_future.wait(); }
		void stop() { m_bRun.store(false); }
		bool working() const { return m_bWorking.load(); }

		virtual void onFinished();
	};

	class DynamicAsyncUploader : public AsyncUploaderBase
	{
	private:
		std::shared_ptr<AssetLoadTask> m_processingTask = nullptr;
		std::unique_ptr<VulkanBuffer> m_stageBuffer = nullptr;

	protected:
		void loadTick();
		void tryReleaseStageBuffer();
		virtual void threadFunction() override;

	public:
		DynamicAsyncUploader(const std::string& name, AsyncUploaderManager& in)
			: AsyncUploaderBase(name, in)
		{

		}

		virtual void onFinished() override;
	};

	class StaticAsyncUploader : public AsyncUploaderBase
	{
	private:
		std::vector<std::shared_ptr<AssetLoadTask>> m_processingTasks;
		std::unique_ptr<VulkanBuffer> m_stageBuffer = nullptr;

	private:
		void loadTick();
		virtual void threadFunction() override;

	public:
		StaticAsyncUploader(const std::string& name, AsyncUploaderManager& in)
			: AsyncUploaderBase(name, in)
		{

		}

		virtual void onFinished() override;
	};

	class AsyncUploaderManager : NonCopyable
	{
	private:
		// Task need to load use static stage buffer.
		struct UploaderContext
		{
			std::condition_variable cv;
			std::mutex mutex;
			std::queue<std::shared_ptr<AssetLoadTask>> tasks;
		};

		UploaderContext m_staticContext;
		UploaderContext m_dynamicContext;

		std::vector<std::unique_ptr<StaticAsyncUploader>> m_staticUploaders;
		std::vector<std::unique_ptr<DynamicAsyncUploader>> m_dynamicUploaders;

		uint32_t m_staticUploaderMaxSize;
		uint32_t m_dynamicUploaderMinSize;

		std::mutex m_m_submitObjectsMutex;
		std::vector<AsyncUploaderBase*> m_submitObjects;
		std::vector<AsyncUploaderBase*> m_pendingObjects;

	public:
		explicit AsyncUploaderManager(
			uint32_t staticUploaderMaxSize, 
			uint32_t dynamicUploaderMinSize);

		void addTask(std::shared_ptr<AssetLoadTask> inTask);
		void tick(const RuntimeModuleTickData& tickData);

		void submitObjects();
		void syncPendingObjects();

		void flushTask();

		std::condition_variable& getStaticCondition()
		{
			return m_staticContext.cv;
		}

		void pushSubmitFunctions(AsyncUploaderBase* f)
		{
			std::lock_guard<std::mutex> lock(m_m_submitObjectsMutex);
			m_submitObjects.push_back(f);
		}

		uint32_t getStaticUploadMaxSize() const { return m_staticUploaderMaxSize; }
		uint32_t getDynamicUploadMinSize() const { return m_dynamicUploaderMinSize; }

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

		void beforeReleaseFlush();

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
}