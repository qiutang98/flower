#include "uploader.h"
#include "context.h"

namespace engine
{
	static const size_t kAsyncStaticUploaderNum  = 4;
	static const size_t kAsyncDynamicUploaderNum = 2;

	static const uint32_t kBufferOffsetRoundUpSize = 128;
	static inline uint32_t getUploadSizeQuantity(AssetLoadTask* task)
	{
		return kBufferOffsetRoundUpSize * 
			divideRoundingUp(task->uploadSize(), kBufferOffsetRoundUpSize);
	}

	static inline std::string getTransferBufferUniqueId()
	{
		static std::atomic<size_t> counter = 0;
		return "M_AsyncUpload_" + std::to_string(counter.fetch_add(1));
	}

	void AsyncUploaderBase::prepareCommandBufferAsync()
	{
		CHECK(m_poolAsync == VK_NULL_HANDLE);
		CHECK(m_commandBufferAsync == VK_NULL_HANDLE);

		// Create async pool.
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = getContext()->getCopyFamily();
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		RHICheck(vkCreateCommandPool(getContext()->getDevice(), &poolInfo, nullptr, &m_poolAsync));

		// Allocated common buffer.
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_poolAsync;
		allocInfo.commandBufferCount = 1;
		RHICheck(vkAllocateCommandBuffers(getContext()->getDevice(), &allocInfo, &m_commandBufferAsync));
	}

	void AsyncUploaderBase::destroyCommandBufferAsync()
	{
		CHECK(m_poolAsync);
		CHECK(m_commandBufferAsync);

		vkFreeCommandBuffers(getContext()->getDevice(), m_poolAsync, 1, &m_commandBufferAsync);
		vkDestroyCommandPool(getContext()->getDevice(), m_poolAsync, nullptr);

		m_poolAsync = VK_NULL_HANDLE;
		m_commandBufferAsync = VK_NULL_HANDLE;
	}

	void AsyncUploaderBase::startRecordAsync()
	{
		vkResetCommandBuffer(m_commandBufferAsync, 0);

		// Begin record.
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_commandBufferAsync, &beginInfo);
	}

	void AsyncUploaderBase::endRecordAsync()
	{
		vkEndCommandBuffer(m_commandBufferAsync);
	}

	void AsyncUploaderBase::onFinished()
	{
		CHECK(m_bWorking);
		m_bWorking = false;
	}

	AsyncUploaderBase::AsyncUploaderBase(const std::string& name, AsyncUploaderManager& in)
		: m_name(name), m_manager(in)
	{
		m_future = std::async(std::launch::async, [this]()
		{
			// Create fence of this uploader state.
			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			vkCreateFence(getDevice(), &fenceInfo, nullptr, &m_fence);
			
			resetFence();
			prepareCommandBufferAsync();

			while (m_bRun.load())
			{
				threadFunction();
			}

			// Before release you must ensure all work finish.
			CHECK(!working());

			destroyCommandBufferAsync();
			vkDestroyFence(getContext()->getDevice(), m_fence, nullptr);
			LOG_INFO("Async uploader {0} destroy.", m_name);
		});

		LOG_INFO("Async uploader {0} create.", m_name);
	}

	void AsyncUploaderBase::resetFence()
	{
		vkResetFences(getDevice(), 1, &m_fence);
	}

	void DynamicAsyncUploader::loadTick()
	{
		ZoneScoped;
		CHECK(!m_processingTask);
		CHECK(m_bWorking == false);

		// Get task from manager.
		m_manager.dynamicTasksAction([&, this](std::queue<std::shared_ptr<AssetLoadTask>>& srcQueue)
		{
			// Empty already.
			if (srcQueue.size() == 0)
			{
				return;
			}

			if (srcQueue.size() > 0)
			{
				std::shared_ptr<AssetLoadTask> processTask = srcQueue.front();
				uint32_t requireSize = getUploadSizeQuantity(processTask.get());
				CHECK(requireSize >= m_manager.getDynamicUploadMinSize());

				m_processingTask = processTask;
				srcQueue.pop();
			}
		});

		// May stole by other thread, return.
		if (!m_processingTask)
		{
			tryReleaseStageBuffer();
			return;
		}

		m_bWorking = true;

		uint32_t requireSize = getUploadSizeQuantity(m_processingTask.get());

		const bool bShouldRecreate =
			   (m_stageBuffer == nullptr)                    // No create yet.
			|| (m_stageBuffer->getSize() < requireSize)      // Size no enough.
			|| (m_stageBuffer->getSize() > 2 * requireSize); // Size too big, waste too much time.

		if (bShouldRecreate)
		{
			m_stageBuffer = std::make_unique<VulkanBuffer>(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				getTransferBufferUniqueId().c_str(),
				requireSize
			);
		}

		startRecordAsync();
		m_stageBuffer->map();
		{
			void* mapped = m_stageBuffer->getMapped();
			RHICommandBufferBase commandBase
			{ 
				.cmd = m_commandBufferAsync, 
				.pool = m_poolAsync,
				.queueFamily = getContext()->getCopyFamily()
			};
			m_processingTask->uploadFunction(0, mapped, commandBase, *m_stageBuffer);
		}
		m_stageBuffer->unmap();
		endRecordAsync();

		m_manager.pushSubmitFunctions(this);
	}

	void DynamicAsyncUploader::tryReleaseStageBuffer()
	{
		CHECK(!m_processingTask);
		m_stageBuffer = nullptr;
	}

	void DynamicAsyncUploader::threadFunction()
	{
		if (!m_manager.dynamicLoadAssetTaskEmpty() && !working())
		{
			loadTick();
		}
		else
		{
			std::unique_lock<std::mutex> lock(m_manager.getDynamicMutex());
			m_manager.getDynamicCondition().wait(lock);
		}
	}

	void DynamicAsyncUploader::onFinished()
	{
		ASSERT(m_processingTask != nullptr, "Awake dynamic async loader but no task feed, fix me!");
		m_processingTask->finishCallback();
		m_processingTask = nullptr;

		AsyncUploaderBase::onFinished();
	}

	void StaticAsyncUploader::loadTick()
	{
		ZoneScoped;

		CHECK(m_processingTasks.empty());
		CHECK(m_bWorking == false);

		// Get static task from manager.
		m_manager.staticTasksAction([&, this](std::queue<std::shared_ptr<AssetLoadTask>>& srcQueue)
		{
			// Empty already.
			if (srcQueue.size() == 0)
			{
				return;
			}

			uint32_t availableSize = m_manager.getStaticUploadMaxSize();
			while (srcQueue.size() > 0)
			{
				std::shared_ptr<AssetLoadTask> processTask = srcQueue.front();
				uint32_t requireSize = getUploadSizeQuantity(processTask.get());
				CHECK(requireSize < m_manager.getDynamicUploadMinSize());

				// Small buffer use static uploader.
				if (availableSize > requireSize)
				{
					m_processingTasks.push_back(processTask);
					availableSize -= requireSize;
					srcQueue.pop();
				}
				else
				{
					// No enough space for new task, break task push.
					break;
				}
			}
		});

		// May stole by other thread, no processing task, return.
		if (m_processingTasks.size() <= 0)
		{
			return;
		}

		// Now can work.
		m_bWorking = true;

		if (m_stageBuffer == nullptr)
		{
			VkDeviceSize baseBufferSize = static_cast<VkDeviceSize>(m_manager.getStaticUploadMaxSize());

			m_stageBuffer = std::make_unique<VulkanBuffer>(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				getTransferBufferUniqueId().c_str(),
				baseBufferSize
			);
		}

		// Do copy action here.
		startRecordAsync();
		m_stageBuffer->map();
		{
			RHICommandBufferBase commandBase
			{
				.cmd = m_commandBufferAsync,
				.pool = m_poolAsync,
				.queueFamily = getContext()->getCopyFamily()
			};
			void* mapped = m_stageBuffer->getMapped();

			uint32_t offsetPos = 0;
			for (auto& uploadingTask : m_processingTasks)
			{
				uploadingTask->uploadFunction(offsetPos, mapped, commandBase, *m_stageBuffer);

				offsetPos += getUploadSizeQuantity(uploadingTask.get());
				mapped = (void*)((char*)mapped + getUploadSizeQuantity(uploadingTask.get()));
			}
		}
		m_stageBuffer->unmap();

		endRecordAsync();
		m_manager.pushSubmitFunctions(this);
	}

	void StaticAsyncUploader::threadFunction()
	{
		if (!m_manager.staticLoadAssetTaskEmpty() && !working())
		{
			loadTick();
		}
		else
		{
			std::unique_lock<std::mutex> lock(m_manager.getStaticMutex());
			m_manager.getStaticCondition().wait(lock);
		}
	}

	void StaticAsyncUploader::onFinished()
	{
		CHECK(m_processingTasks.size() > 0);
		for (auto& uploadingTask : m_processingTasks)
		{
			uploadingTask->finishCallback();
		}
		m_processingTasks.clear();

		AsyncUploaderBase::onFinished();
	}

	AsyncUploaderManager::AsyncUploaderManager(uint32_t staticUploaderMaxSize, uint32_t dynamicUploaderMinSize)
		: m_dynamicUploaderMinSize(dynamicUploaderMinSize * 1024 * 1024), m_staticUploaderMaxSize(staticUploaderMaxSize * 1024 * 1024)
	{
		const auto& copyPools = getContext()->getNormalCopyCommandPools();
		const uint32_t copyFamily = getContext()->getCopyFamily();

		for (size_t i = 0; i < kAsyncStaticUploaderNum; i++)
		{
			std::string name = "StaticAsyncUpload_" + std::to_string(i);
			m_staticUploaders.push_back(
				std::make_unique<StaticAsyncUploader>(name, *this));
		}

		for (size_t i = 0; i < kAsyncDynamicUploaderNum; i++)
		{
			std::string name = "DynamicAsyncUpload_" + std::to_string(i);
			m_dynamicUploaders.push_back(
				std::make_unique<DynamicAsyncUploader>(name, *this));
		}
	}

	void AsyncUploaderManager::addTask(std::shared_ptr<AssetLoadTask> inTask)
	{
		if (getUploadSizeQuantity(inTask.get()) >= getDynamicUploadMinSize())
		{
			dynamicTasksAction([&](std::queue<std::shared_ptr<AssetLoadTask>>& queue) 
				{ queue.push(inTask); });
		}
		else
		{
			staticTasksAction([&](std::queue<std::shared_ptr<AssetLoadTask>>& queue) 
				{ queue.push(inTask); });
		}
	}

	void AsyncUploaderManager::tick(const RuntimeModuleTickData& tickData)
	{
		// Flush submit functions.
		syncPendingObjects();
		submitObjects();

		if (!staticLoadAssetTaskEmpty())
		{
			getStaticCondition().notify_all();
		}
		if (!dynamicLoadAssetTaskEmpty())
		{
			getDynamicCondition().notify_all();
		}
	}

	void AsyncUploaderManager::submitObjects()
	{
		std::lock_guard<std::mutex> lock(m_m_submitObjectsMutex);

		if (!m_submitObjects.empty())
		{
			size_t indexQueue = 0;
			size_t maxIndex = getContext()->getNormalCopyCommandPools().size() - 1;

			for (auto* obj : m_submitObjects)
			{
				VkSubmitInfo submitInfo{};
				submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &obj->getCommandBuffer();

				vkQueueSubmit(getContext()->getNormalCopyCommandPools()[indexQueue].queue, 1, &submitInfo, obj->getFence());

				indexQueue++;
				if (indexQueue > maxIndex)
				{
					indexQueue = 0;
				}

				m_pendingObjects.push_back(obj);
			}
			m_submitObjects.clear();
		}
	}

	void AsyncUploaderManager::syncPendingObjects()
	{
		std::erase_if(m_pendingObjects, [](auto* obj) 
		{ 
			bool bResult = false;
			if (vkGetFenceStatus(getContext()->getDevice(), obj->getFence()) == VK_SUCCESS)
			{
				obj->onFinished();
				obj->resetFence();
				bResult = true;
			}
			return bResult;
		});
	}

	void AsyncUploaderManager::beforeReleaseFlush()
	{
		{
			std::lock_guard lock(m_staticContext.mutex);
			m_staticContext.tasks = {};
		}
		{
			std::lock_guard lock(m_dynamicContext.mutex);
			m_dynamicContext.tasks = {};
		}
		vkDeviceWaitIdle(getDevice());
	}

	bool AsyncUploaderManager::busy()
	{
		bool bAllFree = 
			   staticLoadAssetTaskEmpty()
			&& dynamicLoadAssetTaskEmpty();

		for (size_t i = 0; i < m_staticUploaders.size(); i++)
		{
			bAllFree &= !m_staticUploaders[i]->working();
		}
		for (size_t i = 0; i < m_dynamicUploaders.size(); i++)
		{
			bAllFree &= !m_dynamicUploaders[i]->working();
		}

		return !bAllFree;
	}

	void AsyncUploaderManager::flushTask()
	{
		while (busy())
		{
			getStaticCondition().notify_all();
			getDynamicCondition().notify_all();

			submitObjects();
			syncPendingObjects();
		}
	}

	void AsyncUploaderManager::release()
	{
		flushTask();
		LOG_INFO("Start release async uploader threads...");

		for (size_t i = 0; i < m_staticUploaders.size(); i++)
		{
			m_staticUploaders[i]->stop();
		}
		getStaticCondition().notify_all();
		for (size_t i = 0; i < m_dynamicUploaders.size(); i++)
		{
			m_dynamicUploaders[i]->stop();
		}
		getDynamicCondition().notify_all();

		// Wait all futures.
		for (size_t i = 0; i < m_staticUploaders.size(); i++)
		{
			m_staticUploaders[i]->wait();
			m_staticUploaders[i].reset();
		}
		for (size_t i = 0; i < m_dynamicUploaders.size(); i++)
		{
			m_dynamicUploaders[i]->wait();
			m_dynamicUploaders[i].reset();
		}

		LOG_INFO("All async uploader threads release.");
	}

}