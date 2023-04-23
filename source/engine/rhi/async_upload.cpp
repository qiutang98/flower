#include "async_upload.h"
#include "rhi.h"

namespace engine
{
	static std::string getTransferBufferUniqueId()
	{
		static std::atomic<size_t> counter = 0;
		return "M_AsyncUpload_" + std::to_string(counter.fetch_add(1));
	}

	void AsyncUploaderBase::resetProcessState()
	{
		CHECK(m_bProcessing);

		vkResetFences(m_context->getDevice(), 1, &m_fence);
		vkResetCommandBuffer(m_commandBuffer, 0);
	}

	void AsyncUploaderBase::startRecord()
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
	}

	void AsyncUploaderBase::endRecordAndSubmit()
	{
		vkEndCommandBuffer(m_commandBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_commandBuffer;

		vkQueueSubmit(m_queue, 1, &submitInfo, m_fence);
	}

	AsyncUploaderBase::AsyncUploaderBase(VulkanContext* ct, const std::string& name, AsyncUploaderManager& in, VkQueue inQueue, uint32_t inFamily)
		: m_context(ct), m_name(name), m_manager(in), m_queue(inQueue), m_queueFamily(inFamily)
	{
		m_future = std::async(std::launch::async, [this]()
		{
			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

			vkCreateFence(m_context->getDevice(), &fenceInfo, nullptr, &m_fence);

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.queueFamilyIndex = m_context->getCopyFamily();
			poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			RHICheck(vkCreateCommandPool(m_context->getDevice(), &poolInfo, nullptr, &m_pool));

			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = m_pool;
			allocInfo.commandBufferCount = 1;
			RHICheck(vkAllocateCommandBuffers(m_context->getDevice(), &allocInfo, &m_commandBuffer));

			threadFunction();

			vkFreeCommandBuffers(m_context->getDevice(), m_pool, 1, &m_commandBuffer);
			vkDestroyCommandPool(m_context->getDevice(), m_pool, nullptr);
			vkDestroyFence(m_context->getDevice(), m_fence, nullptr);

			LOG_INFO("Async upload thread {0} exit.", m_name);
		});
	}

	void DynamicAsyncUploader::loadTick()
	{
		// Handle still processing state.
		if (m_bProcessing)
		{
			while (vkGetFenceStatus(m_context->getDevice(), m_fence) != VK_SUCCESS)
			{
				std::this_thread::yield();
			}

			resetProcessState();

			CHECK(m_processingTask != nullptr);
			m_processingTask->finishCallback();
			m_processingTask = nullptr;

			m_bProcessing = false;
		}

		// Task already empty, just return.
		if (m_manager.dynamicLoadAssetTaskEmpty())
		{
			tryReleaseStageBuffer();
			return;
		}

		CHECK(!m_processingTask);

		// Get static task from manager.
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
				uint32_t requireSize = processTask->uploadSize();
				CHECK(requireSize >= m_manager.getDynamicUploadMinSize());

				m_processingTask = processTask;
				m_bProcessing = true;
				srcQueue.pop();
			}
		});

		if (!m_processingTask)
		{
			tryReleaseStageBuffer();
			return;
		}

		uint32_t requireSize = m_processingTask->uploadSize();

		const bool bShouldRecreate =
			   (m_stageBuffer == nullptr)                    // No create yet.
			|| (m_stageBuffer->getSize() < requireSize)      // Size no enough.
			|| (m_stageBuffer->getSize() > 2 * requireSize); // Size too big, waste too much time.

		if (bShouldRecreate)
		{
			m_stageBuffer = std::make_unique<VulkanBuffer>(
				m_context,
				getTransferBufferUniqueId().c_str(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
				requireSize,
				nullptr
			);

		}

		startRecord();
		m_stageBuffer->map();
		{
			void* mapped = m_stageBuffer->getMapped();

			RHICommandBufferBase commandBase{ .cmd = m_commandBuffer, .pool = m_pool, .queueFamily = m_queueFamily };
			m_processingTask->uploadFunction(0, mapped, commandBase, *m_stageBuffer);

		}
		m_stageBuffer->unmap();
		endRecordAndSubmit();
	}

	void DynamicAsyncUploader::tryReleaseStageBuffer()
	{
		CHECK(!m_processingTask && !m_bProcessing);
		m_stageBuffer = nullptr;
	}

	void DynamicAsyncUploader::threadFunction()
	{
		while (m_bRun.load())
		{
			if (!m_manager.dynamicLoadAssetTaskEmpty() || m_bProcessing.load())
			{
				loadTick();
			}
			else
			{
				std::unique_lock<std::mutex> lock(m_manager.getDynamicMutex());
				m_manager.getDynamicCondition().wait(lock);
			}
		}
	}

	void StaticAsyncUploader::loadTick()
	{
		// Handle still processing state.
		if (m_bProcessing)
		{
			while (vkGetFenceStatus(m_context->getDevice(), m_fence) != VK_SUCCESS)
			{
				// Still processing, keep going.
				std::this_thread::yield();
			}


			resetProcessState();

			CHECK(m_processingTasks.size() > 0);
			for (auto& uploadingTask : m_processingTasks)
			{
				uploadingTask->finishCallback();
			}
			m_processingTasks.clear();
			m_bProcessing = false;
		}

		// Task already empty, just return.
		if (m_manager.staticLoadAssetTaskEmpty())
		{
			return;
		}

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
				uint32_t requireSize = processTask->uploadSize();
				CHECK(requireSize < m_manager.getDynamicUploadMinSize());

				// Small buffer use static uploader.
				if (availableSize > requireSize)
				{
					m_processingTasks.push_back(processTask);
					availableSize -= requireSize;
					m_bProcessing = true;
					srcQueue.pop();
				}
				else
				{
					// No enough space for new task, break task push.
					break;
				}
			}
		});

		// No processing task, return.
		if (m_processingTasks.size() <= 0)
		{
			return;
		}

		if (m_stageBuffer == nullptr)
		{
			VkDeviceSize baseBufferSize = static_cast<VkDeviceSize>(m_manager.getStaticUploadMaxSize());

			m_stageBuffer = std::make_unique<VulkanBuffer>(
				m_context,
				getTransferBufferUniqueId().c_str(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
				baseBufferSize,
				nullptr
			);

			
		}

		// Do copy action here.
		startRecord();
		m_stageBuffer->map();
		{
			RHICommandBufferBase commandBase{ .cmd = m_commandBuffer, .pool = m_pool, .queueFamily = m_queueFamily };
			void* mapped = m_stageBuffer->getMapped();

			uint32_t offsetPos = 0;
			for (auto& uploadingTask : m_processingTasks)
			{
				uploadingTask->uploadFunction(offsetPos, mapped, commandBase, *m_stageBuffer);
				offsetPos += uploadingTask->uploadSize();
				mapped = (void*)((char*)mapped + uploadingTask->uploadSize());
			}
		}
		m_stageBuffer->unmap();

		endRecordAndSubmit();
	}

	void StaticAsyncUploader::threadFunction()
	{
		while (m_bRun.load())
		{
			if (!m_manager.staticLoadAssetTaskEmpty() || m_bProcessing.load())
			{
				loadTick();
			}
			else
			{
				std::unique_lock<std::mutex> lock(m_manager.getStaticMutex());
				m_manager.getStaticCondition().wait(lock);
			}
		}
	}

	AsyncUploaderManager::AsyncUploaderManager(VulkanContext* ct, uint32_t staticUploaderMaxSize, uint32_t dynamicUploaderMinSize)
		: m_context(ct), m_dynamicUploaderMinSize(dynamicUploaderMinSize * 1024 * 1024), m_staticUploaderMaxSize(staticUploaderMaxSize * 1024 * 1024)
	{
		const auto& copyPools = m_context->getNormalCopyCommandPools();

		const uint32_t copyFamily = m_context->getCopyFamily();

		// One copy queue one static uploader.
		m_staticUploaders.resize(copyPools.size());
		for (size_t i = 0; i < m_staticUploaders.size(); i++)
		{
			std::string name = "StaticAsyncUpload_" + std::to_string(i);
			m_staticUploaders[i] = std::make_unique<StaticAsyncUploader>(ct, name, *this, copyPools[i].queue, copyFamily);
		}

		// One dynamic uploader is enough.
		constexpr size_t dynamicNum = 1;
		m_dynamicUploaders.resize(dynamicNum);
		for (size_t i = 0; i < m_dynamicUploaders.size(); i++)
		{
			std::string name = "DynamicAsyncUpload_" + std::to_string(i);

			// Dynamic upload always use copy queue #0.
			m_dynamicUploaders[i] = std::make_unique<DynamicAsyncUploader>(ct, name, *this, copyPools[0].queue, copyFamily);
		}
	}

	void AsyncUploaderManager::addTask(std::shared_ptr<AssetLoadTask> inTask)
	{
		if (inTask->uploadSize() >= getDynamicUploadMinSize())
		{
			dynamicTasksAction([&](std::queue<std::shared_ptr<AssetLoadTask>>& queue){ queue.push(inTask); });
			m_dynamicContext.cv.notify_one();
		}
		else
		{
			// If size fit static uploader size, push in queue.
			staticTasksAction([&](std::queue<std::shared_ptr<AssetLoadTask>>& queue) { queue.push(inTask); });
			m_staticContext.cv.notify_one();
		}
	}

	bool AsyncUploaderManager::busy()
	{
		bool bAllFree = true;

		// All tasks free.
		bAllFree &= staticLoadAssetTaskEmpty();
		bAllFree &= dynamicLoadAssetTaskEmpty();

		// Also handle no processing case.
		for (size_t i = 0; i < m_staticUploaders.size(); i++)
		{
			bAllFree &= !m_staticUploaders[i]->isProcessing();
		}
		for (size_t i = 0; i < m_dynamicUploaders.size(); i++)
		{
			bAllFree &= !m_dynamicUploaders[i]->isProcessing();
		}

		return !bAllFree;
	}

	void AsyncUploaderManager::flushTask()
	{
		getStaticCondition().notify_all();
		getDynamicCondition().notify_all();
		while (busy())
		{
			std::this_thread::yield();
		}
	}

	void AsyncUploaderManager::release()
	{
		flushTask();
		LOG_INFO("Start release async uploader threads...");

		for (size_t i = 0; i < m_staticUploaders.size(); i++)
		{
			m_staticUploaders[i]->setStop();
		}
		m_staticContext.cv.notify_all();
		for (size_t i = 0; i < m_dynamicUploaders.size(); i++)
		{
			m_dynamicUploaders[i]->setStop();
		}
		m_dynamicContext.cv.notify_all();

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