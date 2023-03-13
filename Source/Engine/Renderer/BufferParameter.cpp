#include "Pch.h"
#include "BufferParameter.h"

namespace Flower
{
	inline void bufferParamsSizeCheck(size_t i)
	{
		sizeSafeCheck(i, 999);
	}

	BufferParametersRing::BufferParameter::BufferParameter(
		const char* name,
		size_t bufferSize,
		VkBufferUsageFlags bufferUsage,
		VkDescriptorType type,
		EVMAUsageFlags vmaFlags,
		VkMemoryPropertyFlags memoryFlags)
		: m_bufferSize(bufferSize)
	{
		// Create buffer and set for use convenience.
		m_buffer = VulkanBuffer::create(
			name,
			bufferUsage,
			memoryFlags,
			vmaFlags,
			bufferSize,
			nullptr
		);

		VkDescriptorBufferInfo bufInfo = {};
		bufInfo.buffer = m_buffer->getVkBuffer();
		bufInfo.offset = 0;
		bufInfo.range = bufferSize;

		// default set to binding position zero.
		RHI::get()->descriptorFactoryBegin()
			.bindBuffers(0, 1, &bufInfo, type, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(m_set, m_layout);
	}

	BufferParametersRing::BufferParametersManager::BufferParametersManager(
		bool bMultiFrame,
		VkBufferUsageFlags bufferUsage,
		VkDescriptorType type,
		EVMAUsageFlags vmaFlags,
		VkMemoryPropertyFlags memoryFlags)
		: m_bMultiFrame(bMultiFrame)
		, m_bufferUsage(bufferUsage)
		, m_descriptorType(type)
		, m_vmaFlags(vmaFlags)
		, m_memoryFlags(memoryFlags)
	{

	}

	void BufferParametersRing::BufferParametersManager::tick()
	{
		// When multi frame, manager will tick 
		const uint64_t removeThreshold = m_bMultiFrame ? 0 : RHI::GMaxSwapchainCount;

		// Insert unsort free pool here.
		m_freePool.insert(m_freePool.end(), m_unsortFreePool.begin(), m_unsortFreePool.end());
		m_unsortFreePool.clear();

		bufferParamsSizeCheck(m_freePool.size());
		bufferParamsSizeCheck(m_busyPool.size());

		// Remove longtime unused buffer pool.
		m_freePool.erase(std::remove_if(m_freePool.begin(), m_freePool.end(), [&](const BufferParameterMisc& elem)
		{
			return m_tickCount > elem.freeTime + removeThreshold;
		}), m_freePool.end());

		// Sort less buffer.
		// sort order bigger buffer ---> samller buffer
		std::sort(m_freePool.begin(), m_freePool.end(), [](const BufferParameterMisc& A, const BufferParameterMisc& B)
		{
			return A.buffer->getBuffer()->getSize() > B.buffer->getBuffer()->getSize();
		});

		m_tickCount++;
	}

	std::shared_ptr<BufferParametersRing::BufferParametersManager::BufferParamHandle> BufferParametersRing::BufferParametersManager::getParameter(const char* name, size_t bufferSize)
	{
		bool bShouldReuse = m_freePool.size() > 0;
		if (bShouldReuse)
		{
			// Search most suitable buffer for us.
			auto searchMinSuitableBuffer = [&]()
			{
				size_t startPos = 0;
				size_t endPos = m_freePool.size();
				size_t workingIndex = (startPos + endPos) / 2;
				while (workingIndex > startPos && workingIndex < endPos)
				{
					if (m_freePool[workingIndex].buffer->getBuffer()->getSize() == bufferSize)
					{
						return workingIndex;
					}

					// sort order bigger buffer ---> samller buffer
					if (m_freePool[workingIndex].buffer->getBuffer()->getSize() > bufferSize)
					{
						startPos = workingIndex;
					}
					else
					{
						endPos = workingIndex;
					}
					workingIndex = (startPos + endPos) / 2;
				}
				return workingIndex;
			};

			size_t suitableId = searchMinSuitableBuffer();
			const auto reuseSize = m_freePool[suitableId].buffer->getBuffer()->getSize();
			// Suitable reuse size, so just use from pool.
			if (reuseSize >= bufferSize && reuseSize < bufferSize * 2)
			{
				BufferParameterMisc freeParam = m_freePool[suitableId];

				// Rename GPU buffer.
				freeParam.buffer->getBuffer()->setName(name);

				// Busy pool insert.
				size_t busyPos = m_busyPool.size();
				
				if (m_unusedBusyPos.empty())
				{
					m_busyPool.push_back(freeParam);
				}
				else
				{
					busyPos = m_unusedBusyPos.top();
					m_busyPool[busyPos] = freeParam;
					m_unusedBusyPos.pop();
				}

				// Erase free pool resource.
				m_freePool.erase(m_freePool.begin() + suitableId);
				return std::make_shared<BufferParamHandle>(*freeParam.buffer, this, busyPos);
			}
		}

		// No buffer can reuse, nee create new buffer.
		auto newBuffer = std::make_shared<BufferParameter>(
			name,
			bufferSize,
			m_bufferUsage,
			m_descriptorType,
			m_vmaFlags,
			m_memoryFlags);

		BufferParameterMisc newMisc{};
		newMisc.buffer = newBuffer;
		newMisc.freeTime = m_tickCount;
		size_t busyPos = m_busyPool.size();
		if (m_unusedBusyPos.empty())
		{
			m_busyPool.push_back(newMisc);
		}
		else
		{
			busyPos = m_unusedBusyPos.top();
			m_busyPool[m_unusedBusyPos.top()] = newMisc;
			m_unusedBusyPos.pop();
		}

		return std::make_shared<BufferParamHandle>(*newBuffer, this, busyPos);
	}

	// When release handle, will move to free pool.
	BufferParametersRing::BufferParametersManager::BufferParamHandle::~BufferParamHandle()
	{
		if (!m_manager)
		{
			return; // don't care if no manager exist.
		}

#pragma warning(push)
#pragma warning(disable:4297)
		// Safe check.
		ASSERT(m_manager, "Manager must release after all handle release already.");
		CHECK(m_manager->m_busyPool[m_busyPosition].buffer->getBuffer() == buffer.getBuffer());
#pragma warning(pop)

		// Upadte free count and move to unsort free pool
		BufferParameterMisc freeParam = m_manager->m_busyPool[m_busyPosition];
		freeParam.freeTime = m_manager->m_tickCount;
		m_manager->m_unsortFreePool.push_back(freeParam);

		// busy pool never shrink, but use a stack to store unused busy position.
		m_manager->m_unusedBusyPos.push(m_busyPosition);
	}

	BufferParametersRing::BufferParametersManager::BufferParamHandle::BufferParamHandle(
		BufferParameter& inBuffer,
		BufferParametersManager* manager,
		size_t busyPosition)
		: buffer(inBuffer), m_manager(manager), m_busyPosition(busyPosition)
	{

	}

	void BufferParametersRing::tick()
	{
		m_index++;
		if (m_index >= GBackBufferCount)
		{
			m_index = 0;
		}

		bufferParamsSizeCheck(m_managersMap.size());

		std::vector<size_t> shouldRemoveTerm{};
		for (auto& pair : m_managersMap)
		{
			const bool bMultiFrame = pair.second.at(0)->isMultiFrame();
			pair.second.at(bMultiFrame ? m_index : 0)->tick();

			bool canClear = pair.second[0]->canClear();
			if (bMultiFrame)
			{
				for (const auto& p : pair.second)
				{
					canClear &= p->canClear();
				}
			}
			
			if (canClear)
			{
				shouldRemoveTerm.push_back(pair.first);
			}
		}

		for (const auto& key : shouldRemoveTerm)
		{
			m_managersMap.erase(key);
		}
	}

	BufferParamRefPointer BufferParametersRing::getParameter(
		bool bMultiFrame,
		const char* name, 
		size_t bufferSize,
		VkBufferUsageFlags bufferUsage,
		VkDescriptorType type,
		EVMAUsageFlags vmaFlags,
		VkMemoryPropertyFlags memoryFlags)
	{
		BufferTypeHasher typeHaser
		{
			.bMultiFrame = bMultiFrame,
			.bufferUsage = bufferUsage,
			.type = type,
			.vmaFlags = vmaFlags,
			.memoryFlags = memoryFlags,
		};
		size_t typeKey = CRCHash(typeHaser);

		// Init if no this type.
		auto& arrayDatas = m_managersMap[typeKey];
		if (arrayDatas.empty())
		{
			arrayDatas.resize(bMultiFrame ? GBackBufferCount : 1);
			for (size_t i = 0; i < arrayDatas.size(); i++)
			{
				arrayDatas[i] = std::make_unique<BufferParametersManager>(
					bMultiFrame,
					bufferUsage,
					type,
					vmaFlags,
					memoryFlags);
			}
		}

		size_t pos = bMultiFrame ? m_index : 0;
		return arrayDatas.at(pos)->getParameter(name, bufferSize);
	}
}