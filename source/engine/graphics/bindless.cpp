#include "bindless.h"
#include "context.h"
#include "log.h"
#include "resource.h"

namespace engine
{
	static AutoCVarInt32 cVarRHIBindlessTextureMaxCount(
		"r.RHI.BindlessTextureMaxCount",
		"Bindless texture set max count.",
		"RHI",
		20000,
		CVarFlags::ReadOnly
	);

	static AutoCVarInt32 cVarRHIBindlessSamplerMaxCount(
		"r.RHI.BindlessSamplerMaxCount",
		"Bindless sampler set max count.",
		"RHI",
		1000,
		CVarFlags::ReadOnly
	);

	static AutoCVarInt32 cVarRHIBindlessSSBOMaxCount(
		"r.RHI.BindlessSSBOMaxCount",
		"Bindless ssbo set max count.",
		"RHI",
		10000,
		CVarFlags::ReadOnly
	);

	void BindlessBase::initTemplate(const char* name, VkDescriptorType type, uint32_t maxCountConfig, uint32_t maxDeviceLimit)
	{
		m_name = name;
		m_maxDeviceLimitCount = maxDeviceLimit;
		m_maxCountConfig = maxCountConfig;

		// Create bindless binding here.
		VkDescriptorSetLayoutBinding binding{};
		binding.descriptorType = type;
		binding.stageFlags = kCommonShaderStage;
		binding.binding = 0;

		const uint32_t bindlessMaxCountConfig = maxCountConfig;
		LOG_RHI_TRACE("Config max bindless count is {0}, device limit is {1}.", bindlessMaxCountConfig, maxDeviceLimit);

		// Set max descriptor sampler count to a big number.
		CHECK(bindlessMaxCountConfig < maxDeviceLimit);

		binding.descriptorCount = bindlessMaxCountConfig;

		// One binding.
		VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{};
		setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		setLayoutCreateInfo.bindingCount = 1;
		setLayoutCreateInfo.pBindings = &binding;
		setLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;

		const VkDescriptorBindingFlagsEXT flags =
			VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT |
			VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT |
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT |
			VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT;

		VkDescriptorSetLayoutBindingFlagsCreateInfoEXT bindingFlags{};
		bindingFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
		bindingFlags.bindingCount = 1;
		bindingFlags.pBindingFlags = &flags;
		setLayoutCreateInfo.pNext = &bindingFlags;

		RHICheck(vkCreateDescriptorSetLayout(getDevice(), &setLayoutCreateInfo, nullptr, &m_bindlessDescriptorHeap.setLayout));

		VkDescriptorPoolSize  poolSize{};
		poolSize.type = type;
		poolSize.descriptorCount = bindlessMaxCountConfig;

		VkDescriptorPoolCreateInfo poolCreateInfo{};
		poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolCreateInfo.poolSizeCount = 1;
		poolCreateInfo.pPoolSizes = &poolSize;
		poolCreateInfo.maxSets = 1;
		poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;

		RHICheck(vkCreateDescriptorPool(getDevice(), &poolCreateInfo, nullptr, &m_bindlessDescriptorHeap.descriptorPool));

		VkDescriptorSetAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocateInfo.descriptorPool = m_bindlessDescriptorHeap.descriptorPool;
		allocateInfo.pSetLayouts = &m_bindlessDescriptorHeap.setLayout;
		allocateInfo.descriptorSetCount = 1;

		VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{};
		variableInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
		variableInfo.descriptorSetCount = 1;
		allocateInfo.pNext = &variableInfo;

		const uint32_t NumDescriptors = bindlessMaxCountConfig;
		variableInfo.pDescriptorCounts = &NumDescriptors;
		RHICheck(vkAllocateDescriptorSets(getDevice(), &allocateInfo, &m_bindlessDescriptorHeap.descriptorSetUpdateAfterBind));

		getContext()->setResourceName(VK_OBJECT_TYPE_DESCRIPTOR_SET, (uint64_t)m_bindlessDescriptorHeap.descriptorSetUpdateAfterBind, m_name);
		getContext()->setResourceName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, (uint64_t)m_bindlessDescriptorHeap.descriptorPool, m_name);
		getContext()->setResourceName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)m_bindlessDescriptorHeap.setLayout, m_name);
	}

	uint32_t BindlessBase::getCountAndAndOne()
	{
		std::lock_guard lock(m_bindlessElementCountLock);

		uint32_t index = 0;

		const uint32_t maxRHIBindlessCount = m_maxCountConfig;
		const auto maxFreeIndexSize = maxRHIBindlessCount / 4;
		if (m_freeIndex.size() < maxFreeIndexSize)
		{
			// No free index, increment.
			index = m_bindlessElementCount;
			m_bindlessElementCount++;

			if (m_bindlessElementCount >= maxRHIBindlessCount)
			{
				LOG_WARN("Too much item use in this set, current bindless count already reach {0}, and the config max is {1}, the device limit is {2}. ", 
					m_bindlessElementCount, maxRHIBindlessCount, m_maxDeviceLimitCount);

				LOG_WARN("We reset bindless set count now, maybe cause some render error after this.");
				m_bindlessElementCount = 0;
			}
		}
		else
		{
			// Exist free index, pop first one.
			index = *m_freeIndex.begin();
			m_freeIndex.erase(m_freeIndex.begin());
		}
		return index;
	}

	void BindlessBase::freeBindless(uint32_t index)
	{
		std::lock_guard lock(m_bindlessElementCountLock);

		CHECK(!m_freeIndex.contains(index));
		m_freeIndex.insert(index);
	}

	VkDescriptorSetLayout BindlessBase::getSetLayout() const
	{
		return m_bindlessDescriptorHeap.setLayout;
	}

	void BindlessBase::release()
	{
		vkDestroyDescriptorSetLayout(getContext()->getDevice(), m_bindlessDescriptorHeap.setLayout, nullptr);
		vkDestroyDescriptorPool(getContext()->getDevice(), m_bindlessDescriptorHeap.descriptorPool, nullptr);
	}

	VkDescriptorSet BindlessBase::getSet() const
	{
		return m_bindlessDescriptorHeap.descriptorSetUpdateAfterBind;
	}

	void BindlessSampler::init(const char* name)
	{
		initTemplate(name, VK_DESCRIPTOR_TYPE_SAMPLER, cVarRHIBindlessSamplerMaxCount.get(),
			getContext()->getPhysicalDeviceDescriptorIndexingProperties().maxDescriptorSetUpdateAfterBindSamplers);
	}

	uint32_t BindlessSampler::updateSamplerToBindlessDescriptorSet(VkSampler in)
	{
		VkDescriptorImageInfo imageInfo{};
		imageInfo.sampler = in;
		imageInfo.imageView = VK_NULL_HANDLE;
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VkWriteDescriptorSet  write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = getSet();
		write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
		write.dstBinding = 0;
		write.pImageInfo = &imageInfo;
		write.descriptorCount = 1;
		write.dstArrayElement = getCountAndAndOne();

		vkUpdateDescriptorSets(getContext()->getDevice(), 1, &write, 0, nullptr);

		return write.dstArrayElement;
	}

	void BindlessSampler::freeBindlessImpl(uint32_t index, VkSampler fallback)
	{
		if (fallback != VK_NULL_HANDLE)
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.sampler = fallback;
			imageInfo.imageView = VK_NULL_HANDLE;
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			VkWriteDescriptorSet  write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = getSet();
			write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
			write.dstBinding = 0;
			write.pImageInfo = &imageInfo;
			write.descriptorCount = 1;
			write.dstArrayElement = index;

			vkUpdateDescriptorSets(getContext()->getDevice(), 1, &write, 0, nullptr);
		}

		BindlessBase::freeBindless(index);
	}

	void BindlessTexture::init(const char* name)
	{
		initTemplate(name, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, cVarRHIBindlessTextureMaxCount.get(),
			getContext()->getPhysicalDeviceDescriptorIndexingProperties().maxDescriptorSetUpdateAfterBindSampledImages);
	}

	uint32_t BindlessTexture::updateTextureToBindlessDescriptorSet(VkImageView view, VkImageLayout layout)
	{
		VkDescriptorImageInfo imageInfo{};
		imageInfo.sampler = VK_NULL_HANDLE;

		imageInfo.imageView = view;
		imageInfo.imageLayout = layout;

		VkWriteDescriptorSet  write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = getSet();
		write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		write.dstBinding = 0;
		write.pImageInfo = &imageInfo;
		write.descriptorCount = 1;
		write.dstArrayElement = getCountAndAndOne();

		vkUpdateDescriptorSets(getDevice(), 1, &write, 0, nullptr);

		return write.dstArrayElement;
	}

	void BindlessTexture::freeBindlessImpl(uint32_t index, VulkanImage* fallback)
	{
		// If exist fallback input, we change bindless index to this fallback, 
		// so validation will happy to immediately delete current bindless asset. 
		if (fallback)
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.sampler = VK_NULL_HANDLE;
			imageInfo.imageView = fallback->getOrCreateView().view;
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkWriteDescriptorSet  write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = getSet();
			write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			write.dstBinding = 0;
			write.pImageInfo = &imageInfo;
			write.descriptorCount = 1;
			write.dstArrayElement = index;

			vkUpdateDescriptorSets(getDevice(), 1, &write, 0, nullptr);
		}

		BindlessBase::freeBindless(index);
	}

	void BindlessStorageBuffer::init(const char* name)
	{
		initTemplate(name, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cVarRHIBindlessSSBOMaxCount.get(),
			getContext()->getPhysicalDeviceDescriptorIndexingProperties().maxDescriptorSetUpdateAfterBindStorageBuffers);
	}

	uint32_t BindlessStorageBuffer::updateBufferToBindlessDescriptorSet(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = buffer;
		bufferInfo.offset = offset;
		bufferInfo.range = range;

		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = getSet();
		write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write.dstBinding = 0;
		write.pBufferInfo = &bufferInfo;
		write.descriptorCount = 1;
		write.dstArrayElement = getCountAndAndOne();

		vkUpdateDescriptorSets(getDevice(), 1, &write, 0, nullptr);

		return write.dstArrayElement;
	}

	void BindlessStorageBuffer::freeBindlessImpl(uint32_t index, VulkanBuffer* fallback)
	{
		// Fallback same logic with texture.
		if (fallback)
		{
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = fallback->getVkBuffer();
			bufferInfo.offset = 0;
			bufferInfo.range = fallback->getSize();

			VkWriteDescriptorSet write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = getSet();
			write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			write.dstBinding = 0;
			write.pBufferInfo = &bufferInfo;
			write.descriptorCount = 1;
			write.dstArrayElement = index;

			vkUpdateDescriptorSets(getDevice(), 1, &write, 0, nullptr);
		}

		BindlessBase::freeBindless(index);
	}
}