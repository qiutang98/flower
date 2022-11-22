#include "Pch.h"
#include "Bindless.h"
#include "RHI.h"

namespace Flower
{
	namespace Bindless
	{
		BindlessTexture* const Texture = new BindlessTexture();
		BindlessSampler* const Sampler = new BindlessSampler();
	}

	void BindlessBase::initTemplate(VkDescriptorType type)
	{
		// Create bindless binding here.
		VkDescriptorSetLayoutBinding binding{};
		binding.descriptorType = type;
		binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
		binding.binding = 0;

		// Set max descriptor sampler count to a big number.
		CHECK(MAX_BINDLESS_COUNT <
			RHI::get()->getPhysicalDeviceDescriptorIndexingProperties().maxDescriptorSetUpdateAfterBindSamplers)
		binding.descriptorCount = MAX_BINDLESS_COUNT;

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

		RHICheck(vkCreateDescriptorSetLayout(
			RHI::Device,
			&setLayoutCreateInfo,
			nullptr,
			&m_bindlessDescriptorHeap.setLayout));

		VkDescriptorPoolSize  poolSize{};
		poolSize.type = type;
		poolSize.descriptorCount = MAX_BINDLESS_COUNT;

		VkDescriptorPoolCreateInfo poolCreateInfo{};
		poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolCreateInfo.poolSizeCount = 1;
		poolCreateInfo.pPoolSizes = &poolSize;
		poolCreateInfo.maxSets = 1;
		poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;

		RHICheck(vkCreateDescriptorPool(RHI::Device, &poolCreateInfo, nullptr, &m_bindlessDescriptorHeap.descriptorPool));

		VkDescriptorSetAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocateInfo.descriptorPool = m_bindlessDescriptorHeap.descriptorPool;
		allocateInfo.pSetLayouts = &m_bindlessDescriptorHeap.setLayout;
		allocateInfo.descriptorSetCount = 1;

		VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{};
		variableInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
		variableInfo.descriptorSetCount = 1;
		allocateInfo.pNext = &variableInfo;

		const uint32_t NumDescriptors = MAX_BINDLESS_COUNT;
		variableInfo.pDescriptorCounts = &NumDescriptors;
		RHICheck(vkAllocateDescriptorSets(RHI::Device, &allocateInfo, &m_bindlessDescriptorHeap.descriptorSetUpdateAfterBind));
	}

	uint32_t BindlessBase::getCountAndAndOne()
	{
		std::lock_guard lock(m_bindlessElementCountLock);

		uint32_t index = 0;
		static const auto maxFreeIndexSize = MAX_BINDLESS_COUNT / 4;
		if(m_freeIndex.size() < maxFreeIndexSize)
		{
			// No free index, increment.
			index = m_bindlessElementCount;
			m_bindlessElementCount++;
			CHECK(m_bindlessElementCount < MAX_BINDLESS_COUNT && "Too much item loaded in gpu!");
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

	VkDescriptorSetLayout BindlessBase::getSetLayout()
	{
		return m_bindlessDescriptorHeap.setLayout;
	}

	void BindlessBase::release()
	{
		vkDestroyDescriptorSetLayout(RHI::Device, m_bindlessDescriptorHeap.setLayout, nullptr);
		vkDestroyDescriptorPool(RHI::Device, m_bindlessDescriptorHeap.descriptorPool, nullptr);
	}

	VkDescriptorSet BindlessBase::getSet()
	{
		return m_bindlessDescriptorHeap.descriptorSetUpdateAfterBind;
	}

	void BindlessSampler::init()
	{
		initTemplate(VK_DESCRIPTOR_TYPE_SAMPLER);
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

		vkUpdateDescriptorSets(RHI::Device, 1, &write, 0, nullptr);

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

			vkUpdateDescriptorSets(RHI::Device, 1, &write, 0, nullptr);
		}

		BindlessBase::freeBindless(index);
	}

	void BindlessTexture::init()
	{
		initTemplate(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
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

		vkUpdateDescriptorSets(RHI::Device, 1, &write, 0, nullptr);

		return write.dstArrayElement;
	}

	void BindlessTexture::freeBindlessImpl(uint32_t index, std::shared_ptr<VulkanImage> fallback)
	{
		if (fallback)
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.sampler = VK_NULL_HANDLE;

			imageInfo.imageView = fallback->getView(buildBasicImageSubresource());
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkWriteDescriptorSet  write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = getSet();
			write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			write.dstBinding = 0;
			write.pImageInfo = &imageInfo;
			write.descriptorCount = 1;
			write.dstArrayElement = index;

			vkUpdateDescriptorSets(RHI::Device, 1, &write, 0, nullptr);
		}
		
		BindlessBase::freeBindless(index);
		
	}

	void BindlessStorageBuffer::init()
	{
		initTemplate(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	}

	uint32_t BindlessStorageBuffer::updateBufferToBindlessDescriptorSet(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = buffer;
		bufferInfo.offset = offset;
		bufferInfo.range  = range;

		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = getSet();
		write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write.dstBinding = 0;
		write.pBufferInfo = &bufferInfo;
		write.descriptorCount = 1;
		write.dstArrayElement = getCountAndAndOne();

		vkUpdateDescriptorSets(RHI::Device, 1, &write, 0, nullptr);

		return write.dstArrayElement;
	}
	void BindlessStorageBuffer::freeBindlessImpl(uint32_t index, std::shared_ptr<VulkanBuffer> fallback)
	{
		if (fallback != VK_NULL_HANDLE)
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

			vkUpdateDescriptorSets(RHI::Device, 1, &write, 0, nullptr);
		}

		BindlessBase::freeBindless(index);
	}
}