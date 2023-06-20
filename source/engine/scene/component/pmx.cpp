#include "pmx.h"
#include <rhi/rhi.h>

#include <saba/Base/Path.h>
#include <saba/Base/File.h>
#include <saba/Base/UnicodeUtil.h>
#include <saba/Base/Time.h>

namespace engine
{
	PMXComponent::~PMXComponent()
	{

	}

	void PMXComponent::tick(const RuntimeModuleTickData& tickData)
	{
		if (!m_proxy && (!m_pmxUUID.empty()))
		{
			m_proxy = std::make_unique<PMXMeshProxy>(m_pmxUUID);
		}



	}

	bool PMXComponent::setPMX(const UUID& in)
	{
		if (m_pmxUUID != in)
		{
			m_pmxUUID = in;
			m_proxy = nullptr;

			return true;
		}

		return false;
	}

	PMXMeshProxy::PMXMeshProxy(const UUID& uuid)
	{
		auto pmxAsset = std::dynamic_pointer_cast<AssetPMX>(getAssetSystem()->getAsset(uuid));
		auto path = pmxAsset->getPMXFilePath();
		std::string pmxPath = path.string();

		auto pmxModel = std::make_unique<saba::PMXModel>();
		{
			auto ext = saba::PathUtil::GetExt(pmxPath);
			if (ext != "pmx")
			{
				LOG_ERROR("Must select one pmx file.");
				return;
			}

			if (!pmxModel->Load(pmxPath, "image/mmd"))
			{
				LOG_ERROR("Failed to load pmx file {0}.", pmxPath);
				return;
			}
		}

		// Prepare vertex buffers.
		{
			auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			VmaAllocationCreateFlags bufferFlagVMA = {};
			if (getContext()->getGraphicsCardState().bSupportRaytrace)
			{
				// Raytracing accelerate struct, random shader fetch by address.
				bufferFlagBasic |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
				bufferFlagVMA = {};
			}

			auto vbMemSizePosition = uint32_t(sizeof(glm::vec3) * pmxModel->GetVertexCount());
			auto vbMemSizeNormal = uint32_t(sizeof(glm::vec3) * pmxModel->GetVertexCount());
			auto vbMemSizeUv = uint32_t(sizeof(glm::vec2) * pmxModel->GetVertexCount());
			auto vbMemSizePositionLast = uint32_t(sizeof(glm::vec3) * pmxModel->GetVertexCount());


			m_positionBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizePosition);
			m_normalBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizeNormal);
			m_uvBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizeUv);
			m_positionPrevFrameBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizePositionLast);

			m_stageBufferPosition = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizePosition);
			m_stageBufferNormal = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizeNormal);
			m_stageBufferUv = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizeUv);
			m_stageBufferPositionPrevFrame = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizePositionLast);

			m_normalBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_normalBuffer->getVkBuffer(), 0, vbMemSizeNormal);
			m_uvBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_uvBuffer->getVkBuffer(), 0, vbMemSizeUv);
			m_positionBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_positionBuffer->getVkBuffer(), 0, vbMemSizePosition);
			m_positionPrevBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_positionPrevFrameBuffer->getVkBuffer(), 0, vbMemSizePositionLast);

			// Index Buffer
			m_indexType = VK_INDEX_TYPE_UINT32;
			{
				if (pmxModel->GetIndexElementSize() == 1)
				{
					LOG_ERROR("Vulkan is not supported uint8_t index."); // Some machine can use uint8 extension.
					return;
				}
				else if (pmxModel->GetIndexElementSize() == 2)
				{
					m_indexType = VK_INDEX_TYPE_UINT16;
				}
				else if (pmxModel->GetIndexElementSize() == 4)
				{
					m_indexType = VK_INDEX_TYPE_UINT32;
				}
				else
				{
					UN_IMPLEMENT_WARN();
					return;
				}

				// Create buffer
				auto ibMemSize = uint32_t(pmxModel->GetIndexElementSize() * pmxModel->GetIndexCount());
				m_indexBuffer = std::make_unique<VulkanBuffer>(
					getContext(),
					pmxPath,
					bufferFlagBasic | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
					bufferFlagVMA,
					ibMemSize
				);

				// Copy index to GPU.
				auto stageBuffer = std::make_unique<VulkanBuffer>(
					getContext(),
					"CopyBuffer",
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VulkanBuffer::getStageCopyForUploadBufferFlags(),
					ibMemSize,
					const_cast<void*>(pmxModel->GetIndices())
				);

				m_indexBuffer->stageCopyFrom(stageBuffer->getVkBuffer(), ibMemSize, 0, 0);

				m_indicesBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_indexBuffer->getVkBuffer(), 0, ibMemSize);
			}
		}

		// Prepare textures.
		pmxAsset->tryLoadAllTextures(*pmxModel);

		pmxModel->InitializeAnimation();

		m_bInit = true;
		m_mmdModel = std::move(pmxModel);
		m_pmxAsset = pmxAsset;

		getContext()->executeImmediatelyMajorGraphics([this](VkCommandBuffer cmd) {
			updateVertex(cmd);
		});
		
	}

	PMXMeshProxy::~PMXMeshProxy()
	{
		getContext()->waitDeviceIdle();

		if (m_indicesBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_indicesBindless);
			m_indicesBindless = ~0;
		}
		if (m_normalBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_normalBindless);
			m_normalBindless = ~0;
		}
		if (m_uvBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_uvBindless);
			m_uvBindless = ~0;
		}
		if (m_positionBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_positionBindless);
			m_positionBindless = ~0;
		}
		if (m_positionPrevBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_positionPrevBindless);
			m_positionPrevBindless = ~0;
		}
	}
}