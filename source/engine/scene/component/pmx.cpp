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

			auto vbMemSize = uint32_t(sizeof(Vertex) * pmxModel->GetVertexCount());
			m_vertexBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath,
				bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				bufferFlagVMA,
				vbMemSize
			);

			m_stageBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VulkanBuffer::getStageCopyForUploadBufferFlags(),
				vbMemSize
			);

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
			}
		}

		// Prepare textures.
		pmxAsset->tryLoadAllTextures(*pmxModel);

		pmxModel->InitializeAnimation();

		m_bInit = true;
		m_mmdModel = std::move(pmxModel);
		m_pmxAsset = pmxAsset;
	}

	PMXMeshProxy::~PMXMeshProxy()
	{
		getContext()->waitDeviceIdle();
	}
}