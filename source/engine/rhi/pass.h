#pragma once

#include "rhi_misc.h"
#include "resource.h"
#include "render_texture_pool.h"
#include "ssbo_buffers.h"

namespace engine
{
	// Pass simple interface.
	class PassInterface : NonCopyable
	{
	private:
		friend class PassCollector;

		void init(class VulkanContext* context)
		{
			m_context = context;
			onInit();
		}

	protected:
		class VulkanContext* m_context;

		virtual void onInit() {}
		virtual void release() {}
	};

	// GPU pass collector.
	class PassCollector : NonCopyable
	{
	protected:
		// Cache context.
		class VulkanContext* m_context;
		
		// Pass collect map.
		std::unordered_map<const char*, std::unique_ptr<PassInterface>> m_passMap;

	public:
		explicit PassCollector(class VulkanContext* context);
		virtual ~PassCollector();

		// Get by type.
		template<typename PassType>
		PassType* get()
		{
			static_assert(std::is_base_of_v<PassInterface, PassType>);

			const char* passName = typeid(PassType).name();
			if (!m_passMap[passName])
			{
				// Create and init if no exist.
				m_passMap[passName] = std::make_unique<PassType>();
				m_passMap[passName]->init(m_context);
			}

			return dynamic_cast<PassType*>(m_passMap[passName].get());
		}

		// Update all pass.
		void updateAllPasses();
	};

	class PipeResource : NonCopyable
	{
	public:
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

		virtual ~PipeResource();

		virtual VkPipelineBindPoint getBindPoint() const = 0;
	};

	class ComputePipeResources : public PipeResource
	{
	public:
		ComputePipeResources(
			const std::string& shaderPath,
			uint32_t pushConstSize,
			const std::vector<VkDescriptorSetLayout>& inSetLayout);

		virtual VkPipelineBindPoint getBindPoint() const { return VK_PIPELINE_BIND_POINT_COMPUTE; }

		template<typename T>
		void pushConst(VkCommandBuffer cmd, const T* pushConst)
		{
			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(T), pushConst);
		}

		template<typename T>
		void bindAndPushConst(VkCommandBuffer cmd, const T* pushConstv)
		{
			bind(cmd);
			pushConst<T>(cmd, pushConstv);
		}

		void bind(VkCommandBuffer cmd)
		{
			vkCmdBindPipeline(cmd, getBindPoint(), pipeline);
		}

		void bindSet(VkCommandBuffer cmd, const std::vector<VkDescriptorSet>& passSets, uint32_t offset = 0)
		{
			vkCmdBindDescriptorSets(cmd, getBindPoint(), pipelineLayout, offset, (uint32_t)passSets.size(), passSets.data(), 0, nullptr);
		}


	};

    class GraphicPipeResources : public PipeResource
    {
    public:
		GraphicPipeResources(
			const std::string& vertShaderPath,
			const std::string& fragShaderPath,
			const std::vector<VkDescriptorSetLayout>& inSetLayout,
			uint32_t pushConstSize,
			std::vector<VkFormat>&& inColorAttachmentFormats,
			std::vector<VkPipelineColorBlendAttachmentState>&& inBlendState,
			VkFormat depthFormat,
			VkCullModeFlags cullMode = VK_CULL_MODE_FRONT_BIT,
			VkCompareOp zTestComp = VK_COMPARE_OP_GREATER,
			bool bEnableDepthClamp = false,
			bool bEnableDepthBias = false,
			const std::vector<VkVertexInputAttributeDescription>& inputAttributes = {},
			uint32_t vertexStrip = 0,
			VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL,
			bool bZWrite = true);

		virtual VkPipelineBindPoint getBindPoint() const { return VK_PIPELINE_BIND_POINT_GRAPHICS; }

        void bind(VkCommandBuffer cmd)
        {
            vkCmdBindPipeline(cmd, getBindPoint(), pipeline);
        }

        void bindSet(VkCommandBuffer cmd, std::vector<VkDescriptorSet>&& inMeshPassSets, uint32_t offset = 0)
        {
            std::vector<VkDescriptorSet> meshPassSets = inMeshPassSets;
            vkCmdBindDescriptorSets(cmd, getBindPoint(), pipelineLayout, offset, (uint32_t)meshPassSets.size(), meshPassSets.data(), 0, nullptr);
        }

		template<typename T>
		void pushConst(VkCommandBuffer cmd, const T* pushConst)
		{
			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(T), pushConst);
		}

		template<typename T>
		void bindAndPushConst(VkCommandBuffer cmd, const T* pushConstv)
		{
			bind(cmd);
			pushConst<T>(cmd, pushConstv);
		}
    };

	class PushSetBuilder
	{
	public:
		PushSetBuilder(VkCommandBuffer cmd) : m_cmd(cmd) { }

		PushSetBuilder& addBuffer(BufferParameterHandle buffer)
		{
			CacheBindingBuilder builder;
			builder.type = CacheBindingBuilder::EType::buffer;
			builder.buffer = buffer->getBuffer()->getVkBuffer();
			builder.bufferInfo = buffer->getBufferInfo();
			builder.descriptorType = buffer->getType();
			m_cacheBindingBuilder.push_back(builder);
			return *this;
		}

		PushSetBuilder& addBuffer(const VulkanBuffer& bufferIn, VkDescriptorType descriptorType)
		{
			CacheBindingBuilder builder;
			builder.type = CacheBindingBuilder::EType::buffer;
			builder.buffer = bufferIn.getVkBuffer();
			builder.bufferInfo = bufferIn.getDefaultInfo();
			builder.descriptorType = descriptorType;
			m_cacheBindingBuilder.push_back(builder);
			return *this;
		}

		PushSetBuilder& addAS(TLASBuilder* as)
		{
			CacheBindingBuilder builder;
			builder.type = CacheBindingBuilder::EType::as;
			builder.asBuilder = as;
			m_cacheBindingBuilder.push_back(builder);
			return *this;
		}

		PushSetBuilder& addSRV(VulkanImage& image, const VkImageSubresourceRange& range = buildBasicImageSubresource(), VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D)
		{
			CacheBindingBuilder builder;
			builder.type = CacheBindingBuilder::EType::srv;
			builder.image = &image;
			builder.imageRange = range;
			builder.viewType = viewType;
			m_cacheBindingBuilder.push_back(builder);
			return *this;
		}

		PushSetBuilder& addUAV(VulkanImage& image, const VkImageSubresourceRange& range = buildBasicImageSubresource(), VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D)
		{
			CacheBindingBuilder builder;
			builder.type = CacheBindingBuilder::EType::uav;
			builder.image = &image;
			builder.imageRange = range;
			builder.viewType = viewType;
			m_cacheBindingBuilder.push_back(builder);

			return *this;
		}

		PushSetBuilder& addSRV(PoolImageSharedRef image, const VkImageSubresourceRange& range = buildBasicImageSubresource(), VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D)
		{
			return addSRV(image->getImage(), range, viewType);
		}

		PushSetBuilder& addUAV(PoolImageSharedRef image, const VkImageSubresourceRange& range = buildBasicImageSubresource(), VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D)
		{
			return addUAV(image->getImage(), range, viewType);
		}

		void push(PipeResource* pipe);

	private:
		struct CacheBindingBuilder
		{
			enum class EType
			{
				buffer,
				srv,
				uav,
				as,
			} type;

			VulkanImage* image;
			VkImageSubresourceRange imageRange; 
			VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;

			VkBuffer buffer;
			VkDescriptorType descriptorType;
			VkDescriptorBufferInfo bufferInfo;
			TLASBuilder* asBuilder;
		};

		VkCommandBuffer m_cmd;

		std::vector<CacheBindingBuilder> m_cacheBindingBuilder;
	};
}