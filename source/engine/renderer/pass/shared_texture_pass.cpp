#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    class SharedTextureComputePasses : public PassInterface
    {
    public:
        std::unique_ptr<ComputePipeResources> pipeBrdfLut;

        std::unique_ptr<ComputePipeResources> cloudBasicNoisePipe;
        std::unique_ptr<ComputePipeResources> cloudDetailNoisePipe;
    public:
        virtual void onInit() override
        {
            // Config code.
            {
                VkDescriptorSetLayout setLayoutBrdfLut;
                getContext()->descriptorFactoryBegin()
                    .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0) // outLut
                    .buildNoInfoPush(setLayoutBrdfLut);

                pipeBrdfLut = std::make_unique<ComputePipeResources>(
                    "shader/brdf_lut.glsl", 0, std::vector<VkDescriptorSetLayout>{ setLayoutBrdfLut });

                cloudBasicNoisePipe = std::make_unique<ComputePipeResources>("shader/cloud_basic_noise.glsl", 0, std::vector<VkDescriptorSetLayout>{ setLayoutBrdfLut });
                cloudDetailNoisePipe = std::make_unique<ComputePipeResources>("shader/cloud_detailed_noise.glsl", 0, std::vector<VkDescriptorSetLayout>{ setLayoutBrdfLut });
            }
        }

        virtual void release() override
        {
            pipeBrdfLut.reset();
            cloudBasicNoisePipe.reset();
            cloudDetailNoisePipe.reset();
        }
    };


    void SharedTextures::compute(VkCommandBuffer cmd)
    {
        auto* pass = getContext()->getPasses().get<SharedTextureComputePasses>();

        if (!zeroBuffer)
        {
            float data[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            zeroBuffer = std::make_unique<VulkanBuffer>(
                getContext()->getVMABuffer(),
                "Zero buffer",
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VulkanBuffer::getStageCopyForUploadBufferFlags(),
                sizeof(float) * 4,
                data
            );
        }

        if (!uniformGridVertices16x16)
        {
            constexpr int kDimension = 16;
            constexpr int kVerticesCount = 3 * 2 * kDimension * kDimension;
            constexpr float kQuadSize = 1.0f / float(kDimension);

            // Prepare grid vertices.
            std::vector<vec2> positions{};
            positions.reserve(kVerticesCount);
            for (int x = 0; x < kDimension; x++)
            {
                for (int y = 0; y < kDimension; y++)
                {
                    vec2 startPos = vec2(x, y) * kQuadSize;

                    positions.push_back(startPos + vec2(0.0f, 0.0f) * kQuadSize);
                    positions.push_back(startPos + vec2(0.0f, 1.0f) * kQuadSize);
                    positions.push_back(startPos + vec2(1.0f, 0.0f) * kQuadSize);

                    positions.push_back(startPos + vec2(1.0f, 0.0f) * kQuadSize);
                    positions.push_back(startPos + vec2(0.0f, 1.0f) * kQuadSize);
                    positions.push_back(startPos + vec2(1.0f, 1.0f) * kQuadSize);
                }
            }

            uniformGridVertices16x16 = std::make_unique<VulkanBuffer>(
                getContext()->getVMABuffer(),
                "uniform grid 16x16 vertices",
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VulkanBuffer::getStageCopyForUploadBufferFlags(),
                sizeof(positions[0]) * positions.size(),
                positions.data()
            );
        }

        if (!imageFallback_RGBA16f)
        {
            auto desc = buildImageCreateInfoDefault(
                1u, 1u, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

            imageFallback_RGBA16f = std::make_unique<VulkanImage>(getContext()->getVMAFrequencyImage(), "imageFallback_RGBA16f", desc);
        }
        imageFallback_RGBA16f->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());

        if (!brdfLut)
        {
            auto brdfDesc = buildImageCreateInfoDefault(
                256u, 256u, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

            brdfLut = std::make_unique<VulkanImage>(getContext()->getVMAFrequencyImage(), "BRDF Lut", brdfDesc);
        }
        brdfLut->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            pass->pipeBrdfLut->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(*brdfLut)
                .push(pass->pipeBrdfLut.get());
            vkCmdDispatch(cmd, getGroupCount(brdfLut->getExtent().width, 8), getGroupCount(brdfLut->getExtent().height, 8), 1);
        }
        brdfLut->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());


        if (!cloudBasicNoise)
        {
            auto dim = 128u;
            auto desc = buildImageCreateInfoDefault(dim, dim, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            desc.extent.depth = dim;
            desc.imageType = VK_IMAGE_TYPE_3D;
            cloudBasicNoise = std::make_unique<VulkanImage>(getContext()->getVMAFrequencyImage(), "CloudBasicNoise", desc);
        }
        cloudBasicNoise->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            pass->cloudBasicNoisePipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(*cloudBasicNoise, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
                .push(pass->cloudBasicNoisePipe.get());
            vkCmdDispatch(cmd, getGroupCount(cloudBasicNoise->getExtent().width, 8), getGroupCount(cloudBasicNoise->getExtent().height, 8), cloudBasicNoise->getExtent().depth);
        }
        cloudBasicNoise->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());

        if (!cloudDetailNoise)
        {
            auto dim = 64u;
            auto desc = buildImageCreateInfoDefault(dim, dim, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            desc.extent.depth = dim;
            desc.imageType = VK_IMAGE_TYPE_3D;
            cloudDetailNoise = std::make_unique<VulkanImage>(getContext()->getVMAFrequencyImage(), "cloudDetailNoise", desc);
        }
        cloudDetailNoise->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, buildBasicImageSubresource());
        {
            pass->cloudDetailNoisePipe->bind(cmd);
            PushSetBuilder(cmd)
                .addUAV(*cloudDetailNoise, buildBasicImageSubresource(), VK_IMAGE_VIEW_TYPE_3D)
                .push(pass->cloudDetailNoisePipe.get());
            vkCmdDispatch(cmd, getGroupCount(cloudDetailNoise->getExtent().width, 8), getGroupCount(cloudDetailNoise->getExtent().height, 8), cloudDetailNoise->getExtent().depth);
        }
        cloudDetailNoise->transitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buildBasicImageSubresource());
    }
}