#include "../renderer_interface.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"

namespace engine
{
    class SharedTextureComputePasses : public PassInterface
    {
    public:
        VkDescriptorSetLayout setLayoutBrdfLut = VK_NULL_HANDLE;
        std::unique_ptr<ComputePipeResources> pipeBrdfLut;

    public:
        virtual void onInit() override
        {
            // Config code.
            getContext()->descriptorFactoryBegin()
                .bindNoInfo(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0) // outLut
                .buildNoInfoPush(setLayoutBrdfLut);

            pipeBrdfLut = std::make_unique<ComputePipeResources>("shader/brdf_lut.comp.spv", 0, std::vector<VkDescriptorSetLayout>{ setLayoutBrdfLut });
        }

        virtual void release() override
        {
            pipeBrdfLut.reset();
        }
    };

    void SharedTextures::compute(VkCommandBuffer cmd)
    {
        auto* pass = getContext()->getPasses().get<SharedTextureComputePasses>();

        if(!brdfLut)
        {
            auto brdfDesc = buildImageCreateInfoDefault(256u, 256u, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            brdfLut = std::make_unique<VulkanImage>(getContext(), "BRDF Lut", brdfDesc);
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
    }
}