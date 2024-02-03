#include "context.h"
#include "log.h"

#include "../engine.h"
#include <nameof/nameof.hpp>
#include "../asset/asset_common.h"
#include "../asset/asset_manager.h"
#include "../asset/asset_texture.h"
#include "../asset/asset_staticmesh.h"
#include "../renderer/render_functions.h"

namespace engine
{
    static AutoCVarCmd cVarUpdatePasses("cmd.updatePasses", "Update passes shader and pipeline info.");
    static AutoCVarCmd cVarUpdateCloudPasses("cmd.updatePasses.cloud", "Update passes cloud shader and pipeline info.");

    static AutoCVarBool cVarRHIDebugMarkerEnable(
        "r.RHI.DebugMarkerEnable",
        "Enable debug marker or not.",
        "RHI",
        true,
        CVarFlags::ReadOnly
    );

    static AutoCVarInt32 cVarRHIAsyncUploaderStaticSize(
        "r.RHI.AsyncUploaderStaticBufferMaxSize",
        "Async uploader static buffer max size (MB).",
        "RHI",
        16,
        CVarFlags::ReadOnly
    );

    static AutoCVarInt32 cVarRHIAsyncUploaderDynamicSize(
        "r.RHI.AsyncUploaderDynamicBufferSize",
        "Async uploader dynamic buffer min size (MB).",
        "RHI",
        8,
        CVarFlags::ReadOnly
    );

    UUID getBuiltinTexturesUUID(EBuiltinTextures value)
    {
        std::string name = std::format("{2}/Textures/{0}{1}", nameof::nameof_enum(value),
            AssetTexture::getCDO()->getSuffix(), AssetSaveInfo::kBuiltinFileStartChar);
        return name;
    }

    UUID getBuiltinStaticMeshUUID(EBuiltinStaticMeshes value)
    {
        std::string name = std::format("{2}/StaticMesh/{0}{1}", nameof::nameof_enum(value),
            AssetStaticMesh::getCDO()->getSuffix(), AssetSaveInfo::kBuiltinFileStartChar);
        return name;
    }

    VulkanContext* engine::getContext()
    {
        return Engine::get()->getRuntimeModule<VulkanContext>();
    }

    VkDevice engine::getDevice()
    {
        return Engine::get()->getRuntimeModule<VulkanContext>()->getDevice();
    }

    void VulkanContext::registerCheck(Engine* engine)
    {
    }

    bool VulkanContext::init()
    {
        if (m_engine->isWindowApplication())
        {
            m_window = m_engine->getGLFWWindows()->getGLFWWindowHandle();
        }

        // Create vulkan instance.
        initInstance();

        // Select and query gpu infos.
        selectGpuAndQueryGpuInfos();

        // Create surface if application is windows.
        if (m_engine->isWindowApplication())
        {
            RHICheck(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
        }

        // Create logic device for vulkan.
        initDeviceAndQueue();

        initVMA();
        initCommandPools();

        // Init bindless resources.
        m_bindlessSampler.init("BindlessSampler");
        m_bindlessTexture.init("BindlessTexture");
        m_bindlessStorageBuffer.init("BindlessSSBO");

        m_samplerCache.init();

        // Init async uploader.
        m_uploader = std::make_unique<AsyncUploaderManager>(
            cVarRHIAsyncUploaderStaticSize.get(), 
            cVarRHIAsyncUploaderDynamicSize.get());

        uint32_t frameNum = 1;
        if (m_engine->isWindowApplication())
        {
            // Swapchain init.
            m_swapchain.init();
            initPresentContext();
            frameNum = m_swapchain.getBackbufferCount();
        }

        // 16 MB init dynamic uniform buffer size, 8 MB increment when overflow.
        m_dynamicUniformBuffer = std::make_unique<DynamicUniformBuffer>(frameNum, 16, 8);
        m_rtPool               = std::make_unique<RenderTexturePool>();
        m_bufferParameters     = std::make_unique<BufferParameterPool>();
        m_shaderCache          = std::make_unique<ShaderCache>();
        // 1024 MB + 512 MB LRU cache.
        m_lru                  = std::make_unique<LRUAssetCache>(1024, 512); 
        m_passCollector        = std::make_unique<PassCollector>(this);

        initBuiltinAssets();

        return true;
    }

    bool VulkanContext::tick(const RuntimeModuleTickData& tickData)
    {
        m_uploader->tick(tickData);
        m_dynamicUniformBuffer->onFrameStart();

        CVarCmdHandle(cVarUpdatePasses, [&]() 
        { 
            m_passCollector->updateAllPasses(); 
            CVarSystem::get()->setCVar("cmd.clearAllReflectionCapture", true);
        });

        CVarCmdHandle(cVarUpdateCloudPasses, [&]()
        {
            updateCloudPass();
        });

        

        m_rtPool->tick();
        m_bufferParameters->tick();

        return true;
    }

    bool VulkanContext::beforeRelease()
    {
        m_uploader->beforeReleaseFlush();
        vkDeviceWaitIdle(getDevice());
        return true;
    }

    bool VulkanContext::release()
    {
        destroyBuiltinAsset();


        m_lru                  = nullptr;
        m_shaderCache          = nullptr;
        m_bufferParameters     = nullptr;
        m_rtPool               = nullptr;
        m_dynamicUniformBuffer = nullptr;
        m_passCollector        = nullptr;

        if (m_engine->isWindowApplication())
        {
            destroyPresentContext();
            m_swapchain.release();
        }

        m_uploader->release();
        m_samplerCache.release();

        // Release bindless resource.
        m_bindlessSampler.release();
        m_bindlessTexture.release();
        m_bindlessStorageBuffer.release();

        // Release descriptor allocator and layout cache.
        m_descriptorAllocator.release();
        m_descriptorLayoutCache.release();

        destroyCommandPools();
        destroyVMA();

        destroyDevice();

        // Release surface if exist.
        if (m_surface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
            m_surface = VK_NULL_HANDLE;
        }

        // Destroy vulkan instance.
        destroyInstance();

        // Check resource release state.
        {
            const auto gpuResSize = getAllocateGpuResourceSize();
            ASSERT(gpuResSize == 0, "No release all gpu resource! Still exist {} kB in memory!", gpuResSize / 1024);
        }

        return true;
    }

    void VulkanContext::initVMA()
    {
        {
            VmaAllocatorCreateInfo allocatorInfo = {};
            allocatorInfo.physicalDevice = m_gpu;
            allocatorInfo.device = m_device;
            allocatorInfo.instance = m_instance;
            
            // NOTE: When using ray tracing, we must exist device address for buffers indexing.
            if (m_graphicsSupportStates.bSupportRaytrace)
            {
                allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            }

            vmaCreateAllocator(&allocatorInfo, &m_vmaBuffer);
        }


        {
            VmaAllocatorCreateInfo allocatorInfo = {};
            allocatorInfo.physicalDevice = m_gpu;
            allocatorInfo.device = m_device;
            allocatorInfo.instance = m_instance;

            // NOTE: When using ray tracing, we must exist device address for buffers indexing.
            if (m_graphicsSupportStates.bSupportRaytrace)
            {
                allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            }

            vmaCreateAllocator(&allocatorInfo, &m_vmaFrequencyDestroyBuffer);
        }

        {
            VmaAllocatorCreateInfo allocatorInfo = {};
            allocatorInfo.physicalDevice = m_gpu;
            allocatorInfo.device = m_device;
            allocatorInfo.instance = m_instance;

            // NOTE: When using ray tracing, we must exist device address for buffers indexing.
            if (m_graphicsSupportStates.bSupportRaytrace)
            {
                allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            }

            vmaCreateAllocator(&allocatorInfo, &m_vmaFrequencyDestroyImage);
        }

        {
            VmaAllocatorCreateInfo allocatorInfo = {};
            allocatorInfo.physicalDevice = m_gpu;
            allocatorInfo.device = m_device;
            allocatorInfo.instance = m_instance;

            // NOTE: When using ray tracing, we must exist device address for buffers indexing.
            if (m_graphicsSupportStates.bSupportRaytrace)
            {
                allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            }

            vmaCreateAllocator(&allocatorInfo, &m_vmaImage);
        }
    }

    void VulkanContext::destroyVMA()
    {
        if (m_vmaBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_vmaBuffer);
        }

        if (m_vmaImage != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_vmaImage);
        }

        if (m_vmaFrequencyDestroyBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_vmaFrequencyDestroyBuffer);
        }

        if (m_vmaFrequencyDestroyImage != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_vmaFrequencyDestroyImage);
        }
    }

    void VulkanContext::initCommandPools()
    {
        ASSERT(m_device != VK_NULL_HANDLE, "You must init device before init command pools!");

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        // Graphics command pools.
        poolInfo.queueFamilyIndex = m_queues.graphicsFamily;

        m_commandPools.majorGraphics.queue = m_queues.graphcisQueues[0];
        m_commandPools.secondMajorGraphics.queue = m_queues.graphcisQueues[1];

        RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPools.majorGraphics.pool));
        RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPools.secondMajorGraphics.pool));

        ASSERT(m_queues.graphcisQueues.size() > 2, "Your device too old and even don't support more than two graphics queue.");

        // Other normal queue command pools init.
        if (m_queues.graphcisQueues.size() > 2)
        {
            m_commandPools.graphics.resize(m_queues.graphcisQueues.size() - 2);
            uint32_t index = 2;
            for (auto& pool : m_commandPools.graphics)
            {
                pool.queue = m_queues.graphcisQueues[index];
                RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));

                index++;
            }
        }

        // Compute command pools.
        ASSERT(m_queues.computeQueues.size() > 1, "Your device too old and even don't support more than one compute queue.");
        poolInfo.queueFamilyIndex = m_queues.computeFamily;
        m_commandPools.majorCompute.queue = m_queues.computeQueues[0];
        RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPools.majorCompute.pool));
        if (m_queues.computeQueues.size() > 1)
        {
            m_commandPools.computes.resize(m_queues.computeQueues.size() - 1);
            uint32_t index = 1;
            for (auto& pool : m_commandPools.computes)
            {
                pool.queue = m_queues.computeQueues[index];
                RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));
                index++;
            }
        }

        // Copy command pools.
        poolInfo.queueFamilyIndex = m_queues.copyFamily;
        if (m_queues.copyQueues.size() > 0)
        {
            m_commandPools.copies.resize(m_queues.copyQueues.size());
            uint32_t index = 0;
            for (auto& pool : m_commandPools.copies)
            {
                pool.queue = m_queues.copyQueues[index];
                RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));
                index++;
            }
        }
    }

    void VulkanContext::destroyCommandPools()
    {
        // 1.0 priority queue.
        vkDestroyCommandPool(m_device, m_commandPools.majorGraphics.pool, nullptr);

        // 0.8 priority queue.
        vkDestroyCommandPool(m_device, m_commandPools.majorCompute.pool, nullptr);
        vkDestroyCommandPool(m_device, m_commandPools.secondMajorGraphics.pool, nullptr);

        // 0.5 priority queues.
        for (auto& pool : m_commandPools.graphics)
        {
            vkDestroyCommandPool(m_device, pool.pool, nullptr);
        }
        for (auto& pool : m_commandPools.computes)
        {
            vkDestroyCommandPool(m_device, pool.pool, nullptr);
        }
        for (auto& pool : m_commandPools.copies)
        {
            vkDestroyCommandPool(m_device, pool.pool, nullptr);
        }
    }

    void VulkanContext::setResourceName(VkObjectType objectType, uint64_t handle, const char* name) const
    {
        if (!cVarRHIDebugMarkerEnable.get())
        {
            return;
        }

        static PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectName = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(m_device, "vkSetDebugUtilsObjectNameEXT");
        static std::mutex gMutexForSetResource;

        if (setDebugUtilsObjectName && handle && name)
        {
            std::unique_lock<std::mutex> lock(gMutexForSetResource);

            VkDebugUtilsObjectNameInfoEXT nameInfo = {};
            nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType = objectType;
            nameInfo.objectHandle = handle;
            nameInfo.pObjectName = name;

            setDebugUtilsObjectName(m_device, &nameInfo);
        }
    }

    void VulkanContext::setPerfMarkerBegin(VkCommandBuffer cmdBuf, const char* name, const math::vec4& color) const
    {
        if (!cVarRHIDebugMarkerEnable.get())
        {
            return;
        }

        static PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(m_device, "vkCmdBeginDebugUtilsLabelEXT");
        if (cmdBeginDebugUtilsLabel)
        {
            VkDebugUtilsLabelEXT label = {};
            label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
            label.pLabelName = name;
            label.color[0] = color.r;
            label.color[1] = color.g;
            label.color[2] = color.b;
            label.color[3] = color.a;
            cmdBeginDebugUtilsLabel(cmdBuf, &label);
        }
    }

    void VulkanContext::setPerfMarkerEnd(VkCommandBuffer cmdBuf) const
    {
        if (!cVarRHIDebugMarkerEnable.get())
        {
            return;
        }

        static PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(m_device, "vkCmdEndDebugUtilsLabelEXT");

        if (cmdEndDebugUtilsLabel)
        {
            cmdEndDebugUtilsLabel(cmdBuf);
        }
    }

    VkCommandBuffer VulkanContext::createMajorGraphicsCommandBuffer()
    {
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandBufferCount = 1;
        info.commandPool = getMajorGraphicsCommandPool();
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VkCommandBuffer newBuffer;
        RHICheck(vkAllocateCommandBuffers(m_device, &info, &newBuffer));

        return newBuffer;
    }

    void VulkanContext::freeMajorGraphicsCommandBuffer(VkCommandBuffer cmd)
    {
        vkFreeCommandBuffers(m_device, getMajorGraphicsCommandPool(), 1, &cmd);
    }

    void VulkanContext::executeImmediately(VkCommandPool commandPool, VkQueue queue, std::function<void(VkCommandBuffer cb)>&& func) const
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        RHICheck(vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        RHICheck(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        func(commandBuffer);
        RHICheck(vkEndCommandBuffer(commandBuffer));

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        RHICheck(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        RHICheck(vkQueueWaitIdle(queue));

        vkFreeCommandBuffers(m_device, commandPool, 1, &commandBuffer);
    }

    void VulkanContext::executeImmediatelyMajorGraphics(std::function<void(VkCommandBuffer cb)>&& func) const
    {
        executeImmediately(getMajorGraphicsCommandPool(), getMajorGraphicsQueue(), std::move(func));
    }

    DescriptorFactory VulkanContext::descriptorFactoryBegin()
    {
        return DescriptorFactory::begin(&m_descriptorLayoutCache, &m_descriptorAllocator);
    }

    void VulkanContext::waitDeviceIdle() const
    {
        // Also flush async uploader tasks.
        m_uploader->flushTask();

        // Then just wait gpu finish all work.
        vkDeviceWaitIdle(m_device);
    }

    void VulkanContext::pushDescriptorSet(
        VkCommandBuffer commandBuffer,
        VkPipelineBindPoint pipelineBindPoint,
        VkPipelineLayout layout,
        uint32_t set,
        uint32_t descriptorWriteCount,
        const VkWriteDescriptorSet* pDescriptorWrites)
    {
        static auto pushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(m_device, "vkCmdPushDescriptorSetKHR");

        pushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
    }

    void VulkanContext::initBuiltinAssets()
    {
        LOG_RHI_TRACE("Start upload builtin assets.");
        waitDeviceIdle();

        m_uploader->addTask(RawAssetTextureLoadTask::buildFlatTexture(
            nameof::nameof_enum(EBuiltinTextures::white).data(),
            getBuiltinTexturesUUID(EBuiltinTextures::white),
            { 255, 255, 255, 255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildFlatTexture(
            nameof::nameof_enum(EBuiltinTextures::grey).data(),
            getBuiltinTexturesUUID(EBuiltinTextures::grey),
            { 128, 128, 128, 255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildFlatTexture(
            nameof::nameof_enum(EBuiltinTextures::black).data(),
            getBuiltinTexturesUUID(EBuiltinTextures::black),
            { 0, 0, 0, 255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildFlatTexture(
            nameof::nameof_enum(EBuiltinTextures::translucent).data(),
            getBuiltinTexturesUUID(EBuiltinTextures::translucent),
            { 0, 0, 0, 0 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildFlatTexture(
            nameof::nameof_enum(EBuiltinTextures::normal).data(),
            getBuiltinTexturesUUID(EBuiltinTextures::normal),
            { 128, 128, 255, 255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildFlatTexture(
            nameof::nameof_enum(EBuiltinTextures::metalRoughness).data(),
            getBuiltinTexturesUUID(EBuiltinTextures::metalRoughness),
            { 255, 255, 0, 0 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildTexture(
            "image/scene.png",
            getBuiltinTexturesUUID(EBuiltinTextures::sceneIcon),
            VK_FORMAT_R8G8B8A8_UNORM,
            false,
            4
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildTexture(
            "image/cloudweather.png",
            getBuiltinTexturesUUID(EBuiltinTextures::cloudWeather),
            VK_FORMAT_R8G8B8A8_UNORM,
            false,
            4
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildTexture(
            "image/cloudnoise.png",
            getBuiltinTexturesUUID(EBuiltinTextures::cloudNoise),
            VK_FORMAT_R8G8B8A8_UNORM,
            false,
            4
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildTexture(
            "image/T_CurlNoise.png",
            getBuiltinTexturesUUID(EBuiltinTextures::curlNoise),
            VK_FORMAT_R8G8B8A8_UNORM,
            false,
            4
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildTexture(
            "image/material.png",
            getBuiltinTexturesUUID(EBuiltinTextures::materialIcon),
            VK_FORMAT_R8G8B8A8_UNORM,
            false,
            4
        ));

        m_uploader->addTask(AssetRawStaticMeshLoadTask::buildFromPath(
            nullptr,
            "./staticmesh/box.fbx",
            getBuiltinStaticMeshUUID(EBuiltinStaticMeshes::box),
            nullptr
        ));

        m_uploader->addTask(AssetRawStaticMeshLoadTask::buildFromPath(
            nullptr,
            "./staticmesh/sphere.fbx",
            getBuiltinStaticMeshUUID(EBuiltinStaticMeshes::sphere),
            nullptr
        ));

        m_uploader->addTask(AssetRawStaticMeshLoadTask::buildFromPath(
            nullptr,
            "./staticmesh/plane.fbx",
            getBuiltinStaticMeshUUID(EBuiltinStaticMeshes::plane),
            nullptr
        ));

        waitDeviceIdle();

        if (getGraphicsState().bSupportRaytrace)
        {
            for (size_t i = ((size_t)EBuiltinStaticMeshes::min + 1);
                i < size_t(EBuiltinStaticMeshes::max); i++)
            {
                auto mesh = std::static_pointer_cast<GPUStaticMeshAsset>(getBuiltinStaticMesh((EBuiltinStaticMeshes)i));
                mesh->getOrBuilddBLAS();
            }
        }

        LOG_RHI_TRACE("Upload builtin assets success!");
    }

    std::shared_ptr<UploadAssetInterface> VulkanContext::getBuiltinAsset(
        const UUID& uuid) const
    {
        return m_builtinAssets.at(uuid);
    }

    void VulkanContext::destroyBuiltinAsset()
    {
        m_builtinAssets.clear();
    }

    void VulkanContext::insertBuiltinAsset(const UUID& uuid, std::shared_ptr<UploadAssetInterface> asset)
    {
        ASSERT(!m_builtinAssets.contains(uuid), "Builtin asset insert repeat with same uuid!");
        m_builtinAssets[uuid] = asset;
    }
}