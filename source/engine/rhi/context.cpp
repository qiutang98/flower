#include "rhi.h"
#include <asset/asset_system.h>
#include "gpu_asset.h"
#include <asset/asset_staticmesh.h>
#include <asset/asset_texture.h>

namespace engine
{
    static AutoCVarBool cVarRHIDebugMarkerEnable(
        "r.RHI.DebugMarkerEnable",
        "Enable debug marker or not.",
        "RHI",
        true,
        CVarFlags::ReadOnly
    );

    static AutoCVarCmd cVarUpdatePasses("cmd.updatePasses", "Update passes shader and pipeline info.");

    void VulkanContext::registerCheck(Engine* engine)
    {
        ASSERT(engine->isRuntimeModuleEmpty(), "When vulkan context enable, you must set it as the first module when register.")
    }

    bool VulkanContext::init()
    {
        m_state = EContextState::init;

        // Then try init vulkan context.
        try
        {
            // Windows app, assign windows handle.
            if (m_engine->isWindowApp())
            {
                m_window = m_engine->getFramework()->getWindow();
            }

            // Create vulkan instance.
            initInstance();

            // Select gpu.
            selectGPU();

            // Query gpu infos.
            queryGPUInfo();

            // Query depth format support state.
            quertDepthFormatSupportState();

            // Create surface for window.
            if(m_engine->isWindowApp())
            {
                RHICheck(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
            }
            
            // Create logic device for vulkan.
            initDeviceAndQueue();

            initVMA();

            // Init command pools.
            initCommandPools();
            ASSERT(m_copyPools.size() > 0 && m_computePools.size() > 0 && m_graphicsPools.size() > 0,
                "Your graphics card is too old and even no support async vulkan queue. exiting...");

            // Init descriptor allocator and layout cache.
            m_descriptorAllocator.init(this);
            m_descriptorLayoutCache.init(this);

            // Init bindless resources.
            m_bindlessSampler.init("Bindless Sampler", this);
            m_bindlessTexture.init("Bindless Texture", this);
            m_bindlessStorageBuffer.init("Bindless SSBO", this);

            // Init shader cache.
            m_shaderCache.init(this);

            // Init sampler cache, must after bindless sampler.
            m_samplerCache.init(this);
            
            // 64 MB static uploader, 32 MB dynamic uploader.
            m_uploader = std::make_unique<AsyncUploaderManager>(this, 64, 32); 

            // 1024 MB + 512 MB LRU cache.
            m_lru = std::make_unique<LRUAssetCache>(1024, 512);

            uint32_t frameNum = 1;
            if (m_engine->isWindowApp())
            {
                // Swapchain init.
                m_swapchain.init(this);

                initPresentContext();

                frameNum = m_swapchain.getBackbufferCount();
            }

            m_dynamicUniformBuffer = std::make_unique<DynamicUniformBuffer>(this, frameNum, 16, 8); // 16 MB init dynamic uniform buffer size, 8 MB increment when overflow.

            m_rtPool = std::make_unique<RenderTexturePool>(this);
            m_bufferParameters = std::make_unique<BufferParameterPool>();
            m_passCollector = std::make_unique<PassCollector>(this);

            initEngineAssets();

        }
        catch (...)
        {
            LOG_ERROR("Vulkan context init failed!");
            return false;
        }
    

        // Switch state.
        m_state = EContextState::ticking;
        return true;
    }

    bool VulkanContext::tick(const RuntimeModuleTickData& tickData)
    {
        m_dynamicUniformBuffer->onFrameStart();

        // Update passes if need.
        CVarCmdHandle(cVarUpdatePasses, [&]() { m_passCollector->updateAllPasses(); });

        // Update pool state.
        m_rtPool->tick();
        m_bufferParameters->tick();

        return true;
    }

    void VulkanContext::release()
    {
        vkDeviceWaitIdle(m_device);

        // Set bit to know current context state.
        m_state = EContextState::release;

        // Release engine asset.
        m_engineAssets.clear();

        m_dynamicUniformBuffer = nullptr;

        // Clear pass.
        m_passCollector = nullptr;

        // Clear rt pool.
        m_rtPool = nullptr;
        m_bufferParameters = nullptr;

        // Clear lru cache.
        m_lru->clear();

        m_uploader->release();

        if (m_engine->isWindowApp())
        {
            destroyPresentContext();
            m_swapchain.release();
        }

        // Release bindless resource.
        m_bindlessSampler.release();
        m_bindlessTexture.release();
        m_bindlessStorageBuffer.release();

        // Release descriptor allocator and layout cache.
        m_descriptorAllocator.release();
        m_descriptorLayoutCache.release();

        // Sampler cache release.
        m_samplerCache.release();

        // Shader cache release.
        m_shaderCache.release();

        destroyCommandPools();

        destroyVMA();

        destroyDevice();

        if(m_surface != VK_NULL_HANDLE)
        {
            // release surface.
		    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
            m_surface = VK_NULL_HANDLE;
        }

        destroyInstance();
    }

    void VulkanContext::initVMA()
    {
        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = m_gpu;
        allocatorInfo.device = m_device;
        allocatorInfo.instance = m_instance;

        if (m_graphicsSupportStates.bSupportRaytrace)
        {
            allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }

        vmaCreateAllocator(&allocatorInfo, &m_vma);
    }

    void VulkanContext::destroyVMA()
    {
        if (m_vma != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_vma);
        }
    }

    void VulkanContext::initCommandPools()
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        // Graphics command pools.
        ASSERT(m_queues.graphcisQueues.size() > 2, "Your device too old and even don't support more than one graphics queue. Closing...");
        poolInfo.queueFamilyIndex = m_queues.graphicsFamily;

        m_majorGraphicsPool.queue = m_queues.graphcisQueues[0];
        m_secondMajorGraphicsPool.queue = m_queues.graphcisQueues[1];

        RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_majorGraphicsPool.pool));
        RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_secondMajorGraphicsPool.pool));

        // Other normal queue command pools init.
        if (m_queues.graphcisQueues.size() > 2)
        {
            m_graphicsPools.resize(m_queues.graphcisQueues.size() - 2);
            uint32_t index = 2;
            for (auto& pool : m_graphicsPools)
            {
                pool.queue = m_queues.graphcisQueues[index];
                RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool.pool));

                index++;
            }
        }

        // Compute command pools.
        ASSERT(m_queues.computeQueues.size() > 1, "Your device too old and even don't support more than one compute queue.");
        poolInfo.queueFamilyIndex = m_queues.computeFamily;
        m_majorComputePool.queue = m_queues.computeQueues[0];
        RHICheck(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_majorComputePool.pool));
        if (m_queues.computeQueues.size() > 1)
        {
            m_computePools.resize(m_queues.computeQueues.size() - 1);
            uint32_t index = 1;
            for (auto& pool : m_computePools)
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
            m_copyPools.resize(m_queues.copyQueues.size());
            uint32_t index = 0;
            for (auto& pool : m_copyPools)
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
        vkDestroyCommandPool(m_device, m_majorGraphicsPool.pool, nullptr);

        // 0.8 priority queue.
        vkDestroyCommandPool(m_device, m_majorComputePool.pool, nullptr);
        vkDestroyCommandPool(m_device, m_secondMajorGraphicsPool.pool, nullptr);

        // 0.5 priority queues.
        for (auto& pool : m_graphicsPools)
        {
            vkDestroyCommandPool(m_device, pool.pool, nullptr);
        }
        for (auto& pool : m_computePools)
        {
            vkDestroyCommandPool(m_device, pool.pool, nullptr);
        }
        for (auto& pool : m_copyPools)
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

    VkPipelineLayout VulkanContext::createPipelineLayout(const VkPipelineLayoutCreateInfo& info)
    {
        VkPipelineLayout layout;
        RHICheck(vkCreatePipelineLayout(m_device, &info, nullptr, &layout));
        return layout;
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
        vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        func(commandBuffer);

        vkEndCommandBuffer(commandBuffer);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
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

    UUID VulkanContext::getBuiltEngineAssetUUID(EBuiltinEngineAsset type)
    {
    #define CASE_STR(X) case EBuiltinEngineAsset::X: return "EBuiltinEngineAsset::"#X;

        switch (type)
        {
            CASE_STR(Texture_White)
            CASE_STR(Texture_Grey)
            CASE_STR(Texture_Black)
            CASE_STR(Texture_Translucent)
            CASE_STR(Texture_Normal)
            CASE_STR(Texture_Specular)
            CASE_STR(Texture_CloudWeather)
            CASE_STR(Texture_CurlNoise)
            CASE_STR(Texture_Noise)
            CASE_STR(Texture_Sky3d)
            CASE_STR(StaticMesh_Box)
            CASE_STR(StaticMesh_Sphere)
        }

        // Non-implement check.
        CHECK_ENTRY();
        return {};
    #undef CASE_STR
    }

    void VulkanContext::insertEngineAsset(const UUID& uuid, std::shared_ptr<LRUAssetInterface> asset)
    {
        ASSERT(!m_engineAssets.contains(uuid), "Engine asset insert repeat with same uuid!");
        m_engineAssets[uuid] = asset;
    }

    void VulkanContext::insertLRUAsset(const UUID& uuid, std::shared_ptr<LRUAssetInterface> asset)
    {
        m_lru->insert(uuid, asset);
    }

    void VulkanContext::initEngineAssets()
    {
        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineFlatTexture(
            this,
            "EngineWhite",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_White),
            { 255, 255, 255, 255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineFlatTexture(
            this,
            "EngineGray",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Grey),
            { 128, 128, 128, 255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineFlatTexture(
            this,
            "EngineBlack",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Black),
            { 0,0,0,255 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineFlatTexture(
            this,
            "EngineTranslucent",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Translucent),
            { 0,0,0,0 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineFlatTexture(
            this,
            "EngineNormal",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Normal),
            { 125,130,255,0 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineFlatTexture(
            this,
            "EngineSpecular",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Specular),
            { 255, 255, 0, 0 }
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineTexture(
            this,
            "./image/T_CloudWetherMap.png",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_CloudWeather),
            VK_FORMAT_R8G8B8A8_UNORM,
            false
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineTexture(
            this,
            "./image/T_CurlNoise.png",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_CurlNoise),
            VK_FORMAT_R8G8B8A8_UNORM,
            false
        ));
        m_uploader->addTask(RawAssetTextureLoadTask::buildEngineTexture(
            this,
            "./image/T_Noise.png",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Noise),
            VK_FORMAT_R8G8B8A8_UNORM,
            false
        ));

        m_uploader->addTask(RawAssetTextureLoadTask::buildEngine3dTexture(
            this,
            "./image/T_Sky.texture3d",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::Texture_Sky3d),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            { 256, 128, 33 }
        ));

        m_uploader->addTask(AssetRawStaticMeshLoadTask::buildFromPath(
            this,
            "./staticmesh/box.obj",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::StaticMesh_Box),
            m_engineMeshBounds[getBuiltEngineAssetUUID(EBuiltinEngineAsset::StaticMesh_Box)]
        ));

        m_uploader->addTask(AssetRawStaticMeshLoadTask::buildFromPath(
            this,
            "./staticmesh/sphere.obj",
            getBuiltEngineAssetUUID(EBuiltinEngineAsset::StaticMesh_Sphere),
            m_engineMeshBounds[getBuiltEngineAssetUUID(EBuiltinEngineAsset::StaticMesh_Sphere)]
        ));

        m_uploader->flushTask();

        if (getGraphicsCardState().bSupportRaytrace)
        {
            for (size_t i = ((size_t)EBuiltinEngineAsset::StaticMesh_Min + 1);
                i < size_t(EBuiltinEngineAsset::StaticMesh_Max); i++)
            {
                auto mesh = std::static_pointer_cast<GPUStaticMeshAsset>(getEngineAsset((EBuiltinEngineAsset)i));
                mesh->getOrBuilddBLAS();
            }
        }
    }

    std::shared_ptr<LRUAssetInterface> engine::VulkanContext::getEngineAsset(EBuiltinEngineAsset asset) const
    {
        return m_engineAssets.at(getBuiltEngineAssetUUID(asset));
    }

    std::shared_ptr<LRUAssetInterface> VulkanContext::getEngineAsset(const UUID& uuid) const
    {
        return m_engineAssets.at(uuid);
    }

    std::shared_ptr<GPUStaticMeshAsset> VulkanContext::getOrCreateStaticMeshAsset(const UUID& uuid)
    {
        if (m_engineAssets.contains(uuid))
        {
            return std::dynamic_pointer_cast<GPUStaticMeshAsset>(m_engineAssets.at(uuid));
        }

        if (!m_lru->contain(uuid))
        {
            auto asset = std::dynamic_pointer_cast<AssetStaticMesh>(getAssetSystem()->getAsset(uuid));
            auto newTask = AssetStaticMeshLoadFromCacheTask::build(getContext(), asset);
            m_uploader->addTask(newTask);
        }

        return std::dynamic_pointer_cast<GPUStaticMeshAsset>(m_lru->tryGet(uuid));
    }

    std::shared_ptr<GPUImageAsset> VulkanContext::getOrCreateTextureAsset(const UUID& uuid)
    {
        if (m_engineAssets.contains(uuid))
        {
            return std::dynamic_pointer_cast<GPUImageAsset>(m_engineAssets.at(uuid));
        }

        if (!m_lru->contain(uuid))
        {
            auto asset = std::dynamic_pointer_cast<AssetTexture>(getAssetSystem()->getAsset(uuid));
            auto newTask = AssetTextureCacheLoadTask::build(getContext(), asset);
            m_uploader->addTask(newTask);
        }

        return std::dynamic_pointer_cast<GPUImageAsset>(m_lru->tryGet(uuid));
    }

    void VulkanContext::waitDeviceIdle() const
    {
        m_uploader->flushTask();
        vkDeviceWaitIdle(m_device);
    }

    VulkanContext* getContext()
    {
        static VulkanContext* context = Framework::get()->getEngine().getRuntimeModule<VulkanContext>();
        return context;
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
}