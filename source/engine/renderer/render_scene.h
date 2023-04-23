#pragma once

#include <util/util.h>
#include <rhi/rhi.h>
#include <scene/scene_archive.h>

namespace engine
{
    class SceneManager;
    class RenderScene : NonCopyable
    {
    public:
        RenderScene(VulkanContext* context, SceneManager* sceneManager);
        ~RenderScene();

        // Upadte collect scene infos. often called before all renderer logic.
        void tick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd);

        // Static mesh infos.
        bool isStaticMeshExist() const { return !m_staticmeshObjects.empty(); }
        const auto& getStaticMeshObjects() const { return m_staticmeshObjects; }
        BufferParameterHandle& getStaticMeshObjectsGPU() { return m_staticmeshObjectsGPU; }
        const BufferParameterHandle& getStaticMeshObjectsGPU() const { return m_staticmeshObjectsGPU; }

        // Sky infos.
        bool isSkyExist() const { return m_sky.lock() != nullptr; }
        std::shared_ptr<SkyComponent> getSky() { return m_sky.lock(); }
        const auto& getSkyGPU() const { return m_skyGPU; }

        bool shouldRenderSDSM() const;

        const auto& getPostprocessVolumeSetting() const
        {
            return m_postprocessVolumeInfo;
        }

        auto* getTLAS() { return &m_tlasRing->getActive(); }

        bool isTerrainExist() const { return !m_terrainComponents.empty(); }
        auto& getTerrains() { return m_terrainComponents; }

    private:
        void renderObjectCollect(const RuntimeModuleTickData& tickData, class Scene* scene, VkCommandBuffer cmd);

        void lightCollect(class Scene* scene, VkCommandBuffer cmd);

    private:
        VulkanContext* m_context;
        SceneManager* m_sceneManager;

        // Static mesh object info in scene.
        std::vector<GPUStaticMeshPerObjectData> m_staticmeshObjects;
        BufferParameterHandle m_staticmeshObjectsGPU;

        // Sky object info in scene. current only support one sky.
        GPUSkyInfo m_skyGPU;
        std::weak_ptr<SkyComponent> m_sky;


        // Submesh map scene nodes, only collect when require pick.
        std::unordered_map<uint32_t, class SceneNode*> m_submeshIdMapNodes;

        // Scene postprocessing volume info.
        PostprocessVolumeSetting m_postprocessVolumeInfo;

        // Collect scene tlas instance.
        std::unique_ptr<TLASBuilderRing> m_tlasRing;

        std::vector<std::weak_ptr<TerrainComponent>> m_terrainComponents;
    };
}
