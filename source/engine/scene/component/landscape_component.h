#pragma once
#include "../component.h"
#include "../shader/common_header.h"
#include <graphics/graphics.h>

namespace engine
{
	class RenderScene;
	class GPUImageAsset;

	class LandscapeComponent : public Component
	{
		REGISTER_BODY_DECLARE(Component);

	public:
		LandscapeComponent() = default;
		LandscapeComponent(std::shared_ptr<SceneNode> sceneNode) : Component(sceneNode)
		{

		}

		virtual ~LandscapeComponent() = default;

		virtual bool uiDrawComponent() override;
		static const UIComponentReflectionDetailed& uiComponentReflection();


		bool collectLandscape(RenderScene& renderScene, VkCommandBuffer cmd);

		uint32_t getLODCount() const;
		uint32_t getRenderDimension() const;

		bool setAssetUUID(const UUID& in);
		void clearCache();
		void buildCache();

		const GPUImageAsset* getGPUImage() const;
		GPUImageAsset* getGPUImage();

		const vec2& getOffset() const { return m_offset; }
		float getMinHeight() const { return m_minHeight; }
		float getMaxHeight() const { return m_maxHeight; }

		PoolImageSharedRef getHeightMapHZB() { return m_heightMapHzb; }

		PoolImageSharedRef getNormalMap() { return m_normalMap; }

	private:
		std::shared_ptr<GPUImageAsset> m_heightmapImage = nullptr;
		PoolImageSharedRef m_heightMapHzb = nullptr;
		PoolImageSharedRef m_normalMap = nullptr;

	private:
		// Dimension config.
		int32_t m_dimension  = 8192; // 8km * 8km default.

		// Altitude of terrain.
		float   m_maxHeight  = 400.0f;
		float   m_minHeight  = -10.0f;

		// Offset of terrain original.
		math::vec2 m_offset  = ivec2(-4096);

		// Asset height map uuid.
		UUID m_heightmapTextureUUID = {};
	};
}