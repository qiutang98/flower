#include "reflection_probe_component.h"

#include <iconFontcppHeaders/IconsFontAwesome6.h>
#include <editor/widgets/content.h>
#include <editor/editor.h>

#include "../../renderer/render_scene.h"
#include <asset/asset_manager.h>
#include <engine/asset/asset_staticmesh.h>
#include <asset/asset_material.h>
#include <renderer/render_functions.h>
#include <renderer/reflection_capture_renderer.h>

namespace engine
{
	AutoCVarFloat cVarReflectionCaptureZFar(
		"r.reflectionCapture.zFar",
		"z far for reflection capture",
		"Rendering",
		300.0f,
		CVarFlags::ReadAndWrite);

	static inline PerObjectInfo getReflectionProbeRenderProxy()
	{
		PerObjectInfo result;

		auto asset = std::dynamic_pointer_cast<AssetStaticMesh>(
			getAssetManager()->getAsset(getBuiltinStaticMeshUUID(EBuiltinStaticMeshes::sphere)));
		auto gpuAssett = getContext()->getBuiltinStaticMesh(EBuiltinStaticMeshes::sphere);

		CHECK(asset->getSubMeshes().size() == 1);

		const auto& submesh = asset->getSubMeshes()[0];

		result.meshInfoData.meshType = EMeshType_ReflectionCaptureMesh;
		result.meshInfoData.indicesCount = submesh.indicesCount;
		result.meshInfoData.indexStartPosition = submesh.indicesStart;
		result.meshInfoData.indicesArrayId = gpuAssett->getIndices().bindless;
		result.meshInfoData.normalsArrayId = gpuAssett->getNormals().bindless;
		result.meshInfoData.tangentsArrayId = gpuAssett->getTangents().bindless;
		result.meshInfoData.positionsArrayId = gpuAssett->getPositions().bindless;
		result.meshInfoData.uv0sArrayId = gpuAssett->getUV0s().bindless;
		result.meshInfoData.sphereBounds = math::vec4(submesh.bounds.origin, submesh.bounds.radius);
		result.meshInfoData.extents = submesh.bounds.extents;
		result.meshInfoData.submeshIndex = 0;

		result.materialInfoData = buildDefaultBSDFMaterialInfo();
		result.materialInfoData.metalAdd     = 1.0f;
		result.materialInfoData.roughnessMul = 0.0f;

		return result;
	}

	ReflectionProbeComponent::~ReflectionProbeComponent()
	{

	}

	bool ReflectionProbeComponent::uiDrawComponent()
	{
		bool bChangedValue = false;
		{


			if (ImGui::Button("Recapture"))
			{
				m_sceneCapture = nullptr;
			}

			ImGui::PushItemWidth(100.0f);
			{
				auto copy = m_dimension;
				ImGui::DragInt("Dimension", &copy, 128, 128, 1024);
				copy = getNextPOT(copy);

				if (m_dimension != copy)
				{
					m_dimension = copy;
					bChangedValue = true;

					clearCapture();
				}
			}
			ImGui::PopItemWidth();

			ImGui::PushItemWidth(300.0f);
			{
				ImGui::DragFloat3("Min Extent", &m_minExtent.x, 1.0f, -500.0f,   0.0f);
				ImGui::DragFloat3("Max Extent", &m_maxExtent.x, 1.0f,    0.0f, 500.0f);

				ImGui::Checkbox("Is Draw Extent", &m_bDrawExtent);
			}

			ImGui::PopItemWidth();
		}
		return bChangedValue;
	}

	const UIComponentReflectionDetailed& ReflectionProbeComponent::uiComponentReflection()
	{
		static const UIComponentReflectionDetailed reflection =
		{
			.bOptionalCreated = true,
			.iconCreated = ICON_FA_STAR + std::string("  ReflectionProbe"),
		};
		return reflection;
	}

	void ReflectionProbeComponent::collectReflectionProbe(RenderScene& renderScene)
	{
		auto& collector = renderScene.getObjectCollector();
		

		static const PerObjectInfo proxyTemplate = getReflectionProbeRenderProxy();

		auto copyProxy = proxyTemplate;
		copyProxy.modelMatrix = getNode()->getTransform()->getWorldMatrix();
		copyProxy.modelMatrixPrev = getNode()->getTransform()->getPrevWorldMatrix();
		copyProxy.bSelected = getNode()->editorSelected();
		copyProxy.sceneNodeId = getNode()->getId();

		collector.push_back(copyProxy);

		if (m_bDrawExtent)
		{
			renderScene.drawAABBminMax(
				getNode()->getTransform()->getTranslation() + m_minExtent, 
				getNode()->getTransform()->getTranslation() + m_maxExtent);
		}
	}

	class ReflectionCaptureCamera : public CameraInterface
	{
	public:
		ReflectionCaptureCamera(uint faceIndex, vec3 position)
		{
			CHECK(faceIndex < 6);
			const vec3 captureDirections[6] =
			{
				vec3(-1.0,  0.0,  0.0),
				vec3( 1.0,  0.0,  0.0),
				vec3( 0.0, -1.0,  0.0),
				vec3( 0.0,  1.0,  0.0),
				vec3( 0.0,  0.0, -1.0),
				vec3( 0.0,  0.0,  1.0),
			};

			const vec3 upDirections[6] =
			{
				vec3(0.0, 1.0,  0.0),
				vec3(0.0, 1.0,  0.0),
				vec3(0.0, 0.0, -1.0),
				vec3(0.0, 0.0,  1.0),
				vec3(0.0, 1.0,  0.0),
				vec3(0.0, 1.0,  0.0),
			};

			m_viewMatrix = math::lookAt(position, captureDirections[faceIndex] + position, upDirections[faceIndex]);
			m_projectMatrix = math::perspectiveLH_ZO(math::radians(90.0f), 1.0f, cVarReflectionCaptureZFar.get(), 1e-3f);
		}

		virtual math::mat4 getViewMatrix() const override
		{
			return m_viewMatrix;
		}

		virtual math::mat4 getProjectMatrix() const override
		{
			return m_projectMatrix;
		}

	private:
		math::mat4 m_viewMatrix;
		math::mat4 m_projectMatrix;
	};

	void ReflectionProbeComponent::updateReflectionCapture(
		VkCommandBuffer cmd,
		const RuntimeModuleTickData& tickData)
	{
		m_bCaptureOutOfDate = false;
		m_capturePos = getNode()->getTransform()->getTranslation();
		m_prevActiveFrameNumber = tickData.tickCount;


		auto sceneCaptureRaw = getContext()->getRenderTargetPools().createPoolCubeImage(
			"sceneEnvCapture",
			m_dimension,  // Must can divide by 8.
			m_dimension,  // Must can divide by 8.
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			-1
		);

		const auto inCubeViewRangeAll = VkImageSubresourceRange
		{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 6
		};

		// Capture scene to capture cube map and generate mipmaps.
		{
			sceneCaptureRaw->getImage().transitionLayout(
				cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, inCubeViewRangeAll);

			ReflectionCaptureRenderer renderer{ };

			for (uint faceIndex = 0; faceIndex < 6; faceIndex++)
			{
				ReflectionCaptureCamera renderCamera(faceIndex, getNode()->getTransform()->getTranslation());

				auto renderResult = renderer.render(m_dimension, cmd, &renderCamera, tickData);
				renderResult->getImage().transitionTransferSrc(cmd);

				int32_t mipWidth = sceneCaptureRaw->getImage().getExtent().width;
				int32_t mipHeight = sceneCaptureRaw->getImage().getExtent().height;
				
				// Copy to cube map.
				{
					VkImageBlit blit{};
					blit.srcOffsets[0] = { 0, 0, 0 };
					blit.dstOffsets[0] = { 0, 0, 0 };
					blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
					blit.dstOffsets[1] = VkOffset3D{ mipWidth, mipHeight, 1 };

					blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					blit.srcSubresource.mipLevel = 0;
					blit.dstSubresource.mipLevel = 0;
					blit.srcSubresource.baseArrayLayer = 0;
					blit.dstSubresource.baseArrayLayer = faceIndex;
					blit.srcSubresource.layerCount = 1;
					blit.dstSubresource.layerCount = 1;

					vkCmdBlitImage(cmd, 
						renderResult->getImage().getImage(), 
						VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						sceneCaptureRaw->getImage().getImage(),
						VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);


					const auto view = VkImageSubresourceRange
					{
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = faceIndex,
						.layerCount = 1
					};
					sceneCaptureRaw->getImage().transitionLayout(
						cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, view);
				}
			}

			sceneCaptureRaw->getImage().transitionLayout(
				cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inCubeViewRangeAll);
		}

		// Generate convoluted cubemap.
		buildCubemapReflection(cmd, sceneCaptureRaw, m_sceneCapture, m_dimension / 2);
	}

	void ReflectionProbeComponent::tick(const RuntimeModuleTickData& tickData)
	{
		if (!isCaptureOutOfDate())
		{
			if (m_capturePos != getNode()->getTransform()->getTranslation())
			{
				markOutOfDate();
			}
		}

	}

}