#include "Pch.h"
#include "PMXComponent.h"

// TODO: Optimize and clean me, this is a temp version code.
#include "AssetSystem/TextureManager.h"
#include "Engine.h"
#include "AssetSystem/AssetSystem.h"
#include "AssetSystem/AsyncUploader.h"
#include <Saba/Base/Path.h>
#include <Saba/Base/File.h>
#include <Saba/Base/UnicodeUtil.h>
#include <Saba/Base/Time.h>
#include "../../Renderer/RendererInterface.h"
#include "../SceneArchive.h"

#include "../../Renderer/ShadingModel.h"

// NOTE: MMD asset is free but no MIT license.
//       So reading every time from raw asset.
namespace Flower
{
	void PMXMeshProxy::release()
	{
		vkDeviceWaitIdle(RHI::Device);

		m_mmdModel.reset();
		m_vmdAnim.reset();
		m_indexBuffer.reset();
		m_vertexBuffer.reset();
		m_stageBuffer.reset();
	}

	bool PMXMeshProxy::prepareVMD()
	{
		auto vmdAnim = std::make_unique<saba::VMDAnimation>();
		if (!vmdAnim->Create(m_mmdModel))
		{
			LOG_ERROR("Failed to create VMDAnimation.");
			return false;
		}

		bool bExistOneVmd = false;
		for (const auto& vmdPath : m_vmdPath)
		{
			saba::VMDFile vmdFile;
			if (!saba::ReadVMDFile(&vmdFile, vmdPath.c_str()))
			{
				LOG_ERROR("Failed to read VMD file {0}.", vmdPath);
				return false;
			}

			if (!vmdAnim->Add(vmdFile))
			{
				LOG_ERROR("Failed to add VMDAnimation {0}.", vmdPath);
				return false;
			}

			bExistOneVmd = true;
			if (!vmdFile.m_cameras.empty())
			{
				LOG_ERROR("You can't use camera as pmx vmd {0}.", vmdPath);
				return false;
			}
		}

		vmdAnim->SyncPhysics(0.0f);
		m_vmdAnim = std::move(vmdAnim);

		return true;
	}

	bool PMXMeshProxy::preparePMX()
	{
		auto ext = saba::PathUtil::GetExt(m_pmxPath);
		if (ext != "pmx")
		{
			LOG_ERROR("Must select one pmx file.");
			return false;
		}

		auto pmxModel = std::make_unique<saba::PMXModel>();

		std::string resourceDir = saba::PathUtil::GetExecutablePath();
		resourceDir = saba::PathUtil::GetDirectoryName(resourceDir);
		resourceDir = saba::PathUtil::Combine(resourceDir, "media");
		std::string mmdDir = saba::PathUtil::Combine(resourceDir, "mmd");

		if (!pmxModel->Load(m_pmxPath, mmdDir))
		{
			LOG_ERROR("Failed to load pmx file {0}.", m_pmxPath);
			return false;
		}

		m_mmdModel = std::move(pmxModel);
		m_mmdModel->InitializeAnimation();

		return true;
	}

	bool PMXMeshProxy::prepareMaterial()
	{
		// prepare material and upload texture to bindless descriptor set here.
		size_t matCount = m_mmdModel->GetMaterialCount();

		const saba::MMDMaterial* materials = m_mmdModel->GetMaterials();

		// PMX material can keep same name, so can't simple use set and map.
		// Update materials from pmx file.
		m_component->m_materials.reserve(std::max(m_component->m_materials.size(), matCount));
		for (size_t i = 0; i < matCount; i++)
		{
			if (m_component->m_materials.size() <= i)
			{
				m_component->m_materials.push_back(PMXDrawMaterial{ .material = materials[i] });

				m_component->m_materials[i].bTranslucent = m_component->m_materials[i].material.m_alpha < 0.999f;

			}
			else
			{
				m_component->m_materials[i].material = materials[i];
			}
		}


		std::set<std::string> texLoaded{};

		// Load all material's texture.

		for (size_t i = 0; i < matCount; i++)
		{
			saba::MMDMaterial matMMD = materials[i];
			auto& workingMat = m_component->m_materials.at(i);

			CHECK(workingMat.material.m_name == matMMD.m_name);
			CHECK(workingMat.material.m_enName == matMMD.m_enName);

			// texture.
			if (!matMMD.m_texture.empty() && std::filesystem::exists(matMMD.m_texture) 
				&& !texLoaded.contains(matMMD.m_texture) && !TextureManager::get()->isAssetExist(matMMD.m_texture))
			{
				auto task = RawAssetTextureLoadTask::build(
					matMMD.m_texture, // 
					matMMD.m_texture, // UUID use path here.
					VK_FORMAT_R8G8B8A8_SRGB);
				GEngine->getRuntimeModule<AssetSystem>()->addUploadTask(task);

				texLoaded.insert(matMMD.m_texture);
			}

			// Sp texture.
			if (!matMMD.m_spTexture.empty() && std::filesystem::exists(matMMD.m_spTexture)
				&& !texLoaded.contains(matMMD.m_spTexture) && !TextureManager::get()->isAssetExist(matMMD.m_spTexture))
			{
				auto task = RawAssetTextureLoadTask::build(
					matMMD.m_spTexture,
					matMMD.m_spTexture, // UUID use path here.
					VK_FORMAT_R8G8B8A8_SRGB);
				GEngine->getRuntimeModule<AssetSystem>()->addUploadTask(task);

				texLoaded.insert(matMMD.m_spTexture);
			}

			// toon texture.
			if (!matMMD.m_toonTexture.empty() && std::filesystem::exists(matMMD.m_toonTexture)
				&& !texLoaded.contains(matMMD.m_toonTexture) && !TextureManager::get()->isAssetExist(matMMD.m_toonTexture))
			{
				auto task = RawAssetTextureLoadTask::build(
					matMMD.m_toonTexture,
					matMMD.m_toonTexture, // UUID use path here.
					VK_FORMAT_R8G8B8A8_SRGB);
				GEngine->getRuntimeModule<AssetSystem>()->addUploadTask(task);

				texLoaded.insert(matMMD.m_toonTexture);
			}
		}

		GEngine->getRuntimeModule<AssetSystem>()->flushUploadTask();

		for (size_t i = 0; i < matCount; i++)
		{
			saba::MMDMaterial matMMD = materials[i];
			auto& workingMat = m_component->m_materials.at(i);

			CHECK(workingMat.material.m_name == matMMD.m_name);
			CHECK(workingMat.material.m_enName == matMMD.m_enName);

			const uint32_t fallbackWhite = TextureManager::get()->getImage(EngineTextures::GWhiteTextureUUID).get()->getBindlessIndex();

			// texture.
			if (!matMMD.m_texture.empty() && std::filesystem::exists(matMMD.m_texture))
			{
				workingMat.mmdTex = TextureManager::get()->getImage(matMMD.m_texture).get()->getBindlessIndex();
			}
			else
			{
				LOG_WARN("Lose tex {} in material {}, use fallback white.", matMMD.m_texture, matMMD.m_name);
				workingMat.mmdTex = fallbackWhite;
			}

			// Sp texture.
			if (!matMMD.m_spTexture.empty() && std::filesystem::exists(matMMD.m_spTexture))
			{
				workingMat.mmdSphereTex = TextureManager::get()->getImage(matMMD.m_spTexture).get()->getBindlessIndex();
			}
			else
			{
				LOG_WARN("Lose sp tex {} in material {}, use fallback white.", matMMD.m_spTexture, matMMD.m_name);
				workingMat.mmdSphereTex = ~0;
			}

			// toon texture.
			if (!matMMD.m_toonTexture.empty() && std::filesystem::exists(matMMD.m_toonTexture))
			{
				workingMat.mmdToonTex = TextureManager::get()->getImage(matMMD.m_toonTexture).get()->getBindlessIndex();
			}
			else
			{
				LOG_WARN("Lose toon tex {} in material {}, use fallback white.", matMMD.m_toonTexture, matMMD.m_name);
				workingMat.mmdToonTex = ~0;
			}
		}

		return true;
	}

	bool PMXMeshProxy::prepareVertexBuffer()
	{
		auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationCreateFlags bufferFlagVMA = {};

		auto vbMemSize = uint32_t(sizeof(Vertex) * m_mmdModel->GetVertexCount());
		m_vertexBuffer = VulkanBuffer::create2(
			m_pmxPath.c_str(),
			bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			bufferFlagVMA,
			vbMemSize
		);

		m_stageBuffer = VulkanBuffer::create(
			"CopyBuffer",
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			EVMAUsageFlags::StageCopyForUpload,
			vbMemSize
		);

		// Index Buffer
		{
			if (m_mmdModel->GetIndexElementSize() == 1)
			{
				LOG_ERROR("Vulkan is not supported uint8_t index.");
				return false;
			}
			else if (m_mmdModel->GetIndexElementSize() == 2)
			{
				m_indexType = VK_INDEX_TYPE_UINT16;
			}
			else if (m_mmdModel->GetIndexElementSize() == 4)
			{
				m_indexType = VK_INDEX_TYPE_UINT32;
			}
			else
			{
				LOG_ERROR("Unknown index size.[{0}].", m_mmdModel->GetIndexElementSize());
				return false;
			}

			// Create buffer
			auto ibMemSize = uint32_t(m_mmdModel->GetIndexElementSize() * m_mmdModel->GetIndexCount());
			m_indexBuffer = VulkanBuffer::create2(
				m_pmxPath.c_str(),
				bufferFlagBasic | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				bufferFlagVMA,
				ibMemSize
			);

			// Copy index to GPU.

			auto stageBuffer = VulkanBuffer::create(
				"CopyBuffer",
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				EVMAUsageFlags::StageCopyForUpload,
				ibMemSize,
				const_cast<void*>(m_mmdModel->GetIndices())
			);

			m_indexBuffer->stageCopyFrom(
				stageBuffer->getVkBuffer(),
				ibMemSize,
				0,
				0
			);
		}

		return true;
	}

	void PMXMeshProxy::UpdateAnimation(float vmdFrameTime, float physicElapsed)
	{
		m_mmdModel->BeginAnimation();
		m_mmdModel->UpdateAllAnimation(m_vmdAnim.get(), vmdFrameTime * 30.0f, physicElapsed);
		m_mmdModel->EndAnimation();
	}

	PMXMeshProxy::PMXMeshProxy(PMXComponent* InComp)
		: m_component(InComp)
	{
	}

	bool PMXMeshProxy::Ready()
	{
		return
			m_mmdModel &&
			m_vmdAnim &&
			m_vertexBuffer &&
			m_indexBuffer &&
			m_stageBuffer;
	}

	// TODO: Can optimize.
	void PMXMeshProxy::UpdateVertex(VkCommandBuffer cmd)
	{
		size_t vtxCount = m_mmdModel->GetVertexCount();

		std::vector<glm::vec3> positionLast = m_mmdModel->getUpdatePositions();
		m_mmdModel->Update();
		const glm::vec3* position = m_mmdModel->GetUpdatePositions();
		const glm::vec3* normal = m_mmdModel->GetUpdateNormals();
		const glm::vec2* uv = m_mmdModel->GetUpdateUVs();

		glm::vec3* positionLastPtr = &positionLast[0];
		// Update vertices

		auto bufferSize = VkDeviceSize(sizeof(Vertex) * vtxCount);

		// copy vertex buffer gpu. 
		m_stageBuffer->map();
		{
			void* vbStMem = m_stageBuffer->mapped;
			auto v = static_cast<Vertex*>(vbStMem);
			for (size_t i = 0; i < vtxCount; i++)
			{
				v->position = *position;
				v->normal = *normal;
				v->uv = *uv;
				v->positionLast = *positionLastPtr;
				v++;
				position++;
				normal++;
				uv++;
				positionLastPtr++;
			}
		}
		m_stageBuffer->unmap();

		// copy to gpu
		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmd, m_stageBuffer->getVkBuffer(), m_vertexBuffer->getVkBuffer(), 1, &copyRegion);
	}

	void PMXMeshProxy::OnRenderTick(VkCommandBuffer cmd)
	{
		if (!Ready()) return;
		UpdateVertex(cmd);
	}

	void PMXMeshProxy::Setup(PMXInitTrait initTrait)
	{
		m_pmxPath = initTrait.pmxPath;
		m_vmdPath = initTrait.vmdPath;
		release();

		preparePMX();
		prepareVMD();
		prepareMaterial();
		prepareVertexBuffer();

		CHECK(Ready());
	}

	void PMXMeshProxy::SetupCamera(std::string cameraPath)
	{
		m_cameraPath = cameraPath;
		vmdCameraAnim.reset();

		saba::VMDFile vmdFile;
		if (!saba::ReadVMDFile(&vmdFile, cameraPath.c_str()))
		{
			LOG_ERROR("Failed to read VMD file {0}.", cameraPath);
			return;
		}
		if (vmdFile.m_cameras.empty())
		{
			LOG_ERROR("Vmd file {0} no contain camera.", cameraPath);
			return;
		}

		vmdCameraAnim = std::make_unique<saba::VMDCameraAnimation>();
		if (!vmdCameraAnim->Create(vmdFile))
		{
			LOG_ERROR("Failed to create VMDCameraAnimation.");
			return;
		}
	}

	void PMXMeshProxy::OnSceneTick(float vmdFrameTime, float physicElapsed)
	{
		if (!Ready()) return;
		UpdateAnimation(vmdFrameTime, physicElapsed);
		m_currentFrameCameraData = {};

		// updata camera data.
		if (vmdCameraAnim)
		{
			m_currentFrameCameraData.bValidData = true;
			vmdCameraAnim->Evaluate(vmdFrameTime * 30.0f);
		}
	}

	PerFrameMMDCamera PMXMeshProxy::GetCurrentFrameCameraData(float width, float height, float zNear, float zFar, glm::mat4 worldMatrix)
	{
		if (vmdCameraAnim)
		{
			const auto mmdCam = vmdCameraAnim->GetCamera();
			saba::MMDLookAtCamera lookAtCam(mmdCam);

			const bool bReverseZ = true;

			// update current Frame camera data.
			m_currentFrameCameraData.bValidData = true;

			glm::vec3 eyeWorldPos    = worldMatrix * glm::vec4(lookAtCam.m_eye, 1.0f);
			glm::vec3 centerWorldPos = worldMatrix * glm::vec4(lookAtCam.m_center, 1.0f);
			glm::vec3 upWorld        = worldMatrix * glm::vec4(lookAtCam.m_up, 0.0f);

			m_currentFrameCameraData.viewMat = glm::lookAt(eyeWorldPos, centerWorldPos, upWorld);


			auto fov = mmdCam.m_fov;

			m_currentFrameCameraData.projMat = glm::perspectiveFovRH(
				fov,
				width,
				height,
				bReverseZ ? zFar : zNear,
				bReverseZ ? zNear : zFar
			);

			m_currentFrameCameraData.fovy = fov;
			m_currentFrameCameraData.worldPos = eyeWorldPos;

			return m_currentFrameCameraData;
		}
		else
		{
			PerFrameMMDCamera ret{};
			ret.bValidData = false;
			return ret;
		}
	}

	void PMXMeshProxy::OnRenderCollect(
		RendererInterface* renderer, 
		VkCommandBuffer cmd, 
		VkPipelineLayout pipelinelayout, 
		const glm::mat4& modelMatrix, 
		const glm::mat4& modelMatrixPrev,
		bool bTranslucentPass)
	{
		if (!Ready()) return;

		VkBuffer vertexBuffer = m_vertexBuffer->getVkBuffer();
		VkBuffer indexBuffer = m_indexBuffer->getVkBuffer();
		const VkDeviceSize offset = 0;
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, m_indexType);
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);

		// then draw every submesh.
		size_t subMeshCount = m_mmdModel->GetSubMeshCount();
		for (uint32_t i = 0; i < subMeshCount; i++)
		{
			const auto& subMesh = m_mmdModel->GetSubMeshes()[i];
			const auto& material = m_component->m_materials.at(subMesh.m_materialID);

			if (material.bHide)
			{
				continue;
			}

			bool bShouldDraw = true;
			if (bTranslucentPass)
			{
				bShouldDraw = material.bTranslucent;
			}
			else
			{
				bShouldDraw = !material.bTranslucent;
			}

			if (!bShouldDraw)
			{
				continue;
			}

			uint32_t dynamicOffset = renderer->getDynamicBufferRing()->alloc(sizeof(PMXGpuParams));


			PMXGpuParams params{};
			params.modelMatrix = modelMatrix;
			params.modelMatrixPrev = modelMatrixPrev;
			

			params.texId = material.mmdTex;
			params.spTexID = material.mmdSphereTex;
			params.toonTexID = material.mmdToonTex;
			params.pixelDepthOffset = material.pixelDepthOffset;
			params.pmxObjectID = i;
			params.shadingModel = PMXShadingModelToParam((EPMXShadingModel)material.pmxShadingModel);

			memcpy((char*)(renderer->getDynamicBufferRing()->getBuffer()->mapped) + dynamicOffset, &params, sizeof(PMXGpuParams));

			auto set = renderer->getDynamicBufferRing()->getSet();
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 0, 1, &set, 1, &dynamicOffset);


			vkCmdDrawIndexed(cmd, subMesh.m_vertexCount, 1, subMesh.m_beginIndex, 0, 0);
		}
	}

	void PMXMeshProxy::OnShadowRenderCollect(
		RendererInterface* renderer, 
		VkCommandBuffer cmd, 
		VkPipelineLayout pipelinelayout, 
		uint32_t cascadeIndex, 
		const glm::mat4& modelMatrix, 
		const glm::mat4& modelMatrixPrev)
	{
		if (!Ready()) return;

		VkBuffer vertexBuffer = m_vertexBuffer->getVkBuffer();
		VkBuffer indexBuffer = m_indexBuffer->getVkBuffer();
		const VkDeviceSize offset = 0;
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, m_indexType);
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);

		// then draw every submesh.
		size_t subMeshCount = m_mmdModel->GetSubMeshCount();
		for (uint32_t i = 0; i < subMeshCount; i++)
		{
			const auto& subMesh = m_mmdModel->GetSubMeshes()[i];
			const auto& material = m_component->m_materials.at(subMesh.m_materialID);

			if (material.bHide)
			{
				continue;
			}

			if (material.bTranslucent) // TODO: translucent shadow.
			{
				continue;
			}

			uint32_t dynamicOffset = renderer->getDynamicBufferRing()->alloc(sizeof(PMXGpuParams));


			PMXGpuParams params{};
			params.modelMatrix = modelMatrix;
			params.modelMatrixPrev = modelMatrixPrev;


			params.texId = material.mmdTex;
			params.spTexID = material.mmdSphereTex;
			params.toonTexID = material.mmdToonTex;
			params.pixelDepthOffset = material.pixelDepthOffset;
			params.pmxObjectID = i;

			memcpy((char*)(renderer->getDynamicBufferRing()->getBuffer()->mapped) + dynamicOffset, &params, sizeof(PMXGpuParams));

			auto set = renderer->getDynamicBufferRing()->getSet();
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 7, 1, &set, 1, &dynamicOffset);


			vkCmdDrawIndexed(cmd, subMesh.m_vertexCount, 1, subMesh.m_beginIndex, 0, 0);
		}
	}

	PerFrameMMDCamera PMXComponent::getCurrentFrameCameraData(float width, float height, float zNear, float zFar)
	{
		glm::mat4 worldMatrix = m_node.lock()->getTransform()->getWorldMatrix();

		return getProxy()->GetCurrentFrameCameraData(width, height, zNear, zFar, worldMatrix);
	}

	void PMXComponent::onRenderCollect(
		RendererInterface* renderer, 
		VkCommandBuffer cmd, 
		VkPipelineLayout pipelinelayout,
		bool bTranslucentPass)
	{
		if (auto node = m_node.lock())
		{
			auto modelMatrix = node->getTransform()->getWorldMatrix();
			auto modelMatrixPrev = node->getTransform()->getPrevWorldMatrix();
			getProxy()->OnRenderCollect(
				renderer, cmd, pipelinelayout, modelMatrix, modelMatrixPrev, bTranslucentPass);
		}
	}

	bool PMXComponent::isPMXCameraPlaying() const
	{
		return m_bPlayAnimation && bCameraSetupReady;
	}

	void PMXComponent::onShadowRenderCollect(RendererInterface* renderer, VkCommandBuffer cmd, VkPipelineLayout pipelinelayout, uint32_t cascadeIndex)
	{
		if (auto node = m_node.lock())
		{
			auto modelMatrix = node->getTransform()->getWorldMatrix();

			getProxy()->OnShadowRenderCollect(renderer, cmd, pipelinelayout, cascadeIndex, modelMatrix, node->getTransform()->getPrevWorldMatrix());
		}
	}

	void PMXComponent::onRenderTick(VkCommandBuffer cmd)
	{
		if (auto node = m_node.lock())
		{
			getProxy()->OnRenderTick(cmd);
		}
	}

	void PMXComponent::tick(const RuntimeModuleTickData& tickData)
	{
		float dt = tickData.deltaTime;
		if (dt > 1.0f / 30.0f)
		{
			dt = 1.0f / 30.0f;
		}

		if (auto node = m_node.lock())
		{
			if (m_bPMXMeshChanged && !m_pmxPath.empty())
			{
				m_bPMXMeshChanged = false;
				PMXInitTrait initTrait{};
				initTrait.pmxPath = m_pmxPath;

				if (!m_vmdPath.empty())
				{
					initTrait.vmdPath.push_back(m_vmdPath);
				}


				getProxy()->Setup(initTrait);
			}

			if (m_bCameraPathChanged && !m_cameraPath.empty())
			{
				m_bCameraPathChanged = false;
				bCameraSetupReady = true;
				getProxy()->SetupCamera(m_cameraPath);
			}

			if (m_bPlayAnimation)
			{
				m_animationPlayTime += dt;
			}

			m_elapsed = dt;
			getProxy()->OnSceneTick(m_animationPlayTime, m_elapsed);
		}
	}

	PMXMeshProxy* PMXComponent::getProxy()
	{
		if (m_proxy == nullptr)
		{
			m_proxy = std::make_unique<PMXMeshProxy>(this);
		}
		return m_proxy.get();
	}


	void PMXComponent::resetAnimation()
	{
		m_animationPlayTime = 0.0f;
		GEngine->getSoundEngine()->stopAllSounds();
	}

	void PMXComponent::setPlayAnimationState(bool bState)
	{
		if (m_bPlayAnimation != bState)
		{
			if (m_bPlayAnimation)
			{
				GEngine->getSoundEngine()->setAllSoundsPaused();
			}
			else
			{
				GEngine->getSoundEngine()->play2D(m_wavPath.c_str(), false);
			}

			m_bPlayAnimation = bState;
		}
	}

	////

	PMXComponent::~PMXComponent()
	{

	}

	void PMXComponent::setPmxPath(std::string newPath)
	{
		if (m_pmxPath != newPath)
		{
			m_pmxPath = newPath;
			m_bPMXMeshChanged = true;
			markDirty();
		}
	}

	void PMXComponent::setVmdPath(std::string vmdPath)
	{
		if (m_vmdPath != vmdPath)
		{
			m_vmdPath = vmdPath;
			m_bPMXMeshChanged = true;
			markDirty();
		}
	}

	void PMXComponent::setWavPath(std::string wavPath)
	{
		if (m_wavPath != wavPath)
		{
			m_wavPath = wavPath;
			m_bWaveChanged = true;

			markDirty();
		}
	}

	void PMXComponent::setCameraPath(std::string cameraPath)
	{
		if (m_cameraPath != cameraPath)
		{
			m_cameraPath = cameraPath;
			m_bCameraPathChanged = true;
			markDirty();
		}
	}


}