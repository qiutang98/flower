#include "pmx.h"
#include <rhi/rhi.h>

#include <saba/Base/Path.h>
#include <saba/Base/File.h>
#include <saba/Base/UnicodeUtil.h>
#include <saba/Base/Time.h>
#include <asset/asset_wave.h>
#include <util/openal.h>
#include <renderer/render_scene.h>

namespace engine
{
	PMXComponent::~PMXComponent()
	{
		clearAudio();
	}

	void PMXComponent::onGameBegin()
	{
		if (m_bAudioPrepared)
		{
			for (std::size_t i = 0; i < m_audioBufferes.size(); ++i)
			{
				alCall(alBufferData, m_audioBufferes[i], m_audioFormat, &m_aduioDatas[i * kOpenAlBufferSize], kOpenAlBufferSize, m_audioSampleRate);
			}
			m_audioBufferCursor = kOpenAlNumBuffers * kOpenAlBufferSize;

			alCall(alSourceQueueBuffers, m_audioSource, kOpenAlNumBuffers, &m_audioBufferes[0]);

			m_audioState = AL_PLAYING;
			alCall(alSourcePlay, m_audioSource);
		}
	}

	void PMXComponent::onGameStop()
	{
		if (m_bAudioPrepared)
		{
			m_audioBufferCursor = 0;

			m_audioState = AL_STOPPED;
			alCall(alSourceStop, m_audioSource);


			ALint buffersProcessed = 0;
			alCall(alGetSourcei, m_audioSource, AL_BUFFERS_PROCESSED, &buffersProcessed);
			while (buffersProcessed--)
			{
				ALuint buffer;
				alCall(alSourceUnqueueBuffers, m_audioSource, 1, &buffer);
			}
		}
	}

	void PMXComponent::onGameContinue()
	{
		if (m_bAudioPrepared)
		{
			m_audioState = AL_PLAYING;
			alCall(alSourcePlay, m_audioSource);
		}
	}

	void PMXComponent::onGamePause()
	{
		if (m_bAudioPrepared)
		{
			m_audioState = AL_PAUSED;
			alCall(alSourcePause, m_audioSource);
		}
	}

	void PMXComponent::tick(const RuntimeModuleTickData& tickData)
	{
		if (!m_proxy && (!m_pmxUUID.empty()))
		{
			m_proxy = std::make_unique<PMXMeshProxy>(m_pmxUUID, m_vmdUUIDs);
			getRenderer()->getScene()->unvalidAS();
		}

		if (!m_bAudioPrepared && !m_singSong.empty())
		{
			prepareAudio();
		}

		// Runing game.
		if (m_bAudioPrepared)
		{
			if (m_audioState == AL_PLAYING)
			{
				update_stream(m_audioSource, m_audioFormat, m_audioSampleRate, m_aduioDatas, m_audioBufferCursor);
				alCall(alGetSourcei, m_audioSource, AL_SOURCE_STATE, &m_audioState);
			}

		}
	}

	bool PMXComponent::setPMX(const UUID& in)
	{
		if (m_pmxUUID != in)
		{
			m_pmxUUID = in;
			m_proxy = nullptr;
			getRenderer()->getScene()->unvalidAS();
			return true;
		}

		return false;
	}

	bool PMXComponent::setSong(const UUID& in)
	{
		if (m_singSong != in)
		{
			m_singSong = in;
			prepareAudio();
			return true;
		}

		return false;
	}

	void PMXComponent::prepareAudio()
	{
		clearAudio();

		auto waveAsset = std::dynamic_pointer_cast<AssetWave>(getAssetSystem()->getAsset(m_singSong));
		auto wavePath = waveAsset->getWaveFilePath().string();

		AudioFile<double> audioFile;

		// Load song.
		audioFile.load(wavePath);
		audioFile.printSummary();


		std::uint8_t channels = audioFile.getNumChannels();
		m_audioSampleRate = audioFile.getSampleRate();
		std::uint8_t bitsPerSample = audioFile.getBitDepth();


		m_bAudioVolumetric = waveAsset->m_bVolumetric;
		alCall(alGenBuffers, m_audioBufferes.size(), m_audioBufferes.data());

		
		if (channels == 1 && bitsPerSample == 8)
		{
			m_audioFormat = AL_FORMAT_MONO8;

			m_aduioDatas.resize(audioFile.samples[0].size());
			uint8* datas = (uint8*)m_aduioDatas.data();
			for (size_t i = 0; i < audioFile.samples[0].size(); i++)
			{
				datas[i] = AudioSampleConverter<double>::sampleToUnsignedByte(audioFile.samples[0][i]);
			}
		}
		else if (channels == 1 && bitsPerSample == 16)
		{
			m_audioFormat = AL_FORMAT_MONO16;
			m_aduioDatas.resize(audioFile.samples[0].size() * 2);
			int16_t* datas = (int16_t*)m_aduioDatas.data();

			for (size_t i = 0; i < audioFile.samples[0].size(); i++)
			{
				datas[i] = AudioSampleConverter<double>::sampleToSixteenBitInt(audioFile.samples[0][i]);
			}
		}
		else if (channels == 2 && bitsPerSample == 8)
		{
			m_audioFormat = AL_FORMAT_STEREO8;

			m_aduioDatas.resize(audioFile.samples[0].size() * 2);
			uint8* datas = (uint8*)m_aduioDatas.data();

			for (size_t i = 0; i < audioFile.samples[0].size(); i++)
			{
				datas[i * 2 + 0] = AudioSampleConverter<double>::sampleToUnsignedByte(audioFile.samples[0][i]);
				datas[i * 2 + 1] = AudioSampleConverter<double>::sampleToUnsignedByte(audioFile.samples[1][i]);
			}
		}
		else if (channels == 2 && bitsPerSample == 16)
		{
			m_audioFormat = AL_FORMAT_STEREO16;

			m_aduioDatas.resize(audioFile.samples[0].size() * 2 * 2);
			int16_t* datas = (int16_t*)m_aduioDatas.data();

			for (size_t i = 0; i < audioFile.samples[0].size(); i++)
			{
				datas[i * 2 + 0] = AudioSampleConverter<double>::sampleToSixteenBitInt(audioFile.samples[0][i]);
				datas[i * 2 + 1] = AudioSampleConverter<double>::sampleToSixteenBitInt(audioFile.samples[1][i]);
			}
		}
		else
		{
			LOG_ERROR("unrecognised wave format {} channels and {} bps.", channels, bitsPerSample);
			return;
		}



		alCall(alGenSources, 1, &m_audioSource);

		alCall(alSourcef, m_audioSource, AL_PITCH, 1);
		alCall(alSourcef, m_audioSource, AL_GAIN, 1.0f);
		alCall(alSource3f, m_audioSource, AL_VELOCITY, 0, 0, 0);
		alCall(alSourcei, m_audioSource, AL_LOOPING, AL_FALSE);
		alCall(alSource3f, m_audioSource, AL_POSITION, 0, 0, 0);



		m_bAudioPrepared = true;

	}

	void PMXComponent::clearAudio()
	{
		if (m_bAudioPrepared)
		{
			alCall(alDeleteSources, 1, &m_audioSource);
			alCall(alDeleteBuffers, kOpenAlNumBuffers, &m_audioBufferes[0]);
		}


		m_audioState = AL_INITIAL;
		m_bAudioPrepared = false;
	}


	size_t PMXComponent::addVmd(const UUID& in)
	{
		auto result = m_vmdUUIDs.size();
		m_vmdUUIDs.push_back(in);
		m_proxy->rebuildVMD(m_vmdUUIDs);

		return result;
	}

	void PMXComponent::removeVmd(size_t i)
	{
		m_vmdUUIDs.erase(m_vmdUUIDs.begin() + i);
		m_proxy->rebuildVMD(m_vmdUUIDs);
	}

	void PMXComponent::clearVmd()
	{
		m_vmdUUIDs.clear();
		m_proxy->rebuildVMD(m_vmdUUIDs);
	}

	bool PMXMeshProxy::rebuildVMD(const std::vector<UUID>& vmdUUIDs)
	{
		// Clear vmd animation proxy when need rebuild.
		m_vmd = std::make_unique<saba::VMDAnimation>();
		if (!m_vmd->Create(m_mmdModel))
		{
			LOG_ERROR("Failed to create VMDAnimation.");
			return false;
		}


		for (const auto& vmdUUID : vmdUUIDs)
		{
			auto vmdAsset = std::dynamic_pointer_cast<AssetVMD>(getAssetSystem()->getAsset(vmdUUID));
			if (vmdAsset->m_bCamera)
			{
				continue;
			}

			auto vmdPath = vmdAsset->getVMDFilePath().string();
			saba::VMDFile vmdFile;
			if (!saba::ReadVMDFile(&vmdFile, vmdPath.c_str()))
			{
				LOG_ERROR("Failed to read VMD file {0}.", vmdPath);
				return false;
			}

			if (!vmdFile.m_cameras.empty())
			{
				LOG_ERROR("You can't use camera as pmx vmd {0}.", vmdPath);
				continue;
			}

			if (!m_vmd->Add(vmdFile))
			{
				LOG_ERROR("Failed to add VMDAnimation {0}.", vmdPath);
				continue;
			}
		}

		m_vmd->SyncPhysics(0.0f);

		return true;
	}

	PMXMeshProxy::PMXMeshProxy(const UUID& uuid, const std::vector<UUID>& vmdUUIDs)
	{
		auto pmxAsset = std::dynamic_pointer_cast<AssetPMX>(getAssetSystem()->getAsset(uuid));
		auto path = pmxAsset->getPMXFilePath();
		std::string pmxPath = path.string();

		auto pmxModel = std::make_shared<saba::PMXModel>();
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

		pmxModel->InitializeAnimation();

		m_mmdModel = pmxModel;
		m_pmxAsset = pmxAsset;
		if (!vmdUUIDs.empty())
		{
			rebuildVMD(vmdUUIDs);
		}

		// Prepare vertex buffers.
		{
			auto bufferFlagBasic = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			VmaAllocationCreateFlags bufferFlagVMA = {};
			if (getContext()->getGraphicsCardState().bSupportRaytrace)
			{
				// Raytracing accelerate struct, random shader fetch by address.
				bufferFlagBasic |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
				bufferFlagVMA = {};
			}

			auto vbMemSizePosition = uint32_t(sizeof(glm::vec3) * pmxModel->GetVertexCount());
			auto vbMemSizeNormal = uint32_t(sizeof(glm::vec3) * pmxModel->GetVertexCount());
			auto vbMemSizeUv = uint32_t(sizeof(glm::vec2) * pmxModel->GetVertexCount());
			auto vbMemSizePositionLast = uint32_t(sizeof(glm::vec3) * pmxModel->GetVertexCount());


			m_positionBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizePosition);
			m_normalBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizeNormal);
			m_uvBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizeUv);
			m_positionPrevFrameBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizePositionLast);
			m_smoothNormalBuffer = std::make_unique<VulkanBuffer>(getContext(), pmxPath, bufferFlagBasic | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferFlagVMA, vbMemSizeNormal);

			m_stageBufferPosition = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizePosition);
			m_stageBufferNormal = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizeNormal);
			m_stageBufferUv = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizeUv);
			m_stageBufferPositionPrevFrame = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizePositionLast);
			m_stageSmoothNormal = std::make_unique<VulkanBuffer>(getContext(), pmxPath, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanBuffer::getStageCopyForUploadBufferFlags(), vbMemSizeNormal);

			m_normalBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_normalBuffer->getVkBuffer(), 0, vbMemSizeNormal);
			m_uvBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_uvBuffer->getVkBuffer(), 0, vbMemSizeUv);
			m_positionBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_positionBuffer->getVkBuffer(), 0, vbMemSizePosition);
			m_positionPrevBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_positionPrevFrameBuffer->getVkBuffer(), 0, vbMemSizePositionLast);
			m_smoothNormalBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_smoothNormalBuffer->getVkBuffer(), 0, vbMemSizeNormal);

			// Index Buffer
			m_indexType = VK_INDEX_TYPE_UINT32;
			{
				CHECK(pmxModel->GetIndexElementSize() == 4);

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

				m_indicesBindless = getContext()->getBindlessSSBOs().updateBufferToBindlessDescriptorSet(m_indexBuffer->getVkBuffer(), 0, ibMemSize);
			}
		}

		// Prepare textures.
		pmxAsset->tryLoadAllTextures(*pmxModel);

		m_bInit = true;

		getContext()->executeImmediatelyMajorGraphics([this](VkCommandBuffer cmd) 
		{
			updateVertex(cmd);
		});
	}

	PMXMeshProxy::~PMXMeshProxy()
	{
		getContext()->waitDeviceIdle();

		if (m_indicesBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_indicesBindless);
			m_indicesBindless = ~0;
		}
		if (m_normalBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_normalBindless);
			m_normalBindless = ~0;
		}
		if (m_uvBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_uvBindless);
			m_uvBindless = ~0;
		}
		if (m_positionBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_positionBindless);
			m_positionBindless = ~0;
		}
		if (m_positionPrevBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_positionPrevBindless);
			m_positionPrevBindless = ~0;
		}
		if (m_smoothNormalBindless != ~0)
		{
			getContext()->getBindlessSSBOs().freeBindlessImpl(m_smoothNormalBindless);
			m_smoothNormalBindless = ~0;
		}
	}
}