#include "../../scene/component/reflection_probe_component.h"
#include "../../scene/scene_node.h"
#include "../deferred_renderer.h"
#include "../render_scene.h"
#include "../renderer.h"
#include "../scene_textures.h"
#include <algorithm>
#include <random>

namespace engine
{

	AutoCVarFloat cVarReflectionCaptureFarDistanceToUnvalid(
		"r.reflectionCapture.distanceStartClear",
		"Distance to camera which is time to start clear.",
		"Rendering",
		300.0f,
		CVarFlags::ReadAndWrite);

	AutoCVarInt32 cVarReflectionCaptureFramesToUnvalid(
		"r.reflectionCapture.framesStartClear",
		"How many frame pass can start clear this capture.",
		"Rendering",
		240,
		CVarFlags::ReadAndWrite);

	void engine::prepareReflectionCaptureForRender(
		VkCommandBuffer cmd,
		RenderScene* scene,
		const PerFrameData& perframe,
		const RuntimeModuleTickData& tickData,
		const struct AtmosphereTextures& inAtmosphere,
		ReflectionProbeContext& result)
	{
		if (!inAtmosphere.envCapture)
		{
			return;
		}

		CHECK(perframe.renderType != ERendererType_ReflectionCapture);

		// All reflections sort by distance to camera.
		auto reflections = scene->getReflections();
		const vec3 camPos = math::vec3(perframe.camWorldPos);

		auto eraseBegin = std::remove_if(reflections.begin(), reflections.end(), [&](ReflectionProbeComponent* x)
		{
			vec3 minPos = x->getNode()->getTransform()->getTranslation() + x->getMinExtent();
			vec3 maxPos = x->getNode()->getTransform()->getTranslation() + x->getMaxExtent();

			return
				camPos.x > maxPos.x || camPos.y > maxPos.y || camPos.z > maxPos.z ||
				camPos.x < minPos.x || camPos.y < minPos.y || camPos.z < minPos.z;
		});

		// Long time no used far distance capture unvalid cache.
		for (auto i = eraseBegin; i != reflections.end(); i++)
		{
			float dis = math::distance((*i)->getNode()->getTransform()->getTranslation(), math::vec3(perframe.camWorldPos));
			if (dis > cVarReflectionCaptureFarDistanceToUnvalid.get() && 
			   (tickData.tickCount - (*i)->getPreActiveFrameNumber() > cVarReflectionCaptureFramesToUnvalid.get()))
			{
				(*i)->clearCapture();
			}
		}

		reflections.erase(eraseBegin, reflections.end());

		// Try update.
		if (!reflections.empty())
		{
			// Select reflection probe to render scene capture.
			std::ranges::sort(reflections, [&](const auto& captureA, const auto& captureB)
				{
					float disA = math::distance(captureA->getNode()->getTransform()->getTranslation(), math::vec3(perframe.camWorldPos));
					float disB = math::distance(captureB->getNode()->getTransform()->getTranslation(), math::vec3(perframe.camWorldPos));
					return disA < disB;
				});

			// Render reflection probe.
			uint32_t kUpdatePeriod = 15;
			if (tickData.tickCount % kUpdatePeriod == 0)
			{
				size_t renderIndex = ~0;

				// Found no valid, capture first.
				if (renderIndex == ~0)
				{
					for (size_t i = 0; i < reflections.size(); i++)
					{
						if (!reflections[i]->isCaptureValid())
						{
							renderIndex = i;
							break;
						}
					}
				}

				// Found position change, capture second.
				if (renderIndex == ~0)
				{
					for (size_t i = 0; i < reflections.size(); i++)
					{
						if (reflections[i]->isCaptureOutOfDate())
						{
							renderIndex = i;
							break;
						}
					}
				}

				// Now capture.
				if (renderIndex != ~0)
				{
					reflections[renderIndex]->updateReflectionCapture(cmd, tickData);
				}
			}
		}
		else
		{
			return;
		}

		if (!reflections[0]->isCaptureValid())
		{
			return;
		}

		reflections[0]->updateActiveFrameNumber(tickData.tickCount);

		result.probe0 = reflections[0]->getSceneCapture();
		result.probe0Position = reflections[0]->getNode()->getTransform()->getTranslation();
		result.probe0MinExtent   = reflections[0]->getMinExtent();
		result.probe0MaxExtent = reflections[0]->getMaxExtent();

		{
			vec3 minPos = result.probe0Position + result.probe0MinExtent;
			vec3 maxPos = result.probe0Position + result.probe0MaxExtent;

			vec3 dis2Min = (minPos - camPos) / result.probe0MinExtent;
			vec3 dis2Max = (maxPos - camPos) / result.probe0MaxExtent;

			result.probe0ValidState = math::min(math::min(math::min(math::min(math::min(dis2Min.x, dis2Min.y), dis2Min.z), dis2Max.x), dis2Max.y), dis2Max.z);
			result.probe0ValidState = math::smoothstep(0.0f, 1.0f, result.probe0ValidState);
		}

		if (reflections.size() <= 1 || !reflections[1]->isCaptureValid())
		{
			return;
		}

		reflections[1]->updateActiveFrameNumber(tickData.tickCount);
		result.probe1 = reflections[1]->getSceneCapture();
		result.probe1Position = reflections[1]->getNode()->getTransform()->getTranslation();
		result.probe1MinExtent = reflections[1]->getMinExtent();
		result.probe1MaxExtent = reflections[1]->getMaxExtent();

		{
			vec3 minPos = result.probe1Position + result.probe1MinExtent;
			vec3 maxPos = result.probe1Position + result.probe1MaxExtent;

			vec3 dis2Min = (minPos - camPos) / result.probe1MinExtent;
			vec3 dis2Max = (maxPos - camPos) / result.probe1MaxExtent;

			result.probe1ValidState = math::min(math::min(math::min(math::min(math::min(dis2Min.x, dis2Min.y), dis2Min.z), dis2Max.x), dis2Max.y), dis2Max.z);
			result.probe1ValidState = math::smoothstep(0.0f, 1.0f, result.probe1ValidState);
		}
	}
}

