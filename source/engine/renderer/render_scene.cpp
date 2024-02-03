#include "render_scene.h"
#include "renderer.h"
#include <scene/component/staticmesh_component.h>
#include <scene/component/sky_component.h>
#include <scene/scene_manager.h>
#include <scene/component/postprocess_component.h>
#include <scene/component/reflection_probe_component.h>
#include <engine/asset/asset_staticmesh.h>
#include <asset/asset_manager.h>
#include <scene/component/landscape_component.h>

namespace engine
{
	static AutoCVarCmd cVarClearAllReflectionCapture(
		"cmd.clearAllReflectionCapture", "Clear all reflection capture cache.");


	static AutoCVarInt32 cVarTerrainShadowDepthDim("r.terrain.shadowDepthDim", "shadow depth dim of terrain.", "Rendering", 2048, CVarFlags::ReadAndWrite);

	static AutoCVarInt32 cVarCloudShadowDepthDim("r.cloud.shadowDepthDim", "shadow depth dim of cloud.", "Rendering", 768, CVarFlags::ReadAndWrite);

	int32_t engine::getShadowDepthDimTerrain()
	{
		return cVarTerrainShadowDepthDim.get();
	}

	int32_t engine::getShadowDepthDimCloud()
	{
		return cVarCloudShadowDepthDim.get();
	}

	static inline PerObjectInfo getIrradianceProbeRenderProxy()
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
		result.materialInfoData.roughnessAdd = 1.0f;

		return result;
	}

	void RenderScene::tick(const RuntimeModuleTickData& tickData, VkCommandBuffer cmd)
	{
		auto scene = getSceneManager()->getActiveScene();

		// Reset perframe collect data.
		m_perFrameCollect = {};

		scene->loopComponents<StaticMeshComponent>([&](std::shared_ptr<StaticMeshComponent> comp)
		{
			comp->collectRenderObject(*this);
			return false;
		});

		// Find first sky component of scene.
		scene->loopComponents<SkyComponent>([&](std::shared_ptr<SkyComponent> comp)
		{
			if(comp->collectSkyLight(*this))
			{
				m_perFrameCollect.skyComponent = comp.get();
				return true;
			}

			return false;
		});

		// Find first landscape component of scene.
		scene->loopComponents<LandscapeComponent>([&](std::shared_ptr<LandscapeComponent> comp)
		{
			if (comp->collectLandscape(*this, cmd))
			{
				m_perFrameCollect.landscape = comp.get();
				return true;
			}

			return false;
		});

		CVarCmdHandle(cVarClearAllReflectionCapture, [&]() { 
			m_bClearAllRelfectionInThisLoop = true;
		});

		scene->loopComponents<ReflectionProbeComponent>([&](std::shared_ptr<ReflectionProbeComponent> comp)
		{
			comp->collectReflectionProbe(*this);
			m_perFrameCollect.reflections.push_back(comp.get());

			if (m_bClearAllRelfectionInThisLoop)
			{
				comp->clearCapture();
			}
			return false;
		});
		m_bClearAllRelfectionInThisLoop = false;

		if (m_perFrameCollect.objects.size() >= kMaxObjectId)
		{
			LOG_WARN("Too much object in, current num is {}.", m_perFrameCollect.objects.size());
		}

		// Build and upload gpu scene. TODO: Sparse upload to save the performance.
		if (!m_perFrameCollect.objects.empty())
		{
			if(false)
			{
				vec3 distance = m_perFrameCollect.sceneStaticMeshAABB.max - m_perFrameCollect.sceneStaticMeshAABB.min;

				ivec3 gridNum = { 8, 8, 8 };

				static const PerObjectInfo proxyTemplate = getIrradianceProbeRenderProxy();
				const vec3 dF = distance / vec3(gridNum);
				for (int x = 0; x < gridNum.x; x++)
				{
					for (int y = 0; y < gridNum.y; y++)
					{
						for (int z = 0; z < gridNum.z; z++)
						{
							vec3 id = vec3(x, y, z) * dF + vec3(0.5) * dF;
							vec3 position = m_perFrameCollect.sceneStaticMeshAABB.min + id;

							auto copyProxy = proxyTemplate;
							copyProxy.modelMatrix = math::scale(math::translate(mat4(1.0f), position), {0.5f, 0.5f, 0.5f});
							copyProxy.modelMatrixPrev = copyProxy.modelMatrix;
							copyProxy.bSelected   = false;
							copyProxy.sceneNodeId = ~0;

							getObjectCollector().push_back(copyProxy);
						}
					}
				}
			}


			m_perFrameCollect.objectsBufferGPU = getContext()->getBufferParameters().getStaticStorage(
				"objectsBufferGPU", sizeof(m_perFrameCollect.objects[0]) * m_perFrameCollect.objects.size());
			m_perFrameCollect.objectsBufferGPU->updateDataPtr((void*)m_perFrameCollect.objects.data());


			//drawAABBminMax(m_perFrameCollect.sceneStaticMeshAABB.min, m_perFrameCollect.sceneStaticMeshAABB.max);
		}



		// Find first postprocess component of scene.
		scene->loopComponents<PostprocessComponent>([&](std::shared_ptr<PostprocessComponent> comp)
		{
			m_perFrameCollect.postprocessingComponent = comp.get();
			return true;
		});

		// Prepare TLAS.
		tlasPrepare(tickData, scene.get(), cmd);
	}

	bool RenderScene::isTLASValid() const
	{
		return !m_perFrameCollect.cacheASInstances.empty() && m_tlas.isInit();
	}

	// When ensure r0 is inside of sphere.
	// Only exist one positive result, use it.
	float raySphereIntersectInside(
		  vec3  r0  // ray origin
		, vec3  rd  // normalized ray direction
		, vec3  s0  // sphere center
		, float sR) // sphere radius
	{
		float a = dot(rd, rd);

		vec3 s02r0 = r0 - s0;
		float b = 2.0f * dot(rd, s02r0);

		float c = dot(s02r0, s02r0) - (sR * sR);
		float delta = b * b - 4.0f * a * c;

		// float sol0 = (-b - sqrt(delta)) / (2.0 * a);
		float sol1 = (-b + sqrt(delta)) / (2.0f * a);

		// sol1 > sol0, so just return sol1
		return sol1;
	}

	// No intersection: .x > .y.
	vec2 intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) 
	{
		vec3 tMin = (boxMin - rayOrigin) / rayDir; 
		vec3 tMax = (boxMax - rayOrigin) / rayDir; 

		vec3 t1 = min(tMin, tMax); // -inf
		vec3 t2 = max(tMin, tMax); // +inf

		float tNear = math::max(math::max(t1.x, t1.y), t1.z); // -inf filter.
		float tFar  = math::min(math::min(t2.x, t2.y), t2.z); // +inf filter.

		return vec2(tNear, tFar);
	};

	void RenderScene::fillPerframe(PerFrameData& inout, const RuntimeModuleTickData& tickData)
	{
		if (auto* landscape = getLandscape())
		{
			inout.landscape.bLandscapeValid = true;
			inout.landscape.terrainObjectId = landscape->getNode()->getId();
			inout.landscape.bLandscapeSelect = landscape->getNode()->editorSelected();
			inout.landscape.lodCount = landscape->getLODCount();

			// Per terrain
			inout.landscape.terrainDimension = landscape->getRenderDimension();
			inout.landscape.offsetX = landscape->getOffset().x;
			inout.landscape.offsetY = landscape->getOffset().y;
			inout.landscape.minHeight = landscape->getMinHeight();
			inout.landscape.maxHeight = landscape->getMaxHeight();

			inout.landscape.heightmapUUID = landscape->getGPUImage()->getBindlessIndex();
			inout.landscape.hzbUUID = landscape->getHeightMapHZB()->getImage().getOrCreateView().srvBindless;

			// Center position.
			vec3 rayO = math::vec3(inout.camWorldPos);
			vec3 ray = -math::normalize(inout.sunLightInfo.direction);

			inout.landscape.terrainShadowValid = (ray.y > 0.0f) && (rayO.y > inout.landscape.minHeight);
			if(inout.landscape.terrainShadowValid)
			{
				// Current just use half dimension as extent.
				float shadowExtent = float(inout.landscape.terrainDimension / 2);


				float theta = ray.y;
				float tanTheta = tan(theta);
				float d_o = shadowExtent * tanTheta;

				float d0 = ( inout.landscape.maxHeight - inout.landscape.minHeight) / abs(ray.y) + d_o;

				// Then we need to compute cloud's project matrix.
				math::mat4 shadowView = math::lookAtRH(
					// Camera look from cloud top position.
					rayO + ray * d0,
					// Look at center of terrain.
					rayO,
					// Up direction. 
					math::vec3{ 0.0f, 1.0f, 0.0f } // Y up.
				);

				math::mat4 shadowProj = math::orthoRH_ZO(
					-shadowExtent,
					 shadowExtent,
					-shadowExtent,
					 shadowExtent,
					 d0,     // Also reverse z for cloud shadow depth.
					 1e-5f
				);



				// Texel align.
				const float sMapSize = float(getShadowDepthDimTerrain());
				mat4 shadowViewProjMatrix = shadowProj * shadowView;
				vec4 shadowOrigin = vec4(0.0f, 0.0f, 0.0f, 1.0f);
				shadowOrigin = shadowViewProjMatrix * shadowOrigin;
				shadowOrigin *= (sMapSize / 2.0f);

				// Move to center uv pos
				vec3 roundedOrigin = round(vec3(shadowOrigin));
				vec3 roundOffset = roundedOrigin - vec3(shadowOrigin);
				roundOffset = roundOffset * (2.0f / sMapSize);
				roundOffset.z = 0.0f;

				// Push back round offset data to project matrix.
				shadowProj[3][0] += roundOffset.x;
				shadowProj[3][1] += roundOffset.y;

				// Final proj view matrix
				inout.landscape.sunFarShadowViewProj = shadowProj * shadowView;
				inout.landscape.sunFarShadowViewProjInverse = math::inverse(inout.landscape.sunFarShadowViewProj);
			}

		}
		else
		{
			inout.landscape.bLandscapeValid = false;
			inout.landscape.terrainObjectId = 0;
			inout.landscape.bLandscapeSelect = false;
			inout.landscape.terrainShadowValid = false;
		}


		if (auto* skyComp = getSkyComponent())
		{
			inout.bSkyComponentValid = true;

			inout.sunLightInfo = skyComp->getSunInfo();
			inout.atmosphere = skyComp->getAtmosphereParameters();
			inout.skyComponentSceneNodeId = skyComp->getNode()->getId();
			inout.bSkyComponentSelected = skyComp->getNode()->editorSelected();

			inout.cloud = skyComp->getCloudParameters();

#if AP1_COLOR_SPACE
			inout.sunLightInfo.color = convertSRGBColorSpace(inout.sunLightInfo.color);
			inout.sunLightInfo.shadowColor = convertSRGBColorSpace(inout.sunLightInfo.shadowColor);
			inout.atmosphere.absorptionColor = convertSRGBColorSpace(inout.atmosphere.absorptionColor);
			inout.atmosphere.rayleighScatteringColor = convertSRGBColorSpace(inout.atmosphere.rayleighScatteringColor);
			inout.atmosphere.mieScatteringColor = convertSRGBColorSpace(inout.atmosphere.mieScatteringColor);
			inout.atmosphere.mieAbsColor = convertSRGBColorSpace(inout.atmosphere.mieAbsColor);
			inout.atmosphere.mieAbsorption = convertSRGBColorSpace(inout.atmosphere.mieAbsorption);
			inout.atmosphere.groundAlbedo = convertSRGBColorSpace(inout.atmosphere.groundAlbedo);
			inout.cloud.cloudAlbedo = convertSRGBColorSpace(inout.cloud.cloudAlbedo);
#endif
			// Update cloud post info.
			{
				// Update cam world pos, convert to atmosphere unit.
				inout.cloud.camWorldPos = math::vec3(inout.camWorldPos) * 0.001f + glm::vec3{ 0.0f, inout.atmosphere.bottomRadius + kAtmosphereCameraOffset, 0.0f }; // To km.

				// Ray intersect outer sphere.
				vec3 ray = -math::normalize(inout.sunLightInfo.direction);
				float intersectT = raySphereIntersectInside(inout.cloud.camWorldPos, ray, vec3(0.0f), inout.atmosphere.topRadius);

				float startT = inout.cloud.cloudAreaThickness;

				// Then we need to compute cloud's project matrix.
				math::mat4 shadowView = math::lookAtRH(
					// Camera look from cloud top position.
					inout.cloud.camWorldPos + ray * intersectT,
					// Camera look at earth center.
					inout.cloud.camWorldPos,
					// Up direction. 
					math::vec3{ 0.0f, 1.0f, 0.0f } // Y up.
				);

				math::mat4 shadowProj = math::orthoRH_ZO(
					-inout.cloud.cloudShadowExtent,
					 inout.cloud.cloudShadowExtent,
					-inout.cloud.cloudShadowExtent,
					 inout.cloud.cloudShadowExtent,
					 intersectT,     // Also reverse z for cloud shadow depth.
					 1e-2f
				);

				inout.cloud.cloudSpaceViewProject = shadowProj * shadowView;
				inout.cloud.cloudSpaceViewProjectInverse = math::inverse(inout.cloud.cloudSpaceViewProject);
			}
		}
		else
		{
			inout.bSkyComponentValid = false;
			inout.skyComponentSceneNodeId = 0;
			inout.bSkyComponentSelected = false;
		}

		if (auto* postComp = getPostprocessComponent())
		{
			inout.postprocessing = computePostprocessSettingDetail(inout, postComp->getSetting(), tickData.deltaTime);
		}
		else
		{
			inout.postprocessing =
				computePostprocessSettingDetail(inout, defaultPostprocessVolumeSetting(), tickData.deltaTime);
		}
	}

	void RenderScene::drawAABB(vec3 center, vec3 extents)
	{
		float color = 1.0f;

		const vec4 p0 = vec4(center + extents * vec3( 1.0,  1.0,  1.0), color);
		const vec4 p1 = vec4(center + extents * vec3(-1.0,  1.0,  1.0), color);
		const vec4 p2 = vec4(center + extents * vec3( 1.0, -1.0,  1.0), color);
		const vec4 p3 = vec4(center + extents * vec3( 1.0,  1.0, -1.0), color);
		const vec4 p4 = vec4(center + extents * vec3(-1.0, -1.0,  1.0), color);
		const vec4 p5 = vec4(center + extents * vec3( 1.0, -1.0, -1.0), color);
		const vec4 p6 = vec4(center + extents * vec3(-1.0,  1.0, -1.0), color);
		const vec4 p7 = vec4(center + extents * vec3(-1.0, -1.0, -1.0), color);

		m_perFrameCollect.drawLineCPU.push_back(p0);
		m_perFrameCollect.drawLineCPU.push_back(p1);
		m_perFrameCollect.drawLineCPU.push_back(p0);
		m_perFrameCollect.drawLineCPU.push_back(p2);
		m_perFrameCollect.drawLineCPU.push_back(p0);
		m_perFrameCollect.drawLineCPU.push_back(p3);

		m_perFrameCollect.drawLineCPU.push_back(p6);
		m_perFrameCollect.drawLineCPU.push_back(p7);
		m_perFrameCollect.drawLineCPU.push_back(p5);
		m_perFrameCollect.drawLineCPU.push_back(p7);
		m_perFrameCollect.drawLineCPU.push_back(p4);
		m_perFrameCollect.drawLineCPU.push_back(p7);

		m_perFrameCollect.drawLineCPU.push_back(p1);
		m_perFrameCollect.drawLineCPU.push_back(p6);
		m_perFrameCollect.drawLineCPU.push_back(p2);
		m_perFrameCollect.drawLineCPU.push_back(p5);
		m_perFrameCollect.drawLineCPU.push_back(p1);
		m_perFrameCollect.drawLineCPU.push_back(p4);
		m_perFrameCollect.drawLineCPU.push_back(p2);
		m_perFrameCollect.drawLineCPU.push_back(p4);

		m_perFrameCollect.drawLineCPU.push_back(p3);
		m_perFrameCollect.drawLineCPU.push_back(p6);
		m_perFrameCollect.drawLineCPU.push_back(p3);
		m_perFrameCollect.drawLineCPU.push_back(p5);
	}

	void RenderScene::drawAABBminMax(vec3 min, vec3 max)
	{
		auto center = (min + max) * vec3(0.5);
		auto extent = max - center;

		drawAABB(center, extent);
	}

	void RenderScene::tlasPrepare(const RuntimeModuleTickData& tickData, Scene* scene, VkCommandBuffer cmd)
	{
		// When instance is empty, destroy tlas and pre-return.
		if (m_perFrameCollect.cacheASInstances.empty())
		{
			unvalidTLAS();
			return;
		}

		// Sometimes need rebuild. clear here.
		if (m_perFrameCollect.bTLASFullRebuild)
		{
			unvalidTLAS();
		}

		// Update or build TLAS.
		m_tlas.buildTlas(cmd, m_perFrameCollect.cacheASInstances, m_tlas.isInit());
	}

	void AABBBounds::transform(mat4 transformMatrix)
	{
		auto center = (min + max) * vec3(0.5);
		auto extents = max - center;

		vec4 p[8];

		p[0] = transformMatrix * vec4(center + extents * vec3( 1.0,  1.0,  1.0), 1.0f);
		p[1] = transformMatrix * vec4(center + extents * vec3(-1.0,  1.0,  1.0), 1.0f);
		p[2] = transformMatrix * vec4(center + extents * vec3( 1.0, -1.0,  1.0), 1.0f);
		p[3] = transformMatrix * vec4(center + extents * vec3( 1.0,  1.0, -1.0), 1.0f);
		p[4] = transformMatrix * vec4(center + extents * vec3(-1.0, -1.0,  1.0), 1.0f);
		p[5] = transformMatrix * vec4(center + extents * vec3( 1.0, -1.0, -1.0), 1.0f);
		p[6] = transformMatrix * vec4(center + extents * vec3(-1.0,  1.0, -1.0), 1.0f);
		p[7] = transformMatrix * vec4(center + extents * vec3(-1.0, -1.0, -1.0), 1.0f);

		AABBBounds newAABB{ };
		for (const auto& pV : p)
		{
			vec3 newP = pV;
			newAABB.min = math::min(newAABB.min, newP);
			newAABB.max = math::max(newAABB.max, newP);
		}

		*this = newAABB;
	}
}