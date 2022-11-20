#include "Pch.h"
#include "FormatDefine.h"
#include "../AssetCommon.h"
#include "UnicodeUtil.h"

using namespace Flower;
using namespace Importer;
using namespace Importer::PMX;

//
// Copyright(c) 2016-2017 benikabocha.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

bool PMX::readFile(::File& pmx, const char* fileName)
{
	if (!std::filesystem::exists(fileName))
	{
		return false;
	}

	struct AutoDeleteBuffer
	{
		char* buffer = nullptr;
		~AutoDeleteBuffer()
		{
			if (buffer)
			{
				delete[] buffer;
			}
		}
	} autoDeleteBuffer;
	
	uint32_t fileLength = 0;
	{
		std::ifstream is;
		is.open(fileName, std::ios::binary);

		is.seekg(0, std::ios::end);
		fileLength = is.tellg();
		is.seekg(0, std::ios::beg);

		if (fileLength <= 0)
		{
			is.close();
			return false;
		}

		autoDeleteBuffer.buffer = new char[fileLength];
		is.read(autoDeleteBuffer.buffer, fileLength);
		is.close();
	}
	
	uint32_t stepCount = 0;
	char* filePtr = autoDeleteBuffer.buffer;

	#define READ_START_CHECK() if (stepCount >= fileLength) { LOG_ERROR("File step error on start check"); return false; }
	#define READ_END_CHECK()   return stepCount <= fileLength;

	auto readAndStep = [&]<typename T>(T* dest) -> bool
	{
		READ_START_CHECK();

		memcpy((void*)dest, (void*)filePtr, sizeof(T));

		filePtr += sizeof(T);
		stepCount += sizeof(T);

		READ_END_CHECK();
	};

	auto readArrayAndStep = [&]<typename T>(T* destArray, size_t count) -> bool
	{
		READ_START_CHECK();

		memcpy((void*)destArray, (void*)filePtr, sizeof(T) * count);

		filePtr += sizeof(T) * count;
		stepCount += sizeof(T) * count;

		READ_END_CHECK();
	};

	auto readStringAndStep = [&](std::string* dest) -> bool
	{
		READ_START_CHECK();

		uint32_t bufferSize;
		if (!readAndStep(&bufferSize))
		{
			LOG_ERROR("readAndStep string buffer size fail.");
			return false;
		}

		if (bufferSize > 0)
		{
			if (pmx.metaData.encodeType == 0)
			{
				// utf-16
				std::u16string utf16Str(bufferSize / 2, u'\0');
				if (!readArrayAndStep(&utf16Str[0], utf16Str.size()))
				{
					return false;
				}
				
				// convert to utf-8.
				if (!saba::ConvU16ToU8(utf16Str, *dest))
				{
					return false;
				}
			}
			else if(pmx.metaData.encodeType == 1)
			{
				// utf-8
				std::string utf8Str(bufferSize, '\0');
				
				if (!readArrayAndStep(&utf8Str[0], bufferSize))
				{
					return false;
				}

				*dest = utf8Str;
			}
			else
			{
				CHECK_ENTRY();
			}
		}

		READ_END_CHECK();
	};

	auto readIndexAndStep = [&](int32_t* index, uint8_t indexSize)
	{
		READ_START_CHECK();
		switch (indexSize)
		{
		case 1:
		{
			uint8_t idx;
			readAndStep(&idx);
			if (idx != 0xFF)
			{
				*index = (int32_t)idx;
			}
			else
			{
				*index = -1;
			}
		}
		break;
		case 2:
		{
			uint16_t idx;
			readAndStep(&idx);
			if (idx != 0xFFFF)
			{
				*index = (int32_t)idx;
			}
			else
			{
				*index = -1;
			}
		}
		break;
		case 4:
		{
			uint32_t idx;
			readAndStep(&idx);
			*index = (int32_t)idx; // Is this safe?
		}
		break;
		default:
			CHECK_ENTRY();
			return false;
		}
		READ_END_CHECK();
	};

	auto readMetaAndStep = [&]()
	{
		READ_START_CHECK();

		auto& meta = pmx.metaData;

		readAndStep(&meta.magic);
		readAndStep(&meta.version);
		readAndStep(&meta.dataSize);
		readAndStep(&meta.encodeType);
		readAndStep(&meta.appendUvNum);
		readAndStep(&meta.vertexIndexSize);
		readAndStep(&meta.textureIndexSize);
		readAndStep(&meta.materialIndexSize);
		readAndStep(&meta.boneIndexSize);
		readAndStep(&meta.morphIndexSize);
		readAndStep(&meta.rigidbodyIndexSize);

		READ_END_CHECK();
	};

	auto readInfoAndStep = [&]()
	{
		READ_START_CHECK();

		auto& info = pmx.modelInfo;

		readStringAndStep(&info.localName);
		readStringAndStep(&info.globalName);
		readStringAndStep(&info.localComment);
		readStringAndStep(&info.globalComment);

		READ_END_CHECK();
	};

	auto readVertexAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t vertexCount;
		if (!readAndStep(&vertexCount))
		{
			LOG_ERROR("readAndStep vertex count fail.");
			return false;
		}

		auto& vertices = pmx.vertices;
		vertices.resize(vertexCount);
		for (auto& vertex : vertices)
		{
			readAndStep(&vertex.position);
			readAndStep(&vertex.normal);
			readAndStep(&vertex.uv0);
			for (uint8_t i = 0; i < pmx.metaData.appendUvNum; i++)
			{
				readAndStep(&vertex.appendUvs[i]);
			}

			readAndStep(&vertex.weightType);

			switch (vertex.weightType)
			{
			case EVertexWeight::BDEF1:
				readIndexAndStep(&vertex.boneIndices[0], pmx.metaData.boneIndexSize);
				break;
			case EVertexWeight::BDEF2:
				readIndexAndStep(&vertex.boneIndices[0], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[1], pmx.metaData.boneIndexSize);
				readAndStep(&vertex.boneWeights[0]);
				break;
			case EVertexWeight::BDEF4:
				readIndexAndStep(&vertex.boneIndices[0], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[1], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[2], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[3], pmx.metaData.boneIndexSize);
				readAndStep(&vertex.boneWeights[0]);
				readAndStep(&vertex.boneWeights[1]);
				readAndStep(&vertex.boneWeights[2]);
				readAndStep(&vertex.boneWeights[3]);
				break;
			case EVertexWeight::SDEF:
				readIndexAndStep(&vertex.boneIndices[0], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[1], pmx.metaData.boneIndexSize);
				readAndStep(&vertex.boneWeights[0]);
				readAndStep(&vertex.sdefC);
				readAndStep(&vertex.sdefR0);
				readAndStep(&vertex.sdefR1);
				break;
			case EVertexWeight::QDEF:
				readIndexAndStep(&vertex.boneIndices[0], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[1], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[2], pmx.metaData.boneIndexSize);
				readIndexAndStep(&vertex.boneIndices[3], pmx.metaData.boneIndexSize);
				readAndStep(&vertex.boneWeights[0]);
				readAndStep(&vertex.boneWeights[1]);
				readAndStep(&vertex.boneWeights[3]);
				readAndStep(&vertex.boneWeights[4]);
				break;
			default:
				CHECK_ENTRY();
				return false;
			}
			readAndStep(&vertex.edgeMag);
		}

		READ_END_CHECK();
	};

	auto readFaceAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t faceCount = 0;
		if (!readAndStep(&faceCount))
		{
			LOG_ERROR("Read face count error.");
			return false;
		}

		int32_t cacheFaceCount = faceCount;
		faceCount /= 3;

		CHECK(cacheFaceCount == faceCount * 3);

		pmx.faces.resize(faceCount);

		switch (pmx.metaData.vertexIndexSize)
		{
		case 1:
		{
			std::vector<uint8_t> vertices(faceCount * 3);
			readArrayAndStep(vertices.data(), vertices.size());
			for (int32_t faceIdx = 0; faceIdx < faceCount; faceIdx++)
			{
				pmx.faces[faceIdx].vertices[0] = vertices[faceIdx * 3 + 0];
				pmx.faces[faceIdx].vertices[1] = vertices[faceIdx * 3 + 1];
				pmx.faces[faceIdx].vertices[2] = vertices[faceIdx * 3 + 2];
			}
		}
		break;
		case 2:
		{
			std::vector<uint16_t> vertices(faceCount * 3);
			readArrayAndStep(vertices.data(), vertices.size());
			for (int32_t faceIdx = 0; faceIdx < faceCount; faceIdx++)
			{
				pmx.faces[faceIdx].vertices[0] = vertices[faceIdx * 3 + 0];
				pmx.faces[faceIdx].vertices[1] = vertices[faceIdx * 3 + 1];
				pmx.faces[faceIdx].vertices[2] = vertices[faceIdx * 3 + 2];
			}
		}
		break;
		case 4:
		{
			std::vector<uint32_t> vertices(faceCount * 3);
			readArrayAndStep(vertices.data(), vertices.size());
			for (int32_t faceIdx = 0; faceIdx < faceCount; faceIdx++)
			{
				pmx.faces[faceIdx].vertices[0] = vertices[faceIdx * 3 + 0];
				pmx.faces[faceIdx].vertices[1] = vertices[faceIdx * 3 + 1];
				pmx.faces[faceIdx].vertices[2] = vertices[faceIdx * 3 + 2];
			}
		}
		break;
		default:
			return false;
		}

		READ_END_CHECK();
	};

	auto readTextureAndStep = [&]() 
	{
		READ_START_CHECK();

		int32_t texCount = 0;
		if (!readAndStep(&texCount))
		{
			LOG_ERROR("Read texture count error.");
			return false;
		}

		pmx.textures.resize(texCount);

		for (auto& tex : pmx.textures)
		{
			readStringAndStep(&tex.textureName);
		}

		READ_END_CHECK();
	};

	auto readMaterialAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t matCount = 0;
		if (!readAndStep(&matCount))
		{
			LOG_ERROR("Reading material count fail.");
			return false;
		}

		pmx.materials.resize(matCount);

		for (auto& mat : pmx.materials)
		{
			readStringAndStep(&mat.localName);
			readStringAndStep(&mat.globalName);

			readAndStep(&mat.diffuse);
			readAndStep(&mat.specular);
			readAndStep(&mat.specularPower);
			readAndStep(&mat.ambient);

			readAndStep(&mat.drawMode);

			readAndStep(&mat.edgeColor);
			readAndStep(&mat.edgeSize);

			readIndexAndStep(&mat.textureIndex, pmx.metaData.textureIndexSize);
			readIndexAndStep(&mat.sphereTextureIndex, pmx.metaData.textureIndexSize);
			readAndStep(&mat.sphereMode);

			readAndStep(&mat.toonMode);
			if (mat.toonMode == EToonMode::Separate)
			{
				readIndexAndStep(&mat.toonTextureIndex, pmx.metaData.textureIndexSize);
			}
			else if (mat.toonMode == EToonMode::Common)
			{
				uint8_t toonIndex;
				readAndStep(&toonIndex);
				mat.toonTextureIndex = (int32_t)toonIndex;
			}
			else
			{
				return false;
			}

			readStringAndStep(&mat.memo);

			readAndStep(&mat.numFaceVertices);
		}

		READ_END_CHECK();
	};

	auto readBoneAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t boneCount;
		if (!readAndStep(&boneCount))
		{
			LOG_ERROR("Reading bone count error.");
			return false;
		}

		pmx.bones.resize(boneCount);

		for (auto& bone : pmx.bones)
		{
			readStringAndStep(&bone.localName);
			readStringAndStep(&bone.globalName);

			readAndStep(&bone.position);
			readIndexAndStep(&bone.parentBoneIndex, pmx.metaData.boneIndexSize);
			readAndStep(&bone.deformDepth);

			readAndStep(&bone.boneFlag);

			if (((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::TargetShowMode) == 0)
			{
				readAndStep(&bone.positionOffset);
			}
			else
			{
				readIndexAndStep(&bone.linkBoneIndex, pmx.metaData.boneIndexSize);
			}

			if (((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::AppendRotate) ||
				((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::AppendTranslate))
			{
				readIndexAndStep(&bone.appendBoneIndex, pmx.metaData.boneIndexSize);
				readAndStep(&bone.appendWeight);
			}

			if ((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::FixedAxis)
			{
				readAndStep(&bone.fixedAxis);
			}

			if ((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::LocalAxis)
			{
				readAndStep(&bone.localXAxis);
				readAndStep(&bone.localZAxis);
			}

			if ((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::DeformOuterParent)
			{
				readAndStep(&bone.keyValue);
			}

			if ((uint16_t)bone.boneFlag & (uint16_t)EBoneFlags::IK)
			{
				readIndexAndStep(&bone.ikTargetBoneIndex, pmx.metaData.boneIndexSize);
				readAndStep(&bone.ikIterationCount);
				readAndStep(&bone.ikLimit);

				int32_t linkCount;
				if (!readAndStep(&linkCount))
				{
					return false;
				}

				bone.ikLinks.resize(linkCount);
				for (auto& ikLink : bone.ikLinks)
				{
					readIndexAndStep(&ikLink.ikBoneIndex, pmx.metaData.boneIndexSize);
					readAndStep(&ikLink.enableLimit);

					if (ikLink.enableLimit != 0)
					{
						readAndStep(&ikLink.limitMin);
						readAndStep(&ikLink.limitMax);
					}
				}
			}
		}

		READ_END_CHECK();
	};

	auto readMorphAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t morphCount;
		if (!readAndStep(&morphCount))
		{
			LOG_ERROR("Reading morph count error.");
			return false;
		}

		pmx.morphs.resize(morphCount);

		for (auto& morph : pmx.morphs)
		{
			readStringAndStep(&morph.localName);
			readStringAndStep(&morph.globalName);

			readAndStep(&morph.controlPanel);
			readAndStep(&morph.morphType);

			int32_t dataCount;
			if (!readAndStep(&dataCount))
			{
				return false;
			}

			if (morph.morphType == EMorphType::Position)
			{
				morph.positionMorph.resize(dataCount);
				for (auto& data : morph.positionMorph)
				{
					readIndexAndStep(&data.vertexIndex, pmx.metaData.vertexIndexSize);
					readAndStep(&data.position);
				}
			}
			else if (morph.morphType == EMorphType::UV ||
				morph.morphType == EMorphType::AddUV1 ||
				morph.morphType == EMorphType::AddUV2 ||
				morph.morphType == EMorphType::AddUV3 ||
				morph.morphType == EMorphType::AddUV4
				)
			{
				morph.uvMorph.resize(dataCount);
				for (auto& data : morph.uvMorph)
				{
					readIndexAndStep(&data.vertexIndex, pmx.metaData.vertexIndexSize);
					readAndStep(&data.uv);
				}
			}
			else if (morph.morphType == EMorphType::Bone)
			{
				morph.boneMorph.resize(dataCount);
				for (auto& data : morph.boneMorph)
				{
					readIndexAndStep(&data.boneIndex, pmx.metaData.boneIndexSize);
					readAndStep(&data.position);
					readAndStep(&data.quaternion);
				}
			}
			else if (morph.morphType == EMorphType::Material)
			{
				morph.materialMorph.resize(dataCount);
				for (auto& data : morph.materialMorph)
				{
					readIndexAndStep(&data.materialIndex, pmx.metaData.materialIndexSize);
					readAndStep(&data.opType);
					readAndStep(&data.diffuse);
					readAndStep(&data.specular);
					readAndStep(&data.specularPower);
					readAndStep(&data.ambient);
					readAndStep(&data.edgeColor);
					readAndStep(&data.edgeSize);
					readAndStep(&data.textureFactor);
					readAndStep(&data.sphereTextureFactor);
					readAndStep(&data.toonTextureFactor);
				}
			}
			else if (morph.morphType == EMorphType::Group)
			{
				morph.groupMorph.resize(dataCount);
				for (auto& data : morph.groupMorph)
				{
					readIndexAndStep(&data.morphIndex, pmx.metaData.morphIndexSize);
					readAndStep(&data.weight);
				}
			}
			else if (morph.morphType == EMorphType::Flip)
			{
				morph.flipMorph.resize(dataCount);
				for (auto& data : morph.flipMorph)
				{
					readIndexAndStep(&data.morphIndex, pmx.metaData.morphIndexSize);
					readAndStep(&data.weight);
				}
			}
			else if (morph.morphType == EMorphType::Impluse)
			{
				morph.impulseMorph.resize(dataCount);
				for (auto& data : morph.impulseMorph)
				{
					readIndexAndStep(&data.rigidbodyIndex, pmx.metaData.rigidbodyIndexSize);
					readAndStep(&data.localFlag);
					readAndStep(&data.translateVelocity);
					readAndStep(&data.rotateTorque);
				}
			}
			else
			{
				LOG_ERROR("Unsupported Morph Type:[{}]", (int)morph.morphType);
				return false;
			}
		}

		READ_END_CHECK();
	};

	auto readDisplayFrameAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t displayFrameCount;
		if (!readAndStep(&displayFrameCount))
		{
			return false;
		}

		pmx.displayFrames.resize(displayFrameCount);

		for (auto& displayFrame : pmx.displayFrames)
		{
			readStringAndStep(&displayFrame.localName);
			readStringAndStep(&displayFrame.globalName);

			readAndStep(&displayFrame.flag);
			int32_t targetCount;
			if (!readAndStep(&targetCount))
			{
				return false;
			}
			displayFrame.targets.resize(targetCount);
			for (auto& target : displayFrame.targets)
			{
				readAndStep(&target.type);
				if (target.type == DispalyFrame::ETargetType::BoneIndex)
				{
					readIndexAndStep(&target.index, pmx.metaData.boneIndexSize);
				}
				else if (target.type == DispalyFrame::ETargetType::MorphIndex)
				{
					readIndexAndStep(&target.index, pmx.metaData.morphIndexSize);
				}
				else
				{
					return false;
				}
			}
		}

		READ_END_CHECK();
	};

	auto readRigidbodyAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t rbCount;
		if (!readAndStep(&rbCount))
		{
			return false;
		}

		pmx.rigidbodies.resize(rbCount);

		for (auto& rb : pmx.rigidbodies)
		{
			readStringAndStep(&rb.localName);
			readStringAndStep(&rb.globalName);

			readIndexAndStep(&rb.boneIndex, pmx.metaData.boneIndexSize);
			readAndStep(&rb.group);
			readAndStep(&rb.collisionGroup);

			readAndStep(&rb.shape);
			readAndStep(&rb.shapeSize);

			readAndStep(&rb.translate);
			readAndStep(&rb.rotate);

			readAndStep(&rb.mass);
			readAndStep(&rb.translateDimmer);
			readAndStep(&rb.rotateDimmer);
			readAndStep(&rb.repulsion);
			readAndStep(&rb.friction);

			readAndStep(&rb.op);
		}

		READ_END_CHECK();
	};

	auto readJointAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t jointCount;
		if (!readAndStep(&jointCount))
		{
			return false;
		}

		pmx.joints.resize(jointCount);

		for (auto& joint : pmx.joints)
		{
			readStringAndStep(&joint.localName);
			readStringAndStep(&joint.globalName);

			readAndStep(&joint.type);
			readIndexAndStep(&joint.rigidbodyAIndex, pmx.metaData.rigidbodyIndexSize);
			readIndexAndStep(&joint.rigidbodyBIndex, pmx.metaData.rigidbodyIndexSize);

			readAndStep(&joint.translate);
			readAndStep(&joint.rotate);

			readAndStep(&joint.translateLowerLimit);
			readAndStep(&joint.translateUpperLimit);
			readAndStep(&joint.rotateLowerLimit);
			readAndStep(&joint.rotateUpperLimit);

			readAndStep(&joint.springTranslateFactor);
			readAndStep(&joint.springRotateFactor);
		}

		READ_END_CHECK();
	};

	auto readSoftbodyAndStep = [&]()
	{
		READ_START_CHECK();

		int32_t sbCount;
		if (!readAndStep(&sbCount))
		{
			return false;
		}

		pmx.softbodies.resize(sbCount);

		for (auto& sb : pmx.softbodies)
		{
			readStringAndStep(&sb.localName);
			readStringAndStep(&sb.globalName);

			readAndStep(&sb.type);

			readIndexAndStep(&sb.materialIndex, pmx.metaData.materialIndexSize);

			readAndStep(&sb.group);
			readAndStep(&sb.collisionGroup);

			readAndStep(&sb.flag);

			readAndStep(&sb.BLinkLength);
			readAndStep(&sb.numClusters);

			readAndStep(&sb.totalMass);
			readAndStep(&sb.collisionMargin);

			readAndStep(&sb.aeroModel);

			readAndStep(&sb.VCF);
			readAndStep(&sb.DP);
			readAndStep(&sb.DG);
			readAndStep(&sb.LF);
			readAndStep(&sb.PR);
			readAndStep(&sb.VC);
			readAndStep(&sb.DF);
			readAndStep(&sb.MT);
			readAndStep(&sb.CHR);
			readAndStep(&sb.KHR);
			readAndStep(&sb.SHR);
			readAndStep(&sb.AHR);

			readAndStep(&sb.SRHR_CL);
			readAndStep(&sb.SKHR_CL);
			readAndStep(&sb.SSHR_CL);
			readAndStep(&sb.SR_SPLT_CL);
			readAndStep(&sb.SK_SPLT_CL);
			readAndStep(&sb.SS_SPLT_CL);

			readAndStep(&sb.V_IT);
			readAndStep(&sb.P_IT);
			readAndStep(&sb.D_IT);
			readAndStep(&sb.C_IT);

			readAndStep(&sb.LST);
			readAndStep(&sb.AST);
			readAndStep(&sb.VST);

			int32_t arCount;
			if (!readAndStep(&arCount))
			{
				return false;
			}
			sb.anchorRigidbodies.resize(arCount);
			for (auto& ar : sb.anchorRigidbodies)
			{
				readIndexAndStep(&ar.rigidBodyIndex, pmx.metaData.rigidbodyIndexSize);
				readIndexAndStep(&ar.vertexIndex, pmx.metaData.vertexIndexSize);
				readAndStep(&ar.nearMode);
			}

			int32_t pvCount;
			if (!readAndStep(&pvCount))
			{
				return false;
			}
			sb.pinVertexIndices.resize(pvCount);
			for (auto& pv : sb.pinVertexIndices)
			{
				readIndexAndStep(&pv, pmx.metaData.vertexIndexSize);
			}
		}

		READ_END_CHECK();
	};

	auto readFile = [&]()
	{
		if (!readMetaAndStep())
		{
			LOG_ERROR("read meta fail.");
			return false;
		}

		if (!readInfoAndStep())
		{
			LOG_ERROR("Read info fail.");
			return false;
		}

		if (!readVertexAndStep())
		{
			LOG_ERROR("Read vertex Fail.");
			return false;
		}

		if (!readFaceAndStep())
		{
			LOG_ERROR("Read face Fail.");
			return false;
		}

		if (!readTextureAndStep())
		{
			LOG_ERROR("Read texture fail.");
			return false;
		}

		if (!readMaterialAndStep())
		{
			LOG_ERROR("Read material Fail.");
			return false;
		}

		if (!readBoneAndStep())
		{
			LOG_ERROR("Read bone Fail.");
			return false;
		}

		if (!readMorphAndStep())
		{
			LOG_ERROR("Read Morph Fail.");
			return false;
		}

		if (!readDisplayFrameAndStep())
		{
			LOG_ERROR("Read DisplayFrame Fail.");
			return false;
		}

		if (!readRigidbodyAndStep())
		{
			LOG_ERROR("Read Rigidbody Fail.");
			return false;
		}

		if (!readJointAndStep())
		{
			LOG_ERROR("ReadJoint Fail.");
			return false;
		}

		if (stepCount < fileLength)
		{
			if (!readSoftbodyAndStep())
			{
				LOG_ERROR("ReadSoftbody Fail.");
				return false;
			}
		}

		return true;
	};

	return readFile();
}