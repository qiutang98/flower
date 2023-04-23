#pragma once
#include "util.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/GltfMaterial.h>
#include <assimp/material.h>

namespace engine
{
    // Import static mesh by assimp.
	class AssimpStaticMeshImporter
	{
	public:
		explicit AssimpStaticMeshImporter(const std::filesystem::path& inRawMeshPath)
			: m_bImportMaterials(false)
		{

		}

		explicit AssimpStaticMeshImporter(
			const std::filesystem::path& inRawMeshPath, 
			const std::filesystem::path& inProjectRootPath,
			const std::filesystem::path& saveMaterialsPath,
			const std::filesystem::path& saveTexturesPath)
			: m_rawMeshPath(inRawMeshPath)
			, m_projectRootPath(inProjectRootPath)
			, m_textureSavePath(saveTexturesPath)
			, m_materialSavePath(saveMaterialsPath)
			, m_bImportMaterials(true)
		{

		}

		void processNode(aiNode* node, const aiScene* scene)
		{
			for (unsigned int i = 0; i < node->mNumMeshes; i++)
			{
				aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
				m_subMeshInfos.push_back(processMesh(mesh, scene));
			}

			for (unsigned int i = 0; i < node->mNumChildren; i++)
			{
				processNode(node->mChildren[i], scene);
			}
		}

		// NOTE: Exist one warning here, we use move below avoid vector copy.
		const std::vector<StaticMeshSubMesh>& getSubmeshInfo() const;
		const size_t getIndicesCount() const;
		const size_t getVerticesCount() const;

		const std::vector<VertexIndexType>& getIndices() const;
		const std::vector<VertexTangent>& getTangents() const;
		const std::vector<VertexUv0>& getUv0s() const;
		const std::vector<VertexNormal>& getNormals() const;
		const std::vector<VertexPosition>& getPositions() const;

		//
		std::vector<VertexIndexType>&& moveIndices();
		std::vector<VertexTangent>&& moveTangents();
		std::vector<VertexUv0>&& moveUv0s();
		std::vector<VertexNormal>&& moveNormals();
		std::vector<VertexPosition>&& movePositions();

	private:
		StaticMeshSubMesh processMesh(aiMesh* mesh, const aiScene* scene);

	private:
		bool m_bImportMaterials;

		// Raw mesh path.
		std::filesystem::path m_rawMeshPath;
		std::filesystem::path m_projectRootPath;
		std::filesystem::path m_textureSavePath;
		std::filesystem::path m_materialSavePath;

		// Submeshes info.
		std::vector<StaticMeshSubMesh> m_subMeshInfos = { };

		// Indices.
		std::vector<VertexIndexType> m_indices = { };

		// Vertices.
		std::vector<VertexPosition> m_positions = { };
		std::vector<VertexTangent> m_tangents = { };
		std::vector<VertexUv0> m_uv0s = { };
		std::vector<VertexNormal> m_normals = { };

		std::unordered_map<std::filesystem::path, UUID> m_texPathUUIDMap{ };
		std::unordered_map<std::filesystem::path, UUID> m_materialPathUUIDMap { };
	};
}