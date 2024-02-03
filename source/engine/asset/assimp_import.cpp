#include "assimp_import.h"
#include "asset_material.h"
#include "asset_texture.h"

#include <execution>
#include "asset_manager.h"

#pragma warning(disable: 4172)

namespace engine
{
    void AssimpStaticMeshImporter::fillMeshAssetMeta(AssetStaticMesh& mesh) const
    {
        mesh.m_subMeshes     = getSubmeshInfo();
        mesh.m_indicesCount  = getIndicesCount();
        mesh.m_verticesCount = getVerticesCount();

        mesh.m_minPosition = vec3(std::numeric_limits<float>::max());
        mesh.m_maxPosition = vec3(std::numeric_limits<float>::lowest());

        for (const auto& subMesh : mesh.m_subMeshes)
        {
            const auto minPos = subMesh.bounds.origin - subMesh.bounds.extents;
            const auto maxPos = subMesh.bounds.origin + subMesh.bounds.extents;

            mesh.m_maxPosition = math::max(mesh.m_maxPosition, maxPos);
            mesh.m_minPosition = math::min(mesh.m_minPosition, minPos);
        }
    }

    const std::vector<StaticMeshSubMesh>& AssimpStaticMeshImporter::getSubmeshInfo() const { return m_subMeshInfos; }
    const size_t AssimpStaticMeshImporter::getIndicesCount() const { return m_indices.size(); }
    const size_t AssimpStaticMeshImporter::getVerticesCount() const { return m_positions.size(); }

    const std::vector<VertexIndexType>& AssimpStaticMeshImporter::getIndices() const { return m_indices; }
    const std::vector<VertexTangent>& AssimpStaticMeshImporter::getTangents() const { return m_tangents; }
    const std::vector<VertexNormal>& AssimpStaticMeshImporter::getNormals() const { return m_normals; }
    const std::vector<VertexUv0>& AssimpStaticMeshImporter::getUv0s() const { return m_uv0s; }
    const std::vector<VertexPosition>& AssimpStaticMeshImporter::getPositions() const { return m_positions; }

    //
    std::vector<VertexIndexType>&& AssimpStaticMeshImporter::moveIndices() { return std::move(m_indices); }
    std::vector<VertexTangent>&& AssimpStaticMeshImporter::moveTangents() { return std::move(m_tangents); }
    std::vector<VertexNormal>&& AssimpStaticMeshImporter::moveNormals() { return std::move(m_normals); }
    std::vector<VertexUv0>&& AssimpStaticMeshImporter::moveUv0s() { return std::move(m_uv0s); }
    std::vector<VertexPosition>&& AssimpStaticMeshImporter::movePositions() { return std::move(m_positions); }

    StaticMeshSubMesh AssimpStaticMeshImporter::processMesh(aiMesh* mesh, const aiScene* scene)
    {
        // Add a new submesh.
        StaticMeshSubMesh subMeshInfo
        { 
            .indicesStart = (uint32_t)m_indices.size(),
        };

        // load vertices.
        std::vector<VertexTangent> tangents(mesh->mNumVertices);
        std::vector<VertexNormal> normals(mesh->mNumVertices);
        std::vector<VertexUv0> uv0s(mesh->mNumVertices);
        std::vector<VertexPosition> positions(mesh->mNumVertices);
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            positions[i] = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };

            normals[i] = {mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z};

            // Uv.
            if (mesh->mTextureCoords[0])
            {
                uv0s[i] = {mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y};
            }
            else
            {
                uv0s[i] = { 0.0f, 0.0f };
            }

            if (mesh->mTangents)
            {
                // Tangent, need handle uv flip case.
                math::vec3 tangentLoaded{ mesh->mTangents[i].x,  mesh->mTangents[i].y,  mesh->mTangents[i].z };
                math::vec3 bitangentLoaded{ mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z };

                float signTangent = glm::sign(
                    glm::dot(glm::normalize(bitangentLoaded), glm::normalize(glm::cross(normals[i], tangentLoaded))));

                tangents[i] = { tangentLoaded, signTangent };
            }
        }

        // Load indices.
        std::vector<VertexIndexType> indices;
        indices.reserve(mesh->mNumFaces * 3);

        const uint32_t indexOffset = static_cast<uint32_t>(m_positions.size());
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
            {
                indices.push_back(indexOffset + face.mIndices[j]);
            }
        }

        // Insert to data array.
        m_indices.insert(m_indices.end(), indices.begin(), indices.end());
        m_positions.insert(m_positions.end(), positions.begin(), positions.end());
        m_tangents.insert(m_tangents.end(), tangents.begin(), tangents.end());
        m_normals.insert(m_normals.end(), normals.begin(), normals.end());
        m_uv0s.insert(m_uv0s.end(), uv0s.begin(), uv0s.end());

        // Now fill indices count of the submesh.
        subMeshInfo.indicesCount = static_cast<uint32_t>(indices.size());

        // aabb bounds process.
        auto aabbExt = (mesh->mAABB.mMax - mesh->mAABB.mMin) * 0.5f;
        auto aabbCenter = aabbExt + mesh->mAABB.mMin;
        subMeshInfo.bounds = 
        {
            .origin = { aabbCenter.x, aabbCenter.y, aabbCenter.z },
            .extents = { aabbExt.x, aabbExt.y, aabbExt.z},
            .radius = math::distance(math::vec3(mesh->mAABB.mMax.x, mesh->mAABB.mMax.y, mesh->mAABB.mMax.z), subMeshInfo.bounds.origin)
        };

        // standard pbr texture prepare.
        aiString baseColorTextures{};
        aiString normalTextures{};
        aiString metalRoughnessTextures{};
        aiString aoTextures{};
        aiString emissiveTextures{};

        struct ImageImportConfig
        {
            UUID uuid;
            std::shared_ptr<AssetTextureImportConfig> config;
        };

        std::vector<ImageImportConfig> imagePendingConfigs{ };

        auto tryFetechTexture = [&](const char* pathIn, UUID& outUUID, bool bSrgb, float cutoff, ETextureFormat format)
        {
            std::filesystem::path texPath = m_rawMeshPath.parent_path() / utf8::utf8to16(pathIn);

            auto filename = texPath.filename();
            auto saveTexturePath = m_textureSavePath / filename.replace_extension();

            if (m_texPathUUIDMap[texPath].empty())
            {
                ImageImportConfig newConfig { };

                {
                    auto name = saveTexturePath.filename().u16string() + utf8::utf8to16(AssetTexture::getCDO()->getSuffix());
                    auto relativePathUtf8 = buildRelativePathUtf8(getAssetManager()->getProjectConfig().assetPath, saveTexturePath.parent_path());

                    newConfig.uuid = AssetSaveInfo(utf8::utf16to8(name), relativePathUtf8).getUUID();
                }
                
                newConfig.config = std::make_shared<AssetTextureImportConfig>();
                newConfig.config->path = { texPath, saveTexturePath };
                newConfig.config->bGenerateMipmap = true;
                newConfig.config->bSRGB = bSrgb;
                newConfig.config->alphaMipmapCutoff = cutoff;
                newConfig.config->format = format;

                m_texPathUUIDMap[texPath] = newConfig.uuid;
                imagePendingConfigs.push_back(newConfig);
            }
            else
            {
                LOG_TRACE("Texture {} is reusing in material.", pathIn);
            }

            outUUID = m_texPathUUIDMap.at(texPath);
        };

        if (mesh->mMaterialIndex >= 0 && m_bImportMaterials)
        {
            static const std::string materialPrefixName = "Material_";

            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            const std::string materialName = materialPrefixName + material->GetName().C_Str() + AssetMaterial::getCDO()->getSuffix();
            auto materialSavePath = m_materialSavePath / utf8::utf8to16(materialName);

            if(m_materialPathUUIDMap.contains(materialSavePath))
            {
                subMeshInfo.material = m_materialPathUUIDMap.at(materialSavePath);
            }
            else
            {
                AssetSaveInfo materialSaveInfo(materialName, 
                    buildRelativePathUtf8(getAssetManager()->getProjectConfig().assetPath, 
                        materialSavePath.parent_path()));
                auto newMaterial = getAssetManager()->createAsset<AssetMaterial>(materialSaveInfo).lock();
                newMaterial->markDirty();

                // Diffuse map, SRGB, 0.5 cut off alpha, BC3 format.
                newMaterial->cutoff = 0.5f;
                if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
                {
                    material->GetTexture(aiTextureType_DIFFUSE, 0, &baseColorTextures);
                    tryFetechTexture(baseColorTextures.C_Str(), newMaterial->baseColorTexture, true, 0.5f, ETextureFormat::BC3); 
                }
                {
                    C_STRUCT aiColor4D diffuse;
                    if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
                    {
                        newMaterial->baseColorMul.x = diffuse.r;
                        newMaterial->baseColorMul.y = diffuse.g;
                        newMaterial->baseColorMul.z = diffuse.b;
                        newMaterial->baseColorMul.w = diffuse.a;
                    }
                }


                // Normal map, linear, 1.0 cut off alpha, BC5 format.
                if (material->GetTextureCount(aiTextureType_HEIGHT) > 0)
                {
                    material->GetTexture(aiTextureType_HEIGHT, 0, &normalTextures);
                    tryFetechTexture(normalTextures.C_Str(), newMaterial->normalTexture, false, 1.0f, ETextureFormat::BC5);
                }

                // MetalRoughness,  linear, 1.0 cut off alpha, used in GB channel, don't care alpha, so use BC1 default.
                if (material->GetTextureCount(aiTextureType_SPECULAR) > 0)
                {
                    material->GetTexture(aiTextureType_SPECULAR, 0, &metalRoughnessTextures);
                    tryFetechTexture(metalRoughnessTextures.C_Str(), newMaterial->metalRoughnessTexture, false, 1.0f, ETextureFormat::BC1);
                }

                // Ambient,  linear, 1.0 cut off alpha.
                if (material->GetTextureCount(aiTextureType_AMBIENT) > 0)
                {
                    material->GetTexture(aiTextureType_AMBIENT, 0, &aoTextures);
                    tryFetechTexture(aoTextures.C_Str(), newMaterial->aoTexture, false, 1.0f, ETextureFormat::BC4R8);
                }

                // Emissive, SRGB, don't care alpha, so use BC1 default.
                if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0)
                {
                    material->GetTexture(aiTextureType_EMISSIVE, 0, &emissiveTextures);
                    tryFetechTexture(emissiveTextures.C_Str(), newMaterial->emissiveTexture, true, 1.0f, ETextureFormat::BC1);
                }

                subMeshInfo.material = newMaterial->getSaveInfo().getUUID();
                if (newMaterial->save())
                {
                    m_materialPathUUIDMap[materialSavePath] = newMaterial->getSaveInfo().getUUID();
                }
                else
                {
                    LOG_ERROR("Failed to save material meta asset, the material {} import fail!", 
                        utf8::utf16to8(materialSavePath.u16string()));
                    subMeshInfo.material = {};
                }
            }
            
            const auto& meta = AssetTexture::uiGetAssetReflectionInfo();
            std::for_each(std::execution::par, imagePendingConfigs.begin(), imagePendingConfigs.end(), [&](const ImageImportConfig& item)
            {
                meta.importConfig.importAssetFromConfigThreadSafe(item.config);
            });
        }
        else // No material found, keep empty.
        {
            subMeshInfo.material = {};
        }

        return subMeshInfo;
    }

}