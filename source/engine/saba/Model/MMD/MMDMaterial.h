//
// Copyright(c) 2016-2017 benikabocha.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#ifndef SABA_MODEL_MMD_MMDMATERIAL_H_
#define SABA_MODEL_MMD_MMDMATERIAL_H_

#include <string>
#include <cstdint>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace saba
{
	struct MMDMaterial
	{
		MMDMaterial();
		enum class SphereTextureMode
		{
			None,
			Mul,
			Add,
		};

		std::string     m_enName;
		std::string     m_name;

		glm::vec3		m_diffuse;
		float			m_alpha;
		glm::vec3		m_specular;
		float			m_specularPower;
		glm::vec3		m_ambient;
		uint8_t			m_edgeFlag;
		float			m_edgeSize;
		glm::vec4		m_edgeColor;
		std::string		m_texture;
		std::string		    m_spTexture;
		SphereTextureMode	m_spTextureMode;
		std::string		    m_toonTexture;
		glm::vec4		m_textureMulFactor;
		glm::vec4		m_spTextureMulFactor;
		glm::vec4		m_toonTextureMulFactor;
		glm::vec4		m_textureAddFactor;
		glm::vec4		m_spTextureAddFactor;
		glm::vec4		m_toonTextureAddFactor;
		bool			m_bothFace;
		bool			m_groundShadow;
		bool			m_shadowCaster;
		bool			m_shadowReceiver;

		auto operator<=>(const MMDMaterial&) const = default;

		template<class Archive> void serialize(Archive& archive, std::uint32_t const version)
		{
			archive(this->m_enName);
			archive(this->m_name);

			archive(this->m_diffuse);
			archive(this->m_alpha);
			archive(this->m_specular);
			archive(this->m_specularPower);
			archive(this->m_ambient);
			archive(this->m_edgeFlag);
			archive(this->m_edgeSize);
			archive(this->m_edgeColor);
			archive(this->m_texture);

			archive(this->m_spTexture);

			uint32_t spTextureMode = uint32_t(this->m_spTextureMode);
			archive(spTextureMode); //
			this->m_spTextureMode = saba::MMDMaterial::SphereTextureMode(spTextureMode);

			archive(this->m_toonTexture);
			archive(this->m_textureMulFactor);
			archive(this->m_spTextureMulFactor);

			archive(this->m_toonTextureMulFactor);
			archive(this->m_textureAddFactor);
			archive(this->m_spTextureAddFactor);
			archive(this->m_toonTextureAddFactor);

			archive(this->m_bothFace);
			archive(this->m_groundShadow);
			archive(this->m_shadowCaster);
			archive(this->m_shadowReceiver);
		}
	};
}

#endif // !SABA_MODEL_MMD_MMDMATERIAL_H_
