#include "pmx.h"

namespace engine
{
	PMXComponent::~PMXComponent()
	{
	}
	bool PMXComponent::setPMX(const UUID& in)
	{
		if (m_pmxUUID != in)
		{
			m_pmxUUID = in;


			UN_IMPLEMENT_WARN();
			return true;
		}

		return false;
	}
}