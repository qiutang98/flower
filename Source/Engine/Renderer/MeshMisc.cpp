#include "Pch.h"
#include "MeshMisc.h"

namespace Flower
{
	void StaticMeshRenderBounds::toExtents(
		const StaticMeshRenderBounds& in,
		float& zmin,
		float& zmax,
		float& ymin,
		float& ymax,
		float& xmin,
		float& xmax,
		float scale)
	{
		xmax = in.origin.x + in.extents.x * scale;
		xmin = in.origin.x - in.extents.x * scale;
		ymax = in.origin.y + in.extents.y * scale;
		ymin = in.origin.y - in.extents.y * scale;
		zmax = in.origin.z + in.extents.z * scale;
		zmin = in.origin.z - in.extents.z * scale;
	}

	StaticMeshRenderBounds StaticMeshRenderBounds::combine(
		const StaticMeshRenderBounds& b0,
		const StaticMeshRenderBounds& b1)
	{
		float zmin0, zmin1, zmax0, zmax1, ymin0, ymin1, ymax0, ymax1, xmin0, xmin1, xmax0, xmax1;

		toExtents(b0, zmin0, zmax0, ymin0, ymax0, xmin0, xmax0);
		toExtents(b1, zmin1, zmax1, ymin1, ymax1, xmin1, xmax1);

		float xmax = std::max(xmax0, xmax1);
		float ymax = std::max(ymax0, ymax1);
		float zmax = std::max(zmax0, zmax1);
		float xmin = std::min(xmin0, xmin1);
		float ymin = std::min(ymin0, ymin1);
		float zmin = std::min(zmin0, zmin1);

		StaticMeshRenderBounds ret{};

		ret.origin.x = (xmax + xmin) * 0.5f;
		ret.origin.y = (ymax + ymin) * 0.5f;
		ret.origin.z = (zmax + zmin) * 0.5f;

		ret.extents.x = (xmax - xmin) * 0.5f;
		ret.extents.y = (ymax - ymin) * 0.5f;
		ret.extents.z = (zmax - zmin) * 0.5f;

		ret.radius = std::sqrt(ret.extents.x * ret.extents.x + ret.extents.y * ret.extents.y + ret.extents.z * ret.extents.z);
		return ret;
	}

}