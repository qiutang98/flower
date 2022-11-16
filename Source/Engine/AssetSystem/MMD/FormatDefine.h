#pragma once
#include "Pch.h"

namespace Flower 
{

namespace Importer
{
	// Cpp machine size check.
	static_assert(sizeof(float) == 4);
	static_assert(sizeof(char) == 1);

	struct vec2
	{
		float x, y;
	};

	struct vec3
	{
		float x, y, z;
	};

	struct vec4
	{
		float x, y, z, w;
	};

	struct quat
	{
		float x, y, z, w;
	};
	
	// Strict check avoid memory align cause some problem.
	static_assert(sizeof(vec2) == sizeof(float) * 2);
	static_assert(sizeof(vec3) == sizeof(float) * 3);
	static_assert(sizeof(vec4) == sizeof(float) * 4);
	static_assert(sizeof(quat) == sizeof(float) * 4);
}

namespace Importer::PMX
{
	// Meta data.
	struct MetaData
	{
		char magic[4];
		float version;
		uint8_t dataSize;
		uint8_t encodeType;
		uint8_t appendUvNum;
		uint8_t vertexIndexSize;  
		uint8_t textureIndexSize; 
		uint8_t materialIndexSize;
		uint8_t boneIndexSize;
		uint8_t morphIndexSize;
		uint8_t rigidbodyIndexSize;
	};

	struct ModelInfo
	{
		std::string localName;
		std::string globalName;

		std::string localComment;
		std::string globalComment;
	};

	enum class EVertexWeight : uint8_t
	{
		BDEF1 = 0U,
		BDEF2,
		BDEF4,
		SDEF,
		QDEF,
	};

	struct Vertex
	{
		vec3 position;
		vec3 normal;
		vec2 uv0;
		vec4 appendUvs[4];
		EVertexWeight weightType;
		int32_t	boneIndices[4];
		float boneWeights[4];
		vec3 sdefC;
		vec3 sdefR0;
		vec3 sdefR1;
		float edgeMag;
	};

	struct Face
	{
		uint32_t vertices[3];
	};

	struct Texture
	{
		std::string textureName;
	};

	enum class EDrawModeFlags : uint8_t
	{
		BothFace = 0x01,
		GroundShadow = 0x02,
		CastSelfShadow = 0x04,
		RecieveSelfShadow = 0x08,
		DrawEdge = 0x10,
		VertexColor = 0x20,
		DrawPoint = 0x40,
		DrawLine = 0x80,
	};

	enum class ESphereMode : uint8_t
	{
		None,
		Mul,
		Add,
		SubTexture,
	};

	enum class EToonMode : uint8_t
	{
		Separate,	
		Common,
	};

	struct Material
	{
		std::string	localName;
		std::string	globalName;
		vec4 diffuse;
		vec3 specular;
		float specularPower;
		vec3 ambient;
		EDrawModeFlags drawMode;
		vec4 edgeColor;
		float edgeSize;
		int32_t	textureIndex;
		int32_t	sphereTextureIndex;
		ESphereMode sphereMode;
		EToonMode toonMode;
		int32_t	toonTextureIndex;
		std::string	memo;
		int32_t	numFaceVertices;
	};

	enum class EBoneFlags : uint16_t
	{
		TargetShowMode = 0x0001,
		AllowRotate = 0x0002,
		AllowTranslate = 0x0004,
		Visible = 0x0008,
		AllowControl = 0x0010,
		IK = 0x0020,
		AppendLocal = 0x0080,
		AppendRotate = 0x0100,
		AppendTranslate = 0x0200,
		FixedAxis = 0x0400,
		LocalAxis = 0x800,
		DeformAfterPhysics = 0x1000,
		DeformOuterParent = 0x2000,
	};

	struct IKLink
	{
		int32_t	ikBoneIndex;
		unsigned char enableLimit;
		vec3 limitMin;
		vec3 limitMax;
	};

	struct Bone
	{
		std::string	localName;
		std::string	globalName;
		vec3 position;
		int32_t	parentBoneIndex;
		int32_t	deformDepth;

		EBoneFlags	boneFlag;
		vec3 positionOffset;
		int32_t	linkBoneIndex;

		int32_t	appendBoneIndex;
		float appendWeight;

		vec3 fixedAxis;
		vec3 localXAxis;
		vec3 localZAxis;
		int32_t	keyValue;

		int32_t	ikTargetBoneIndex;
		int32_t	ikIterationCount;
		float	ikLimit;

		std::vector<IKLink> ikLinks;
	};

	enum class EMorphType : uint8_t
	{
		Group,
		Position,
		Bone,
		UV,
		AddUV1, 
		AddUV2,
		AddUV3,
		AddUV4,
		Material,
		Flip,
		Impluse,
	};

	struct Morph
	{
		std::string	localName;
		std::string	globalName;

		uint8_t	controlPanel;
		EMorphType morphType;

		struct PositionMorph
		{
			int32_t	vertexIndex;
			vec3 position;
		};

		struct UVMorph
		{
			int32_t	vertexIndex;
			vec4 uv;
		};

		struct BoneMorph
		{
			int32_t	boneIndex;
			vec3 position;
			quat quaternion;
		};

		struct MaterialMorph
		{
			enum class EOpType : uint8_t
			{
				Mul,
				Add,
			};

			int32_t	materialIndex;
			EOpType	opType;
			vec4 diffuse;
			vec3 specular;
			float specularPower;
			vec3 ambient;
			vec4 edgeColor;
			float edgeSize;
			vec4 textureFactor;
			vec4 sphereTextureFactor;
			vec4 toonTextureFactor;
		};

		struct GroupMorph
		{
			int32_t	morphIndex;
			float weight;
		};

		struct FlipMorph
		{
			int32_t	morphIndex;
			float weight;
		};

		struct ImpulseMorph
		{
			int32_t	rigidbodyIndex;
			uint8_t	localFlag;
			vec3	translateVelocity;
			vec3	rotateTorque;
		};

		std::vector<PositionMorph>	positionMorph;
		std::vector<UVMorph>		uvMorph;
		std::vector<BoneMorph>		boneMorph;
		std::vector<MaterialMorph>	materialMorph;
		std::vector<GroupMorph>		groupMorph;
		std::vector<FlipMorph>		flipMorph;
		std::vector<ImpulseMorph>	impulseMorph;
	};

	struct DispalyFrame
	{
		std::string	localName;
		std::string	globalName;

		enum class ETargetType : uint8_t
		{
			BoneIndex,
			MorphIndex,
		};
		struct Target
		{
			ETargetType	type;
			int32_t		index;
		};

		enum class EFrameType : uint8_t
		{
			DefaultFrame,
			SpecialFrame,
		};

		EFrameType flag;
		std::vector<Target>	targets;
	};

	struct Rigidbody
	{
		std::string	localName;
		std::string	globalName;

		int32_t	boneIndex;
		uint8_t	group;
		uint16_t collisionGroup;

		enum class EShape : uint8_t
		{
			Sphere,
			Box,
			Capsule,
		};
		EShape shape;
		vec3 shapeSize;

		vec3 translate;
		vec3 rotate;

		float mass;
		float translateDimmer;
		float rotateDimmer;
		float repulsion;
		float friction;

		enum class EOperation : uint8_t
		{
			Static,
			Dynamic,
			DynamicAndBoneMerge
		};
		EOperation op;
	};

	struct Joint
	{
		std::string	localName;
		std::string	globalName;

		enum class JointType : uint8_t
		{
			SpringDOF6,
			DOF6,
			P2P,
			ConeTwist,
			Slider,
			Hinge,
		};

		JointType type;
		int32_t	rigidbodyAIndex;
		int32_t	rigidbodyBIndex;

		vec3 translate;
		vec3 rotate;

		vec3 translateLowerLimit;
		vec3 translateUpperLimit;
		vec3 rotateLowerLimit;
		vec3 rotateUpperLimit;

		vec3 springTranslateFactor;
		vec3 springRotateFactor;
	};

	struct Softbody
	{
		std::string	localName;
		std::string	globalName;

		enum class ESoftbodyType : uint8_t
		{
			TriMesh,
			Rope,
		};
		ESoftbodyType type;

		int32_t	materialIndex;

		uint8_t	group;
		uint16_t collisionGroup;

		enum class ESoftbodyMask : uint8_t
		{
			BLink = 0x01,
			Cluster = 0x02,
			HybridLink = 0x04,
		};
		ESoftbodyMask flag;

		int32_t	BLinkLength;
		int32_t	numClusters;

		float totalMass;
		float collisionMargin;

		enum class EAeroModel : int32_t
		{
			kAeroModelV_TwoSided,
			kAeroModelV_OneSided,
			kAeroModelF_TwoSided,
			kAeroModelF_OneSided,
		};
		int32_t	aeroModel;

		// Config
		float	VCF;
		float	DP;
		float	DG;
		float	LF;
		float	PR;
		float	VC;
		float	DF;
		float	MT;
		float	CHR;
		float	KHR;
		float	SHR;
		float	AHR;

		// Cluster
		float	SRHR_CL;
		float	SKHR_CL;
		float	SSHR_CL;
		float	SR_SPLT_CL;
		float	SK_SPLT_CL;
		float	SS_SPLT_CL;

		// Interation
		int32_t	V_IT;
		int32_t	P_IT;
		int32_t	D_IT;
		int32_t	C_IT;

		// Material
		float	LST;
		float	AST;
		float	VST;

		struct AnchorRigidbody
		{
			int32_t	rigidBodyIndex;
			int32_t	vertexIndex;
			uint8_t	nearMode; // 0:FF 1:ON
		};
		std::vector<AnchorRigidbody> anchorRigidbodies;
		std::vector<int32_t> pinVertexIndices;
	};

	struct File
	{
		MetaData metaData;
		ModelInfo modelInfo;

		std::vector<Vertex>		vertices;
		std::vector<Face>		faces;
		std::vector<Texture>		textures;
		std::vector<Material>	materials;
		std::vector<Bone>		bones;
		std::vector<Morph>		morphs;
		std::vector<DispalyFrame>	displayFrames;
		std::vector<Rigidbody>	rigidbodies;
		std::vector<Joint>		joints;
		std::vector<Softbody>	softbodies;
	};

	extern bool readFile(File& inout, const char* fileName);
}

}