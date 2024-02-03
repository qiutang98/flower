#include "scene.h"
#include <rttr/registration.h>
#include "../ui/ui.h"

#include "../serialization/serialization.h"

namespace engine
{

	Scene::Scene(const AssetSaveInfo& saveInfo)
		: AssetInterface(saveInfo)
	{

	}

	const Scene* Scene::getCDO()
	{
		static const Scene kSceneCDO = {};
		return &kSceneCDO;
	}

	void Scene::onPostAssetConstruct()
	{
		m_root = createNode(kRootId, "Root");
	}

	VulkanImage* Scene::getSnapshotImage()
	{
		static auto* icon = &getContext()->getBuiltinTexture(EBuiltinTextures::sceneIcon)->getSelfImage();

		return icon;
	}

	const AssetReflectionInfo& Scene::uiGetAssetReflectionInfo()
	{
		const static AssetReflectionInfo kInfo =
		{
			.name = "Scene",
			.icon = ICON_FA_CHESS_KING,
			.decoratedName = std::string("  ") + ICON_FA_CHESS_KING + std::string("     Scene"),
			.importConfig = { .bImportable = false, }
		};
		return kInfo;
	}

	bool Scene::assetIsScene(const char* ext)
	{
		if (ext == getCDO()->getSuffix())
		{
			return true;
		}
		return false;
	}

	std::shared_ptr<SceneNode> Scene::createNode(size_t id, const std::string& name)
	{
		auto node = SceneNode::create(id, name, getptr<Scene>());
		CHECK(!m_sceneNodes[node->getId()].lock());
		m_sceneNodes[node->getId()] = node;

		return node;
	}

	size_t Scene::requireSceneNodeId()
	{
		ASSERT(m_currentId < std::numeric_limits<uint>::max(), "GUID max than max object id value.");

		m_currentId++;
		return m_currentId;
	}

	void Scene::tick(const RuntimeModuleTickData& tickData)
	{
		// All node tick.
		loopNodeTopToDown([tickData](std::shared_ptr<SceneNode> node)
			{
				node->tick(tickData);
			}, m_root);
	}

	void Scene::onGameBegin()
	{
		loopNodeTopToDown([](std::shared_ptr<SceneNode> node)
			{
				node->onGameBegin();
			}, m_root);
	}

	void Scene::onGameStop()
	{
		loopNodeTopToDown([](std::shared_ptr<SceneNode> node)
			{
				node->onGameStop();
			}, m_root);
	}

	void Scene::onGameContinue()
	{
		loopNodeTopToDown([](std::shared_ptr<SceneNode> node)
			{
				node->onGameContinue();
			}, m_root);
	}

	void Scene::onGamePause()
	{
		loopNodeTopToDown([](std::shared_ptr<SceneNode> node)
			{
				node->onGamePause();
			}, m_root);
	}

	void Scene::deleteNode(std::shared_ptr<SceneNode> node)
	{
		// Delete node will erase node loop from top to down.
		loopNodeTopToDown(
			[&](std::shared_ptr<SceneNode> nodeLoop)
			{
				m_sceneNodes.erase(nodeLoop->getId());
			},
			node);

		// Cancel node's parent relationship.
		node->unparent();

		markDirty();
	}

	std::shared_ptr<SceneNode> Scene::createNode(const std::string& name, std::shared_ptr<SceneNode> parent)
	{
		// Use require id to avoid guid repeat problem.
		auto result = createNode(requireSceneNodeId(), name);

		setParent(parent ? parent : m_root, result);
		markDirty();
		return result;
	}

	void Scene::addChild(std::shared_ptr<SceneNode> child)
	{
		m_root->addChild(child);
	}

	void Scene::loopNodeDownToTop(
		const std::function<void(std::shared_ptr<SceneNode>)>& func,
		std::shared_ptr<SceneNode> node)
	{
		auto& children = node->getChildren();
		for (auto& child : children)
		{
			loopNodeDownToTop(func, child);
		}

		func(node);
	}

	void Scene::loopNodeTopToDown(
		const std::function<void(std::shared_ptr<SceneNode>)>& func,
		std::shared_ptr<SceneNode> node)
	{
		func(node);

		auto& children = node->getChildren();
		for (auto& child : children)
		{
			loopNodeTopToDown(func, child);
		}
	}

	std::shared_ptr<SceneNode> Scene::findNode(const std::string& name) const
	{
		if (name == m_root->getName())
		{
			return m_root;
		}

		for (auto& rootChild : m_root->getChildren())
		{
			std::queue<std::shared_ptr<SceneNode>> traverseNodes{};
			traverseNodes.push(rootChild);

			while (!traverseNodes.empty())
			{
				auto& node = traverseNodes.front();
				traverseNodes.pop();

				if (node->getName() == name)
				{
					return node;
				}

				for (auto& childNode : node->getChildren())
				{
					traverseNodes.push(childNode);
				}
			}
		}

		return nullptr;
	}

	std::vector<std::shared_ptr<SceneNode>> Scene::findNodes(const std::string& name) const
	{
		std::vector<std::shared_ptr<SceneNode>> results{ };

		for (auto& rootChild : m_root->getChildren())
		{
			std::queue<std::shared_ptr<SceneNode>> traverseNodes{};
			traverseNodes.push(rootChild);

			while (!traverseNodes.empty())
			{
				auto& node = traverseNodes.front();
				traverseNodes.pop();

				if (node->getName() == name)
				{
					results.push_back(node);
				}

				for (auto& childNode : node->getChildren())
				{
					traverseNodes.push(childNode);
				}
			}
		}

		if (name == m_root->getName())
		{
			results.push_back(m_root);
		}

		return results;
	}

	bool Scene::setParent(std::shared_ptr<SceneNode> parent, std::shared_ptr<SceneNode> son)
	{
		bool bNeedSet = false;

		if (parent == son)
		{
			return false;
		}

		auto oldP = son->getParent();

		if (oldP == nullptr || (!son->isSon(parent) && parent->getId() != oldP->getId()))
		{
			bNeedSet = true;
		}

		if (bNeedSet)
		{
			son->setParent(parent);
			return true;
		}

		return false;
	}

	// Sync scene node tree's transform form top to down to get current result.
	void Scene::flushSceneNodeTransform()
	{
		loopNodeTopToDown([](std::shared_ptr<SceneNode> node)
			{
				node->getTransform()->updateWorldTransform();
			}, m_root);
	}

	bool Scene::existNode(size_t id) const
	{
		if (m_sceneNodes[id].lock())
		{
			return true;
		}

		m_sceneNodes.erase(id);
		return false;
	}

	std::shared_ptr<SceneNode> Scene::getNode(size_t id) const
	{
		return m_sceneNodes.at(id).lock();
	}

	bool Scene::removeComponent(std::shared_ptr<SceneNode> node, const std::string& type)
	{
		if (node->hasComponent(type))
		{
			node->removeComponent(type);

			markDirty();
			return true;
		}

		return false;
	}

	bool Scene::saveImpl()
	{
		std::shared_ptr<AssetInterface> asset = getptr<Scene>();
		return saveAsset(asset, getSavePath(), false);
	}

	void Scene::unloadImpl()
	{

	}
}