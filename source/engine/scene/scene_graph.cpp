#include "scene_graph.h"
#include "scene.h"
#include <util/framework.h>
#include "scene_archive.h"
#include <asset/asset.h>
#include <asset/asset_system.h>

namespace engine
{
	std::shared_ptr<Scene> Scene::create(std::string name)
	{
		auto newScene = std::make_shared<Scene>();
		newScene->m_assetNameUtf8 = name;
		newScene->m_initName = name;
		return newScene;
	}

	bool Scene::saveActionImpl()
	{
		std::shared_ptr<AssetInterface> asset = getptr<Scene>();
		return saveAsset(asset, getSavePath(), getSuffix(), false);
	}

	std::shared_ptr<SceneNode> Scene::createNode(size_t id, const std::string& name)
	{
		auto node = SceneNode::create(id, name, getptr<Scene>());
		CHECK(!m_cacheSceneNodeMaps[node->getId()].lock());
		m_cacheSceneNodeMaps[node->getId()] = node;

		return node;
	}

	size_t Scene::requireId()
	{
		CHECK(m_currentId < SIZE_MAX && "GUID max than size_t's max value.");

		m_currentId++;
		return m_currentId;
	}

	SceneManager* Scene::getManager()
	{
		if (m_manager == nullptr)
		{
			m_manager = Framework::get()->getEngine().getRuntimeModule<SceneManager>();
		}

		return m_manager;
	}

	bool Scene::init()
	{
		m_root = createNode(kRootId, m_initName);
		return true;
	}

	void Scene::shrinkCacheComponent(const char* id)
	{
		auto& cacheWeakPtr = m_cacheSceneComponents[id];

		cacheWeakPtr.erase(std::remove_if(cacheWeakPtr.begin(), cacheWeakPtr.end(),
			[](const std::weak_ptr<Component>& p)
			{
				return p.lock().get() == nullptr;
			}),
			cacheWeakPtr.end()
		);
		m_cacheSceneComponentsShrinkAlready[id] = true;
	}

	bool Scene::setName(const std::string& name)
	{
		return m_root->setName(name);
	}

	void Scene::tick(const RuntimeModuleTickData& tickData)
	{
		// All node tick.
		loopNodeTopToDown([tickData](std::shared_ptr<SceneNode> node)
		{
			node->tick(tickData);
		}, m_root);
	}

	std::stringstream Scene::saveToStream()
	{
		std::stringstream storage(std::ios::in | std::ios::out | std::ios::binary);
		cereal::BinaryOutputArchive archive(storage);
		archive(*this);
		return std::move(storage);
	}

	void Scene::loadFromStream(std::stringstream&& s)
	{
		cereal::BinaryInputArchive archive(s);
		archive(*this);
		setDirty();
	}

	void Scene::deleteNode(std::shared_ptr<SceneNode> node)
	{
		loopNodeTopToDown(
			[&](std::shared_ptr<SceneNode> nodeLoop) 
			{
				m_cacheSceneNodeMaps.erase(nodeLoop->getId());
			},
			node);

		// Loop delete nodes.
		node->selfDelete();
	}

	std::shared_ptr<SceneNode> Scene::createNode(const std::string& name, std::shared_ptr<SceneNode> parent)
	{
		// use require id to avoid guid repeat problem.
		m_nodeCount ++;
		auto result = createNode(requireId(), name);

		setParent(parent ? parent : m_root, result);
		setDirty();
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

	std::shared_ptr<SceneNode> Scene::findNode(const std::string& name)
	{
		if (name == m_root->getName())
		{
			return m_root;
		}

		for (auto& root_node : m_root->getChildren())
		{
			std::queue<std::shared_ptr<SceneNode>> traverseNodes{};
			traverseNodes.push(root_node);

			while (!traverseNodes.empty())
			{
				auto& node = traverseNodes.front();
				traverseNodes.pop();

				if (node->getName() == name)
				{
					return node;
				}

				for (auto& child_node : node->getChildren())
				{
					traverseNodes.push(child_node);
				}
			}
		}

		return nullptr;
	}

	std::vector<std::shared_ptr<SceneNode>> Scene::findNodes(const std::string& name)
	{
		std::vector<std::shared_ptr<SceneNode>> results{ };

		for (auto& root_node : m_root->getChildren())
		{
			std::queue<std::shared_ptr<SceneNode>> traverse_nodes{};
			traverse_nodes.push(root_node);

			while (!traverse_nodes.empty())
			{
				auto& node = traverse_nodes.front();
				traverse_nodes.pop();

				if (node->getName() == name)
				{
					results.push_back(node);
				}

				for (auto& child_node : node->getChildren())
				{
					traverse_nodes.push(child_node);
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
		},m_root);
	}

	bool Scene::existNode(size_t id)
	{
		if (m_cacheSceneNodeMaps[id].lock())
		{
			return true;
		}

		m_cacheSceneNodeMaps.erase(id);
		return false;
	}
}