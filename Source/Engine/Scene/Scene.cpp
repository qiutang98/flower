#include "Pch.h"
#include "Scene.h"
#include "SceneManager.h"
#include "../Engine.h"

namespace Flower
{
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
			m_manager = GEngine->getRuntimeModule<SceneManager>();
		}

		return m_manager;
	}

	std::shared_ptr<Scene> Scene::create(std::string name)
	{
		auto newScene = std::shared_ptr<Scene>(new Scene());
		newScene->m_initName = name;
		return newScene;
	}

	Scene::~Scene()
	{
		m_root.reset();
	}

	bool Scene::init()
	{
		m_root = SceneNode::create(ROOT_ID, m_initName, shared_from_this());
		return true;
	}

	bool Scene::setDirty(bool bDirty)
	{
		if (m_bDirty != bDirty)
		{
			m_bDirty = bDirty;
			return true;
		}

		return false;
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
		m_lazyDestroyComponents.tick();

		// update all transforms.
		loopNodeTopToDown([tickData](std::shared_ptr<SceneNode> node)
		{
			node->tick(tickData);
		}, m_root);
	}

	const std::string& Scene::getName() const
	{
		return m_root->getName();
	}

	void Scene::deleteNode(std::shared_ptr<SceneNode> node)
	{
		node->selfDelete();
	}

	std::shared_ptr<SceneNode> Scene::createNode(const std::string& name, std::shared_ptr<SceneNode> parent)
	{
		// use require id to avoid guid repeat problem.
		m_nodeCount ++;
		auto result = SceneNode::create(requireId(), name, shared_from_this());

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

	std::shared_ptr<SceneNode> Scene::getRootNode()
	{
		return m_root;
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

	// sync scene node tree's transform form top to down to get current result.
	void Scene::flushSceneNodeTransform()
	{
		loopNodeTopToDown([](std::shared_ptr<SceneNode> node)
		{
			node->getTransform()->updateWorldTransform();
		},m_root);
	}
}