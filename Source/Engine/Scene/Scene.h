#pragma once

#include "../Core/Core.h"
#include "../RuntimeModule.h"
#include "../Renderer/RendererCommon.h"
#include "Component.h"
#include "SceneNode.h"

namespace Flower
{
	class SceneManager;

	class Scene : public std::enable_shared_from_this<Scene>
	{
		friend class cereal::access;
		friend SceneNode;

	private: // Runtime update data.
		SceneManager* m_manager = nullptr;

		// Root node id.
		const size_t ROOT_ID = 0;

		// Is scene dirty?
		bool m_bDirty = false;

		// Is cache scene component dirty already. see m_cacheSceneComponents below.
		std::unordered_map<const char*, bool> m_cacheSceneComponentsShrinkAlready;

		

	private: // Serialize area.

		// Cache scene node index. use for runtime guid.
		size_t m_currentId = ROOT_ID;

		// Owner of the root node.
		std::shared_ptr<SceneNode> m_root;

		// Init name.
		std::string m_initName;

		// Cache scene components, no include transform.
		std::unordered_map<const char*, std::vector<std::weak_ptr<Component>>> m_cacheSceneComponents;

		

		size_t m_nodeCount = 0;

	private:
		// require guid of scene node in this scene.
		size_t requireId();
		

		SceneManager* getManager();
		void shrinkCacheComponent(const char* id);

	public:
		bool init();
		void tick(const RuntimeModuleTickData& tickData);

	public:
		// Just for cereal, don't use it in runtime.
		Scene() = default;

		virtual ~Scene();

		static std::shared_ptr<Scene> create(std::string name = "Untitled");

		bool isDirty() const { return m_bDirty; }
		auto getptr() { return shared_from_this(); }
		size_t getCurrentGUID() const { return m_currentId; }
		size_t getNodeCount() const { return m_nodeCount; }
		const std::string& getName() const;
		bool setDirty(bool bDirty = true);
		bool setName(const std::string& name);
		void deleteNode(std::shared_ptr<SceneNode> node);
		std::shared_ptr<SceneNode> createNode(const std::string& name, std::shared_ptr<SceneNode> parent = nullptr);
		void addChild(std::shared_ptr<SceneNode> child);
		std::shared_ptr<SceneNode> getRootNode();
		bool setParent(std::shared_ptr<SceneNode> parent, std::shared_ptr<SceneNode> son);

		

		// post-order loop.
		void loopNodeDownToTop(
			const std::function<void(std::shared_ptr<SceneNode>)>& func, 
			std::shared_ptr<SceneNode> node);
		
		// pre-order loop.
		void loopNodeTopToDown(
			const std::function<void(std::shared_ptr<SceneNode>)>& func, 
			std::shared_ptr<SceneNode> node);

		// loop the whole graph to find first same name scene node, this is a slow function.
		std::shared_ptr<SceneNode> findNode(const std::string& name);

		// find all same name nodes, this is a slow function.
		std::vector<std::shared_ptr<SceneNode>> findNodes(const std::string& name);

		const std::vector<std::weak_ptr<Component>>& getComponents(const char* id) const
		{
			return m_cacheSceneComponents.at(id);
		}

		bool hasComponent(const char* id) const
		{
			auto component = m_cacheSceneComponents.find(id);
			return (component != m_cacheSceneComponents.end() && !component->second.empty());
		}

		template<typename T>
		void loopComponents(std::function<void(std::shared_ptr<T>)>&& func)
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
			if (hasComponent<T>() && !m_cacheSceneComponentsShrinkAlready[typeid(T).name()])
			{
				shrinkCacheComponent(typeid(T).name());
			}

			if (hasComponent<T>())
			{
				for (std::weak_ptr<T>& p : getComponents<T>())
				{
					if (auto pShared = p.lock())
					{
						if (pShared->getNode()->hasComponent<T>())
						{
							func(pShared);
						}
					}
				}
			}
		}

		// update whole graph's transform.
		void flushSceneNodeTransform();

		template<typename T>
		void addComponent(std::shared_ptr<T> component, std::shared_ptr<SceneNode> node)
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");

			if (component && !node->hasComponent<T>())
			{
				node->setComponent(component);
				m_cacheSceneComponents[typeid(T).name()].push_back(component);
				m_bDirty = true;
			}
		}

		template <typename T>
		bool hasComponent() const
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
			return hasComponent(typeid(T).name());
		}

		template<typename T>
		bool removeComponent(std::shared_ptr<SceneNode> node)
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
			return removeComponent(typeid(T).name());
		}

		bool removeComponent(std::shared_ptr<SceneNode> node, const char* type)
		{
			if (node->hasComponent(type))
			{
				setDirty();
				node->removeComponent(type);
				m_cacheSceneComponentsShrinkAlready[type] = false;
				return true;
			}

			return false;
		}

		template <class T>
		std::vector<std::weak_ptr<T>> getComponents() const
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");

			std::vector<std::weak_ptr<T>> result;
			auto id = typeid(T).name();
			if (hasComponent(id))
			{
				const auto& sceneComponents = getComponents(id);

				result.resize(sceneComponents.size());
				std::transform(sceneComponents.begin(), sceneComponents.end(), result.begin(),
					[](const std::weak_ptr<Component>& component) -> std::weak_ptr<T>
					{
						return std::static_pointer_cast<T>(component.lock());
					});
			}
			return result;
		}
	};
}