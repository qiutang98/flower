#pragma once

#include "../Core/Core.h"
#include "../RuntimeModule.h"
#include "../Renderer/RendererCommon.h"
#include "Component.h"
#include "SceneNode.h"

namespace Flower
{
	class SceneManager;

	// Simple scene graph implement.
	class Scene : public std::enable_shared_from_this<Scene>
	{
		ARCHIVE_DECLARE;
		friend SceneNode;

	private: // Runtime update data.

		// Cache scene manager.
		SceneManager* m_manager = nullptr;

		// Root node id.
		const size_t ROOT_ID = 0;

		// Is scene dirty?
		bool m_bDirty = false;

		// Is cache scene component dirty already. see m_cacheSceneComponents below.
		// When some sence node remove component, need to call scene update cache in next tick.
		std::unordered_map<std::string, bool> m_cacheSceneComponentsShrinkAlready = {};


#pragma region SerializeField
	////////////////////////////// Serialize area //////////////////////////////
	private: 
		// Cache scene node index. use for runtime guid.
		size_t m_currentId = ROOT_ID;

		// Owner of the root node.
		std::shared_ptr<SceneNode> m_root;

		// Init name.
		std::string m_initName;

		// Cache scene components, no include transform.
		std::unordered_map<std::string, std::vector<std::weak_ptr<Component>>> m_cacheSceneComponents;

		// How many node exist here.
		size_t m_nodeCount = 0;

		std::string m_savePath = {};

	////////////////////////////// Serialize area //////////////////////////////
#pragma endregion SerializeField
	private: // Functions.

		// require guid of scene node in this scene.
		size_t requireId();
		
		// get scene manager.
		SceneManager* getManager();

		// shrink cache component.
		void shrinkCacheComponent(const char* id);

	public:
		// Just for cereal, don't use it in runtime.
		Scene() = default;

		// Create scene function.
		static std::shared_ptr<Scene> create(std::string name = "Untitled");

		// Destroy call.
		virtual ~Scene();

		// Scene init.
		bool init();

		// Tick every frame.
		void tick(const RuntimeModuleTickData& tickData);

		const std::string& getSavePath() const
		{
			return m_savePath;
		}

	public: // Getter function.
		
		// This scene already edit? need save?
		bool isDirty() const 
		{ 
			return m_bDirty; 
		}

		// Get shared ptr.
		auto getptr() 
		{ 
			return shared_from_this(); 
		}

		// Current useful node guid.
		size_t getCurrentGUID() const 
		{ 
			return m_currentId; 
		}

		// How many node exist.
		size_t getNodeCount() const 
		{ 
			return m_nodeCount; 
		}

		// Get scene name.
		const std::string& getName() const
		{
			return m_root->getName();
		}

		void setSavePath(const std::string& path)
		{
			m_savePath = path;
		}

		// Get root node.
		std::shared_ptr<SceneNode> getRootNode()
		{
			return m_root;
		}

		// Get components.
		const std::vector<std::weak_ptr<Component>>& getComponents(const char* id) const
		{
			return m_cacheSceneComponents.at(id);
		}

		// Check exist component or not.
		bool hasComponent(const char* id) const
		{
			auto component = m_cacheSceneComponents.find(id);
			return (component != m_cacheSceneComponents.end() && !component->second.empty());
		}

	
	public: // Simple setter.

		// Set scene dirty state.
		bool setDirty(bool bDirty = true);

		// Change scene name.
		bool setName(const std::string& name);

		// Delete one node, also include it's child nodes.
		void deleteNode(std::shared_ptr<SceneNode> node);

		// Create node.
		std::shared_ptr<SceneNode> createNode(const std::string& name, std::shared_ptr<SceneNode> parent = nullptr);

		// Add child for root node.
		void addChild(std::shared_ptr<SceneNode> child);

		// Set node's parent relationship.
		bool setParent(std::shared_ptr<SceneNode> parent, std::shared_ptr<SceneNode> son);

		// post-order loop.
		void loopNodeDownToTop(const std::function<void(std::shared_ptr<SceneNode>)>& func, std::shared_ptr<SceneNode> node);
		
		// pre-order loop.
		void loopNodeTopToDown(const std::function<void(std::shared_ptr<SceneNode>)>& func, std::shared_ptr<SceneNode> node);

		// loop the whole graph to find first same name scene node, this is a slow function.
		std::shared_ptr<SceneNode> findNode(const std::string& name);

		// find all same name nodes, this is a slow function.
		std::vector<std::shared_ptr<SceneNode>> findNodes(const std::string& name);

		// update whole graph's transform.
		void flushSceneNodeTransform();

	public:
		// Loop scene's components.
		template<typename T>
		void loopComponents(std::function<bool(std::shared_ptr<T>)>&& func)
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
							if (func(pShared))
							{
								return;
							}
						}
					}
				}
			}
		}

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