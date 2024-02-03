#pragma once

#include "component.h"
#include "scene_node.h"

namespace engine
{
	// Simple scene graph implement.
	class Scene : public AssetInterface
	{
		REGISTER_BODY_DECLARE(AssetInterface);

		friend SceneNode;

	public:
		Scene() = default;
		explicit Scene(const AssetSaveInfo& saveInfo);

		virtual ~Scene() = default;
		static const Scene* getCDO();

	public:
		// ~AssetInterface.
		virtual EAssetType getType() const override { return EAssetType::darkscene; }
		virtual void onPostAssetConstruct() override;
		virtual VulkanImage* getSnapshotImage() override;
		static const AssetReflectionInfo& uiGetAssetReflectionInfo();

		static bool assetIsScene(const char* ext);

	protected:
		virtual bool saveImpl() override;
		virtual void unloadImpl() override;
		// ~AssetInterface.

	public:
		// Tick every frame.
		void tick(const RuntimeModuleTickData& tickData);

		void onGameBegin();
		void onGamePause();
		void onGameContinue();
		void onGameStop();

		// Get root node.
		std::shared_ptr<SceneNode> getRootNode() { return m_root; }

	public: 
		// ~Scene node operator.
		
		// Root node id.
		static const size_t kRootId = 0;

		// How many node exist.
		size_t getNodeCount() const { return m_sceneNodes.size(); }

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
		std::shared_ptr<SceneNode> findNode(const std::string& name) const;

		// find all same name nodes, this is a slow function.
		std::vector<std::shared_ptr<SceneNode>> findNodes(const std::string& name) const;

		// update whole graph's transform.
		void flushSceneNodeTransform();

		// Node is exist or not.
		bool existNode(size_t id) const;

		// Get node with check.
		std::shared_ptr<SceneNode> getNode(size_t id) const;

	protected:
		// require guid of scene node in this scene.
		size_t requireSceneNodeId();

		// Create scene node.
		std::shared_ptr<SceneNode> createNode(size_t id, const std::string& name);

	public:
		// ~Component operator

		// Get components.
		inline const std::vector<std::weak_ptr<Component>>& getComponents(const std::string& id) const
		{
			return m_components.at(id);
		}

		template <typename T>
		inline const std::vector<std::weak_ptr<Component>>& getComponents() const
		{
			std::string type = rttr::type::get<T>().get_name().data();
			return getComponents(type);
		}

		template <typename T>
		inline std::vector<std::weak_ptr<Component>>& getComponents()
		{
			std::string type = rttr::type::get<T>().get_name().data();
			return m_components.at(type);
		}

		// Check exist component or not.
		inline bool hasComponent(const std::string& id) const
		{
			auto component = m_components.find(id);
			return (component != m_components.end() && !component->second.empty());
		}

		// Loop scene's components.
		template<typename T> void loopComponents(std::function<bool(std::shared_ptr<T>)>&& func);

		// Add component for node.
		template<typename T> bool addComponent(std::shared_ptr<T> component, std::shared_ptr<SceneNode> node);
		bool addComponent(const std::string& type, std::shared_ptr<Component> component, std::shared_ptr<SceneNode> node);

		template <typename T>
		bool hasComponent() const
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
			std::string type = rttr::type::get<T>().get_name().data();

			return hasComponent(type);
		}

		template<typename T>
		bool removeComponent(std::shared_ptr<SceneNode> node)
		{
			static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
			std::string type = rttr::type::get<T>().get_name().data();

			return removeComponent(type);
		}

		bool removeComponent(std::shared_ptr<SceneNode> node, const std::string& type);

	private:
		// Cache scene node index. use for runtime guid.
		size_t m_currentId = kRootId;

		// Owner of the root node.
		std::shared_ptr<SceneNode> m_root = nullptr;

		// Cache scene components, no include transform.
		std::unordered_map<std::string, std::vector<std::weak_ptr<Component>>> m_components;

		// Cache scene node maps.
		mutable std::unordered_map<size_t, std::weak_ptr<SceneNode>> m_sceneNodes;
	};


	template<typename T>
	inline void Scene::loopComponents(std::function<bool(std::shared_ptr<T>)>&& func)
	{
		static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
		if (!hasComponent<T>())
		{
			return;
		}
		
		auto& compWeakPtrs = getComponents<T>();

		// Loop all component.
		size_t unvalidComponentNum = 0U;
		for (auto& p : compWeakPtrs)
		{
			if (auto pShared = p.lock())
			{
				// Some function require pre-return after find first component.
				if (func(std::static_pointer_cast<T>(pShared)))
				{
					return;
				}
			}
			else
			{
				unvalidComponentNum++;
			}
		}

		// Try shrink scene components if need.
		static const size_t kShrinkComponentNumMin = 10U;
		if (unvalidComponentNum > kShrinkComponentNumMin)
		{
			compWeakPtrs.erase(std::remove_if(compWeakPtrs.begin(), compWeakPtrs.end(),
				[](const std::weak_ptr<Component>& p)
				{
					return !p.lock();
				}),
				compWeakPtrs.end());
		}
	}

	inline bool Scene::addComponent(
		const std::string& type, 
		std::shared_ptr<Component> component, 
		std::shared_ptr<SceneNode> node)
	{
		if (component && !node->hasComponent(type))
		{
			node->setComponent(type, component);
			m_components[type].push_back(component);
			markDirty();

			return true;
		}
	}

	template<typename T>
	inline bool Scene::addComponent(std::shared_ptr<T> component, std::shared_ptr<SceneNode> node)
	{
		static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");

		if (component && !node->hasComponent<T>())
		{
			node->setComponent(component);
			std::string type = rttr::type::get<T>().get_name().data();

			m_components[type].push_back(component);
			markDirty();

			return true;
		}

		return false;
	}
}