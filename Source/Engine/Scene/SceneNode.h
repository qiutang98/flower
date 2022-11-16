#pragma once

#include "Component/Transform.h"

namespace Flower
{
    class Scene;

    class SceneNode : public std::enable_shared_from_this<SceneNode>
    {
        friend Scene;
    private:
        SceneNode() = default;

    private:
        bool m_bVisibility = true;
        bool m_bStatic = true;

        // Id of scene node.
        size_t m_id;

        std::string m_runTimeIdName;

        // The scene node depth of the scene tree.
        size_t m_depth = 0; 

        // Inspector name.
        std::string m_name  = "SceneNode"; 

        // Reference of parent.
        std::weak_ptr<SceneNode> m_parent;

        // Reference of scene.
        std::weak_ptr<Scene> m_scene;

        // Owner of components.
        std::unordered_map<const char*, std::shared_ptr<Component>> m_components;

        // Owner of children.
        std::vector<std::shared_ptr<SceneNode>> m_children;

    private:
        void setVisibilityImpl(bool bState, bool bForce);
        void setStaticImpl(bool bState, bool bForce);

    public:
        bool getVisibility() const { return m_bVisibility; }
        bool getStatic() const { return m_bStatic; }

        void markDirty();


        bool canSetNewVisibility();
        bool canSetNewStatic();

        void setVisibility(bool bState) { setVisibilityImpl(bState, false); }
        void setStatic(bool bState) { setStaticImpl(bState, false); }

        void tick(const RuntimeModuleTickData& tickData);

        static std::shared_ptr<SceneNode> create(const size_t id, const std::string& name, std::shared_ptr<Scene> scene);

        virtual ~SceneNode();

        const auto& getId()       const { return m_id; }
        const auto& getRuntimeIdName() const { return m_runTimeIdName; }
        const auto& getName()     const { return m_name; }
        const auto& getChildren() const { return m_children; }
        auto& getChildren()             { return m_children; }
        const auto& getDepth()    const { return m_depth; }
        const bool  isRoot()      const { return m_id == 0; }

        std::shared_ptr<Component> getComponent(const char* id);

        template <typename T>
        std::shared_ptr<T> getComponent()
        {
            static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
            return std::dynamic_pointer_cast<T>(getComponent(typeid(T).name()));
        }

        auto getTransform() { return getComponent<Transform>(); }
        auto getParent() { return m_parent.lock(); }
        auto getPtr() { return shared_from_this(); }
        auto getScene()  { return m_scene.lock(); }

        bool hasComponent(const char* id);

        template <class T> 
        bool hasComponent() 
        { 
            static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
            return hasComponent(typeid(T).name());
        }

        bool setName(const std::string& in);

        // p is the son of this node.
        bool isSon(std::shared_ptr<SceneNode> p);

        // p is the direct son of this node.
        bool isSonDirectly(std::shared_ptr<SceneNode> p);

        

    private:
        // Update self node depth and child's.
        void updateDepth();

        void addChild(std::shared_ptr<SceneNode> child);
        void removeChild(std::shared_ptr<SceneNode> o);
        void removeChild(size_t id);

        // set p as this node's new parent.
        void setParent(std::shared_ptr<SceneNode> p);

        template<typename T>
        void setComponent(std::shared_ptr<T> component)
        {
            static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");

            component->setNode(getPtr());
            auto it = m_components.find(typeid(T).name());
            if (it != m_components.end())
            {
                it->second = component;
            }
            else
            {
                m_components.insert(std::make_pair(typeid(T).name(), component));
            }
        }

        // Delete this node.
        void selfDelete();

        // remove component.
        void removeComponent(const char* type);
    };
}