#pragma once

#include "scene_common.h"

#include "Component/Transform.h"

namespace engine
{
    class SceneNode : public std::enable_shared_from_this<SceneNode>
    {
        REGISTER_BODY_DECLARE();
        friend Scene;

    public:
        // Just provide for cereal, don't use it runtime.
        SceneNode() = default;
        virtual ~SceneNode();

        static std::shared_ptr<SceneNode> create(
            const size_t id, 
            const std::string& name, 
            std::shared_ptr<Scene> scene);

        void tick(const RuntimeModuleTickData& tickData);

        void onGameBegin();
        void onGamePause();
        void onGameContinue();
        void onGameStop();

        const auto& getId() const { return m_id; }
        const auto& getName() const { return m_name; }
        const auto& getChildren() const { return m_children; }
        auto& getChildren() { return m_children; }
        const bool isRoot() const;

        bool getVisibility() const { return m_bVisibility; }
        bool getStatic() const { return m_bStatic; }

        bool editorSelected();

        std::shared_ptr<Component> getComponent(const std::string& id);

        template <typename T>
        std::shared_ptr<T> getComponent()
        {
            static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");

            std::string type = rttr::type::get<T>().get_name().data();
            return std::dynamic_pointer_cast<T>(getComponent(type));
        }

        auto getTransform() { return getComponent<Transform>(); }
        auto getParent() { return m_parent.lock(); }
        auto getPtr() { return shared_from_this(); }
        auto getScene() { return m_scene.lock(); }

        bool hasComponent(const std::string& id) const;

        template <class T>
        bool hasComponent() const
        {
            static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");
            std::string type = rttr::type::get<T>().get_name().data();

            return hasComponent(type);
        }

        bool setName(const std::string& in);

        bool isSon(std::shared_ptr<SceneNode> p) const; // p is the son of this node.
        bool isSonDirectly(std::shared_ptr<SceneNode> p) const; // p is the direct son of this node.

        void markDirty();

        bool canSetNewVisibility();
        bool canSetNewStatic();

        void setVisibility(bool bState) { setVisibilityImpl(bState, false); }
        void setStatic(bool bState) { setStaticImpl(bState, false); }

    private:
        void addChild(std::shared_ptr<SceneNode> child);

        void removeChild(std::shared_ptr<SceneNode> o);
        void removeChild(size_t id);

        // set p as this node's new parent.
        void setParent(std::shared_ptr<SceneNode> p);

        template<typename T>
        void setComponent(std::shared_ptr<T> component);

        void setComponent(const std::string& type, std::shared_ptr<Component> component);

        // Remove parent relationship.
        bool unparent();

        // remove component.
        void removeComponent(const std::string& id);

        // Set node view state.
        void setVisibilityImpl(bool bState, bool bForce);

        // Set node static state.
        void setStaticImpl(bool bState, bool bForce);

    private:
        // This node visibility state.
        bool m_bVisibility = true;

        // This node static state.
        bool m_bStatic = true;

        // Id of scene node.
        size_t m_id;

        // Inspector name, utf8 encode.
        u8str m_name = "untitled";

        // Reference of parent.
        std::weak_ptr<SceneNode> m_parent;

        // Reference of scene.
        std::weak_ptr<Scene> m_scene;

        // Owner of components.
        std::unordered_map<std::string, std::shared_ptr<Component>> m_components;

        // Owner of children.
        std::vector<std::shared_ptr<SceneNode>> m_children;
    };

    template<typename T>
    inline void SceneNode::setComponent(std::shared_ptr<T> component)
    {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component.");

        component->setNode(getPtr());
        std::string type = rttr::type::get<T>().get_name().data();

        auto it = m_components.find(type);
        if (it != m_components.end())
        {
            it->second = component;
        }
        else
        {
            m_components.insert(std::make_pair(type, component));
        }
    }

    inline void SceneNode::setComponent(
        const std::string& type, 
        std::shared_ptr<Component> component)
    {
        component->setNode(getPtr());

        auto it = m_components.find(type);
        if (it != m_components.end())
        {
            it->second = component;
        }
        else
        {
            m_components.insert(std::make_pair(type, component));
        }
    }
}