#pragma once

#include <deque>
#include <memory>
#include <util/util.h>

class Undo
{
public:
    explicit Undo(size_t maxUndoItem) : m_maxUndoItemNum(maxUndoItem)
    {

    }

    struct Entry
    {
        virtual const char* getType() const { return "BaseCommand"; }
        virtual void undo() { }
        virtual void redo() { }
    };

    struct Track
    {
        std::deque<std::unique_ptr<Entry>> undone;   
        std::vector<std::unique_ptr<Entry>> redone; 
    };

    bool hasUndo() const { return !m_track.undone.empty(); }
    bool hasRedo() const { return !m_track.redone.empty(); }

    template<typename EntryT, typename ...Args>
    inline static std::unique_ptr<EntryT> createEntry(Args&&... params)
    {
        return std::make_unique<EntryT>(params...);
    }

    void clear()
    {
        m_track.undone.clear();
        m_track.redone.clear();
    }

    void done(std::unique_ptr<Entry>&& cmd)
    {
        // Push undone stack.
        m_track.undone.push_back(std::move(cmd));

        // Pop front if too many undone.
        if (m_track.undone.size() >= m_maxUndoItemNum)
        {
            m_track.undone.pop_front();
        }

        // Clear redone stack.
        m_track.redone.clear();
    }

    bool undo()
    {
        if (m_track.undone.empty())
        {
            return false;
        }

        // Pop back.
        std::unique_ptr<Entry> cmd = std::move(m_track.undone.back());
        m_track.undone.pop_back();

        LOG_TRACE("Undo action: {0}.", cmd->getType());

        // Execute undo.
        cmd->undo();

        // Push to redo.
        m_track.redone.push_back(std::move(cmd));

        return true;
    }

    bool redo()
    {
        if (m_track.redone.empty())
        {
            return false;
        }
            
        // Pop back.
        std::unique_ptr<Entry> cmd = std::move(m_track.redone.back());
        m_track.redone.pop_back();

        LOG_TRACE("Redo action: {0}.", cmd->getType());

        // Execute redo.
        cmd->redo();

        // Push to done.
        m_track.undone.push_back(std::move(cmd));

        return true;
    }

private:
    // Max undo item.
    size_t m_maxUndoItemNum;

    // Store track.
    Track m_track;
};

template<typename T> class UndoBasic : public Undo::Entry
{
public:
    UndoBasic(T* data) : m_dataRef(data), m_dataCopy(*data) { }

    virtual void undo() { std::swap(m_dataCopy, *m_dataRef); }
    virtual void redo() { std::swap(m_dataCopy, *m_dataRef); }

private:
    T* m_dataRef;
    T  m_dataCopy;     
};

template<typename T> class UndoArchiveBasic : public Undo::Entry
{
public:
    UndoArchiveBasic(const T& data) : m_dataRef(data), m_dataCopy(*data) 
    {
    
    }

    virtual void undo() { std::swap(m_dataCopy, *m_dataRef); }
    virtual void redo() { std::swap(m_dataCopy, *m_dataRef); }

private:
    T* m_dataRef;
    T  m_dataCopy;
};

template<typename T>
class UndoBasicScope
{
public:
    explicit UndoBasicScope(Undo* undo, T* value)
    {
        m_undo = undo;
        m_entry = Undo::createEntry<UndoBasic<T>>(value);
    }

    ~UndoBasicScope()
    {
        m_undo->done(std::move(m_entry));
    }
private:
    std::unique_ptr<UndoBasic<T>> m_entry;
    Undo* m_undo;
};

