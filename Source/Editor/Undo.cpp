#include "Pch.h"
#include "Undo.h"

Undo::~Undo()
{
	clear();
}

Undo& Undo::get()
{
    static Undo undo;
    return undo;
}

void Undo::clear()
{
    for (Entry* cmd : m_track.done)
    {
        destroyEntry(cmd);
    }
    m_track.done.clear();

    for (Entry* cmd : m_track.undone)
    {
        destroyEntry(cmd);
    }
    m_track.undone.clear();
}

void* Undo::allocateEntry(size_t size, size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

void Undo::destroyEntry(Entry* cmd)
{
    if (!cmd)
    {
        return;
    }

    cmd->~Entry();
    _aligned_free(cmd);
}

void Undo::done(Entry* cmd)
{
    m_track.done.push_back(cmd);

    // Erase redo stack when a new entry is added to the undo stack
    for (Entry* cmd : m_track.undone)
    {
        destroyEntry(cmd);
    }
    m_track.undone.clear();
}

bool Undo::undo()
{
    if (m_track.done.empty())
    {
        return false;
    }

    Entry* cmd = m_track.done.back();
    m_track.done.pop_back();

    cmd->undo();
    m_track.undone.push_back(cmd);

    return true;
}

bool Undo::redo()
{
    if (m_track.undone.empty())
    {
        return false;
    }

    Entry* cmd = m_track.undone.back();
    m_track.undone.pop_back();

    cmd->redo();
    m_track.done.push_back(cmd);

    return true;
}