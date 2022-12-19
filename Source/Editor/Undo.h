#pragma once
#include "Pch.h"

class Undo final
{
public:
    explicit Undo() = default;
    ~Undo();

    static Undo& get();

	struct Entry
	{
        Entry() = default;
        virtual ~Entry() = default;

        virtual const char* getType() const { return "Base Command"; }

		virtual void undo() { }
		virtual void redo() { }
	};

	struct Track
	{
		std::vector<Entry*> done; // undo stack.
		std::vector<Entry*> undone; // redo stack.
	};

	void clear();

    template<typename EntryT, typename ...Args>
    static EntryT* createEntry(Args&&... params)
    {
        void* block = allocateEntry(sizeof(EntryT), alignof(EntryT));
        return new (block) EntryT(params...);
    }

    static void destroyEntry(Entry* cmd);

    // Add entry.
    void done(Entry* cmd);

    bool undo();
    bool redo();

    bool hasUndo() const { return !m_track.done.empty(); }
    bool hasRedo() const { return !m_track.undone.empty(); }

private:
	static void* allocateEntry(size_t size, size_t alignment);

	Track m_track;
};