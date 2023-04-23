#include "delegate.h"
#include <mutex>

namespace engine
{
    DelegateHandle::HandleType DelegateHandle::getNewID()
    {
        static HandleType runtimeID = 0;

        // Valid state check.
        // uint64_t is very large and should never overflow.
        if (runtimeID == kInvalidID)
		{
			runtimeID = 0; 
		}

        // Increment and return.
        return runtimeID ++;
    }
}