#include "threadpool.h"

namespace engine
{
	ThreadPool* ThreadPool::getDefault()
	{
		static ThreadPool threadpool(true);
		return &threadpool;
	}
}