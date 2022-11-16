#pragma once

namespace Flower
{
	template <typename T>
	class Singleton
	{
	private:
		Singleton() { }
		Singleton(const Singleton& rhs) { }
		Singleton& operator= (const Singleton& rhs) { }

	public:
		static T* get()
		{
			// After c++ 11 this process is thread safe.
			static T singleton{ };
			return &singleton;
		}
	};
}