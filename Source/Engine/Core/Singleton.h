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
		// No dll export safe, but current enough.
		static T* get()
		{
			// After c++ 11 this process is thread safe.
			static T singleton{ };
			return &singleton;
		}
	};
}