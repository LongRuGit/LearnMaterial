#ifndef SINGLETON_H
#define SINGLETON_H

#include "CNonCopyable.h"

template<typename T>
class Singleton :public CNonCopyable
{
public:
	static T & Instance()
	{
		static T instance;
		return instance;
	}
private:
};
#endif // !SINGLETON_H
