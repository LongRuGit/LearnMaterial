#include "SortClass.h"


SortSequence::SortClass::SortClass():
m_pData(nullptr)
{
}

SortSequence::SortClass::~SortClass()
{
}

SortSequence::SortClass & SortSequence::SortClass::operator=(const SortClass& ist)
{
	if (&ist!=this)
	{
		SortSequence::SortClass sortTemp(ist);
		char * pTemp = sortTemp.m_pData;
		sortTemp.m_pData = m_pData;
		m_pData = pTemp;
	}
	return *this;
}
