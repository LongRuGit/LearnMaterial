#ifndef CNONCOPYABLE_H
#define CNONCOPYABLE_H

class CNonCopyable
{
private:
	CNonCopyable(const CNonCopyable&);
	const CNonCopyable& operator=(const CNonCopyable&);
protected:
	CNonCopyable() { }
	~CNonCopyable() { }
};

#endif 
