#include "Common.h"
//#pragma pack(n)

//C语言的Strcpy()函数不考虑内存覆盖
char * my_strcpy(char * strDest, const char * strSrc)
{
	assert((strDest != nullptr) && (strSrc != nullptr));
	char * pCharAdress = strDest;
	while ((*pCharAdress++ = *strSrc++) != '\0')
	{
	}
	return strDest;
}

//strcpy能把strSrc的内容复制到strDest，为什么还要char * 类型的返回值？
//答：为了实现链式表达式。                                    
//例如       int length = strlen(strcpy(strDest, “hello world”));
//考虑内存覆盖
char * newMy_Strcpy(char *strDest, const char *strSrc)
{
	assert((strDest != nullptr) && (strSrc != nullptr));
	size_t len = strlen(strSrc)+1;
	if (strDest < strSrc || strSrc + len < strDest)
	{
		char * pResult = strDest;
		while (len--)
		{
			*pResult++ = *strSrc++;
		}
	}
	else
	{
		char * pResult = strDest;
		pResult += len-1;
		strSrc += len-1;
		while (len--)
		{
			*pResult-- = *strSrc--;
		}
	}
	return strDest;
}

char * my_strCat(char * pre, const char * next)
{
	assert((pre != nullptr) && (next != nullptr));
	size_t len = strlen(pre);
	char * pTemp = pre + len;
	while ((*pTemp++ = *next++) != '\0')
	{
	}
	return pre;
}

char * my_strnpy(char * strDest, const char *strSrc, size_t len)
{
	assert((strDest != nullptr) && (strSrc != nullptr));
	char * pTemp = strDest;
	while ((*pTemp++ = *strSrc++) != '\0'&&len)
	{
		--len;
	}
	while (len--)
	{
		*pTemp++ = '\0';
	}
	return strDest;
}

void * my_memcpy(void * pDes, const void * pSrc, size_t len)
{
	assert((pDes != nullptr) && (pSrc != nullptr));
	char * pTempDes = (char *)pDes;
	const char * pTempSrc = (const char *)pSrc;
	while (len--)
	{
		*pTempDes++ = *pTempSrc++;
	}
	return pDes;
}

//考虑内存覆盖memmove考虑内存覆盖
void * my_memcpyNew(void * pDes, const void * pSrc, size_t len)
{
	assert(pDes);
	assert(pSrc);

	if ((char*)pDes <= (char*)pSrc || (char*)pDes >= ((char*)pSrc + len))//不交叉
	{
		char* d = (char*)pDes;
		char* s = (char*)pSrc;
		while (len--)
		{
			*d++ = *s++;
		}
	}
	else					//交叉
	{
		char* d = (char*)pDes + len;
		const char* s = (const char*)pSrc + len;
		while (len--)
		{
			*d-- = *s--;
		}
	}
	return pDes;
}

//判断内存是大端还是小端模式
bool IsSmallEnd()
{
	int a = 0x12345678;
	char *p = (char *)&a;
	if (*p == 0x78)
	{
		cout << "small End\n";
		return true;
	}
	else if (*p == 0x12)
	{
		cout << "big End\n";
	}
	return false;
}
//通过union进行判断
union TempUn
{
	char c;
	int i;
}un;
//un.i = 1;
//if (un.c)
//return "small"
//else
//{
//	return "big"
//}

//LRU最近最少使用算法实现
class LRU
{
public:
	LRU(int val=0);
	~LRU();
	int Get(const int key);
	void Put(const int key, const int value);
	LRU &operator++();
	const LRU operator++(int);
private:
	int m_maxCapacity;
	list<pair<int,int>> m_List;
	unordered_map<int, list<pair<int, int>>::iterator> m_HashM;
};

LRU::LRU(int val/*=0*/):
m_maxCapacity(val)
{

}

LRU::~LRU()
{
}

int LRU::Get(const int key)
{
	if (m_HashM.count(key) == 0)
		return -1;
	auto pairTemp = *m_HashM[key];
	m_List.push_front(pairTemp);
	m_List.erase(m_HashM[key]);
	m_HashM[key] = m_List.begin();
	return pairTemp.second;
}

void LRU::Put(const int key, const int value)
{
	if (m_HashM.count(key))
	{
		Get(key);
		m_List.begin()->second = value;
		return;
	}
	else
	{
		if (m_HashM.size() == m_maxCapacity)
		{
			m_HashM.erase(m_List.back().first);
			m_List.pop_back();
		}
		m_List.push_front(make_pair(key, value));
		m_HashM[key] = m_List.begin();
	}
}

LRU & LRU::operator++()
{
	//*this += 1;
	return *this;
}

const LRU LRU::operator++(int)
{
	LRU temp = *this;
	++*this;
	return temp;
}

//最近最不常用算法
class LFUCache {
private:
	struct Item
	{
		int val;
		int freq;
		Item* next;
		Item* prev;
		Item(int v, int f) :val(v), freq(f), next(NULL), prev(NULL){}
	};
	unordered_map<int, pair<int, Item*>> dataMap;
	unordered_map<int, pair<Item*, Item*>> freqMap;
	int minFreq;
	int cap;

	void removeItem(Item* item)
	{
		item->prev->next = item->next;
		item->next->prev = item->prev;
		item->next = NULL;
		item->prev = NULL;
	}

	void insertItemToEnd(Item* tail, Item* item)
	{
		//双向链表
		tail->prev->next = item;
		item->prev = tail->prev;
		item->next = tail;
		tail->prev = item;
	}

	void createHeadTailinFreqMap(int freq)
	{
		if (freqMap.count(freq) == 0)
		{
			Item* head = new Item(-1, -1);
			Item* tail = new Item(-1, -1);
			head->next = tail;
			tail->prev = head;
			freqMap[freq] = { head, tail };
		}
	}

	void removeHeadTailInFreqMap(int freq)
	{
		if (freqMap[freq].first->next == freqMap[freq].second)
		{
			delete freqMap[freq].first;
			delete freqMap[freq].second;
			freqMap.erase(freq);
		}
	}
public:
	LFUCache(int capacity) {
		this->cap = capacity;
		minFreq = 0;
	}

	int get(int key) {
		if (dataMap.count(key) == 0)
			return -1;
		Item* item = dataMap[key].second;
		removeItem(item);
		if (freqMap[item->freq].first->next == freqMap[item->freq].second)
		{
			removeHeadTailInFreqMap(item->freq);
			if (minFreq == item->freq)
				minFreq += 1;
		}
		item->freq += 1;
		if (freqMap.count(item->freq) == 0)
		{
			createHeadTailinFreqMap(item->freq);
		}
		insertItemToEnd(freqMap[item->freq].second, item);

		return dataMap[key].first;
	}

	void put(int key, int value) {
		if (this->cap == 0)
			return;
		if (dataMap.count(key) == 0)
		{
			Item* item = new Item(key, 1);

			if (dataMap.size() == this->cap)
			{
				Item* del = freqMap[minFreq].first->next;
				removeItem(del);
				dataMap.erase(del->val);
				delete del, del = NULL;
				if (freqMap[minFreq].first->next == freqMap[minFreq].second && minFreq > 1)
				{
					removeHeadTailInFreqMap(minFreq);
				}
				minFreq = 1;
				if (freqMap.count(minFreq) == 0)
				{
					createHeadTailinFreqMap(item->freq);
				}
				insertItemToEnd(freqMap[minFreq].second, item);
			}
			else
			{
				minFreq = 1;
				if (freqMap.count(minFreq) == 0)
				{
					createHeadTailinFreqMap(item->freq);
				}
				insertItemToEnd(freqMap[minFreq].second, item);
			}
			dataMap[key] = { value, item };
		}
		else
		{
			Item* item = dataMap[key].second;
			dataMap[key].first = value;
			removeItem(item);
			if (freqMap[item->freq].first->next == freqMap[item->freq].second)
			{
				removeHeadTailInFreqMap(item->freq);
				if (minFreq == item->freq)
					minFreq += 1;
			}
			item->freq += 1;
			if (freqMap.count(item->freq) == 0)
			{
				createHeadTailinFreqMap(item->freq);
			}
			insertItemToEnd(freqMap[item->freq].second, item);
		}
	}
};


//mutex mu;//线程互斥对象
////懒汉板单例模式
//class Singleton
//{	
//public:
//	static Singleton* GetInstance()
//	{
//		
//		if (SingletonInstance == NULL)
//		{
//			mu.lock();
//			if (SingletonInstance==nullptr)
//			{
//				SingletonInstance = new Singleton;
//			}
//			mu.unlock();
//		}
//			
//		return SingletonInstance;
//	}
//private:
//	static Singleton * SingletonInstance;
//	Singleton();
//	~Singleton();
//	Singleton(Singleton&);
//	Singleton& operator=(const Singleton&);
//};
//
//Singleton* Singleton::SingletonInstance = NULL;
//
////饿汉模式单例模式实例运行时立即初始化
//class SingHunggery
//{
//public:
//	static SingHunggery& GetInstance()
//	{
//		return instance;
//	}
//private:
//	static SingHunggery instance;
//	SingHunggery();
//	~SingHunggery();
//	SingHunggery(SingHunggery&);
//	SingHunggery& operator=(const SingHunggery&);
//};
//
//SingHunggery SingHunggery::instance;

//两个栈实现队列
class QueueBySatck
{
public:
	QueueBySatck();
	~QueueBySatck();
	void myPush(int value)
	{
		stack1.push(value);
	}

	int myPop()
	{
		if (stack1.empty() && stack2.empty())
			return -1;
		
		if (stack2.empty())
		{
			while (!stack1.empty())
			{
				stack2.push(stack1.top());
				stack1.pop();
			}
		}
		int value = stack2.top();
		stack2.pop();
		return value;
	}

	int myFront()
	{
		if (stack1.empty() && stack2.empty())
			return -1;
		return stack2.top();
	}
private:
	stack<int> stack1;
	stack<int> stack2;
};

QueueBySatck::QueueBySatck()
{
}

QueueBySatck::~QueueBySatck()
{
}

//两个队列实现一个栈
class StackByQueue
{
public:
	StackByQueue();
	~StackByQueue();
	void myPush(int value)
	{
		if (queue1.empty())
		{
			queue2.push(value);
		}
		else
		{
			queue1.push(value);
		}
	}

	int myPop()
	{
		if (queue1.empty() && queue2.empty())
			return -1;
		if (queue1.empty())
		{
			while (queue2.size() > 1)
			{
				queue1.push(queue2.front());
				queue2.pop();
			}
			int value = queue2.front();
			queue2.pop();
			return value;
		}
		else
		{
			while (queue1.size() > 1)
			{
				queue2.push(queue1.front());
				queue1.pop();
			}
			int value = queue1.front();
			queue1.pop();
			return value;
		}
	}

	int myTop()
	{
		if (queue1.empty() && queue2.empty())
			return -1;
		return queue2.front();
	}

private:
	queue<int> queue1;
	queue<int> queue2;
};

StackByQueue::StackByQueue()
{
}

StackByQueue::~StackByQueue()
{
}


//将这个问题进一步抽象，已知random_m()随机数生成器的范围是[1, m] 求random_n()生成[1, n]范围的函数，m < n && n <= m *m
int random_n()
{
	int val = 0;
	int n = 10;
	int t=100; // t为n最大倍数，且满足 t <= m * m  若m是7 n是10 则t=40
	do {
		//val = m * (random_m() - 1) + random_m();
	} while (val > t);
	return val%n+1;
}

//数据流中的中位数
class DataMid
{
public:
	DataMid();
	~DataMid();
	void put(const int value)
	{
		m_maxQue.push(value);
		m_minQue.push(m_maxQue.top());
		m_maxQue.pop();
		if (m_maxQue.size()<m_minQue.size())
		{
			m_maxQue.push(m_minQue.top());
			m_minQue.pop();
		}
	}
	double GetMidValue()
	{
		if (m_maxQue.empty())
			return -1;
		if (m_maxQue.size() > m_minQue.size())
		{
			return m_maxQue.top();
		}
		else
		{
			return (m_maxQue.top() + m_minQue.top()) / 2.0;
		}
	}

private:
	priority_queue<int> m_maxQue;
	priority_queue<int, vector<int>, greater<int>> m_minQue;
};

DataMid::DataMid()
{
}

DataMid::~DataMid()
{
}

//判断一个序列是不是栈混洗序列
bool IsTrueStac(const vector<int>& vecPush, const vector<int>&vecPop)
{
	if (vecPush.size() != vecPop.size()||vecPop.empty()||vecPush.empty())
		return false;
	const int length = vecPush.size();
	int newIndex = 0;
	stack<int> stacHelp;
	for (int i = 0; i < length; ++i)
	{
		stacHelp.push(vecPush[i]);
		while (!stacHelp.empty() && newIndex<length&&vecPop[newIndex]==stacHelp.top())
		{
			stacHelp.pop();
			++newIndex;
		}
	}
	return stacHelp.empty();
}

//求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
int sum_Solution(int n)
{
	int ans = n;
	ans&&(ans += sum_Solution(n - 1));
	return ans;
}

//序列化二叉树
class Codec {
public:
	// Encodes a tree to a single string.
	string serialize(TreeNode* root) {
		if (!root) return "";
		string res;
		SerHelp(root, res);
		if (!res.empty())
		{
			res.pop_back();
		}
		return res;
	}

	void SerHelp(TreeNode * root, string &res)
	{
		if (!root)
		{
			res += "#,";
			return;
		}
		res += to_string(root->val) + ",";
		SerHelp(root->left, res);
		SerHelp(root->right, res);
	}

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) {
		int pos = 0;
		int n = data.size();
		if (n <= 0) return nullptr;
		TreeNode *root = desHelp(data, pos, n);
		return root;
	}

	TreeNode * desHelp(string&data, int &index, const int length)
	{
		if (index >= length)
			return NULL;
		if (data[index] == '#')
		{
			index += 2;
			return NULL;
		}
		string Number;
		while (index < length&&data[index] != ',')
		{
			Number.push_back(data[index++]);
		}
		TreeNode * root = new TreeNode(stoi(Number));
		++index;
		root->left = desHelp(data, index, length);
		root->right = desHelp(data, index, length);
		return root;
	}
};

/******************************************************************************
 函数名称： maxSlidingWindow
 功能说明： 滑动窗口的最大值
 参    数： vector<int> & nums 
 参    数： int k 窗口大小
 返 回 值： std::vector<int>
 作    者： Ru Long
 日    期： 2020/02/28
******************************************************************************/
vector<int> maxSlidingWindow(vector<int>& nums, int k)
{
	if (nums.empty() || k < 1)
		return{};
	deque<int> deHelp;
	for (int i = 0; i < k&&i<nums.size();++i)
	{
		while (!deHelp.empty()&&nums[i]>=nums[deHelp.back()])
		{
			deHelp.pop_back();
		}
		deHelp.push_back(i);
	}
	vector<int> res;
	for (int i = k; i < nums.size();++i)
	{
		res.push_back(nums[deHelp.front()]);
		while (!deHelp.empty() && nums[i] >= nums[deHelp.back()])
		{
			deHelp.pop_back();
		}
		deHelp.push_back(i);
		if (!deHelp.empty()&&(i - deHelp.front())>= k)
		{
			deHelp.pop_front();
		}
	}
	res.push_back(nums[deHelp.front()]);
	return res;
}

//创建一个不能被继承的类
template<typename T> class MakeFinal
{
	friend T;
public:
private:
	MakeFinal();
	~MakeFinal();
};

class FinalClass:virtual public MakeFinal<FinalClass>
{
public:
	FinalClass();
	~FinalClass();
private:
};

//创建一个只能在栈上建立对象的类
class CreateOnlyStack
{
public:
	CreateOnlyStack();
	~CreateOnlyStack();

private:
	void * operator new(size_t);
	void operator delete(void *);
};

//创建一个只能在堆建立对象的类
class CreateOnlyHeap
{
public:
	static CreateOnlyHeap* GetCreateOnlyHeap(){
		return new CreateOnlyHeap();
	}
	void Destory()
	{
		delete this;
	}
protected:
CreateOnlyHeap();
~CreateOnlyHeap();
};


//KMP算法
void GetNextVec(const string & str, vector<int>&nextVec)
{
	if (str.empty())
	{
		return;
	}
	nextVec[0] = -1;
	int i = 0, j = -1;
	while (i<str.size()-1)
	{
		if (j == -1 || str[i] == str[j])
		{
			++j;
			++i;
			nextVec[i] = j;
		}
		else
		{
			j = nextVec[j];
		}
	}
}

int KMP(const string &target, const string&istr)
{
	if (target.empty()||istr.empty())
	{
		return -1;
	}
	vector<int> nextVec(istr.size());
	int i = 0, j = 0;
	const int lenstr = istr.size();
	//不能直接用istr.size()来判断。因为负数和一个无符号比较会统一转换成无符号数
	while (i < target.size() && j < lenstr)
	{
		if (j == -1 || target[i] == istr[j])
		{
			++j;
			++i;
		}
		else
		{
			j = nextVec[j];
		}
	}
	if (j == istr.size())
		return i - j;
	return -1;
}

//利用不均匀的抛硬币产生均匀的硬币 0概率为0.4 1概率为0.6
int foo()
{
	return 0;
}

int Coin()
{
	while (true)
	{
		int a = foo();
		if (a == foo())
		{
			return a;
		}
	}
}

//计算一个数的平方根
int GetSqrt(const int target)
{
	if (target<1)
	{
		return 0;
	}
	int left = 1, right = target;
	while (right-left>1)
	{
		int mid = left + (right - left) / 2;
		if (target/mid<mid)
		{
			right = mid;
		}
		else
		{
			left = mid;
		}
	}
	return left;
}

struct  nodeList
{
	int value;
	nodeList * pre;
	nodeList * next;
	nodeList(int iValue = 0) :value(iValue), pre(nullptr), next(nullptr){}
};

class myList
{
public:
	myList() :head(nullptr), tail(nullptr){}
	~myList();
	void push_back(int value)
	{
		nodeList * node = new nodeList(value);
		if (tail == nullptr)
		{
			node->next = tail;
			head = tail = node;
		}
		else
		{
			tail->next = node;
			node->pre = tail;
			tail = node;
		}
	}

	void pop_back()
	{
		if (head == tail)
		{
			delete head;
			head = tail = nullptr;
		}
		else
		{
			nodeList * pre = tail->pre;
			delete tail;
			tail = pre;
		}
	}

	void pop_front()
	{
		if (head == tail)
		{
			delete head;
			head = tail = nullptr;
		}
		else
		{
			nodeList * nextTemp = head->next;
			delete head;
			head = nextTemp;
		}
	}

	nodeList * find(const int key)
	{
		nodeList * begin = head;
		while (begin)
		{
			if (begin->value == key)
			{
				break;
			}
			begin = begin->next;
		}
		return begin;
	}
	void insert(nodeList * pos, const int value)
	{
		if (!pos)
		{
			return;
		}
		nodeList * pTemp = new nodeList(value);
		if (tail == pos)
		{
			tail->next = pTemp;
			pTemp->pre = tail;
			tail = pTemp;
		}
		else
		{
			pTemp->next = pos->next;
			pTemp->pre = pos;
			pTemp->next->pre = pTemp;
			pos->next = pTemp;
		}
	}

	void erase(nodeList *pos)
	{
		if (!pos)
		{
			return;
		}
		if (head == tail)
		{
			assert(head == pos);
			delete head;
			head = tail = nullptr;
			return;
		}
		if (pos == head)
		{
			head = head->next;
			head->pre = nullptr;
		}
		else if (pos == tail)
		{
			tail = tail->next;
			tail->next = nullptr;
		}
		else
		{
			pos->pre->next = pos->next;
			pos->next->pre = pos->pre;
		}
		delete pos;
		pos = nullptr;
	}

private:
	nodeList * head;
	nodeList * tail;
};

//智能指针的简单实现,成员变量是一个普通的指针用于解应用，一个int * 指针用于计数

//随机洗牌算法的实现
void Shuffe(vector<int>&nums)
{
	if (nums.empty())
	{
		return;
	}
	for (int i = nums.size() - 1; i > 0;--i)
	{
		srand((unsigned int)time(nullptr));
		swap(nums[i], nums[rand() % (i + 1)]);
	}
}

// class Block
// {
// public:
// 	size_t size = 0;
// 	bool free = false;
// 	int debug = 0;
// };
// 
// const int BLOCK_SIZE = sizeof(Block);
// std::list<Block *> mem_list;
// 
// Block *RequestSpace(size_t size) //需要处理分配异常
// {
// 	class Block *block;
// 	block = reinterpret_cast<Block *>(sbrk(0));			//得到当前堆顶部的指针
// 	void *request = sbrk(size + BLOCK_SIZE);			//sbrk(foo)递增堆大小foo并返回指向堆的上一个顶部的指针
// 	assert(reinterpret_cast<void *>(block) == request); 
// 	if (request == reinterpret_cast<void *>(-1))
// 	{
// 		throw std::bad_alloc();
// 		return NULL;
// 	}
// 	mem_list.push_back(block); //将其放到链表末尾
// 	block->size = size;
// 	block->free = false;
// 	return block;
// }
// Block *FindFreeBlock(size_t size)
// {
// 	for (auto t : mem_list)
// 	{
// 		if ((t->free) && (t->size >= size))
// 			return t;
// 	}
// 	return NULL;
// }
// 
// void *MyMalloc(size_t size)
// {
// 	if (size <= 0)
// 		return NULL;
// 	class Block *block;
// 	int mem_size = mem_list.size();
// 	if (mem_size == 0)
// 	{
// 		block = RequestSpace(size);
// 	}
// 	else
// 	{
// 		block = FindFreeBlock(size);
// 		if (!block) //有空余的空间，但是没有合适的空间可以分配给申请的大小
// 		{
// 			block = RequestSpace(size);
// 		}
// 		else //找到一块合适的内存块
// 		{
// 			block->free = false;
// 			// block->debug = xxxxxx;
// 		}
// 	}
// 	return (block + 1);
// 	/*这是为什么？
// 	答：这是因为他申请内存的时候都是sizeof(Block) + size,所以+1，直接让其指向原始内存　*/
// }
// void *MyCalloc(size_t n, size_t size)
// {
// 	size_t sum_size = n * size;
// 	void *ptr = MyMalloc(sum_size);
// 	memset(ptr, 0, sum_size);
// 	return ptr;
// }
// class Block *GetBlobkPtr(void *ptr)
// {
// 	return (class Block *)ptr - 1;
// }
// void MyFree(void *ptr)
// {
// 	if (!ptr)
// 		return;
// 	else
// 	{
// 		class Block *block_ptr = GetBlobkPtr(ptr);
// 		assert(block_ptr->free == false);
// 		block_ptr->free = true;
// 	}
// }
// void *MyRealloc(void *ptr, size_t size)
// {
// 	if (!ptr)
// 	{
// 		return MyMalloc(size);
// 	}
// 	class Block *block = GetBlobkPtr(ptr);
// 	if (block->size >= size)
// 		return ptr; //空间足够了
// 	void *new_ptr = MyMalloc(size);
// 	memcpy(new_ptr, ptr, block->size);
// 	MyFree(ptr);
// 	return new_ptr;
// }


//让你设计一个微信发红包的api，你会怎么设计，不能有人领到的红包里面没钱，红包数值精确到分。
//** 思路** ：
//假设一共有 N 元，一共有 K 个人，则可以每个人拿到的钱为 random(N - (K - 1) * 0.01)，然后更新N，直到最后一个人就直接 N。
