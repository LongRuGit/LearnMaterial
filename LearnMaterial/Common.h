#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <string>
#include <stack>
#include <iostream>
#include <set>
#include <map>
#include <bitset>
#include "math.h"
#include <queue>
#include <memory>
#include <list>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <climits>
#include <assert.h>
#include <utility>
#include <functional>
#include <time.h>

using namespace std;

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

using namespace std;

//并查集模板算法的实现，主要判断是否有连通性
class UnionFind
{
public:
	void Initialize(const int &iNumber)
	{
		m_vecParent.resize(iNumber);
		for (int i = 0; i < iNumber; ++i)
		{
			m_vecParent[i] = i;
		}
	}

	int find(int iNode)
	{
		while (iNode != m_vecParent[iNode])
		{
			//路径压缩算法
			m_vecParent[iNode] = m_vecParent[m_vecParent[iNode]];
			iNode = m_vecParent[iNode];
		}
		return m_vecParent[iNode];
	}

	void Union(int iLeftNode,int iRightNode)
	{
		m_vecParent[find(iLeftNode)] = find(iRightNode);
	}

private:
	vector<int> m_vecParent;
};

#endif 