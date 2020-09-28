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
	TreeNode * left;
	TreeNode * right;
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

	void Union(int iLeftNode,int iRightNode)
	{
		int x = find(iLeftNode);
		int y = find(iRightNode);
		if (x!=y)
		{
			m_vecParent[x] = y;
		}
	}

	bool IsConnect(int leftNode, int rightNode)
	{
		int x = find(leftNode);
		int y = find(rightNode);
		return x == y;
	}

private:
	vector<int> m_vecParent;

	int find(int iNode)
	{
		return iNode == m_vecParent[iNode] ? iNode : (m_vecParent[iNode] = find(m_vecParent[iNode]));
	}
};

#endif 