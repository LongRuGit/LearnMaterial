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

#endif 