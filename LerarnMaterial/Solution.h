#ifndef SOLUTION_H
#define SOLUTION_H
#include <string>
#include <vector>
#include <unordered_map>
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
#include "Common.h"
using namespace std;
class Solution
{
public:
	Solution();
	~Solution();
	/******************************************************************************
	 函数名称： longestPalindrome
	 功能说明： 找到最长的回文子字符串,从中间向2边开花求，马车算法
	 参    数： string s 
	 返 回 值： std::string
	 作    者： Ru Long
	 日    期： 2019/10/26
	******************************************************************************/
	string longestPalindrome(string s);
	void DfsParent(vector<string> &istrVec,string istr,int l,int r);
	/******************************************************************************
	 函数名称： generateParenthesis
	 功能说明： 生成可能的n对括号的正确组合
	 参    数： int n 
	 返 回 值： std::vector<std::string>
	 作    者： Ru Long
	 日    期： 2019/10/28
	******************************************************************************/
	vector<string> generateParenthesis(int n);
	/******************************************************************************
	 函数名称： swapPairs
	 功能说明： 两两交换2个相邻的节点
	 参    数： ListNode * head 
	 返 回 值： ListNode*
	 作    者： Ru Long
	 日    期： 2019/10/28
	******************************************************************************/
	ListNode* swapPairs(ListNode* head);
	/******************************************************************************
	 函数名称： divide
	 功能说明： 给定2个整数，要求不使用乘法除法和mod运算，返回结果
	 参    数： int dividend 
	 参    数： int divisor 
	 返 回 值： int
	 作    者： Ru Long
	 日    期： 2019/10/28
	******************************************************************************/
	int divide(int dividend, int divisor);
private:
	int expandAroundCenter(const string &s, int left, int right);
	
};

#endif 
