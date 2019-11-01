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
	/******************************************************************************
	 函数名称： nextPermutation
	 功能说明： 返回下一个最近且比这个数大的数，如果没有按升序排列
	 参    数： vector<int> & nums 
	 返 回 值： void
	 作    者： Ru Long
	 日    期： 2019/10/30
	******************************************************************************/
	void nextPermutation(vector<int>& nums);
	/******************************************************************************
	 函数名称： search
	 功能说明： 在一个未知点旋转的升序数组中，找到某个目标值，要logn
	 参    数： vector<int> & nums 
	 参    数： int target 
	 返 回 值： int
	 作    者： Ru Long
	 日    期： 2019/10/30
	******************************************************************************/
	int search(vector<int>& nums, int target);
	/******************************************************************************
	 函数名称： searchRange
	 功能说明： 在一个升序数组中找到最左和最右为target的数
	 参    数： vector<int> & nums 
	 参    数： int target 
	 返 回 值： std::vector<int>
	 作    者： Ru Long
	 日    期： 2019/10/31
	******************************************************************************/
	vector<int> searchRange(vector<int>& nums, int target);
	/******************************************************************************
	 函数名称： isValidSudoku
	 功能说明： 判断是否是一个有效的数独，其中每一行和每一列1-9只能出现一次,在3*3的方格中也只能出现一次
	 参    数： vector<vector<char>> & board 
	 返 回 值： bool
	 作    者： Ru Long
	 日    期： 2019/10/31
	******************************************************************************/
	bool isValidSudoku(vector<vector<char>>& board);
	void DFSSum(int start, int target);
	/******************************************************************************
	 函数名称： combinationSum
	 功能说明： 给定一个无重复数组和目标值，使得数组中的值要等于target，数组中的值可以无限选择,采用回溯算法
	 参    数： vector<int> & candidates 
	 参    数： int target 
	 返 回 值： std::vector<std::vector<int>>
	 作    者： Ru Long
	 日    期： 2019/10/31
	******************************************************************************/
	vector<vector<int>> combinationSum(vector<int>& candidates, int target);
	void DFSSum2(int start, int target);
	/******************************************************************************
	 函数名称： combinationSum2
	 功能说明： 给定一个数组和目标值，使得数组中的值要等于target，数组中的值只能使用一次,元素有重复
	 参    数： vector<int> & candidates 
	 参    数： int target 
	 返 回 值： std::vector<std::vector<int>>
	 作    者： Ru Long
	 日    期： 2019/11/01
	******************************************************************************/
	vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
	/******************************************************************************
	 函数名称： multiply
	 功能说明： 大数相乘，两个字符串的乘积
	 参    数： string num1 
	 参    数： string num2 
	 返 回 值： std::string
	 作    者： Ru Long
	 日    期： 2019/11/01
	******************************************************************************/
	string multiply(string num1, string num2);
	/******************************************************************************
	 函数名称： Select
	 功能说明： 寻找第k小的元素
	 参    数： T a[] 
	 参    数： int n 
	 参    数： int k 
	 返 回 值： T
	 作    者： Ru Long
	 日    期： 2019/11/01
	******************************************************************************/
	template<typename T>
	T Select(T a[], int n, int k);
private:
	int expandAroundCenter(const string &s, int left, int right);
	template<typename T>
	T Select(T a[], int leftEnd, int rightEnd, int k);
	
};

template<typename T>
T Solution::Select(T a[], int n, int k)
{
	if (k<1||k>n)
	{
		return T[0];
	}
	int maxPos = 0;
	for (int i = 1; i < n;++i)
	{
		if (a[i]>a[maxPos])
		{
			maxPos = i;
		}
	}
	swap(a[n - 1], a[maxPos]);
	return Select(a, 0, n - 1, k);
}

template<typename T>
T Solution::Select(T a[], int leftEnd, int rightEnd, int k)
{
	if (leftEnd>=rightEnd)
	{
		return a[leftEnd];
	}
	int leftCur = leftEnd;
	int rightCur = rightEnd + 1;
	T privot = a[leftEnd];
	while (true)
	{
		do 
		{
			++leftCur;
		} while (a[leftCur]<privot);
		do 
		{
			--rightCur;
		} while (a[rightCur]>privot);
		if (leftCur>=rightCur)
		{
			break;
		}
		swap(a[leftCur], a[rightCur]);
	}
	if (rightCur-leftEnd+1==k)
	{
		return privot;
	}
	a[leftEnd] = a[rightCur];
	a[rightCur] = privot;
	if (rightCur-leftEnd+1<k)
	{
		return Select(a, rightCur + 1, rightEnd, k - rightEnd + leftEnd -1);
	}
	return Select(a,leftEnd,rightCur-1,k);
}

#endif 
