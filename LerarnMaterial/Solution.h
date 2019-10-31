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
	/******************************************************************************
	 函数名称： combinationSum
	 功能说明： 给定一个数组和目标值，使得数组中的值要等于target，数组中的值可以无限选择
	 参    数： vector<int> & candidates 
	 参    数： int target 
	 返 回 值： std::vector<std::vector<int>>
	 作    者： Ru Long
	 日    期： 2019/10/31
	******************************************************************************/
	vector<vector<int>> combinationSum(vector<int>& candidates, int target);
private:
	int expandAroundCenter(const string &s, int left, int right);
	
};

#endif 
