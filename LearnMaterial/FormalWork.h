#pragma once
#include "Common.h"
#include "Singleton.h"

class FormalWork :public Singleton<FormalWork>
{
public:
	/******************************************************************************
	 函数名称： calcEquation
	 功能说明： 除法求值-399
	 参    数： vector<vector<string>> & equations 
	 参    数： vector<double> & values 
	 参    数： vector<vector<string>> & queries 
	 返 回 值： std::vector<double>
	 作    者： Ru Long
	 日    期： 2021/01/06
	******************************************************************************/
	vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries);
	/******************************************************************************
	 函数名称： findCircleNum
	 功能说明： 省份数量-547
	 参    数： vector<vector<int>> & isConnected 
	 返 回 值： int
	 作    者： Ru Long
	 日    期： 2021/01/07
	******************************************************************************/
	int findCircleNum(vector<vector<int>>& isConnected);
	/******************************************************************************
	 函数名称： rotate
	 功能说明： 旋转数组-189
	 参    数： vector<int> & nums 
	 参    数： int k 
	 返 回 值： void
	 作    者： Ru Long
	 日    期： 2021/01/08
	******************************************************************************/
	void rotate(vector<int>& nums, int k);
	/******************************************************************************
	 函数名称： summaryRanges
	 功能说明： 汇总区间228
	 参    数： vector<int> & nums 
	 返 回 值： std::vector<std::string>
	 作    者： Ru Long
	 日    期： 2021/01/10
	******************************************************************************/
	vector<string> summaryRanges(vector<int>& nums);
};

