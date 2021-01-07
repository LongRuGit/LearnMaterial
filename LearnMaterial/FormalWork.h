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
};

