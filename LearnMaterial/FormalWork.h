#pragma once
#include "Common.h"
#include "Singleton.h"

class FormalWork :public Singleton<FormalWork>
{
public:
	/******************************************************************************
	 �������ƣ� calcEquation
	 ����˵���� ������ֵ-399
	 ��    ���� vector<vector<string>> & equations 
	 ��    ���� vector<double> & values 
	 ��    ���� vector<vector<string>> & queries 
	 �� �� ֵ�� std::vector<double>
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/06
	******************************************************************************/
	vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries);
	/******************************************************************************
	 �������ƣ� findCircleNum
	 ����˵���� ʡ������-547
	 ��    ���� vector<vector<int>> & isConnected 
	 �� �� ֵ�� int
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/07
	******************************************************************************/
	int findCircleNum(vector<vector<int>>& isConnected);
	/******************************************************************************
	 �������ƣ� rotate
	 ����˵���� ��ת����-189
	 ��    ���� vector<int> & nums 
	 ��    ���� int k 
	 �� �� ֵ�� void
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/08
	******************************************************************************/
	void rotate(vector<int>& nums, int k);
	/******************************************************************************
	 �������ƣ� summaryRanges
	 ����˵���� ��������228
	 ��    ���� vector<int> & nums 
	 �� �� ֵ�� std::vector<std::string>
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/10
	******************************************************************************/
	vector<string> summaryRanges(vector<int>& nums);
};

