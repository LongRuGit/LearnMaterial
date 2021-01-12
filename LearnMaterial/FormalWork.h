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
	/******************************************************************************
	 �������ƣ� smallestStringWithSwaps
	 ����˵���� 1202-�����ַ���
	 ��    ���� string s 
	 ��    ���� vector<vector<int>> & pairs 
	 �� �� ֵ�� std::string
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/11
	******************************************************************************/
	string smallestStringWithSwaps(string s, vector<vector<int>>& pairs);
	/******************************************************************************
	 �������ƣ� sortItems
	 ����˵���� ��Ŀ����-1203
	 ��    ���� int n 
	 ��    ���� int m 
	 ��    ���� vector<int> & group 
	 ��    ���� vector<vector<int>> & beforeItems 
	 �� �� ֵ�� std::vector<int>
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/12
	******************************************************************************/
	vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems);
};

