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
	/******************************************************************************
	 �������ƣ� findRedundantConnection
	 ����˵���� 684-��������
	 ��    ���� vector<vector<int>> & edges 
	 �� �� ֵ�� std::vector<int>
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/13
	******************************************************************************/
	vector<int> findRedundantConnection(vector<vector<int>>& edges);
	/******************************************************************************
	 �������ƣ� prefixesDivBy5
	 ����˵���� �ɱ�5����-1018
	 ��    ���� vector<int> & A 
	 �� �� ֵ�� std::vector<bool>
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/14
	******************************************************************************/
	vector<bool> prefixesDivBy5(vector<int>& A);
	/******************************************************************************
	 �������ƣ� removeStones
	 ����˵���� �Ƴ�����ʯ��-947
	 ��    ���� vector<vector<int>> & stones 
	 �� �� ֵ�� int
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/01/15
	******************************************************************************/
	int removeStones(vector<vector<int>> &stones);
	/******************************************************************************
	 �������ƣ� findNumOfValidWords
	 ����˵���� ������-1178
	 ��    ���� vector<string> & words 
	 ��    ���� vector<string> & puzzles 
	 �� �� ֵ�� std::vector<int>
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/02/26
	******************************************************************************/
	vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles);
	/******************************************************************************
	 �������ƣ� longestSubstring
	 ����˵���� ������ K ���ظ��ַ�����Ӵ�-395
	 ��    ���� string s 
	 ��    ���� int k 
	 �� �� ֵ�� int
	 ��    �ߣ� Ru Long
	 ��    �ڣ� 2021/02/27
	******************************************************************************/
	int longestSubstring(string s, int k);
};

