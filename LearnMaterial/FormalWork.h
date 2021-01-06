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
};

