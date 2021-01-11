#include "FormalWork.h"

std::vector<double> FormalWork::calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries)
{
	if (equations.empty())
	{
		return{};
	}
	unordered_map<string, int> store;  //�洢��Ӧ��string��Ӧ�Ľڵ�index
	int node = 0;
	const int OrignNumber = 10000;
	//ͳ�ƽڵ�����
	for (int i = 0; i < equations.size(); ++i)
	{
		if (store.count(equations[i][0]) == 0)
		{
			store[equations[i][0]] = node++;
		}
		if (store.count(equations[i][1]) == 0)
		{
			store[equations[i][1]] = node++;
		}
	}
	//��ʼ��ͼ
	vector<vector<double>> graph(node, vector<double>(node, OrignNumber));
	for (int i = 0; i < equations.size();++i)
	{
		graph[store[equations[i][0]]][store[equations[i][1]]] = values[i];
		graph[store[equations[i][1]]][store[equations[i][0]]] = 1.0/values[i];
	}
	//ʹ�ø��������㷨
	for (int k = 0; k < node; ++k)
	{
		for (int i = 0; i < node; ++i)
		{
			for (int j = 0; j < node; ++j)
			{
				if (graph[i][k] != OrignNumber&&graph[k][j] != OrignNumber&&graph[i][j] == OrignNumber)
				{
					graph[i][j] = graph[i][k] * graph[k][j];
				}
			}
		}
	}
	std::vector<double> ret(queries.size(), -1);
	for (int i = 0; i < queries.size();++i)
	{
		if (store.count(queries[i][0]) && store.count(queries[i][1]))
		{
			if (graph[store[queries[i][0]]][store[queries[i][1]]] != OrignNumber)
			{
				ret[i] = graph[store[queries[i][0]]][store[queries[i][1]]];
			}
		}
	}
	return ret;
}

int GetFather(vector<int>& parent, int node)
{
	return parent[node]==node?node:parent[node]=GetFather(parent,parent[node]);
}

void Union(vector<int>& parent, int left, int right)
{
	parent[GetFather(parent, left)] = GetFather(parent, right);
}

int FormalWork::findCircleNum(vector<vector<int>>& isConnected)
{
	if (isConnected.empty())
	{
		return 0;
	}
	vector<int> parent(isConnected[0].size());
	for (int i = 0; i < parent.size();++i)
	{
		parent[i] = i;
	}
	for (int k = 0; k < isConnected.size();++k)
	{
		for (int i = k+1; i < isConnected[k].size(); ++i)
		{
			if (isConnected[k][i] == 1)
			{
				Union(parent, k, i);
			}
		}
	}
	std::set<int> ret;
	for (auto &it:parent)
	{
		ret.insert(GetFather(parent,it));
	}
	return ret.size();
}

void FormalWork::rotate(vector<int>& nums, int k)
{
	if (nums.empty())
	{
		return;
	}
	k = k%nums.size();
	reverse(nums.begin(), nums.begin() + nums.size() - k);
	reverse(nums.begin() + nums.size() - k, nums.end());
	reverse(nums.begin(), nums.end());
}

std::vector<std::string> FormalWork::summaryRanges(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	if (nums.empty())
		return{};
	std::set<int> sHelp(nums.begin(), nums.end());
	vector<string> ret;
	for (auto it = sHelp.begin(); it != sHelp.end();)
	{
		auto preIt = it;
		auto nexIt = it;
		++nexIt;
		while (nexIt != sHelp.end() && *nexIt==*preIt+1)
		{
			preIt = nexIt;
			++nexIt;
		}
		if (*preIt == *it)
		{
			ret.emplace_back(to_string(*it));
			++it;
		}
		else
		{
			ret.emplace_back(to_string(*it) + "->" + to_string(*preIt));
			it = nexIt;
		}
	}
	return ret;
}

std::string FormalWork::smallestStringWithSwaps(string s, vector<vector<int>>& pairs)
{
	if (pairs.empty()||s.empty())
	{
		return s;
	}
	vector<int> parent(s.size());
	for (int i = 0; i < parent.size(); ++i)
	{
		parent[i] = i;
	}
	for (int k = 0; k < pairs.size(); ++k)
	{
		Union(parent, pairs[k][0],pairs[k][1]);
	}
	unordered_map<int, priority_queue<int,vector<int>,greater<int>>> hashM;
	for (int i = 0; i < parent.size();++i)
	{
		hashM[GetFather(parent,i)].push(s[i]);
	}
	for (int i = 0; i < s.size();++i)
	{
		s[i] = hashM[GetFather(parent, i)].top();
		hashM[GetFather(parent, i)].pop();
	}
	return s;
}
