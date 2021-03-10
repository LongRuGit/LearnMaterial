#include "FormalWork.h"

std::vector<double> FormalWork::calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries)
{
	if (equations.empty())
	{
		return{};
	}
	unordered_map<string, int> store;  //存储对应的string对应的节点index
	int node = 0;
	const int OrignNumber = 10000;
	//统计节点数量
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
	//初始化图
	vector<vector<double>> graph(node, vector<double>(node, OrignNumber));
	for (int i = 0; i < equations.size();++i)
	{
		graph[store[equations[i][0]]][store[equations[i][1]]] = values[i];
		graph[store[equations[i][1]]][store[equations[i][0]]] = 1.0/values[i];
	}
	//使用弗洛伊德算法
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

vector<int> topSort(vector<int>& deg, vector<vector<int>>& graph, vector<int>& items) 
{
	queue<int> Q;
	for (auto& item : items)
	{
		if (deg[item] == 0) 
		{
			Q.push(item);
		}
	}
	vector<int> res;
	while (!Q.empty())
	{
		int u = Q.front();
		Q.pop();
		res.emplace_back(u);
		for (auto& v : graph[u])
		{
			if (--deg[v] == 0)
			{
				Q.push(v);
			}
		}
	}
	return res.size() == items.size() ? res : vector<int>{};
}

std::vector<int> FormalWork::sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems)
{
	vector<vector<int>> groupItem(n + m);

	// 组间和组内依赖图
	vector<vector<int>> groupGraph(n + m);
	vector<vector<int>> itemGraph(n);

	// 组间和组内入度数组
	vector<int> groupDegree(n + m, 0);
	vector<int> itemDegree(n, 0);

	vector<int> id;
	for (int i = 0; i < n + m; ++i) 
	{
		id.emplace_back(i);
	}

	int leftId = m;
	// 给未分配的 item 分配一个 groupId
	for (int i = 0; i < n; ++i)
	{
		if (group[i] == -1)
		{
			group[i] = leftId;
			leftId += 1;
		}
		groupItem[group[i]].emplace_back(i);
	}
	// 依赖关系建图
	for (int i = 0; i < n; ++i) 
	{
		int curGroupId = group[i];
		for (auto& item : beforeItems[i]) 
		{
			int beforeGroupId = group[item];
			if (beforeGroupId == curGroupId) 
			{
				itemDegree[i] += 1;
				itemGraph[item].emplace_back(i);
			}
			else
			{
				groupDegree[curGroupId] += 1;
				groupGraph[beforeGroupId].emplace_back(curGroupId);
			}
		}
	}

	// 组间拓扑关系排序
	vector<int> groupTopSort = topSort(groupDegree, groupGraph, id);
	if (groupTopSort.size() == 0) 
	{
		return vector<int>{};
	}
	vector<int> ans;
	// 组内拓扑关系排序
	for (auto& curGroupId : groupTopSort)
	{
		int size = groupItem[curGroupId].size();
		if (size == 0)
		{
			continue;
		}
		vector<int> res = topSort(itemDegree, itemGraph, groupItem[curGroupId]);
		if (res.size() == 0) 
		{
			return vector<int>{};
		}
		for (auto& item : res)
		{
			ans.emplace_back(item);
		}
	}
	return ans;
}

std::vector<int> FormalWork::findRedundantConnection(vector<vector<int>>& edges)
{
	if (edges.empty())
	{
		return{};
	}
	vector<int> parent(edges.size() + 1);
	for (int i = 0; i < parent.size();++i)
	{
		parent[i] = i;
	}
	for (auto ed:edges)
	{
		int node1 = ed[0], node2 = ed[1];
		if (GetFather(parent,node1)!=GetFather(parent,node2))
		{
			Union(parent, node1, node2);
		}
		else
		{
			return ed;
		}
	}
	return{};
}

vector<bool> FormalWork::prefixesDivBy5(vector<int>& A)
{
	if (A.empty())
	{
		return{};
	}
	vector<bool> ret(A.size());
	int num = 0;
	for (int i = 0; i < A.size(); ++i)
	{
		num = num * 2 + A[i];
		num %= 5;
		ret[i] = num % 5 == 0 ? true : false;
	}
	return ret;
}

class DisjoinSetUnion
{
public:
	int find(int x)
	{
		if (!f.count(x))
		{
			f[x] = x;
			rank[x] = 1;
		}
		return f[x] == x ? x : f[x] = find(f[x]);
	}

	void UnionSet(int x, int y)
	{
		int fx = find(x), fy = find(y);
		if (fx==fy)
		{
			return;
		}
		if (rank[fx]<rank[fy])
		{
			std::swap(fx, fy);
		}
		rank[fx] += rank[fy];
		f[fy] = fx;
	}

	int numerOfConnectedCompnent()
	{
		int num = 0;
		for (auto &it:f)
		{
			if (it.first==it.second)
			{
				++num;
			}
		}
		return num;
	}

private:
	unordered_map<int, int> f, rank;
};

int FormalWork::removeStones(vector<vector<int>> &stones)
{
	if (stones.empty())
	{
		return 0;
	}
	int n = stones.size();
	DisjoinSetUnion dsu;
	for (int i = 0; i < n;++i)
	{
		dsu.UnionSet(stones[i][0], stones[i][1] + 10001);
	}
	return n - dsu.numerOfConnectedCompnent();
}

int CountNumberOfOne(int target)
{
	int ret = 0;
	while (target)
	{
		target &= (target - 1);
		++ret;
	}
	return ret;
}

std::vector<int> FormalWork::findNumOfValidWords(vector<string>& words, vector<string>& puzzles)
{
	if (words.empty()||puzzles.empty())
	{
		return{};
	}
	unordered_map<int,int> hashM;
	for (int i = 0; i < words.size();++i)
	{
		int mask = 0;
		for (auto &ch:words[i])
		{
			mask |= (1 << (ch - 'a'));
		}
		if (CountNumberOfOne(mask)<=7)
		{
			++hashM[mask];
		}
	}
	vector<int> ret;
	for (auto &puzz:puzzles)
	{
		int mask = 0;
		for (int i = 1; i < 7;++i)
		{
			mask |= (1 << (puzz[i] - 'a'));
		}
		int total = 0;
		int subset = mask;
		do 
		{
			int s = subset | (1 << (puzz[0] - 'a'));
			if (hashM.count(s))
			{
				total += hashM[s];
			}
			//求二进制中所有集合的子集
			subset = (subset - 1) & mask;
		} while (subset != mask);
		ret.push_back(total);
	}
	return ret;
}

int FormalWork::longestSubstring(string s, int k)
{
	if (s.size()<k)
	{
		return 0;
	}
	vector<int> num(26);
	for (auto &it:s)
	{
		++num[it - 'a'];
	}
	int split = 0;
	while (split < s.size() && num[s[split] - 'a' ]>=k)
	{
		++split;
	}
	if (split==s.size())
	{
		return split;
	}
	int leftRet = longestSubstring(s.substr(0, split), k);
	while (split < s.size() && num[s[split] - 'a'] < k)
	{
		++split;
	}
	return std::max(leftRet, longestSubstring(s.substr(split), k));
}

std::vector<int> FormalWork::countBits(int num)
{
	vector<int> ret(num+1,0);
	for (int i = 1; i <= num;++i)
	{
		int pre = i&(i - 1);
		ret[i] = 1 + ret[pre];
	}
	return ret;
}

int FormalWork::maxEnvelopes(vector<vector<int>>& envelopes)
{
	if (envelopes.empty())
	{
		return 0;
	}
	sort(envelopes.begin(), envelopes.end(), [&](const vector<int>& e1, const vector<int>& e2)
	{
		return (e1[0]<e2[0]) || (e1[0] == e2[0] && e1[1]>e2[1]);
	});
	vector<int> num(envelopes.size(), 1);
	int ret = 1;
	for (int i = 0; i < envelopes.size();++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (envelopes[i][1]>envelopes[j][1])
			{
				num[i] = std::max(num[i], 1+num[j]);
			}
			ret = std::max(num[i], ret);
		}
	}
	return ret;
}

std::vector<int> FormalWork::nextGreaterElements(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	vector<int> ret(nums.size(),-1);
	stack<int> stac;
	//循环数组
	for (int i = 0; i < 2 * nums.size() - 1;++i)
	{
		while (!stac.empty() && nums[stac.top()] < nums[i%nums.size()])
		{
			ret[stac.top()] = nums[i%nums.size()];
			stac.pop();
		}
		stac.push(i%nums.size());
	}
	return ret;
}

bool CheckIsBackNew(const string& target,int left,int right)
{
	while (left<right)
	{
		if (target[left]!=target[right])
		{
			return false;
		}
		++left;
		--right;
	}
	return true;
}

void BFSPartion(vector<vector<string>>& ret, vector<string>& path, int start, const string &Target)
{
	if (start==Target.size())
	{
		ret.push_back(path);
		return;
	}
	for (int i = start+1; i <= Target.size();++i)
	{
		if (CheckIsBackNew(Target, start, i - 1))
		{
			path.push_back(Target.substr(start, i - start));
			BFSPartion(ret, path, i, Target);
			path.pop_back();
		}
	}
}

std::vector<std::vector<std::string>> FormalWork::partition(string s)
{
	if (s.empty())
	{
		return{};
	}
	vector<vector<string>> ret;
	vector<string> path;
	BFSPartion(ret, path, 0, s);
	return ret;
}

int FormalWork::minCut(string s)
{
	if (s.empty())
	{
		return 0;
	}
	const int len = s.size();
	vector<vector<bool>> g(len,vector<bool>(len, true));
	for (int i = len - 1; i >= 0;--i)
	{
		for (int j = len - 1; j >= i+1;--j)
		{
			g[i][j] = (s[i] == s[j]) && g[i + 1][j - 1];
		}
	}
	vector<int> num(len, INT_MAX);
	num[0] = 0;
	for (int i = 1; i < len;++i)
	{
		if (g[0][i])
		{
			num[i] = 0;
		}
		else
		{
			for (int j = 0; j < i; ++j)
			{
				if (g[j + 1][i])
				{
					num[i] = std::min(num[i], num[j] + 1);
				}
			}
		}
	}
	return num.back();
}

std::string FormalWork::removeDuplicates(string S)
{
	if (S.empty())
	{
		return "";
	}
	string ret;
	for (int i = 0; i < S.size();++i)
	{
		if (!ret.empty() && ret.back() == S[i])
		{	
			ret.pop_back();
		}
		else
		{
			ret.push_back(S[i]);
		}
	}
	return ret;
}

int FormalWork::calculate(string s)
{
	if (s.empty())
	{
		return 0;
	}
	stack<int> numStac;
	numStac.push(1);
	int ret = 0;
	int sign = 1;
	int index = 0;
	while (index<s.size())
	{
		if (s[index]==' ')
		{
			++index;
		}
		else if (s[index]=='+')
		{
			sign = numStac.top();
			++index;
		}
		else if (s[index]=='-')
		{
			sign = -numStac.top();
			++index;
		}
		else if (s[index]=='(')
		{
			numStac.push(sign);
			++index;
		}
		else if (s[index]==')')
		{
			numStac.pop();
			++index;
		}
		else
		{
			int temp = 0;
			while (index<s.size()&&s[index]>='0'&&s[index]<='9')
			{
				temp = temp * 10 + (s[index] - '0');
				++index;
			}
			ret += sign*temp;
		}
	}
	return ret;
}
