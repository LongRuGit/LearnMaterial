#include "Common.h"
#include "Solution.h"
#include <tuple>
#include <random>
#include <regex>
#include "SortClass.h"
#include "SolutionMediumNew.h"
#include "AutumnMove.h"
#include "FormalWork.h"

void printSudo(vector<vector<char>>& bb) 
{
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++) 
		{
			cout << bb[i][j] << ' ';
		}
		cout << endl;
	}
}

template<typename T>
void PrintVec(const vector<T>& nums)
{
	cout << "\n";
	for (auto &it:nums)
	{
		cout << it << " ";
	}
	cout << "\n";
}

//取二进制（非符号位）的最高位1
uint64_t hight_bitCur(uint64_t x){//0010 1100 0000 0000 0000 0000 0000 0000 0000 0001
	x = x | (x >> 1);              //0011 1110 0000 0000 0000 0000 0000 0000 0000 0000
	x = x | (x >> 2);              //0011 1111 1000 0000 0000 0000 0000 0000 0000 0000
	x = x | (x >> 4);              //0011 1111 1111 1000 0000 0000 0000 0000 0000 0000
	x = x | (x >> 8);              //0011 1111 1111 1111 1111 1000 0000 0000 0000 0000
	x = x | (x >> 16);             //0011 1111 1111 1111 1111 1111 1111 1111 1111 1111
	x = x | (x >> 32);
	// 如果数特别大， 这里感觉会溢出， 所以这里只使用于小于数据最大值1/2的数。
	return (x + 1);        //0100 0000 0000 0000 0000 0000 0000 0000 0000 0000
}

ListNode* GetListNode(vector<int>& nums)
{
	ListNode* ret = new ListNode;
	ListNode* cur = ret;
	for (auto &it : nums)
	{
		cur->next = new ListNode(it);
		cur = cur->next;
	}
	cur = ret->next;
	delete ret;
	return cur;
}


typedef std::vector<int> HFHZGraghLoop;
int pMatrix[100][100];
int V;

static bool LoopBIsSameToA(const HFHZGraghLoop& iA, const HFHZGraghLoop& iB)
{
	if (iA.size() != iB.size())
	{
		return false;
	}
	for (size_t i = 0; i < iA.size(); ++i)
	{
		if (iA[i] != iB[i])
		{
			return false;
		}
	}

	return true;
}

void UniqueLoops(std::vector<HFHZGraghLoop>& ioLoops)
{
	if (ioLoops.size() > 1)
	{
		//排序去重，默认所有的loop都是重复的
		std::vector<HFHZGraghLoop> allSortedLoops;
		//记录allSortedLoops是ioLoops中的哪几个loop
		std::vector<size_t>	allSortedLoopsIndex;
		std::vector<HFHZGraghLoop> tempLoops = ioLoops;
		for (size_t i = 0; i < tempLoops.size(); ++i)
		{
			std::sort(tempLoops[i].begin(), tempLoops[i].end());
			if (std::unique(tempLoops[i].begin(), tempLoops[i].end()) == tempLoops[i].end())
			{
				//说明没有重复的
				allSortedLoops.push_back(tempLoops[i]);
				allSortedLoopsIndex.push_back(i);
			}
		}
		//1 -- 表示需要的结果
		//0 -- 表示没有被检查过
		//-1-- 表示重复的结果
		std::vector<int> allUniqueLoop(allSortedLoops.size(), 0);
		for (size_t i = 0; i < allSortedLoops.size(); ++i)
		{
			if (allUniqueLoop[i] == 0)
			{
				allUniqueLoop[i] = 1;
			}
			else
			{
				continue;
			}
			for (size_t j = i + 1; j < allSortedLoops.size(); ++j)
			{
				if (allUniqueLoop[j] != 0)
					continue;
				if (LoopBIsSameToA(allSortedLoops[i], allSortedLoops[j]))
				{
					allUniqueLoop[j] = -1;
				}
			}
		}
		std::vector<HFHZGraghLoop> tempResult;
		for (size_t i = 0; i < allUniqueLoop.size(); ++i)
		{
			if (allUniqueLoop[i] == 1)
			{
				//修改时间：2020/09/29 ，修改人：konghaijiao
				//修改原因：返回的loop之前弄错了
				tempResult.push_back(ioLoops[allSortedLoopsIndex[i]]);
			}
		}
		ioLoops = tempResult;
	}
}

void ShortestCirclesFromVertex(int iVertex, std::vector<HFHZGraghLoop>& ovCircles, bool ibWeigth/* = true*/)
{
	ovCircles.clear();

	std::vector<HFHZGraghLoop> allMinLoops;
	// To store length of the shortest cycle 
	int ans = INT_MAX;
	// Make distance maximum 
	std::vector<int> dist(V, (int)(1e9));

	// Take a imaginary parent 
	std::vector<int> par(V, -1);

	// Distance of source to source is 0 
	dist[iVertex] = 0;
	std::queue<int> q;

	// Push the source element 
	q.push(iVertex);

	// Continue until queue is not empty 
	while (!q.empty())
	{
		// Take the first element 
		int x = q.front();
		q.pop();

		// Traverse for all it's childs 
		for (int child = 0; child < V; ++child)
		{
			//0 -- 说明两者没有直接相连
			if (0 == pMatrix[x][child])
			{
				continue;
			}
			//修改时间：2020/07/20 ，修改人：konghaijiao
			//修改原因：增加权重
			int weight = ibWeigth ? pMatrix[x][child] : 1;
			// If it is not visited yet 
			if (dist[child] == (int)(1e9))
			{
				// Increase distance by 1（权重） 
				dist[child] = weight + dist[x];

				// Change parent 
				par[child] = x;

				// Push into the queue 
				q.push(child);
			}
			// If it is already visited 
			else if (par[x] != child && par[child] != x)
			{
				//ans = min(ans, dist[x] + dist[child] + 1);
				if (ans >= dist[x] + dist[child] + weight)
				{
					if (INT_MAX != ans)
					{
						if (ans > dist[x] + dist[child] + weight)
						{
							//需要清空之前存储的loop
							allMinLoops.clear();
						}
						//存储新的loop
						int tempIndex = x;
						HFHZGraghLoop tempLoop;
						while (par[tempIndex] != iVertex)
						{
							tempLoop.push_back(tempIndex);
							tempIndex = par[tempIndex];
						}
						if (par[tempIndex] == iVertex)
						{
							tempLoop.push_back(tempIndex);
						}
						tempIndex = child;
						while (par[tempIndex] != iVertex)
						{
							//tempLoop.push_back(tempIndex);
							tempLoop.insert(tempLoop.begin(), tempIndex);
							tempIndex = par[tempIndex];
						}
						if (par[tempIndex] == iVertex)
						{
							//tempLoop.push_back(tempIndex);
							tempLoop.insert(tempLoop.begin(), tempIndex);
						}
						tempLoop.push_back(iVertex);
						allMinLoops.push_back(tempLoop);
					}
					ans = dist[x] + dist[child] + weight;
				}
			}
		}
	}
	//去重
	UniqueLoops(allMinLoops);
	ovCircles = allMinLoops;
}

void MinCircles(std::vector<HFHZGraghLoop>& ovCircles)
{
	std::vector<HFHZGraghLoop> allMinLoops;
	// For all vertices 
	for (int i = 0; i < V; i++)
	{
		std::vector<HFHZGraghLoop> tempMinLoops;
		ShortestCirclesFromVertex(i, tempMinLoops, false);
		for (size_t j = 0; j < tempMinLoops.size(); ++j)
		{
			allMinLoops.push_back(tempMinLoops[j]);
		}
	}
	std::vector<HFHZGraghLoop> tempMinLoops;
	ShortestCirclesFromVertex(0, tempMinLoops, false);
	for (size_t j = 0; j < tempMinLoops.size(); ++j)
	{
		allMinLoops.push_back(tempMinLoops[j]);
	}
	//去重
	UniqueLoops(allMinLoops);
	ovCircles = allMinLoops;
}

//两个顶点之间的最短路径
/*
佛洛伊德算法思想：
1）设置两个矩阵A和Path，初始时将图的邻接矩阵赋值给A,将矩阵Path中元素全部设置为-1
2）以顶点k为中间顶点，k取0——n-1(n为图中顶点个位)，为图中所有顶点对{i,j}进行如下检测与修改：
如果A[i][j]>A[i][k]+A[k][j],则将A[i][j]改为A[i][k]+A[k][j]的值，将path[i][j]改为k,否则
什么都不做。
佛洛伊德算法的时间复杂度分析：
由算法代码可知，本算法的主要部分是一个三层顺序，取内层循环的操作作为基本操作，则
基本操作的执行次数为n^3,因此时间复杂度为O(n^3)。
*/
#define INF 100//INF为比图中任何权值都大的数
#define maxSize 4   //图的顶点数
#define number 8   //图的边数
typedef struct {//图的定义
	int edges[maxSize][maxSize];//邻接矩阵的定义
	int n, e;   //分别为顶点数和边数
}MGraph;
void Floyd(MGraph g, int A[][maxSize], int path[][maxSize]);//佛洛伊德算法
void printPath(int u, int v, int path[][maxSize]);/*递归输出从u到v的最短路径顶点序列*/

void Floyd(MGraph g, int A[][maxSize], int path[][maxSize])
{
	int i, j, k;
	//对数组A[][]和path[][]进行初始化
	for (i = 0; i < g.n; ++i)
	{
		for (j = 0; j < g.n; ++j)
		{
			A[i][j] = g.edges[i][j];
			path[i][j] = -1;
		}
	}

	/*下面这个三层循环是本算法的主要操作，完成了以k为中间点对所有顶点对{i,j}
	进行检测和修改*/
	for (k = 0; k < g.n; ++k)
	{
		for (i = 0; i < g.n; ++i)
		{
			for (j = 0; j < g.n; ++j)
			{
				if (A[i][j] >(A[i][k] + A[k][j]))
				{
					A[i][j] = A[i][k] + A[k][j];
					path[i][j] = k;
				}
			}
		}
	}
}

void printPath(int u, int v, int path[][maxSize])	/*递归输出从u到v的最短路径顶点序列*/
{
	if (path[u][v] == -1)
		return;
	else
	{
		cout << path[u][v] << " ";
		int mid = path[u][v];
		printPath(u, mid, path);//处理mid前半段路径
		printPath(mid, v, path);//处理mid后半段路径
	}
}

//源节点到目标节点的最短路径 迪杰斯特拉算法

struct GraphDijkNode
{
	int node;
	int weight;
	bool operator<(const GraphDijkNode& right)
	{
		return weight > right.weight;
	}
};

void Dijkstra(vector<vector<int>>& graph, const int &orignVertex,vector<int>& dis,vector<int>&fatherNode)
{
	if (graph.empty() || orignVertex<0 || orignVertex>=graph.size())
	{
		return;
	}
	const int V = graph.size();
	vector<bool> visit(V, true);
	for (int i = 0; i < V; ++i)
	{
		dis[i] = graph[orignVertex][i];
		fatherNode[i] = 0;
	}
	visit[orignVertex] = false;
	dis[orignVertex] = 0;
	//使用默认版本
	for (int i = 0; i < V;++i)
	{
		int ans = INT_MAX,newNode;
		for (int j = 0; j < V;++j)
		{
			if (visit[j]&&dis[j]<ans)
			{
				ans = dis[j];
				newNode = j;
			}
		}
		visit[newNode] = false;
		if (ans==INT_MAX)
		{
			break;
		}
		for (int j = 0; j < V;++j)
		{
			if (visit[j]&&(graph[newNode][j]+dis[newNode])<dis[j])
			{
				dis[j] = graph[newNode][j] + dis[newNode];
				fatherNode[j] = newNode;
			}
		}
	}
	//使用堆进行优化
	priority_queue<GraphDijkNode> prioH;
	GraphDijkNode orT;
	orT.node = orignVertex;
	orT.weight = 0;
	prioH.push(orT);
	while (!prioH.empty())
	{
		GraphDijkNode nodeG = prioH.top();
		prioH.pop();
		if (!visit[nodeG.node])
		{
			continue;
		}
		visit[nodeG.node] = false;
		for (int j = 0; j < V; ++j)
		{
			if (visit[j] && (graph[nodeG.node][j] + dis[nodeG.node]) < dis[j])
			{
				dis[j] = graph[nodeG.node][j] + dis[nodeG.node];
				fatherNode[j] = nodeG.node;
				GraphDijkNode newNode;
				newNode.node = j;
				newNode.weight = dis[j];
				prioH.push(newNode);
			}
		}
	}
}

int main(int argc,char* argv)
{
	//使用智能指针
	shared_ptr<Solution> s_ptr;
	s_ptr->generateParenthesis(3);
	tuple<int, double, string> tupStruct;
	tupStruct = make_tuple(1, 2.6, "adawd");
	cout << get<0>(tupStruct)<<endl;
	cout << get<1>(tupStruct) << endl;
	cout << get<2>(tupStruct) << endl;
	s_ptr->divide(1,1);
	string pattern("[^c]ei");
	pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
	regex reg(pattern);
	smatch results;
	string test_str = "receipt freind theif receive";
	if (regex_search(test_str, results, reg))
	{
		cout << results.str() << endl;
	}
	//确定随机数的范围
	uniform_int_distribution<unsigned> u(0, 100);
	default_random_engine e;
	vector<int> nums1 = { 1, 1, 2, 1, 2, 2, 1,5,6,7,8,9,-1,-2,-5};
	vector<int> nums2 = { 197, 130, 1 };
	int nums[7] = {0};
	for (int i = 0; i < 7;i++)
	{
		nums[i] = u(e);
	}
// 	unique_ptr<SortSequence::SortClass> uniSolt_ptr;
// 	cout << "\n请输入需要的排序0-冒泡排序1-选择排序2-插入排序3-希尔排序4-归并排序5-快速排序6-堆排序7-基数排序\n";
// 	int temp = 0;
// 	cin >> temp;
// 	cout << "请输入要排序的数\n";
// 	vector<int> pNum;
// 	int pNumber = 0;
// 	while (cin >> pNumber)
// 	{
// 		pNum.push_back(pNumber);
// 	}
// 	switch (temp)
// 	{
// 	case 0:
// 		uniSolt_ptr->BuppleSort(pNum);
// 		break;
// 	case 1:
// 		uniSolt_ptr->SelectSort(pNum);
// 		break;
// 	case 2:
// 		uniSolt_ptr->InsertSort(pNum);
// 		break;
// 	case 3:
// 		uniSolt_ptr->ShellSort(pNum);
// 		break;
// 	case 4:
// 		uniSolt_ptr->MergeSort(pNum);
// 		break;
// 	case 5:
// 		uniSolt_ptr->QuickSort(pNum);
// 		break;
// 	case 6:
// 		uniSolt_ptr->HeapSort(pNum);
// 		break;
// 	case 7:
// 		uniSolt_ptr->RadixSort(pNum);
// 		break;
// 	default:
// 		break;
// 	}
// 	cout << "排序结果\n";
// 	for (auto &it:pNum)
// 	{
// 		cout << it << " ";
// 	} 
	cout << endl;
	vector<string> res = { "a==b", "b!=a" };
	vector<char> a1({ '.', '1', '.', '.', '7', '.', '.', '.', '.' });
	vector<char> a2({ '.', '.', '.', '.', '.', '1', '2', '9', '.' });
	vector<char> a3({ '2', '.', '.', '.', '.', '.', '.', '.', '6' });
	vector<char> a4({ '.', '.', '7', '.', '.', '.', '.', '.', '.' });
	vector<char> a5({ '.', '.', '.', '.', '8', '.', '1', '6', '4' });
	vector<char> a6({ '.', '.', '.', '.', '4', '.', '.', '.', '.' });
	vector<char> a7({ '.', '.', '.', '.', '5', '8', '.', '7', '.' });
	vector<char> a8({ '6', '.', '.', '.', '.', '2', '.', '.', '.' });
	vector<char> a9({ '5', '4', '.', '.', '.', '3', '9', '.', '.' });
	vector<vector<char>> aa({ a1, a2, a3, a4, a5, a6, a7, a8, a9 });
	int number = 0;
    vector<int> vec;
    srand((unsigned int)time(NULL));
    for (int i = 0; i < 10; ++i)
    {
        vec.emplace_back(rand());
    }
	//AutumnMove::Instance().findContinuousSequence(9);
	vector<int> numsNode = { 1, 2, 3, 4, 5 };
	vector<vector<int>> numsT = { { 0, 3}, {1,2}};
	FormalWork::Instance().minCut("aab");
	FormalWork::Instance().calculate2("1*2-3/4+5*6-7*8+9/10");
	system("pause");
	return 0;
}