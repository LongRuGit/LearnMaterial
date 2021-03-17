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

//ȡ�����ƣ��Ƿ���λ�������λ1
uint64_t hight_bitCur(uint64_t x){//0010 1100 0000 0000 0000 0000 0000 0000 0000 0001
	x = x | (x >> 1);              //0011 1110 0000 0000 0000 0000 0000 0000 0000 0000
	x = x | (x >> 2);              //0011 1111 1000 0000 0000 0000 0000 0000 0000 0000
	x = x | (x >> 4);              //0011 1111 1111 1000 0000 0000 0000 0000 0000 0000
	x = x | (x >> 8);              //0011 1111 1111 1111 1111 1000 0000 0000 0000 0000
	x = x | (x >> 16);             //0011 1111 1111 1111 1111 1111 1111 1111 1111 1111
	x = x | (x >> 32);
	// ������ر�� ����о�������� ��������ֻʹ����С���������ֵ1/2������
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
		//����ȥ�أ�Ĭ�����е�loop�����ظ���
		std::vector<HFHZGraghLoop> allSortedLoops;
		//��¼allSortedLoops��ioLoops�е��ļ���loop
		std::vector<size_t>	allSortedLoopsIndex;
		std::vector<HFHZGraghLoop> tempLoops = ioLoops;
		for (size_t i = 0; i < tempLoops.size(); ++i)
		{
			std::sort(tempLoops[i].begin(), tempLoops[i].end());
			if (std::unique(tempLoops[i].begin(), tempLoops[i].end()) == tempLoops[i].end())
			{
				//˵��û���ظ���
				allSortedLoops.push_back(tempLoops[i]);
				allSortedLoopsIndex.push_back(i);
			}
		}
		//1 -- ��ʾ��Ҫ�Ľ��
		//0 -- ��ʾû�б�����
		//-1-- ��ʾ�ظ��Ľ��
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
				//�޸�ʱ�䣺2020/09/29 ���޸��ˣ�konghaijiao
				//�޸�ԭ�򣺷��ص�loop֮ǰŪ����
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
			//0 -- ˵������û��ֱ������
			if (0 == pMatrix[x][child])
			{
				continue;
			}
			//�޸�ʱ�䣺2020/07/20 ���޸��ˣ�konghaijiao
			//�޸�ԭ������Ȩ��
			int weight = ibWeigth ? pMatrix[x][child] : 1;
			// If it is not visited yet 
			if (dist[child] == (int)(1e9))
			{
				// Increase distance by 1��Ȩ�أ� 
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
							//��Ҫ���֮ǰ�洢��loop
							allMinLoops.clear();
						}
						//�洢�µ�loop
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
	//ȥ��
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
	//ȥ��
	UniqueLoops(allMinLoops);
	ovCircles = allMinLoops;
}

//��������֮������·��
/*
���������㷨˼�룺
1��������������A��Path����ʼʱ��ͼ���ڽӾ���ֵ��A,������Path��Ԫ��ȫ������Ϊ-1
2���Զ���kΪ�м䶥�㣬kȡ0����n-1(nΪͼ�ж����λ)��Ϊͼ�����ж����{i,j}�������¼�����޸ģ�
���A[i][j]>A[i][k]+A[k][j],��A[i][j]��ΪA[i][k]+A[k][j]��ֵ����path[i][j]��Ϊk,����
ʲô��������
���������㷨��ʱ�临�Ӷȷ�����
���㷨�����֪�����㷨����Ҫ������һ������˳��ȡ�ڲ�ѭ���Ĳ�����Ϊ������������
����������ִ�д���Ϊn^3,���ʱ�临�Ӷ�ΪO(n^3)��
*/
#define INF 100//INFΪ��ͼ���κ�Ȩֵ�������
#define maxSize 4   //ͼ�Ķ�����
#define number 8   //ͼ�ı���
typedef struct {//ͼ�Ķ���
	int edges[maxSize][maxSize];//�ڽӾ���Ķ���
	int n, e;   //�ֱ�Ϊ�������ͱ���
}MGraph;
void Floyd(MGraph g, int A[][maxSize], int path[][maxSize]);//���������㷨
void printPath(int u, int v, int path[][maxSize]);/*�ݹ������u��v�����·����������*/

void Floyd(MGraph g, int A[][maxSize], int path[][maxSize])
{
	int i, j, k;
	//������A[][]��path[][]���г�ʼ��
	for (i = 0; i < g.n; ++i)
	{
		for (j = 0; j < g.n; ++j)
		{
			A[i][j] = g.edges[i][j];
			path[i][j] = -1;
		}
	}

	/*�����������ѭ���Ǳ��㷨����Ҫ�������������kΪ�м������ж����{i,j}
	���м����޸�*/
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

void printPath(int u, int v, int path[][maxSize])	/*�ݹ������u��v�����·����������*/
{
	if (path[u][v] == -1)
		return;
	else
	{
		cout << path[u][v] << " ";
		int mid = path[u][v];
		printPath(u, mid, path);//����midǰ���·��
		printPath(mid, v, path);//����mid����·��
	}
}

//Դ�ڵ㵽Ŀ��ڵ�����·�� �Ͻ�˹�����㷨

struct GraphDijkNode
{
	int node;
	int weight;
	bool operator<(const GraphDijkNode& right) const
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
	//ʹ��Ĭ�ϰ汾
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
	//ʹ�öѽ����Ż�
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
	//ʹ������ָ��
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
	//ȷ��������ķ�Χ
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
// 	cout << "\n��������Ҫ������0-ð������1-ѡ������2-��������3-ϣ������4-�鲢����5-��������6-������7-��������\n";
// 	int temp = 0;
// 	cin >> temp;
// 	cout << "������Ҫ�������\n";
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
// 	cout << "������\n";
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
	int numberN = 0;
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