#include "Common.h"
#include "Solution.h"
#include <tuple>
#include <random>
#include <regex>
#include "SortClass.h"

int main()
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
	vector<int> nums1 = { 2,0,0};
	s_ptr->search(nums1,0);
	int nums[7] = {0};
	for (int i = 0; i < 7;i++)
	{
		nums[i] = u(e);
	}
	unique_ptr<SortClass> uniSolt_ptr;
	uniSolt_ptr->MergeSort(nums, 7);
	for (auto iter = begin(nums); iter != end(nums); ++iter)
	{
		cout << *iter << " ";
	}
	cout << endl;
	s_ptr->simplifyPath("/a//b////c/d//././/..");
	system("pause");
	return 0;
}