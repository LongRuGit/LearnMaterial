#include "Solution.h"


Solution::Solution()
{
}


Solution::~Solution()
{
}

std::string Solution::longestPalindrome(string s)
{
	if (s.size() <= 1)
	{
		return s;
	}
	int len = s.size();
	string res;
	int iStart = 0, iEnd = 0;
	for (int i = 0; i < len; ++i)
	{
		int len1 = expandAroundCenter(s, i, i);
		int len2 = expandAroundCenter(s, i, i + 1);
		int lenT = max(len1, len2);
		if (lenT > iEnd - iStart)
		{
			iStart = i - (lenT - 1) / 2;
			iEnd = lenT;
		}
	}
	return s.substr(iStart, iEnd);
// 	int imaxlen = 1;
// 	int istart = 0;
// 	//使用动态规划
// 	vector < vector <bool>> dp(len, vector<bool>(len));
// 	//初始化动态规划的数组
// 	for (int i = 0; i < len; ++i)
// 	{
// 		dp[i][i] = true;
// 		for (int j = 0; j < i; ++j)
// 		{
// 			if (s[j] == s[i] && ((i - j) == 1 || dp[j + 1][i - 1]))
// 			{
// 				dp[j][i] = true;
// 				if (i - j + 1 > imaxlen)
// 				{
// 					imaxlen = i - j + 1;
// 					istart = j;
// 				}
// 			}
// 			else
// 				dp[j][i] = false;
// 
// 		}
// 	}
// 	return s.substr(istart, imaxlen);
}
int ilength = 0;
void Solution::DfsParent(vector<string> &istrVec, string istr, int l, int r)
{
	if (l<r)
	{
		return;
	}
	if (l==ilength&&r==ilength)
	{
		istrVec.push_back(istr);
		return;
	}
	if (l == 0)
	{
		DfsParent(istrVec, istr + '(', l+1,r);
	}
	else if (l==ilength&&r<=ilength)
	{
		DfsParent(istrVec, istr + ')', l, r+1);
	}
	else
	{
		DfsParent(istrVec, istr + '(', l + 1, r);
		DfsParent(istrVec, istr + ')', l, r + 1);
	}
}

std::vector<std::string> Solution::generateParenthesis(int n)
{
	if (n<=0)
	{
		return{};
	}
	ilength = n;
	vector<std::string> res;
	string str = "";
	DfsParent(res, str, 0, 0);
	return res;
}

ListNode* Solution::swapPairs(ListNode* head)
{
	if (nullptr==head)
	{
		return nullptr;
	}
	ListNode * Lo = new ListNode(0);
	Lo->next = head;
	ListNode * pre = Lo;
	while (head != nullptr&&head->next != nullptr)
	{
		ListNode * temp1 = head->next->next;
		ListNode * temp2 = head->next;
		ListNode * temp3 = head;
		pre->next = temp2;
		temp2->next = temp3;
		temp3->next = temp1;
		head = temp1;
		pre = temp3;
	}
	return Lo->next;
}

int Solution::divide(int dividend, int divisor)
{
	if (divisor == 0)
	{
		return INT_MAX;
	}
	if (dividend==0)
	{
		return 0;
	}
	if (dividend==INT_MIN&&divisor==-1)
	{
		return INT_MAX;
	}
	unsigned int res = 0;
	bool flag = true;
	flag = (dividend^divisor)>=0?true:false;
	unsigned int mtyDividend = dividend == INT_MIN ? (unsigned int)(INT_MAX)+1 : abs(dividend);
	unsigned int mtyDivisor = divisor == INT_MIN ? (unsigned int)(INT_MAX)+1 : abs(divisor);
	while (mtyDividend >= mtyDivisor)
	{
		unsigned int temp = mtyDivisor, mask = 1;
		while (temp&&mtyDividend>=temp)
		{
			mtyDividend -= temp;
			res += mask;
			mask <<= 1;
			temp <<= 1;
		}
	}
	if (!flag)
	{
		res = ~(int)(res-1);
	}
	return res;
}

void Solution::nextPermutation(vector<int>& nums)
{
	if (nums.empty())
	{
		return;
	}
	int ipos=-1;
	//找到最后一个逆序对，在ipos之后一定为降序,然后在ipos之后找到一个比ipos大的最小数
	for (int i = 1; i < nums.size();++i)
	{
		if (nums[i]>nums[i-1])
		{
			ipos = i;
		}
	}
	if (ipos != -1)
	{
		//按照字典序找到更小的数与之交换
		int iNewPos = ipos;
		int minNumber = INT_MAX;
		for (int i = ipos + 1; i < nums.size(); ++i)
		{
			if (nums[i]>nums[ipos-1] && nums[i] < minNumber)
			{
				minNumber = nums[i];
				iNewPos = i;
			}
		}
		swap(nums[ipos - 1], nums[iNewPos]);
		sort(nums.begin()+ipos,nums.end());
	}
	else
		sort(nums.begin(), nums.end());
}

int Solution::search(vector<int>& nums, int target)
{
	if (nums.empty())
	{
		return -1;
	}
	int left = 0, right = nums.size() - 1;
	while (left<right)
	{
		int mid = left + (right - left) / 2;
		if (target>nums[mid]&&target<nums[0])
		{
			left = mid + 1;
		}
		else if (nums[0] <= nums[mid] && (target>nums[mid] || target < nums[0]))
		{
			left = mid + 1;
		}
		else
			right = mid;
	}
	return left==right&&nums[left]==target?left:-1;
}

int Solution::expandAroundCenter(const string &s, int left, int right)
{
	int L = left, R = right;
	while (L>=0&&R<s.size()&&s[L]==s[R])
	{
		--L;
		++R;
	}
	return R-L-1;
}
