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
	for (size_t i = 1; i < nums.size();++i)
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
		for (size_t i = ipos + 1; i < nums.size(); ++i)
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

std::vector<int> Solution::searchRange(vector<int>& nums, int target)
{
	if (nums.empty())
	{
		return{ -1, -1 };
	}
	//找到最边的数
	int leftPos = -1, rightPos = -1;
	int lo = 0, hi = nums.size() - 1;
	while (lo < hi)
	{
		int mid = lo + (hi - lo) / 2;
		if (nums[mid] <target)
		{
			lo = mid + 1;
		}
		else 
		{
			hi = mid;
		}
	}
	if (nums[lo] != target)
	{
		return{ -1, -1 };
	}
	leftPos = lo;
	lo = 0, hi = nums.size() - 1;
	while (lo<hi)
	{
		int mid = lo + (hi - lo) / 2;
		if (nums[mid] <= target)
		{
			lo = mid + 1;
		}
		else 
		{
			hi = mid;
		}
	}
	rightPos =(nums[lo]==target)?lo:--lo;
	return {leftPos,rightPos};
}

bool Solution::isValidSudoku(vector<vector<char>>& board)
{
	if (board.empty())
	{
		return true;
	}
	//通过分区域来判断
	vector<unordered_set<int>> row(9), colun(9), block(9);
	for (int i = 0; i < 9;++i)
	{
		for (int j = 0; j < 9;++j)
		{
			int iblo = (i / 3) * 3 + j / 3;
			if (board[i][j]>='1'&&board[i][j]<='9')
			{
				if (row[i].count(board[i][j]) || colun[j].count(board[i][j]) || block[iblo].count(board[i][j]))
				{
					return false;
				}
				row[i].insert(board[i][j]);
				colun[j].insert(board[i][j]);
				block[iblo].insert(board[i][j]);
			}
		}
	}
	return true;
}
std::vector<std::vector<int>> res;
vector<int> path;
vector<int> candiatesTemp;
void Solution::DFSSum(int start, int target)
{
	if (target==0)
	{
		res.push_back(path);
		return;
	}
	for (size_t i = start; i < candiatesTemp.size() && target - candiatesTemp[i] >= 0;++i)
	{
		path.push_back(candiatesTemp[i]);
		DFSSum(i, target - candiatesTemp[i]);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::combinationSum(vector<int>& candidates, int target)
{
	if (candidates.empty())
	{
		return{};
	}
	sort(candidates.begin(), candidates.end());
	candiatesTemp = candidates;
	DFSSum(0, target);
	return res;
}

void Solution::DFSSum2(int start, int target)
{
	if (target == 0)
	{
		res.push_back(path);
		return;
	}
	for (size_t i = start; i < candiatesTemp.size()&&target - candiatesTemp[i] >= 0;++i)
	{
		//去重
		if (i>start&&candiatesTemp[i]==candiatesTemp[i-1])
		{
			continue;
		}
		path.push_back(candiatesTemp[i]);
		DFSSum2(i + 1, target - candiatesTemp[i]);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::combinationSum2(vector<int>& candidates, int target)
{
	if (candidates.empty())
	{
		return{};
	}
	sort(candidates.begin(), candidates.end());
	candiatesTemp = candidates;
	DFSSum2(0, target);
	return res;
}

std::string Solution::multiply(string num1, string num2)
{
	if (num1.empty()||num2.empty())
	{
		return "";
	}
	int iKey = 0;
	int icout = 0;
	string res;
	for (int i = num2.size() - 1; i >= 0; --i)
	{
		string strTemp;
		for (int j = num1.size() - 1; j >= 0; --j)
		{
			int  iTemp = (num1[j] - '0')*(num2[i] - '0');
			iTemp += iKey;
			iKey = iTemp / 10;
			iTemp %= 10;
			strTemp.push_back(iTemp+'0');
		}
		if (iKey!=0)
		{
			strTemp.push_back(iKey + '0');
		}
		iKey = 0;
		reverse(strTemp.begin(), strTemp.end());
		if (!res.empty())
		{
			for (int l = 1; l < icout+1;l++)
			{
				strTemp.push_back(res[res.size() - l-1]);
			}
			strTemp.push_back(res[res.size() - 1]);
			int k = strTemp.size() - icout;
			int m = res.size() - icout;
			while (k>=0||m>=0)
			{
				int a = (m >= 0) ? (res[m] - '0') : 0;
				int b = (k >= 0) ? (strTemp[k] - '0') : 0;
				int  iTemp = a+b+iKey;
				iKey = iTemp / 10;
				iTemp %= 10;
				strTemp[k] = char(iTemp + '0');
				--k;
				--m;
			}
			if (iKey != 0)
			{
				strTemp = (char)(iKey + '0') + strTemp;
			}
		}
		icout++;
		res = strTemp;
	}
	return res;
}

int Solution::expandAroundCenter(const string &s, int left, int right)
{
	size_t L = left, R = right;
	while (L>=0&&R<s.size()&&s[L]==s[R])
	{
		--L;
		++R;
	}
	return R-L-1;
}
