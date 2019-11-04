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
		if (i>(size_t)start&&candiatesTemp[i] == candiatesTemp[i - 1])
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
	if (num1=="0"||num2=="0")
	{
		return "0";
	}
	vector<int> vecRes;
	vecRes.resize(num1.length() + num2.length());
	for (int i = num1.size() - 1; i >= 0;--i)
	{
		for (int j = num2.size() - 1; j >= 0; --j)
		{
			vecRes[i + j + 1] += (num1[i] - '0')*(num2[j] - '0');
		}
	}
	int up = 0;
	for (int i = vecRes.size() - 1; i >= 0;--i)
	{
		vecRes[i] += up;
		up = vecRes[i] / 10;
		vecRes[i] %= 10;
	}
	string res;
	//将首位的0去除
	for (size_t i = 0; i < vecRes.size();++i)
	{
		if (vecRes[i]!=0)
		{
			for (size_t j = i; j < vecRes.size();++j)
			{
				res.push_back((char)(vecRes[j] + '0'));
			}
			break;
		}
		
	}
	return res;
}
std::vector<std::vector<int>> resPer;
int iSize = 0;
void Solution::DFSpermute(vector<int> path, int i, unordered_map<int, bool> &hashM)
{
	if (i == iSize)
	{
		resPer.push_back(path);
		return;
	}
	for (auto iter = hashM.begin(); iter != hashM.end();++iter)
	{
		if (iter->second==true)
		{
			//状态转换,将其在待选序列中去除
			iter->second = false;
			path.push_back(iter->first);
			DFSpermute(path, i + 1, hashM);
			path.pop_back();
			iter->second = true;
		}
	}
}

std::vector<std::vector<int>> Solution::permute(vector<int>& nums)
{
	if (nums.empty())
	{
		return {};
	}
	iSize = nums.size();
	unordered_map<int, bool> hashM;
	for (auto it:nums)
	{
		hashM[it] = true;
	}
	vector<int> path;
	DFSpermute(path, 0, hashM);
	return resPer;
}

void Solution::DFSpermuteUnique(vector<int> path, int i, vector<pair<int, int>> &iInt)
{
	if (i == iSize)
	{
		resPer.push_back(path);
		return;
	}
	//去重
	int ipos = -1;
	for (size_t j = 0; j < iInt.size(); ++j)
	{
		if (ipos>=0&&iInt[j].first == iInt[ipos].first&&i == path.size())
		{
			continue;
		}
		if (iInt[j].second==0)
		{
			iInt[j].second = 1;
			path.push_back(iInt[j].first);
			DFSpermuteUnique(path, i + 1, iInt);
			ipos = j;
			path.pop_back();
			iInt[j].second = 0;
		}
	}
}

std::vector<std::vector<int>> Solution::permuteUnique(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	iSize = nums.size();
	sort(nums.begin(), nums.end());
	vector<pair<int, int>> VecInt;
	for (auto it : nums)
	{
		VecInt.push_back(make_pair(it, 0));
	}
	vector<int> path;
	DFSpermuteUnique(path, 0,VecInt);
	return resPer;
}

void Solution::rotate(vector<vector<int>>& matrix)
{
	if (matrix.empty()||matrix.size()==1)
	{
		return;
	}
	int length = matrix.size();
	//转置矩阵后，在反转
	/*for (int i = 0; i <length;++i)
	{
	for (int j = i; j < length;++j)
	{
	swap(matrix[i][j], matrix[j][i]);
	}
	}
	for (int i = 0; i < length;++i)
	{
	reverse(matrix[i].begin(), matrix[i].end());
	}*/
	//一层一层旋转
	for (int loop = 0; loop < length / 2; ++loop)
	{
		for (int i = loop, j = loop; i < length - 1 - loop;++i)
		{
			int pre = matrix[i][j];
			for (int time = 1; time <= 4; ++time)
			{
				int temp = i;
				i = j;
				j = length - temp - 1;
				swap(pre, matrix[i][j]);
			}
		}
	}
}

std::vector<std::vector<std::string>> Solution::groupAnagrams(vector<string>& strs)
{
	if (strs.empty())
	{
		return{};
	}
	unordered_map<string, vector<string>> hashM;
	for (auto it : strs)
	{
		string strTemp = it;
		sort(strTemp.begin(), strTemp.end());
		hashM[strTemp].push_back(it);
	}
	vector<vector<string>> res;
	for (auto it:hashM)
	{
		res.push_back(it.second);
	}
	return res;
}

double Solution::myPow(double x, int n)
{
	if (n==0)
	{
		return 1;
	}
	if (n==1)
	{
		return x;
	}
	if (n==-1)
	{
		return 1 / x;
	}
	double dhalf = myPow(x, n / 2);
	double rest = myPow(x, n % 2);
	return rest*dhalf*dhalf;
}

std::vector<int> Solution::spiralOrder(vector<vector<int>>& matrix)
{
	if (matrix.empty())
	{
		return {};
	}
	//一下4个数分别为4个角的边界
	int u = 0;                      //上边界
	int d = matrix.size() - 1;      //下边界
	int l = 0;                      //左边界
	int r = matrix[0].size() - 1;   //右边界
	vector<int> res;
	while (true)
	{
		for (int i = l; i <= r;++i)
		{
			res.push_back(matrix[u][i]);
		}
		if (++u>d)
		{
			break;
		}
		for (int i = u; i <= d;++i)
		{
			res.push_back(matrix[i][r]);
		}
		if (--r<l)
		{
			break;
		}
		for (int i = r; i >= l;--i)
		{
			res.push_back(matrix[d][i]);
		}
		if (--d<u)
		{
			break;
		}
		for (int i = d; i >= u;--i)
		{
			res.push_back(matrix[i][l]);
		}
		if (++l>r)
		{
			break;
		}
	}
	return res;
}

bool Solution::canJump(vector<int>& nums)
{
	if (nums.empty())
	{
		return true;
	}
	if (nums[0]==0)
	{
		if (nums.size() == 1)
		{
			return true;
		}
		else
			return false;
	}
	for (size_t i = 0; i < nums.size();++i)
	{
		if (nums[i]==0)
		{
			bool bkey = false;
			for (int j = i - 1; j >= 0;--j)
			{
				if (j+nums[j]>(int)i||(j+nums[j])==nums.size()-1)
				{
					bkey = true;
					break;
				}
			}
			if (!bkey)
			{
				return false;
			}
		}
	}
	return true;
}

std::vector<std::vector<int>> Solution::merge(vector<vector<int>>& intervals)
{
	if (intervals.empty())
	{
		return{};
	}
	map<int, int> hashM;
	//将最大的区间找到
	for (auto it:intervals)
	{
		if (hashM.count(it[0]))
		{
			hashM[it[0]] = max(it[1], hashM[it[0]]);
		}
		else
		{
			hashM[it[0]] = it[1];
		}
	}
	vector<vector<int>> res;
	for (auto iter = hashM.begin(); iter != hashM.end();)
	{
		auto it = iter;
		it++;
		int maxDistance = iter->second;
		//将能覆盖的最远距离存放
		while (it!=hashM.end()&&it->first<=maxDistance)
		{
			maxDistance = max(it->second,maxDistance);
			++it;
		}
		vector<int> vecTemp;
		vecTemp.push_back(iter->first);
		vecTemp.push_back(maxDistance);
		iter = it;
		res.push_back(vecTemp);
	}
	return res;
}

std::vector<std::vector<int>> Solution::generateMatrix(int n)
{
	if (n==0)
	{
		return {};
	}
	int topLine = 0;  //上边线
	int rightLine = n - 1;//右边线
	int lowerLine = n - 1;//下边线
	int leftLine = 0;//左边线
	int icount = 1;
	vector<vector<int>> res(n, vector<int>(n));
	while (true)
	{
		//给上边线赋值
		for (int i = leftLine; i <=rightLine;++i)
		{
			res[topLine][i] = icount++;
		}
		if (++topLine>lowerLine)
		{
			break;
		}
		//给右边线赋值
		for (int i = topLine; i <= lowerLine; ++i)
		{
			res[i][rightLine] = icount++;
		}
		if (--rightLine < leftLine)
		{
			break;
		}
		//给下边线赋值
		for (int i = rightLine; i >= leftLine;--i)
		{
			res[lowerLine][i] = icount++;
		}
		if (--lowerLine<topLine)
		{
			break;
		}
		//给左边线赋值
		for (int i = lowerLine; i >= topLine; --i)
		{
			res[i][leftLine] = icount++;
		}
		if (++leftLine > rightLine)
		{
			break;
		}
	}
	return res;
}

void Solution::DfsGetPermutation(set<int> &iSet, int iNumber, string &res)
{
	if (iSet.size()==1)
	{
		res += to_string(*iSet.begin());
		return;
	}
	int length = iSet.size()-1;
	int m = 1;
	for (int i = 1; i <= length;++i)
	{
		m *= i;
	}
	if (m*(m + 1)<iNumber)
	{
		return;
	}
	int quo = iNumber / m;
	int remain = iNumber % m;
	if (remain == 0)
	{
		auto it = iSet.begin();
		while (--quo>0)
		{
			++it;
		}
		res += to_string(*it);
		iSet.erase(it);
	}
	else
	{
		auto it = iSet.begin();
		while (quo>0)
		{
			++it;
			--quo;
		}
		res += to_string(*it);
		iSet.erase(it);
	}
	DfsGetPermutation(iSet, remain, res);
}

std::string Solution::getPermutation(int n, int k)
{
	if (n<=0)
	{
		return "";
	}
	set<int> hash;
	for (int i = 1; i <= n;++i)
	{
		hash.insert(i);
	}
	string res;
	DfsGetPermutation(hash, k, res);
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
