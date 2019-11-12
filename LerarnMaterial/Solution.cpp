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

std::string Solution::getPermutation(int n, int k)
{
	string result;
	string num = "123456789";
	vector<int> f(n, 1);
	//保存i的阶乘
	for (auto i = 1; i < n;++i)
	{
		f[i] = f[i - 1] * i;
	}
	--k; //为了使[0,i!-1]个排列对应到num[i-1]
	for (int i = n; i >= 1;--i)
	{
		int j = k / f[i - 1];
		k %= f[i - 1];
		result.push_back(num[j]);
		num.erase(j, 1);
	}
	return result;
}

ListNode* Solution::rotateRight(ListNode* head, int k)
{
	if (head == nullptr)
	{
		return head;
	}
	int len = 1;
	ListNode * back = head;
	while (back->next != nullptr)
	{
		back = back->next;
		len++;
	}
	back->next = head;
	k = k%len;
	for (int i = 1; i < len - k + 1; ++i)
	{
		back = back->next;
		head = head->next;
	}
	back->next = NULL;
	return head;
}

int Solution::uniquePaths(int m, int n)
{
	vector<vector<int>> dp(m+1, vector<int>(n+1));
	for (int i = 1; i <= m;i++)
	{
		for (int j = 1; j <= n;j++)
		{
			if (i == 1 || j == 1)
			{
				dp[i][j] = 1;
			}
			else
			{
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
	}
	return dp[m][n];
}

int Solution::minPathSum(vector<vector<int>>& grid)
{
	if (grid.empty())
	{
		return 0;
	}
	vector<vector<int>> dp(grid.size() + 1, vector<int>(grid[0].size() + 1));
	for (auto i = 1; i <= grid.size(); i++)
	{
		for (auto j = 1; j <= grid[0].size(); j++)
		{
			if (i == 1)
			{
				dp[i][j] = dp[i][j - 1] + grid[i - 1][j - 1];
			}
			else if (j==1)
			{
				dp[i][j] = dp[i-1][j] + grid[i - 1][j - 1];
			}
			else
			{
				dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
			}
		}
	}
	return dp[grid.size()][grid[0].size()];
}

std::string Solution::simplifyPath(string path)
{
	if (path.empty())
	{
		return path;
	}
	path.append("/");
	stack<string> stac;
	int len = 0;
	for (int i = 0; i < path.size();++i)
	{
		if (path[i] == '/')
		{
			if (len!=0)
			{
				string tmp = path.substr(i - len, len);
				if (tmp == ".")
				{

				}
				else if (tmp == "..")
				{
					if (!stac.empty())
					{
						stac.pop();
					}
				}
				else
				{
					stac.push(tmp);
				}
			}
			len = 0;
		}
		else
			len++;
	}
	if (stac.empty())
	{
		return "/";
	}
	string res;
	while (!stac.empty())
	{
		res = '/' + stac.top() + res;
		stac.pop();
	}
	return res;
}

void Solution::setZeroes(vector<vector<int>>& matrix)
{
	if (matrix.empty())
	{
		return;
	}
	set<int> row;
	set<int> col;
	for (int i = 0; i < matrix.size();++i)
	{
		for (int j = 0; j < matrix[i].size();++j)
		{
			if (matrix[i][j]==0)
			{
				row.insert(i);
				col.insert(j);
			}
		}
	}
	for (auto it:row)
	{
		for (auto &iter:matrix[it])
		{
			iter = 0;
		}
	}
	for (int i = 0; i < matrix.size();++i)
	{
		for (auto it:col)
		{
			matrix[i][it] = 0;
		}
	}
}

bool Solution::searchMatrix(vector<vector<int>>& matrix, int target)
{
	//2次二分查找
	if (matrix.empty()||matrix[0].empty())
	{
		return false;
	}
	int rowl = 0,rowr=matrix.size();
	int coll = 0, colr = matrix[0].size();
	while (rowl<rowr)
	{
		int midrow = rowl + (rowr - rowl) / 2;
		if (matrix[midrow][matrix[midrow].size()-1]<target)
		{
			rowl = midrow + 1;
		}
		else if (matrix[midrow][matrix[midrow].size() - 1]>target)
		{
			rowr = midrow;
		}
		else
		{
			return true;
		}
	}
	if (rowl==matrix.size())
	{
		return false;
	}
	while (coll<colr)
	{
		int midclo = coll + (colr - coll) / 2;
		if (matrix[rowl][midclo]<target)
		{
			coll = midclo + 1;
		}
		else if (matrix[rowl][midclo]>target)
		{
			colr = midclo;
		}
		else
		{
			return true;
		}
	}
	if (coll==matrix[0].size())
	{
		return false;
	}
	return matrix[rowl][coll] == target;
}

void Solution::sortColors(vector<int>& nums)
{
	//3路快排
	if (nums.empty())
	{
		return;
	}
	int ipos = -1;
	int left = 0, right = nums.size();
	while (left<right)
	{
		if (nums[left] == 0)
		{
			swap(nums[++ipos], nums[left++]);
		}
		else if (nums[left] == 2)
		{
			swap(nums[left], nums[--right]);
		}
		else
		{
			left++;
		}
	}
}

void Solution::DFSCombine(std::vector<vector<int>> & res, std::vector<int> &number, int start, int k, vector<int> &path)
{
	if (path.size() == k)
	{
		res.push_back(path);
		return;
	}
	if (start==number.size())
	{
		return;
	}
	//剪枝做优化
	for (int i = start; i <=number.size()-(k-path.size());i++)
	{
		path.push_back(number[i]);
		DFSCombine(res, number, i + 1, k, path);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::combine(int n, int k)
{
	if (n<k||k==0)
	{
		return{};
	}
	vector<int> numNumber;
	for (int i = 1; i <= n;++i)
	{
		numNumber.push_back(i);
	}
	vector<vector<int>> res;
	vector<int> path;
	DFSCombine(res, numNumber, 0, k, path);
	return res;
}
vector<int> numNumber;
void Solution::DFSSubsets(std::vector<vector<int>> & res, int start, vector<int> &path)
{
	res.push_back(path);
	//剪枝做优化
	for (int i = start; i < numNumber.size(); i++)
	{
		path.push_back(numNumber[i]);
		DFSSubsets(res, i + 1, path);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::subsets(vector<int>& nums)
{
	//按位进行运算
	int total = 1 << nums.size();
	std::vector<vector<int>>  res(total);
	for (int i = 0; i < total;++i)
	{
		for (int j = i, k = 0; j; j >>= 1,k++)
		{
			if (j&1==1)
			{
				res[i].push_back(nums[k]);
			}
		}
	}
	return res;
}
int dir[4][4] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
bool Solution::DFSexist(int x, int y, int index, vector<vector<char>>&board, string &word, vector<vector<bool>>&visit)
{
	if (index==word.size()-1)
	{
		return word[index] == board[x][y];
	}
	if (word[index]==board[x][y])
	{
		visit[x][y] = false;
		for (int i = 0; i < 4;++i)
		{
			int newx = x + dir[i][0];
			int newy = y + dir[i][1];
			if (newx>=0&&newx<board.size()&&newy>=0&&newy<board[0].size()&&visit[newx][newy])
			{
				if (DFSexist(newx, newy, index + 1, board, word, visit))
				{
					return true;
				}
			}
		}
		visit[x][y] = true;
	}
	return false;
}

bool Solution::exist(vector<vector<char>>& board, string word)
{
	if (word.empty())
	{
		return true;
	}
	if (board.empty())
	{
		return false;
	}
	std::vector<std::vector<bool>> vecPos(board.size(),std::vector<bool>(board[0].size(),true));
	for (int i = 0; i < board.size();++i)
	{
		for (int j = 0; j < board[i].size();++j)
		{
			if (DFSexist(i, j, 0, board, word, vecPos))
			{
				return true;
			}
		}
	}
	return false;
}

int Solution::removeDuplicates(vector<int>& nums)
{
	if (nums.empty())
	{
		return 0;
	}
	int ipos = 1;
	if (nums.size()<=2)
	{
		return nums.size();
	}
	for (int i = 2; i < nums.size();++i)
	{
		if (nums[i]!=nums[ipos-1])
		{
			nums[++ipos] = nums[i];
		}
	}
	return ipos+1;
}


ListNode* Solution::deleteDuplicates(ListNode* head)
{
	if (nullptr==head)
	{
		return head;
	}
	ListNode *newNode = new ListNode(0);
	ListNode *l = head;
	ListNode *r = head;
	ListNode *p = newNode;
	int len = 0;
	while (r)
	{
		//判断是否相等，相等时就计数+1
		if (r->val==l->val)
		{
			len++;
			r = r->next;
		}
		else
		{
			if (len<=1)
			{
				p->next = l;
				p = p->next;
			}
			l = r;
			len = 0;
		}
	}
	if (len>1)
	{
		p->next = NULL;
	}
	else
	{
		p->next = l;
	}
	return newNode->next;
}

ListNode* Solution::partition(ListNode* head, int x)
{
	if (head==nullptr)
	{
		return head;
	}
	ListNode *left = new ListNode(0);
	ListNode * right = new ListNode(0);
	ListNode * pl = left;
	ListNode * pr = right;
	while (head)
	{
		if (head->val<x)
		{
			pl->next = head;
			pl = pl->next;
		}
		else
		{
			pr->next = head;
			pr = pr->next;
		}
		head = head->next;
	}
	pr->next = nullptr;
	pl->next = right->next;
	return left->next;
}

std::vector<int> Solution::grayCode(int n)
{
	if (n==0)
	{
		return {0};
	}
	vector<int> res(pow(2, n), 0);
	for (int i = 1; i <= n;i++)
	{
		int len = pow(2, i);
		for (int j = len; j >= len / 2; --j)
		{
			//逆序的格雷码
			res[j] = res[len - 1 - j] + len/2;
		}
	}
	return res;
}

ListNode * Solution::mergeTwoLsits(ListNode* p1, ListNode * p2)
{
	if (nullptr==p1)
	{
		return p2;
	}
	if (nullptr==p2)
	{
		return p1;
	}
	ListNode * newHead = new ListNode(0);
	ListNode *Temp = newHead;
	while (p1!=nullptr&&p2!=nullptr)
	{
		if (p1->val<p2->val)
		{
			Temp->next = p1;
			p1 = p1->next;
		}
		else
		{
			Temp->next = p2;
			p2 = p2->next;
		}
		Temp = Temp->next;
	}
	Temp->next = p1 == nullptr ? p2 : p1;
	return newHead->next;
}

ListNode* Solution::mergeKLists(vector<ListNode*>& lists)
{
	if (lists.empty())
	{
		return nullptr;
	}
	ListNode * newNode = new ListNode(0);
	newNode->next = lists[0];
	for (int i = 1; i < lists.size();++i)
	{
		newNode->next = mergeTwoLsits(newNode->next, lists[i]);
	}
	return newNode->next;
}
std::vector<std::vector<int>> resultDup;
vector<int> numsBer;
void Solution::DFSWithDup(vector<int>&path, int start, int idepth)
{
	if (idepth == path.size())
	{
		resultDup.push_back(path);
		return;
	}
	for (int i = start; i <= numsBer.size() - (idepth - path.size()); ++i)
	{
		if (i != start&&numsBer[i] == numsBer[i - 1])
		{
			continue;
		}
		path.push_back(numsBer[i]);
		DFSWithDup(path, i + 1,idepth);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::subsetsWithDup(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	sort(nums.begin(), nums.end());
	numsBer = nums;
	for (int i = 0; i <=nums.size(); ++i)
	{
		vector<int> path;
		DFSWithDup(path, 0,i);
	}
	return resultDup;
}

ListNode* Solution::reverseBetween(ListNode* head, int m, int n)
{
	if (head==nullptr)
	{
		return head;
	}
	ListNode * newHead = new ListNode(0);
	newHead->next = head;
	int icount = 1;
	ListNode * reverseStartBefore = nullptr;
	while (head)
	{
		if (icount==m)
		{
			break;
		}
		reverseStartBefore = head;
		head = head->next;
		icount++;
	}
	//记录链表的4个节点
	ListNode * reverseStart = head;
	ListNode * pre = nullptr;
	ListNode * cur = nullptr;
	//反转链表
	while (icount<=n&&head)
	{
		cur = head->next;
		head->next = pre;
		pre = head;
		head = cur;
		icount++;
	}
	reverseStart->next = head;
	if (reverseStartBefore==nullptr)
	{
		return pre;
	}
	reverseStartBefore->next = pre;
	return newHead->next;
}

void Solution::DFSToAddAdress(std::vector<std::string> &res, std::vector<string> &path, string &s, int istart, int iend)
{
	//iend表名还需要几个数
	if (iend == 0)
	{
		if (istart!=s.size())
		{
			return;
		}
		string result;
		for (int i = 0; i < path.size()-1; ++i)
		{
			result += path[i] + '.';
		}
		result += path[path.size() - 1];
		res.push_back(result);
		return;
	}
	iend--;
	int num = 0;
	for (int i = istart; i < s.size() - iend;++i)
	{
		num = num * 10 + s[i] - '0';
		if (num>255)
		{
			break;
		}
		path.push_back(to_string(num));
		DFSToAddAdress(res, path, s, i + 1, iend);
		path.pop_back();
		//IP地址不能以0开头
		if (num==0)
		{
			break;
		}
	}
}

std::vector<std::string> Solution::restoreIpAddresses(string s)
{
	if (s.empty())
	{
		return {};
	}
	std::vector<std::string> res;
	std::vector<string> path;
	if (s.size()>=4)
	{
		DFSToAddAdress(res, path, s, 0, 4);
	}
	return res;
}

void Solution::DFSinorderTraversal(TreeNode * root, std::vector<int> &res)
{
	if (root==nullptr)
	{
		return;
	}
	if (root->left!=nullptr)
	{
		DFSinorderTraversal(root->left, res);
	}
	res.push_back(root->val);
	if (root->right!=nullptr)
	{
		DFSinorderTraversal(root->right, res);
	}
}

std::vector<int> Solution::inorderTraversal(TreeNode* root)
{
	//迭代版本的中序遍历
	if (root==nullptr)
	{
		return{};
	}
	std::vector<int> res;
	stack<TreeNode *> stacNode;
	//先让左子树进栈，访问当前节点，在让右子树进栈
	while (root)
	{
		stacNode.push(root);
		root = root->left;
	}
	while (!stacNode.empty())
	{
		auto nodeTemp = stacNode.top();
		res.push_back(nodeTemp->val);
		stacNode.pop();
		nodeTemp = nodeTemp->right;
		while (nodeTemp)
		{
			stacNode.push(nodeTemp);
			nodeTemp = nodeTemp->left;
		}
	}
	return res;
}

std::vector<int> Solution::preorderTraversal(TreeNode* root)
{
	if (root==nullptr)
	{
		return{};
	}
	std::stack<TreeNode *> stac;
	std::vector<int> res;
	stac.push(root);
	while (!stac.empty())
	{
		TreeNode * tempNode = stac.top();
		stac.pop();
		res.push_back(tempNode->val);
		if (tempNode->right)
		{
			stac.push(tempNode->right);
		}
		if (tempNode->left)
		{
			stac.push(tempNode->left);
		}
	}
	return res;
}

std::vector<TreeNode*> Solution::DFSGenerateTree(int left, int right)
{
	vector<TreeNode *> ans;
	if (left>right)
	{
		ans.push_back(NULL);
		return ans;
	}
	for (int i = left; i <= right;++i)
	{
		//找每个节点所对应的的左子树和右子树的几何
		vector<TreeNode*> leftNodeVec = DFSGenerateTree(left, i - 1);
		vector<TreeNode*> rightNodeVec = DFSGenerateTree(i+1, right);
		for (auto itLeft:leftNodeVec)
		{
			for (auto itRight:rightNodeVec)
			{
				TreeNode * newNode = new TreeNode(i);
				newNode->left = itLeft;
				newNode->right = itRight;
				ans.push_back(newNode);
			}
		}
	}
	return ans;
}

std::vector<TreeNode*> Solution::generateTrees(int n)
{
	if (n<=0)
	{
		return{};
	}
	return DFSGenerateTree(1, n);
}

int Solution::numTrees(int n)
{
	//卡特兰数
	if (n <= 0)
	{
		return 0;
	}
	if (n==1)
	{
		return 1;
	}
	long C = 1;
	for (int i = 0; i < n;++i)
	{
		C = C * 2 * (2 * i + 1) / (i + 2);
	}
	return C;
}

bool Solution::isValidBST(TreeNode* root)
{
	if (NULL==root)
	{
		return true;
	}
	if (root->left==NULL&&root->right==NULL)
	{
		return true;
	}
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
