#include "Solution.h"


Solution::Solution()
{
}


Solution::~Solution()
{
}

Solution& Solution::operator ++()
{
	//*this += 1;
	return *this;
}

const Solution Solution::operator ++(int)
{
	Solution oldValue = *this;
	++(*this);
	return oldValue;
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
int Solution::expandAroundCenter(const string &s, int left, int right)
{
	size_t L = left, R = right;
	while (L >= 0 && R < s.size() && s[L] == s[R])
	{
		--L;
		++R;
	}
	return R - L - 1;
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
	if (matrix.empty())
		return false;
	int row = matrix.size();
	int col = matrix[0].size();
	int left = 0;
	int right = row*col - 1;
	int mid = 0, priot = 0;
	while (left <= right)
	{
		mid = left + (right - left) / 2;
		priot = matrix[mid / col][mid%col];
		if (priot == target)
			return true;
		else if (priot > target)
			right = mid - 1;
		else
			left = mid + 1;
	}
	return false;
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

std::vector<int> Solution::postorderTraversal(TreeNode* root)
{
	if (NULL==root)
	{
		return{};
	}
	std::vector<int> res;
	stack<TreeNode*> stacTree;
	stacTree.push(root);
	while (!stacTree.empty())
	{
		auto node = stacTree.top();
		res.push_back(node->val);
		stacTree.pop();
		if (node->left)
		{
			stacTree.push(node->left);
		}
		if (node->right)
		{
			stacTree.push(node->right);
		}
	}
	reverse(res.begin(),res.end());
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
	std::stack<TreeNode *> stacNode;
	TreeNode *pre = nullptr;
	//利用二叉树搜索树的中序遍历是一个递增序列，记录上一次的节点值
	//先让左子树进栈，访问当前节点，在让右子树进栈
	while (root)
	{
		stacNode.push(root);
		root = root->left;
	}
	while (!stacNode.empty())
	{
		TreeNode * tempNode = stacNode.top();
		stacNode.pop();
		if (pre!=nullptr&&tempNode->val<=pre->val)
		{
			return false;
		}
		pre = tempNode;
		tempNode = tempNode->right;
		while (tempNode)
		{
			stacNode.push(tempNode);
			tempNode = tempNode->left;
		}
	}
	return true;
}

void Solution::DFSGetPath(std::vector<TreeNode *> & vecPath, TreeNode * root, TreeNode * target, bool &ikey)
{
	if (ikey)
	{
		return;
	}
	if (root->val==target->val)
	{
		vecPath.push_back(root);
		ikey = true;
		return;
	}
	if (NULL==root)
	{
		return;
	}
	vecPath.push_back(root);
	if (root->left)
	{
		DFSGetPath(vecPath, root->left, target, ikey);
		if (!ikey)
		{
			vecPath.pop_back();
		}
		else
		{
			return;
		}
	}
	if (root->right)
	{
		DFSGetPath(vecPath, root->right, target, ikey);
		if (!ikey)
		{
			vecPath.pop_back();
		}
		else
		{
			return;
		}
	}
}

TreeNode* Solution::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
// 	if (root==NULL)
// 	{
// 		return root;
// 	}
// 	if (p->val==q->val)
// 	{
// 		return q;
// 	}
// 	std::vector<TreeNode*> pathTar1;
// 	std::vector<TreeNode*> pathTar2;
// 	bool ibkey1 = false;
// 	DFSGetPath(pathTar1, root, p, ibkey1);
// 	bool ibkey2 = false;
// 	DFSGetPath(pathTar2, root, q, ibkey2);
// 	if (pathTar1.size()==1)
// 	{
// 		return pathTar1[0];
// 	}
// 	if (pathTar2.size() == 1)
// 	{
// 		return pathTar2[0];
// 	}
// 	if (pathTar1.size()>pathTar2.size())
// 	{
// 		int ipos = pathTar1.size() - 2;
// 		for (; ipos >= 0; --ipos)
// 		{
// 			for (int j = pathTar2.size() - 2; j >= 0; --j)
// 			{
// 				if (pathTar1[ipos]->val == pathTar2[j]->val)
// 				{
// 					return pathTar1[ipos];
// 				}
// 			}
// 		}
// 	}
// 	else
// 	{
// 		int ipos = pathTar2.size() - 2;
// 		for (; ipos >= 0; --ipos)
// 		{
// 			for (int j = pathTar1.size() - 2; j >= 0; --j)
// 			{
// 				if (pathTar1[ipos]->val == pathTar2[j]->val)
// 				{
// 					return pathTar2[ipos];
// 				}
// 			}
// 		}
// 	}
// 	return root;
	//查找的目标节点可能在一侧或则是2侧
	if (!root||root->val==p->val||root->val==q->val)
	{
		return root;
	}
	TreeNode * left = lowestCommonAncestor(root->left, p, q);
	TreeNode * right = lowestCommonAncestor(root->right, p, q);
	if (left&&right)
	{
		return root;
	}
	return left ? left : right;
}

TreeNode* Solution::buildTreeByPreAnIn(vector<int>& preorder, vector<int>& inorder)
{
	if (preorder.empty()||inorder.empty())
	{
		return NULL;
	}
	if (preorder.size()==1)
	{
		TreeNode * node=new TreeNode(preorder[0]);
		node->left = NULL;
		node->right = NULL;
	}
	TreeNode * root = new TreeNode(preorder[0]);
	int it = find(inorder.begin(), inorder.end(), root->val) - inorder.begin();
	vector<int> preLeft(preorder.begin()+1, preorder.begin() + it+1);
	vector<int> preRight(preorder.begin() + it+1, preorder.end());
	vector<int> inorderLeft(inorder.begin(), inorder.begin() + it);
	vector<int> inorderRight(inorder.begin()+it+1, inorder.end());
	root->left = buildTreeByPreAnIn(preLeft, inorderLeft);
	root->right = buildTreeByPreAnIn(preRight, inorderRight);
	return root;
}

TreeNode* Solution::buildTreeByInAnPos(vector<int>& inorder, vector<int>& postorder)
{
	if (inorder.empty() || postorder.empty())
	{
		return NULL;
	}
	if (postorder.size() == 1)
	{
		TreeNode * node = new TreeNode(postorder[0]);
		node->left = NULL;
		node->right = NULL;
		return node;
	}
	TreeNode * root = new TreeNode(postorder[postorder.size()-1]);
	int it = find(inorder.begin(), inorder.end(), root->val) - inorder.begin();
	vector<int> inorderLeft(inorder.begin(), inorder.begin() + it);
	vector<int> inorderRight(inorder.begin() + it + 1, inorder.end());
	vector<int> posLeft(postorder.begin(), postorder.begin() + it);
	vector<int> posRight(postorder.begin() + it, postorder.end()-1);
	root->left = buildTreeByInAnPos(inorderLeft, posLeft);
	root->right = buildTreeByInAnPos(inorderRight, posRight);
	return root;
}

TreeNode* Solution::GenerateBST(std::vector<ListNode*> &vecNode, int left, int right)
{
	if (left == right)
	{
		return NULL;
	}
	int mid = left + (right - left)/2;
	TreeNode * root = new TreeNode(vecNode[mid]->val);
	root->left = GenerateBST(vecNode, left, mid);
	root->right = GenerateBST(vecNode, mid+1, right);
	return root;
}

TreeNode* Solution::sortedListToBST(ListNode* head)
{
	if (head==NULL)
	{
		return NULL;
		return NULL;
	}
	std::vector<ListNode *> vecNode;
	while (head)
	{
		vecNode.push_back(head);
		head = head->next;
	}
	return GenerateBST(vecNode, 0, vecNode.size());
}

void Solution::DFSFindPath(std::vector<vector<int>> &res, std::vector<int> &path,TreeNode * root, int sum)
{
	if (root==NULL)
	{
		return;
	}
	path.push_back(root->val);
	if (root->left==nullptr&&root->right==nullptr)
	{
		if (sum-root->val==0)
		{
			res.push_back(path);
		}
		return;
	}
	sum -= root->val;
	if (root->left)
	{
		DFSFindPath(res, path, root->left, sum);
		path.pop_back();
	}
	if (root->right)
	{
		DFSFindPath(res, path, root->right, sum);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::pathSum(TreeNode* root, int sum)
{
	if (root==NULL)
	{
		return {};
	}
	std::vector<vector<int>> result;
	std::vector<int> path;
	DFSFindPath(result, path, root, sum);
	return result;
}
TreeNode* pre = NULL;
void Solution::flatten(TreeNode* root)
{
	if (root==NULL)
	{
		return;
	}
	if (pre)
	{
		pre->left = NULL;
		pre->right = root;
	}
	pre = root;
	TreeNode * copyNode = pre->right;
	flatten(root->left);
	flatten(copyNode);
}

int Solution::minimumTotal(vector<vector<int>>& triangle)
{
	if (triangle.empty())
	{
		return 0;
	}
	vector<int> dp(triangle.size()+1);
	//从下向上的动态规划
	for (int i = triangle.size() - 1; i >= 0;i--)
	{
		for (int j = 0; j < triangle[i].size();++j)
		{
			dp[j] =min(dp[j+1],dp[j])+triangle[i][j];
		}
	}
	return dp[0];
}

int Solution::ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
	//图的广度优先搜索
	unordered_set<string> hashWord;
	unordered_set<string> unSetWord;
	for (auto it:wordList)
	{
		hashWord.insert(it);
	}
	if (hashWord.count(endWord)==0)
	{
		return 0;
	}
	unSetWord.insert(beginWord);
	queue<string> queStr;
	int len = 1;
	queStr.push(beginWord);
	while (!queStr.empty())
	{
		int iSize = queStr.size();
		while (iSize--)
		{
			auto ss = queStr.front();
			queStr.pop();
			for (int i = 0; i < ss.size();++i)
			{
				string wordTemp = ss;
				for (char ch = 'a'; ch <= 'z';++ch)
				{
					wordTemp[i] = ch;
					if (hashWord.count(wordTemp) == 0 || unSetWord.count(wordTemp))
					{
						continue;
					}
					if (wordTemp==endWord)
					{
						return len + 1;
					}
					unSetWord.insert(wordTemp);
					queStr.push(wordTemp);
				}
			}
		}
		len++;
	}
	return 0;
}

void Solution::DFSFindPathSumNumber(std::vector<int> &path, TreeNode * root, int &sum)
{
	if (root==nullptr)
	{
		return;
	}
	path.push_back(root->val);
	if (root->left==nullptr&&root->right==nullptr)
	{
		int iNumber = 0;
		for (auto it:path)
		{
			iNumber = 10 * iNumber + it;
		}
		sum += iNumber;
		return;
	}
	if (root->left)
	{
		DFSFindPathSumNumber(path, root->left, sum);
		path.pop_back();
	}
	if (root->right)
	{
		DFSFindPathSumNumber(path, root->right, sum);
		path.pop_back();
	}
}

void Solution::DFSBoard(vector<vector<char>>& board, int x, int y)
{
	if (board[x][y] =='Y')
	{
		return;
	}
	for (int i = 0; i < 4;++i)
	{
		int newx = x + dir[i][0];
		int newy = y + dir[i][1];
		if (newx >= 0 && newx < board.size()&&newy>=0&&newy<board[0].size())
		{
			DFSBoard(board, newx, newy);
		}
	}
}

void Solution::solve(vector<vector<char>>& board)
{
	if (board.empty())
	{
		return;
	}
	for (int i = 0; i < board[0].size(); ++i)
	{
		if (board[0][i] == 'O')
		{
			DFSBoard(board, 0, i);
		}
		if (board[board.size() - 1][i] == 'O')
		{
			DFSBoard(board, board.size() - 1, i);
		}
	}
	for (int i = 0; i < board.size(); ++i)
	{
		if (board[i][0] == 'O')
		{
			DFSBoard(board, i, 0);
		}
		if (board[i][board[0].size() - 1] == 'O')
		{
			DFSBoard(board, i, board[0].size() - 1);
		}
	}
	for (int i = 0; i < board.size(); ++i)
	{
		for (int j = 0; j < board[i].size(); ++j)
		{
			if (board[i][j] == 'Y')
			{
				board[i][j] = 'O';
			}
			else
			{
				board[i][j] = 'X';
			}
		}
	}
}

int Solution::sumNumbers(TreeNode* root)
{
	if (root==NULL)
	{
		return 0;
	}
	int sum = 0;
	vector<int> path;
	DFSFindPathSumNumber(path, root, sum);
	return sum;
}

void Solution::DFSpartitionStr(std::vector<vector<string>> & res, vector<string> & path, int start, string &s)
{
	if (start==s.size())
	{
		res.push_back(path);
		return;
	}
	int prePartion = start;
	for (int i = start; i < s.size();++i)
	{
		if (!CheckStr(s,prePartion,i))
		{
			continue;
		}
		path.push_back(s.substr(prePartion, i-prePartion+ 1));
		DFSpartitionStr(res, path, i + 1, s);
		path.pop_back();
	}
}

bool Solution::CheckStr(string & s, int left, int right)
{
	if (left>right)
	{
		return false;
	}
	while (left<right)
	{
		if (s[left] != s[right])
		{
			return false;
		}
		left++;
		right--;
	}
	return true;
}

std::vector<std::vector<std::string>> Solution::partitionStr(string s)
{
	if (s.empty())
	{
		return{};
	}
	vector<vector<string>> res;
	vector<string> path;
	DFSpartitionStr(res, path, 0, s);
	return res;
}

int Solution::canCompleteCircuit(vector<int>& gas, vector<int>& cost)
{
	if (gas.empty()||cost.empty())
	{
		return -1;
	}
	int total = 0;
	int sum = 0;
	int start = 0;
	for (int i = 0; i < gas.size();++i)
	{
		total += gas[i] - cost[i];
		if (sum>=0)
		{
			sum += gas[i] - cost[i];
		}
		else
		{
			//找到最长子序列和的起点
			sum = gas[i] - cost[i];
			start = i;
		}
	}
	return total>=0?start:-1;
}

bool Solution::wordBreak(string s, vector<string>& wordDict)
{
	//宽度优先搜索
// 	if (s.empty())
// 	{
// 		return true;
// 	}
// 	if (wordDict.empty())
// 	{
// 		return false;
// 	}
// 	unordered_set<string> unSet;
// 	for (auto it : wordDict)
// 	{
// 		unSet.insert(it);
// 	}
// 	queue<int> q;
// 	q.push(0);
// 	std::vector<int> signVec(s.size() + 1, 0);
// 	while (!q.empty())
// 	{
// 		int start = q.front();
// 		q.pop();
// 		if (signVec[start] == 0)
// 		{
// 			for (int end = start + 1; end <= s.size(); ++end)
// 			{
// 				if (unSet.count(s.substr(start, end - start)))
// 				{
// 					q.push(end);
// 					if (end == s.size())
// 					{
// 						return true;
// 					}
// 				}
// 			}
// 			signVec[start] = 1;
// 		}
// 	}
	//动态规划
	if (s.empty())
	{
		return true;
	}
	if (wordDict.empty())
	{
		return false;
	}
	unordered_set<string> unSetWord;
	for (auto it : wordDict)
	{
		unSetWord.insert(it);
	}
	std::vector<bool> dp(s.size() + 1, false);
	dp[0] = true;
	for (int i = 1; i <= s.size();++i)
	{
		for (int j = 0; j <i;++j)
		{
			if (dp[j] && unSetWord.count(s.substr(j,i-j)))
			{
				dp[i] = true;
				break;
			}
		}
	}
	return false;
}

int Solution::singleNumber(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	int res = 0;
	for (int i = 0; i < 32;++i)
	{
		int mask = 1 << i;
		int cnt = 0;
		for (auto it:nums)
		{
			if (it&mask!=0)
			{
				cnt++;
			}
		}
		if (cnt%3!=0)
		{
			//将结果复制到输出值中
			res |= mask;
		}
	}
	return res;
}

void Solution::reorderList(ListNode* head)
{
	if (head == NULL)
	{
		return;
	}
	ListNode * slow = head;
	ListNode * fast = head;
	while (fast&&fast->next)
	{
		fast = fast->next->next;
		slow = slow->next;
	}
	//反转slow后的链表
	ListNode * pre = NULL; 
	ListNode * cur = nullptr;
	while (slow)
	{
		cur = slow->next;
		slow->next = pre;
		pre = slow;
		slow = cur;
	}
	slow = pre;
	ListNode * newHead = new ListNode(0);
	newHead->next = head;
	while (slow&&slow->next)
	{
		cur = head->next;
		head->next = slow;
		pre = slow->next;
		slow->next = cur;
		head = cur;
		slow = pre;
	}
	if (!fast)
	{
		head->next = slow;
	}
	head = newHead->next;
}

ListNode* Solution::insertionSortList(ListNode* head)
{
	if (head==NULL)
	{
		return head;
	}
	ListNode * nodeHead = new ListNode(0);
	ListNode * cur = NULL;
	ListNode * pre = NULL;
	ListNode * temp = NULL;
	while (head)
	{
		temp = head->next;
		cur = nodeHead->next;
		pre = nodeHead;
		while (cur)
		{
			if (cur->val>head->val)
			{
				pre->next = head;
				head->next = cur;
				break;
			}
			pre = cur;
			cur = cur->next;
		}
		if (!cur)
		{
			pre->next =head;
			head->next = NULL;
		}
		head = temp;
	}
	return nodeHead->next;
}

void Solution::SwapList(ListNode * p1, ListNode * p2)
{
	if (p1->val==p2->val)
	{
		return;
	}
	p1->val ^= p2->val;
	p2->val ^= p1->val;
	p1->val ^= p2->val;
}

void Solution::QuickSort(ListNode * left, ListNode * right)
{
	if (left == right || left == NULL || right == NULL) return;
	int t = right->val;
	ListNode* prev = NULL;
	ListNode* curr = left;
	ListNode* p = left;
	while (p != right) 
	{
		if (p->val < t)
		{
			SwapList(curr, p);
			prev = curr;
			curr = curr->next;
		}
		p = p->next;
	}
	if (curr != right) 
	{
		SwapList(curr, right);
		curr = curr->next;
	}
	QuickSort(left, prev);
	QuickSort(curr, right);
}

ListNode* Solution::sortList(ListNode* head)
{
	if (head==NULL)
	{
		return head;
	}
	ListNode *tail = head;
	while (tail!=NULL&&tail->next!=NULL)
	{
		tail = tail->next;
	}
	QuickSort(head, tail);
	return head;
}

int Solution::evalRPN(vector<string>& tokens)
{
	if (tokens.empty())
	{
		return 0;
	}
	stack<int> stacNumber;
	int res = 0;
	int pre = 0;
	int cur = 0;
	for (auto it:tokens)
	{
		if (it=="+")
		{
			cur = stacNumber.top();
			stacNumber.pop();
			pre = stacNumber.top();
			stacNumber.pop();
			cur += pre;
			stacNumber.push(cur);
			res = cur;
		}
		else if (it == "-")
		{
			cur = stacNumber.top();
			stacNumber.pop();
			pre = stacNumber.top();
			stacNumber.pop();
			cur = pre-cur;
			stacNumber.push(cur);
			res = cur;
		}
		else if (it == "*")
		{
			cur = stacNumber.top();
			stacNumber.pop();
			pre = stacNumber.top();
			stacNumber.pop();
			cur = pre * cur;
			stacNumber.push(cur);
			res = cur;
		}
		else if (it == "/")
		{
			cur = stacNumber.top();
			stacNumber.pop();
			pre = stacNumber.top();
			stacNumber.pop();
			cur = pre / cur;
			stacNumber.push(cur);
			res = cur;
		}
		else
		{
			res = atoi(it.c_str());
			stacNumber.push(res);
		}
	}
	return res;
}

std::string Solution::reverseWords(string s)
{
	if (s.empty())
	{
		return s;
	}
	std::vector<string> resVec;
	s += " ";
	int icount = 0;
	for (int i = 0; i < s.size();)
	{
		if (s[i]!=' ')
		{
			icount = 0;
			while (i<s.size()&&s[i]!=' ')
			{
				icount++;
				i++;
			}
			resVec.push_back(s.substr(i-icount, icount));
		}
		i++;
	}
	reverse(resVec.begin(), resVec.end());
	string res;
	if (!resVec.empty())
	{
		for (int i = 0; i < resVec.size() - 1; ++i)
		{
			res += resVec[i] + " ";
		}
		res += resVec[resVec.size() - 1];
	}
	return res;
}

int Solution::maxProduct(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	if (nums.size()==1)
	{
		return nums[0];
	}
	//记录之前的最大值和最小值
	int maxNumber = 0;
	int minNumber = 0;
	int res = INT_MIN;
	for (auto it:nums)
	{
		if (it==0)
		{
			maxNumber = minNumber = 0;
		}
		else if (it>0)
		{
			maxNumber = max(it*maxNumber,it);
			minNumber = minNumber*it;
		}
		else
		{
			int temp = maxNumber;
			maxNumber = max(it*minNumber, it);
			minNumber = min(it*temp, it);
		}
		res = max(maxNumber, res);
	}
	return res;
}

int Solution::findMin(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	//旋转数组只能与右边界比较
	int lhs = 0, rhs = nums.size()-1;
	while (lhs<rhs)
	{
		int mid = lhs + (rhs - lhs) / 2;
		if (nums[mid]>nums[rhs])
		{
			lhs = mid + 1;
		}
		else
		{
			rhs = mid;
		}
	}
	return nums[lhs];
}

int Solution::findMinSame(vector<int>& nums)
{
	if (nums.empty())
		return -1;
	int lhs = 0, rhs = nums.size() - 1;
	while (lhs<rhs)
	{
		int mid = lhs + (rhs - lhs) / 2;
		if (nums[mid]>nums[rhs])
		{
			lhs = mid + 1;
		}
		else if (nums[mid]<nums[rhs])
		{
			rhs = mid;
		}
		else
		{
			//排除右边界
			--rhs;
		}
	}
	return nums[lhs];
}

int Solution::findPeakElement(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	int mid = 0;
	int lhs = 0, rhs = nums.size() - 1;
	//左右都是负无穷，mid和mid+1递增的话，右边肯定会递减的
	while (lhs<rhs)
	{
		mid = lhs + (rhs - lhs) / 2;
		if (mid<nums.size()-1&&nums[mid]<nums[mid+1])
		{
			lhs = mid+1;
		}
		else
		{
			rhs = mid;
		}
	}
	return lhs;
}

int Solution::compareVersion(string version1, string version2)
{
	if (version1.empty()||version2.empty())
	{
		return 0;
	}
	version1+=".";
	version2+= ".";
	int lhs = 0, rhs = 0;
	int ver1 = 0, ver2 = 0;
	int ipos = 0;
	while (lhs < version1.size()||rhs<version2.size())
	{
		if (lhs==version1.size())
		{
			ver1 = 0;
		}
		else
		{
			ipos = version1.find(".",lhs);
			if (ipos==string::npos)
			{
				ver1 = 0;
				ipos = version1.size();
			}
			else
			{
				ver1 = atoi(version1.substr(lhs, ipos - lhs).c_str());
				lhs = ipos + 1;
			}
		}
		if (rhs == version2.size())
		{
			ver2 = 0;
		}
		else
		{
			ipos = version2.find(".", rhs);
			if (ipos == string::npos)
			{
				ver2 = 0;
				ipos = version2.size();
			}
			else
			{
				ver2 = atoi(version2.substr(rhs, ipos-rhs).c_str());
				rhs = ipos + 1;
			}
		}
		if (ver1<ver2)
		{
			return -1;
		}
		else if (ver1>ver2)
		{
			return 1;
		}
	}
	return 0;
}

std::string Solution::fractionToDecimal(int numerator, int denominator)
{
	if (denominator==0)
	{
		return "";
	}
	if (numerator==0)
	{
		return "0";
	}
	string res;
	long long num = static_cast<long long>(numerator);
	long long den = static_cast<long long>(denominator);
	if ((num^den) < 0)
	{
		res += "-";
	}
	num = abs(num);
	den = abs(den);
	res += to_string(num / den);
	num = num%den;
	if (num)
	{
		res += ".";
	}
	unordered_map<long long, int> hashM;
	int index = 0; //记录循环开始的位置
	while (num&&hashM.count(num)==0)
	{
		hashM[num] = index++;
		num = num * 10;
		res += to_string(num / den);
		num = num%den;
	}
	if (hashM.count(num))
	{
		res += "()";
		int cur = res.size() - 2;
		while (index-->hashM[num])
		{
			swap(res[cur], res[cur - 1]);
			--cur;
		}
	}
	return res;
}

std::string Solution::largestNumber(vector<int>& nums)
{
	if (nums.empty())
	{
		return "";
	}
	std::string res;
	std::vector<string> strVec;
	for (auto it:nums)
	{
		strVec.push_back(to_string(it));
	}
	auto f_sort = [](const string & lhs, const string & rhs)
	{
		return lhs + rhs < rhs + lhs;
	};
	sort(strVec.begin(), strVec.end(), f_sort);
	for (auto it = strVec.rbegin(); it!=strVec.rend();++it)
	{
		res += *it;
	}
	if (res[0]=='0')
	{
		return "0";
	}
	return res;
}

std::vector<std::string> Solution::findRepeatedDnaSequences(string s)
{
	if (s.length()<=10)
	{
		return {};
	}
	std::unordered_map<string, int> hashMStr;
	for (int i = 0; i <=s.size() - 10;++i)
	{
		hashMStr[s.substr(i, 10)]++;
	}
	std::vector<string> resVec;
	for (auto it:hashMStr)
	{
		if (it.second>1)
		{
			resVec.push_back(it.first);
		}
	}
	return resVec;
}
void DFSHelpr(std::vector<int> &res, int depth, TreeNode *root)
{
	if (root==nullptr)
	{
		return;
	}
	if (depth==res.size())
	{
		res.push_back(root->val);
	}
	DFSHelpr(res, depth + 1, root->right);
	DFSHelpr(res, depth + 1, root->left);
}
std::vector<int> Solution::rightSideView(TreeNode* root)
{
	if (root==nullptr)
	{
		return{};
	}
	std::vector<int> resVec;
// 	std::queue<TreeNode*> q;
// 	q.push(root);
// 	while (!q.empty())
// 	{
// 		int isize = q.size();
// 		
// 		while (isize--)
// 		{
// 			auto node = q.front();
// 			q.pop();
// 			if (node->left!=nullptr)
// 			{
// 				q.push(node->left);
// 			}
// 			if (node->right)
// 			{
// 				q.push(node->right);
// 			}
// 			if (isize==0)
// 			{
// 				resVec.push_back(node->val);
// 			}
// 		}
// 	}
	DFSHelpr(resVec, 0, root);
	return resVec;
}
void BFSLands(vector<vector<bool>> &visited, vector<vector<char>>& grid, int pointx, int pointy)
{
	if (!visited[pointx][pointy])
	{
		return;
	}
	visited[pointx][pointy] = false;
	for (int i = 0; i < 4;++i)
	{
		int newx = pointx + dir[i][0];
		int newy = pointy + dir[i][1];
		if (newx >= 0 && newx < grid.size() && newy >= 0 && newy < grid[0].size()&&visited[newx][newy])
		{
			BFSLands(visited, grid,newx,newy);
		}
	}
}

int Solution::numIslands(vector<vector<char>>& grid)
{
	if (grid.empty())
	{
		return 0;
	}
	int icou=0;
	vector<vector<bool>> visited(grid.size(), std::vector<bool>(grid[0].size(), true));
	for (int i = 0; i < grid.size();i++)
	{
		for (int j = 0; j < grid[i].size();++j)
		{
			if (visited[i][j]&&grid[i][j]=='1')
			{
				BFSLands(visited, grid, i, j);
				++icou;
			}
		}
	}
	return icou;
}

int Solution::rangeBitwiseAnd(int m, int n)
{
	//高位比较是否相等
	int cur = 0;
	while (m!=n)
	{
		m = m >> 1;
		n = n >> 1;
		++cur;
	}
	return m<<cur;
}
void DFSSearch(int i,std::vector<int> & flag,std::vector<std::vector<int>> &outDegree)
{
	flag[i] = -1;
	for (auto &it:outDegree[i])
	{
		if (flag[it]==-1)
		{
			return;
		}
		else if (flag[it]==0)
		{
			DFSSearch(it, flag, outDegree);
		}
	}
	flag[i] = 1;
}
bool Solution::canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
	if (numCourses==0||prerequisites.empty())
	{
		return true;
	}
	//BFS
	//获取每个节点入度和出度
// 	std::vector<int> inDegree(numCourses, 0);
// 	std::vector<std::vector<int>> outDegree(numCourses, std::vector<int>(0));
// 	for (auto &it:prerequisites)
// 	{
// 		inDegree[it[0]]++;
// 		outDegree[it[1]].push_back(it[0]);
// 	}
// 	queue<int> q;
// 	for (int i = 0; i < inDegree.size();++i)
// 	{
// 		if (inDegree[i]==0)
// 		{
// 			q.push(i);
// 		}
// 	}
// 	int cnt = 0;
// 	while (!q.empty())
// 	{
// 		int k = q.front();
// 		q.pop();
// 		++cnt;
// 		for (auto it:outDegree[k])
// 		{
// 			if (--inDegree[it]==0)
// 			{
// 				q.push(it);
// 			}
// 		}
// 	}
// 	return cnt == numCourses;
	//DFS构建邻接矩阵,判断是否有环
	std::vector<int> flag(numCourses, 0);
	std::vector<std::vector<int>> outDegree(numCourses);
	for (auto &it : prerequisites)
	{
		outDegree[it[1]].push_back(it[0]);
	}
	for (int i = 0; i < numCourses;++i)
	{
		//说明有环
		if (flag[i]==-1)
		{
			return false;
		}
		else if (flag[i]==0)
		{
			DFSSearch(i, flag, outDegree);
		}
	}
	for (auto &it:flag)
	{
		if (it==-1)
		{
			return false;
		}
	}
	return true;
}

int Solution::minSubArrayLen(int s, vector<int>& nums)
{
	if (nums.empty())
	{
		return 0;
	}
	int res = INT_MAX;
	//使用双指针,找到以某个位置未开始的和
	int left = 0;
	int sum = 0;
	for (int i = 0; i < nums.size();++i)
	{
		sum += nums[i];
		while (sum>=s)
		{
			res = min(res, i - left + 1);
			sum -= nums[left++];
		}
	}
	return res==INT_MAX?0:res;
}

std::vector<int> Solution::findOrder(int numCourses, vector<vector<int>>& prerequisites)
{
	//BFS解法,DFS解法使用栈，将节点导入即可
	std::vector<int> inDegree(numCourses,0);
	std::vector<std::vector<int>> outDegree(numCourses);
	for (auto &it:prerequisites)
	{
		++inDegree[it[0]];
		outDegree[it[1]].push_back(it[0]);
	}
	queue<int> que;
	for (int i = 0; i < inDegree.size();++i)
	{
		if (inDegree[i]==0)
		{
			que.push(i);
		}
	}
	std::vector<int> res;
	while (!que.empty())
	{
		auto node = que.front();
		que.pop();
		res.push_back(node);
		for (auto it : outDegree[node])
		{
			if (--inDegree[it] == 0)
			{
				que.push(it);
			}
		}
	}
	if (res.size()!=numCourses)
	{
		return{};
	}
	return res;
}

int Solution::rob(vector<int>& nums)
{
	if (nums.size()==1)
	{
		return nums[0];
	}
	std::vector<int> dp1(nums.size() + 1);
	std::vector<int> dp2(nums.size() + 1);
	for (int i = 2; i <=nums.size();++i)
	{
		dp1[i] = max(nums[i - 2] + dp1[i - 2], dp1[i - 1]);
		dp2[i] = max(nums[i - 1] + dp2[i - 2], dp2[i - 1]);
	}
	return max(dp1[nums.size()], dp2[nums.size()]);
}

int Solution::findKthLargest(vector<int>& nums, int k)
{
	if (nums.empty())
	{
		return -1;
	}
	priority_queue<int, vector<int>, std::greater<int> > queLow;
	for (auto &it:nums)
	{
		queLow.push(it);
		if (queLow.size()>k)
		{
			queLow.pop();
		}
	}
	return queLow.top();
}
std::vector<vector<int>> resCom;
void BACKSerach(int sum, int &target, std::vector<int> &path, int iNumber,int istart)
{
	if (iNumber==0)
	{
		if (sum==target)
		{
			resCom.push_back(path);
		}
		return;
	}
	for (int i = istart; i <= 10-iNumber;++i)
	{
		path.push_back(i);
		BACKSerach(sum + i, target, path, iNumber - 1, i + 1);
		path.pop_back();
	}
}

std::vector<std::vector<int>> Solution::combinationSum3(int k, int n)
{
	std::vector<int> path;
	BACKSerach(0, n, path, k, 1);
	return resCom;
}

int Solution::maximalSquare(vector<vector<char>>& matrix)
{
	//最大正方形的边长为左 上 左上 最小边长+1
	if (matrix.empty())
	{
		return 0;
	}
	std::vector<std::vector<int>> dp(matrix.size(),std::vector<int>(matrix[0].size(),0));
	int res = 0;
	for (int i = 0; i < matrix.size();++i)
	{
		for (int j = 0; j < matrix[i].size();++j)
		{
			if (i==0||j==0)
			{
				if (matrix[i][j]=='1')
				{
					dp[i][j] = 1;
					res = max(res,1);
				}
			}
			else
			{
				if (matrix[i][j]=='1')
				{
					int line = INT_MAX;
					line = min(dp[i - 1][j], dp[i][j - 1]);
					line = min(dp[i - 1][j - 1], line);
					dp[i][j] = line + 1;
					res = max(dp[i][j], res);
				}
			}
		}
	}
	return res*res;
}
int countHigh(TreeNode* root)
{
	int high = 0;
	while (root)
	{
		++high;
		root = root->left;
	}
	return high;
}
int Solution::countNodes(TreeNode* root)
{
	if (root==NULL)
	{
		return 0;
	}
	int leftHigh = countHigh(root->left);
	int rightHigh = countHigh(root->right);
	//移位运算符的优先级低于加号运算符
	if (leftHigh==rightHigh)
	{
		return (1 << leftHigh) + countNodes(root->right);
	}
	else
	{
		return (1 << rightHigh) + countNodes(root->left);
	}
}

std::vector<std::string> Solution::summaryRanges(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	int pre = nums[0];
	int ipos = 0;
	std::vector<std::string> resVec;
	for (int i = 1; i < nums.size();)
	{
		while (i<nums.size()&&nums[i-1]+1==nums[i])
		{
			++i;
		}
		if (i==nums.size())
		{
			break;
		}
		if (i - ipos == 1)
		{
			resVec.push_back(to_string(pre));
		}
		else
		{
			resVec.push_back(to_string(pre) + "->" + to_string(nums[i - 1]));
		}
		pre = nums[i];
		ipos = i;
		++i;
	}
	if (pre==nums[nums.size()-1])
	{
		resVec.push_back(to_string(pre));
	}
	else
	{
		resVec.push_back(to_string(pre) + "->" + to_string(nums[nums.size() - 1]));
	}
	return resVec;
}

std::vector<int> Solution::majorityElement(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	std::vector<int> resVec;
	//摩尔投票
	int num1 = nums[0], num2 = nums[0];
	int count1 = 0, count2 = 0;
	for (auto it:nums)
	{
		if (it==num1)
		{
			++count1;
			continue;
		}
		if (it==num2)
		{
			++count2;
			continue;
		}
		if (count1==0)
		{
			num1 = it;
			++count1;
			continue;
		}
		if (count2==0)
		{
			num2 = it;
			++count2;
			continue;
		}
		//若此时两个候选人的票数都不为0，且当前元素不投AB，那么A,B对应的票数都要--;
		--count1;
		--count2;
	}
	count1 = 0;
	count2 = 0;
	for (auto it:nums)
	{
		if (it==num1)
		{
			++count1;
		}
		else if (it==num2)
		{
			++count2;
		}
	}
	if (count1>nums.size()/3)
	{
		resVec.push_back(num1);
	}
	if (count2>nums.size() / 3)
	{
		resVec.push_back(num2);
	}
	return resVec;
}

int Solution::kthSmallest(TreeNode* root, int k)
{
	if (root==nullptr)
	{
		return -1;
	}
	int res = 0;
	//二叉搜索树第k小的元素
	stack<TreeNode*> stacRoot;
	while (root)
	{
		stacRoot.push(root);
		root = root->left;
	}
	while (!stacRoot.empty())
	{
		auto note = stacRoot.top();
		stacRoot.pop();
		if (--k==0)
		{
			res = note->val;
			break;
		}
		note = note->right;
		while (note)
		{
			stacRoot.push(note);
			note = note->left;
		}
	}
	return res;
}

std::vector<int> Solution::productExceptSelf(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	int left = 1;
	int right = 1;
	std::vector<int> outVec(nums.size(),1);
	for (int i = 0; i < nums.size();++i)
	{
		outVec[i] *= left;
		left *= nums[i];

		outVec[nums.size() - i - 1] *= right;
		right *= nums[nums.size() - i - 1];
	}
	return outVec;
}

bool Solution::searchMatrix2(vector<vector<int>>& matrix, int target)
{
	if (matrix.empty())
	{
		return false;
	}
	int col = matrix[0].size()-1;
	int row = 0;
	int prito = 0;
	//从右上角开始寻找
	while (true)
	{
		if (col<0||row==matrix.size())
		{
			return false;
		}
		prito = matrix[row][col];
		if (target == prito)
		{
			return true;
		}
		else if (target > prito)
		{
			++row;
		}
		else
		{
			--col;
		}
	}
	return false;
}
std::map<string, std::vector<int>> mapDiff;
std::vector<int> Solution::diffWaysToCompute(string input)
{
	std::vector<int> res;
	if (mapDiff.count(input))
	{
		return mapDiff[input];
	}
	for (int i = 0; i < input.size();++i)
	{
		if (!(input[i]>='0'&&input[i]<='9'))
		{
			std::vector<int> difLeft = diffWaysToCompute(input.substr(0, i));
			std::vector<int> difRight = diffWaysToCompute(input.substr(i + 1));
			for (auto &itL:difLeft)
			{
				for (auto &itR:difRight)
				{
					if (input[i]=='+')
					{
						res.push_back(itL + itR);
					}
					else if (input[i]=='-')
					{
						res.push_back(itL - itR);
					}
					else if (input[i]=='*')
					{
						res.push_back(itL*itR);
					}
				}
			}
		}
	}
	//最后一个数字
	if (res.empty())
	{
		res.push_back(atoi(input.c_str()));
	}
	mapDiff[input] = res;
	return res;
}

std::vector<int> Solution::singleNumber2(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	std::vector<int> res(2, 0);
	int sign = 0;
	for (auto it:nums)
	{
		sign ^= it;
	}
	//找到不同的位进按位取反
	//~sign + 1;
	sign &= -sign;
	for (auto &it:nums)
	{
		if (it&sign)
		{
			res[0] ^= it;
		}
		else
		{
			res[1] ^= it;
		}
	}
	return res;
}

int Solution::nthUglyNumber(int n)
{
	//三指针法
	std::vector<int> dp(n, 1), ugly(3, 0);
	for (int i = 1; i <n;++i)
	{
		int a = dp[ugly[0]] * 2, b = dp[ugly[1]] * 3, c = dp[ugly[2]] * 5;
		int next = min(a, min(b, c));
		if (next==a)
		{
			++ugly[0];
		}
		if (next==b)
		{
			++ugly[1];
		}
		if (next==c)
		{
			++ugly[2];
		}
		dp[i] = next;
	}
	return dp[n-1];
}

std::vector<string> Solution::convertStringToSuffix(string iStr)
{
	if (iStr.empty())
	{
		return{};
	}
	iStr.push_back('#');
	std::vector<string> resStr;
	stack<char> stacSign;
	stacSign.push('#');
	cout << iStr[0] << " ";
	std::string strTemp;
	for (int i = 0; i < iStr.size();)
	{
		if (iStr[i]<='9'&&iStr[i]>='0')
		{
			strTemp = iStr[i];
			++i;
			while (i < iStr.size()&&iStr[i]<='9'&&iStr[i]>='0')
			{
				strTemp.push_back(iStr[i]);
				++i;
			}
			resStr.push_back(strTemp);
		}
		else if (iStr[i]=='(')
		{
			stacSign.push(iStr[i]);
			++i;
		}
		else if (iStr[i]==')')
		{
			while (stacSign.top()!='(')
			{
				strTemp = stacSign.top();
				resStr.push_back(strTemp);
				stacSign.pop();
			}
			stacSign.pop();
			++i;
		}
		else if (iStr[i]=='+'||iStr[i]=='-')
		{
			while (!stacSign.empty()&&stacSign.top()!='('&&stacSign.top()!='#')
			{
				strTemp = stacSign.top();
				resStr.push_back(strTemp);
				stacSign.pop();
			}
			stacSign.push(iStr[i]);
			++i;
		}
		else if (iStr[i]=='*'||iStr[i]=='/')
		{
			while (!stacSign.empty() && (stacSign.top() == '*' || stacSign.top() == '/'))
			{
				strTemp = stacSign.top();
				resStr.push_back(strTemp);
				stacSign.pop();
			}
			stacSign.push(iStr[i]);
			++i;
		}
		else if (iStr[i] == '#')
		{
			while (!stacSign.empty() && stacSign.top() != '#')
			{
				strTemp = stacSign.top();
				resStr.push_back(strTemp);
				stacSign.pop();
			}
			stacSign.pop();
			break;
		}
		else
		{
			++i;
		}
	}
	return resStr;
}

int Solution::hIndex(vector<int>& citations)
{
	if (citations.empty())
	{
		return 0;
	}
	sort(citations.begin(), citations.end());
	for (int i = 0; i < citations.size();++i)
	{
		if (citations[i]>=citations.size()-i)
		{
			return citations.size() - i;
		}
	}
	return 0;
}

int Solution::hIndex2(vector<int>& citations)
{
	if (citations.empty())
	{
		return 0;
	}
	int size = citations.size();
	int left = 0, right = citations.size();
	while (left<right)
	{
		int mid = left + (right - left) / 2;
		int iTemp = citations[mid] + mid;
		if (iTemp>=size)
		{
			right = mid;
		}
		else
		{
			left = mid+1;
		}
	}
	return size-left;
}
bool BackNumber(int &iTarget, int inumber, std::vector<int> &resNumber,int iadd,int start)
{
	if (iadd>iTarget)
	{
		return false;
	}
	if (inumber==0)
	{
		if (iadd==iTarget)
		{
			return true;
		}
		return false;
	}
	for (int i = start; i <resNumber.size();++i)
	{
		if (BackNumber(iTarget, inumber - 1, resNumber, iadd + resNumber[i], start))
		{
			return true;
		}
	}
	return false;
}

int Solution::numSquares(int n)
{
	if (n==1)
	{
		return n;
	}
	std::vector<int> resNumber;
	for (int i = 1; i <= sqrt(n);++i)
	{
		resNumber.push_back(i*i);
	}
	reverse(resNumber.begin(), resNumber.end());
	for (int i = 1; i <= n; ++i)
	{
		if (BackNumber(n,i,resNumber,0,0))
		{
			return i;
		}
	}
	return -1;
}

int Solution::findDuplicate(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	int left = 0,right=nums.size();
	while (left<right)
	{
		//如果选择右中位数 right=mid-1 left=mid 若选择左中位数mid不加1.left=mid+1；
		int mid = left + (right - left+1) / 2;
		int icount = 0;
		for (auto it:nums)
		{
			if (it<mid)
			{
				++icount;
			}
		}
		if (icount>mid-1)
		{
			right = mid-1;
		}
		else
		{
			left = mid;
		}
	}
	return left;
}

