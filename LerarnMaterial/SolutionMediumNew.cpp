#include "SolutionMediumNew.h"


SolutionMediumNew::SolutionMediumNew()
{
}


SolutionMediumNew::~SolutionMediumNew()
{
}

int SolutionMediumNew::findRepeatNumber(vector<int>& nums)
{
	if (nums.size()<2)
	{
		return -1;
	}
	for (int i = 0; i < nums.size();++i)
	{
		while (i!=nums[i])
		{
			if (nums[i]==nums[nums[i]])
			{
				return nums[i];
			}
			swap(nums[i], nums[nums[i]]);
		}
	}
	return -1;
}

bool SolutionMediumNew::findNumberIn2DArray(vector<vector<int>>& matrix, int target)
{
	if (matrix.empty())
	{
		return false;
	}
	int row = 0, col = matrix[0].size() - 1;
	while (row<matrix.size()&&col>=0)
	{
		if (matrix[row][col]<target)
		{
			++row;
		}
		else if (matrix[row][col]>target)
		{
			--col;
		}
		else
		{
			return true;
		}
	}
	return false;
}

std::string SolutionMediumNew::replaceSpace(string s)
{
	if (s.empty())
	{
		return s;
	}
	int iCount = 0;
	for (auto &it:s)
	{
		if (it==' ')
		{
			++iCount;
		}
	}
	string res(s.size() + 2 * iCount,' ');
	int index = s.size() - 1, newIndex = res.size() - 1;
	while (index>=0&&newIndex>=0)
	{
		if (s[index]!=' ')
		{
			res[newIndex--] = s[index--];
		}
		else
		{
			res[newIndex--] = '0';
			res[newIndex--] = '2';
			res[newIndex--] = '%';
			--index;
		}
	}
	return res;
}

std::vector<int> SolutionMediumNew::reversePrint(ListNode* head)
{
	if (!head)
	{
		return{};
	}
	vector<int> res;
	while (head)
	{
		res.push_back(head->val);
		head = head->next;
	}
	reverse(res.begin(), res.end());
	return res;
}

TreeNode * HelpCreateTreeNode(vector<int>::iterator preBegin, vector<int>::iterator preEnd, vector<int>::iterator inoBegin, vector<int>::iterator inoEnd)
{
	if (preBegin==preEnd)
	{
		return nullptr;
	}
	TreeNode * root = new TreeNode(*preBegin);
	auto disNode = find(inoBegin, inoEnd, *preBegin);
	root->left = HelpCreateTreeNode(preBegin + 1, preBegin + (disNode - inoBegin) + 1, inoBegin, disNode);
	root->right = HelpCreateTreeNode(preBegin + (disNode - inoBegin) + 1, preEnd, disNode + 1, inoEnd);
	return root;
}

TreeNode* SolutionMediumNew::buildTree(vector<int>& preorder, vector<int>& inorder)
{
	return HelpCreateTreeNode(preorder.begin(), preorder.end(), inorder.begin(), inorder.end());
}

int SolutionMediumNew::fib(int n)
{
	if (n<=0)
	{
		return 0;
	}
	if (n<3)
	{
		return 1;
	}
	int pre = 1, cur = 1;
	for (int i = 2; i < n;++i)
	{
		int iTemp = cur;
		cur += pre;
		if (cur>1000000007)
		{
			cur %= 1000000007;
		}
		pre = iTemp;
	}
	return cur;
}

int SolutionMediumNew::minArray(vector<int>& numbers)
{
	if (numbers.empty())
	{
		return -1;
	}
	int leftCur = 0, rightCur = numbers.size()-1;
	if (numbers[0]<numbers.back())
	{
		return numbers[0];
	}
	while (leftCur<rightCur)
	{
		int mid = leftCur + (rightCur - leftCur) / 2;
		if (numbers[mid]<numbers[rightCur])
		{
			rightCur = mid;
		}
		else if (numbers[mid] == numbers[rightCur])
		{
			--rightCur;
		}
		else 
		{
			leftCur = mid + 1;
		}
	}
	return numbers[leftCur];
}
int direc[4][2] = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
bool DFSExist(vector<vector<char>>& board, const string &word, int indexX,int indexY,int ipos, vector<vector<bool>>&visit)
{
	if (ipos==word.size()-1)
	{
		return board[indexX][indexY] == word.back();
	}
	if (board[indexX][indexY]==word[ipos])
	{
		visit[indexX][indexY] = false;
		for (int i = 0; i < 4; ++i)
		{
			int newX = indexX + direc[i][0];
			int newY = indexY + direc[i][1];
			if (newX >= 0 && newX < board.size() && newY >= 0 && newY < board[0].size() && visit[newX][newY])
			{
				if (DFSExist(board, word, newX, newY, ipos+1, visit))
				{
					return true;
				}
			}
		}
		visit[indexX][indexY] = true;
	}
	return false;
}

bool SolutionMediumNew::exist(vector<vector<char>>& board, string word)
{
	if (board.empty()||word.empty())
	{
		return false;
	}
	vector<vector<bool>> visited(board.size(), vector<bool>(board[0].size(), true));
	for (int i = 0; i < board.size();++i)
	{
		for (int j = 0; j < board[0].size();++j)
		{
			if (visited[i][j] && DFSExist(board, word, i,j,0,visited))
			{
				return true;
			}
		}
	}
	return false;
}

int SolutionMediumNew::cuttingRope(int n)
{
	if (n<=0)
	{
		return 0;
	}
	if (n<3)
	{
		return 1;
	}
	if (n==3)
	{
		return 2;
	}
	if (n==4)
	{
		return 4;
	}
	vector<int> dp(n + 1);
	dp[0] = 0;
	dp[1] = 1;
	dp[2] = 2;
	dp[3] = 3;
	dp[4] = 4;
	for (int i = 5; i <= n;++i)
	{
		for (int j = 1; 2 * j <=i;++j)
		{
			dp[i] = max(dp[i], dp[j] * dp[i - j]);
		}
	}
	return dp[n];
}

double SolutionMediumNew::myPow(double x, int n)
{
	if (x==0)
	{
		return 0;
	}
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
	double halfNum = myPow(x, n / 2);
	double key = myPow(x, n % 2);
	return key*halfNum*halfNum;
}

bool SolutionMediumNew::isMatch(string s, string p)
{
	if (p.empty())
	{
		return s.empty();
	}
	bool bKey = !s.empty()&&((s[0] == p[0]) || p[0] == '.');
	if (p.size()>1&&p[1]=='*')
	{
		return isMatch(s, p.substr(2)) || (bKey&&isMatch(s.substr(1), p));
	}
	return bKey&&isMatch(s.substr(1), p.substr(1));
}

bool SolutionMediumNew::isNumber(string s)
{
	if (s.empty())
	{
		return false;
	}
	int ipos = 0;
	while (s[ipos]==' ')
	{
		++ipos;
	}
	if (ipos!=0)
	{
		return isNumber(s.substr(ipos));
	}
	while (s.back()==' ')
	{
		s.pop_back();
	}
	int pointNumber = 0;
	int number = 0;
	if (s[ipos]=='+'||s[ipos]=='-')
	{
		++ipos;
	}
	while (ipos<s.size()&&s[ipos]!='E'&&s[ipos]!='e')
	{
		if (s[ipos]=='.')
		{
			++pointNumber;
		}
		else if (s[ipos] >= '0'&&s[ipos] <= '9')
		{
			++number;
		}
		else
		{
			return false;
		}
		++ipos;
	}
	if (pointNumber>1||number==0)
	{
		return false;
	}
	if (ipos==s.size())
	{
		return true;
	}
	++ipos;
	if (ipos==s.size())
	{
		return false;
	}
	if (s[ipos]=='+'||s[ipos]=='-')
	{
		++ipos;
	}
	if (ipos==s.size())
	{
		return false;
	}
	while (ipos<s.size()&&s[ipos]>='0'&&s[ipos]<='9')
	{
		++ipos;
	}
	return ipos == s.size();
}

std::vector<int> SolutionMediumNew::exchange(vector<int>& nums)
{
	if (nums.size()<2)
	{
		return nums;
	}
	int odd = 0, even = nums.size() - 1;
	while (odd<even)
	{
		while (odd<even&&nums[odd]%2==1)
		{
			++odd;
		}
		while (odd<even&&nums[even]%2==0)
		{
			--even;
		}
		if (odd>=even)
		{
			break;
		}
		swap(nums[odd], nums[even]);
	}
	return nums;
}

std::vector<int> SolutionMediumNew::exchangeThree(vector<int>& nums)
{
	if (nums.size()<2)
	{
		return nums;
	}
	int leftCur = -1, rightCur = nums.size(), index = 0;
	while (index<rightCur)
	{
		if (nums[index]==0)
		{
			swap(nums[index++], nums[++leftCur]);
		}
		else if (nums[index]==2)
		{
			//因为右边的数字还不定
			swap(nums[index], nums[--rightCur]);
		}
		else
		{
			++index;
		}
	}
	return nums;
}

ListNode* SolutionMediumNew::getKthFromEnd(ListNode* head, int k)
{
	if (head==nullptr&&k<1)
	{
		return nullptr;
	}
	ListNode * fastNode = head;
	while (k&&fastNode)
	{
		fastNode = fastNode->next;
		--k;
	}
	if (k!=0&&fastNode==nullptr)
	{
		return nullptr;
	}
	while (fastNode&&head)
	{
		fastNode = fastNode->next;
		head = head->next;
	}
	return head;
}

ListNode* SolutionMediumNew::reverseList(ListNode* head)
{
	if (head==nullptr)
	{
		return head;
	}
	ListNode * pre = nullptr;
	ListNode * pNext = nullptr;
	while (head)
	{
		pNext = head->next;
		head->next = pre;
		pre = head;
		head = pNext;
	}
	return pre;
}

ListNode* SolutionMediumNew::mergeTwoLists(ListNode* l1, ListNode* l2)
{
	if (!l1)
	{
		return l2;
	}
	if (!l2)
	{
		return l1;
	}
	ListNode * newHead = new ListNode(0);
	ListNode * cur = newHead;
	while (l1&&l2)
	{
		if (l1->val>l2->val)
		{
			cur->next = l2;
			l2 = l2->next;
		}
		else
		{
			cur->next = l1;
			l1 = l1->next;
		}
		cur = cur->next;
	}
	cur->next = l1 ? l1 : l2;
	cur = newHead->next;
	delete newHead;
	return cur;
}

bool helpCheck(TreeNode * pRoot1, TreeNode * pRoot2)
{
	if (pRoot2==nullptr)
	{
		return true;
	}
	if (pRoot1==nullptr)
	{
		return false;
	}
	if (pRoot1->val!=pRoot2->val)
	{
		return false;
	}
	return helpCheck(pRoot1->left, pRoot2->left) && helpCheck(pRoot1->right, pRoot2->right);
}

bool SolutionMediumNew::CheckSameTree(TreeNode* pRoot1, TreeNode* pRoot2)
{
	if (nullptr==pRoot2)
	{
		return false;
	}
	if (nullptr==pRoot1)
	{
		return false;
	}
	return helpCheck(pRoot1, pRoot2) || CheckSameTree(pRoot1->left, pRoot2) || CheckSameTree(pRoot1->right, pRoot2);
}

TreeNode* SolutionMediumNew::mirrorTree(TreeNode* root)
{
	if (nullptr==root)
	{
		return root;
	}
// 	TreeNode * newRoot = new TreeNode(root->val);
// 	newRoot->left = mirrorTree(root->right);
// 	newRoot->right = mirrorTree(root->left);
// 	return newRoot;
	stack<TreeNode*> stacHelp;
	stacHelp.push(root);
	while (!stacHelp.empty())
	{
		TreeNode * nodeTemp = stacHelp.top();
		stacHelp.pop();
		if (nullptr==nodeTemp)
		{
			continue;
		}
		swap(nodeTemp->left, nodeTemp->right);
		stacHelp.push(nodeTemp->left);
		stacHelp.push(nodeTemp->right);
	}
	return root;
}

bool helpSymetric(TreeNode* root1, TreeNode * root2)
{
	if (nullptr==root1&&nullptr==root2)
	{
		return true;
	}
	if (nullptr==root1||nullptr==root2)
	{
		return false;
	}
	if (root1->val!=root2->val)
	{
		return false;
	}
	return helpSymetric(root1->left, root2->right) && helpSymetric(root1->right, root2->left);
}

bool SolutionMediumNew::isSymmetric(TreeNode* root)
{
	return helpSymetric(root, root);
}

bool SolutionMediumNew::validateStackSequences(vector<int>& pushed, vector<int>& popped)
{
	if (pushed.size()!=popped.size())
	{
		return false;
	}
	if (pushed.empty()&&popped.empty())
	{
		return true;
	}
	int newIndex = 0;
	stack<int> helpStac;
	for (auto &it:pushed)
	{
		helpStac.push(it);
		while (!helpStac.empty()&&newIndex<popped.size()&&popped[newIndex]==helpStac.top())
		{
			helpStac.pop();
			++newIndex;
		}
	}
	return helpStac.empty();
}

bool HelpVerify(vector<int>&sequence, int start, int end)
{
	if (start>=end)
	{
		return true;
	}
	int countLeft = 0;
	for (int i = start; i < end; ++i)
	{
		if (sequence[i]>sequence[end])
		{
			break;
		}
		++countLeft;
	}
	for (int i = start + countLeft; i < end;++i)
	{
		if (sequence[i]<sequence[end])
		{
			return false;
		}
	}
	return HelpVerify(sequence, start, start + countLeft - 1) && HelpVerify(sequence, start + countLeft,end-1);
}

bool SolutionMediumNew::verifyPostorder(vector<int>& sequence)
{
	return HelpVerify(sequence, 0, sequence.size() - 1);
}

void helpSum(vector<vector<int>>&res, vector<int>&path, const int target, int add,TreeNode * root)
{
	if (nullptr==root)
	{
		return;
	}
	path.push_back(root->val);
	add += root->val;
	if (nullptr==root->left&&nullptr==root->right)
	{
		if (add==target)
		{
			res.push_back(path);
		}
		path.pop_back();
		return;
	}
	if (root->left)
	{
		helpSum(res, path, target, add, root->left);
	}
	if (root->right)
	{
		helpSum(res, path, target, add, root->right);
	}
	path.pop_back();
}

std::vector<std::vector<int>> SolutionMediumNew::pathSum(TreeNode* root, int sum)
{
	if (nullptr==root)
	{
		return{};
	}
	vector<vector<int>> res;
	vector<int> path;
	helpSum(res, path, sum, 0, root);
	return res;
}

Node* SolutionMediumNew::copyRandomList(Node* head)
{
// 	if (nullptr == head)
// 		return head;
// 	unordered_map<Node*,Node*> hashM;
// 	Node * temp = head;
// 	while (temp)
// 	{
// 		hashM[temp] = new Node(temp->val);
// 		temp = temp->next;
// 	}
// 	temp = head;
// 	while (temp)
// 	{
// 		hashM[temp]->next = hashM[temp->next];
// 		hashM[temp]->random = hashM[temp->random];
// 		temp = temp->next;
// 	}
// 	return hashM[head];
	if (nullptr==head)
	{
		return head;
	}
	Node * pTemp = head;
	//复制链表
	while (pTemp)
	{
		Node * pClone = new Node(pTemp->val);
		pClone->next = pTemp->next;
		pTemp->next = pClone;
		pTemp = pClone->next;
	}
	//复制随机指针
	pTemp = head;
	while (pTemp)
	{
		if (pTemp->random)
		{
			pTemp->next->random = pTemp->random->next;
		}
		pTemp = pTemp->next->next;
	}
	//拆开
	Node * pLeftNode = head;
	Node * pRes = head->next;
	Node *pRightNode = pRes;
	while (pLeftNode&&pRightNode)
	{
		pLeftNode->next = pLeftNode->next->next;
		if (pRightNode->next!=nullptr)
		{
			pRightNode->next = pRightNode->next->next;
		}
		pLeftNode = pLeftNode->next;
		pRightNode = pRightNode->next;
	}
	return pRes;
}

void HelpPer(vector<string>&res, string&path, string s,int start)
{
	if (path.size()==s.size())
	{
		res.push_back(path);
		return;
	}
	for (int i = start; i < s.size();++i)
	{
		if (i!=start&&s[i]==s[start])
		{
			continue;
		}
		swap(s[start], s[i]);
		path.push_back(s[start]);
		HelpPer(res, path, s, start + 1);
		path.pop_back();
	}
}

std::vector<std::string> SolutionMediumNew::permutation(string s)
{
	if (s.empty())
	{
		return{};
	}
	sort(s.begin(), s.end());
	string path;
	vector<string> res;
	HelpPer(res, path, s, 0);
	return res;
}

int SolutionMediumNew::majorityElement(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	int count = 1;
	int res = nums[0];
	for (int i = 1; i < nums.size();++i)
	{
		if (res==nums[i])
		{
			++count;
		}
		else
		{
			--count;
			if (count==0)
			{
				res = nums[i];
				++count;
			}
		}
	}
	return res;
}

int partion(vector<int>&arr, int start, int end)
{
	if (start>=end)
	{
		return start;
	}
	int leftCur = start;
	int rightCur = end + 1;
	int prio = arr[start];
	while (leftCur<rightCur)
	{
		do 
		{
			++leftCur;
		} while (leftCur<rightCur&&arr[leftCur]<prio);
		do 
		{
			--rightCur;
		} while (arr[rightCur]>prio);
		if (leftCur>=rightCur)
		{
			break;
		}
		swap(arr[leftCur], arr[rightCur]);
	}
	swap(arr[start], arr[rightCur]);
	return rightCur;
}

std::vector<int> SolutionMediumNew::getLeastNumbers(vector<int>& arr, int k)
{
	if (arr.empty()||k<1||k>arr.size())
	{
		return{};
	}
	int index = 0;
	int start = 0;
	int end = arr.size()-1;
	while (index!=k-1)
	{
		index = partion(arr, start, end);
		if (index>k-1)
		{
			end = index-1;
		}
		else if (index<k-1)
		{
			start = index+1;
		}
	}
	vector<int> res(arr.begin(),arr.begin()+k);
	return res;
}

int SolutionMediumNew::maxSubArray(vector<int>& nums)
{
	if (nums.empty())
	{
		return -1;
	}
	int pre = 0;
	int res = INT_MIN;
	for (auto &it:nums)
	{
		pre = max(pre + it, it);
		res = max(res, pre);
	}
	return res;
}

int SolutionMediumNew::countDigitOne(int n)
{
	if (n<1)
	{
		return 0;
	}
	int rightNumber = 0;
	int count = 0;
	int res = 0;
	while (n)
	{
		int ikey = n % 10;
		n /= 10;
		if (ikey==0)
		{
			res += n*pow(10, count);
		}
		else if (ikey == 1)
		{
			res += n*pow(10, count) + rightNumber + 1;
		}
		else
		{
			res += (n + 1)*pow(10, count);
		}
		rightNumber += ikey*pow(10, count);
		++count;
	}
	return res;
}

int SolutionMediumNew::findNthDigit(int n)
{
	if (n<0)
	{
		return -1;
	}
	if (n<10)
	{
		return n;
	}
	int count = 2;
	n -= 10;
	while (n>9*pow(10,count-1)*count)
	{
		n -= 9 * pow(10, count - 1)*count;
		++count;
	}
	string strTemp = to_string(n / count + pow(10, count - 1));
	return strTemp[n%count] - '0';
}

std::string SolutionMediumNew::minNumber(vector<int>& numbers)
{
	if (numbers.empty())
	{
		return "";
	}
	sort(numbers.begin(), numbers.end(), [](const int &lhs,const int& rhs)
	{
		return to_string(lhs) + to_string(rhs) < to_string(rhs) + to_string(lhs);
	});
	string res;
	for (auto &it:numbers)
	{
		res += to_string(it);
	}
	return res;
}

int SolutionMediumNew::translateNum(int num)
{
	if (num<0)
	{
		return 0;
	}
	string strTemp = to_string(num);
	vector<int> dp(strTemp.size());
	dp[0] = 1;
	for (int i = 1; i < strTemp.size();++i)
	{
		dp[i] += dp[i - 1];
		if (strTemp[i - 1] != '0')
		{
			if ((strTemp[i-1]-'0')*10+strTemp[i]-'0'<=25)
			{
				if (i>1)
				{
					dp[i] += dp[i - 2];
				}
				else
				{
					++dp[i];
				}
			}
		}
	}
	return dp.back();
}

int SolutionMediumNew::maxValue(vector<vector<int>>& grid)
{
	if (grid.empty())
	{
		return 0;
	}
	vector<vector<int>> dp(grid.size(), vector<int>(grid[0].size()));
	dp[0][0] = grid[0][0];
	for (int i = 1; i < grid.size();++i)
	{
		dp[i][0] += dp[i - 1][0] + grid[i][0];
	}
	for (int i = 1; i < grid[0].size();++i)
	{
		dp[0][i] += dp[0][i-1] + grid[0][i];
	}
	for (int i = 1; i < grid.size();++i)
	{
		for (int j = 1; j < grid[0].size(); ++j)
		{
			dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
		}
	}
	return dp[grid.size()-1][grid[0].size()-1];
}

int SolutionMediumNew::lengthOfLongestSubstring(string s)
{
	if (s.empty())
	{
		return 0;
	}
	unordered_set<int> hashSet;
	int left = 0, right = 0;
	int res = 1;
	while (right < s.size())
	{
		while (right < s.size() && hashSet.count(s[right]) == 0)
		{
			hashSet.insert(s[right++]);
		}
		res = max(res, right - left);
		while (left < right&&s[left] != s[right])
		{
			hashSet.erase(s[left++]);
		}
		++left;
		++right;
	}
	return res;
}

int SolutionMediumNew::nthUglyNumber(int n)
{
	if (n<=1)
	{
		return 1;
	}
	vector<int> dp(n,1);
	vector<int> ugly(3);
	for (int i = 1; i < n;++i)
	{
		int a = dp[ugly[0]] * 2, b=dp[ugly[1]] * 3, c=dp[ugly[2]] * 5;
		int minNum = min(a, min(b, c));
		if (minNum==a)
		{
			++ugly[0];
		}
		if (minNum==b)
		{
			++ugly[1];
		}
		if (minNum==c)
		{
			++ugly[2];
		}
		dp[i] = minNum;
	}
	return dp.back();
}

char SolutionMediumNew::firstUniqChar(string str)
{
	if (str.empty())
	{
		return ' ';
	}
	unordered_map<int, int> hashM;
	for (auto &it:str)
	{
		++hashM[it];
	}
	int ipos = -1;
	for (int i = 0; i < str.size();++i)
	{
		if (hashM[str[i]]==1)
		{
			ipos = i;
			break;
		}
	}
	return ipos == -1 ? ' ' : str[ipos];
}

int mergeCount(vector<int>&nums, vector<int>&helpNums, int start, int end)
{
	if (start==end)
	{
		helpNums[start] = nums[end];
		return 0;
	}
	const int length = (end - start) / 2;
	int leftCount = mergeCount(nums, helpNums, start,start + length);
	int rightCount = mergeCount(nums, helpNums, start + length + 1, end);
	int count = 0;
	int leftCur = start + length, rightCur = end;
	int index = end;
	while (leftCur >= start&&rightCur >= start + length + 1)
	{
		if (nums[leftCur] > nums[rightCur])
		{
			count += rightCur - start - length;
			helpNums[index--] = nums[leftCur--];
		}
		else
		{
			helpNums[index--] = nums[rightCur--];
		}
	}
	while (leftCur >= start)
	{
		helpNums[index--] = nums[leftCur--];
	}
	while (rightCur >= start + length + 1)
	{
		helpNums[index--] = nums[rightCur--];
	}
	for (int i = start; i <= end;++i)
	{
		nums[i] = helpNums[i];
	}
	return count + leftCount+rightCount;
}

int SolutionMediumNew::reversePairs(vector<int>& nums)
{
	if (nums.size()<2)
	{
		return 0;
	}
	vector<int> helpNums(nums.size());
	return mergeCount(nums, helpNums, 0, nums.size() - 1);
}

ListNode * SolutionMediumNew::getIntersectionNode(ListNode *pHead1, ListNode *pHead2)
{
	if (nullptr==pHead1||nullptr==pHead2)
	{
		return nullptr;
	}
	ListNode * nodeTempA = pHead1;
	ListNode * nodeTempB = pHead2;
	int countA = 0, countB = 0;
	while (nodeTempA)
	{
		++countA;
		nodeTempA = nodeTempA->next;
	}
	while (nodeTempB)
	{
		++countB;
		nodeTempB = nodeTempB->next;
	}
	nodeTempA = pHead1;
	nodeTempB = pHead2;
	int num = countA - countB;
	if (num>0)
	{
		while (num--)
		{
			nodeTempA = nodeTempA->next;
		}
	}
	else
	{
		while (num++)
		{
			nodeTempB = nodeTempB->next;
		}
	}
	while (nodeTempA&&nodeTempB&&nodeTempA!=nodeTempB)
	{
		nodeTempA = nodeTempA->next;
		nodeTempB = nodeTempB->next;
	}
	return nodeTempA == nodeTempB ? nodeTempA : nullptr;
}

int SolutionMediumNew::search(vector<int>& data, int k)
{
	if (data.empty()||k<data.front()||k>data.back())
	{
		return 0;
	}
	int leftCur = 0, rightCur = data.size();
	while (leftCur<rightCur)
	{
		int mid = leftCur + (rightCur - leftCur) / 2;
		if (data[mid]>=k)
		{
			rightCur = mid;
		}
		else
		{
			leftCur = mid + 1;
		}
	}
	int ipos = leftCur;
	if (ipos==data.size())
	{
		return 0;
	}
	leftCur = 0, rightCur = data.size();
	while (leftCur<rightCur)
	{
		int mid = leftCur + (rightCur - leftCur) / 2;
		if (data[mid] <= k)
		{
			leftCur = mid + 1;
		}
		else
		{
			rightCur = mid;
		}
	}
	return leftCur - ipos;
}

int SolutionMediumNew::missingNumber(vector<int>& nums)
{
	if (nums.empty())
	{
		return 0;
	}
	int left = 0, right = nums.size();
	while (left<right)
	{
		int mid = left + (right - left) / 2;
		if (nums[mid]>mid)
		{
			right = mid;
		}
		else
		{
			left = mid + 1;
		}
	}
	return left;
}

void HelpKthLa(TreeNode* root, int &res, int &k)
{
	if (nullptr==root)
	{
		return;
	}
	HelpKthLa(root->right, res, k);
	--k;
	if (k==0)
	{
		res = root->val;
		return;
	}
	HelpKthLa(root->left, res, k);
}

int SolutionMediumNew::kthLargest(TreeNode* root, int k)
{
	if (nullptr==root||k<1)
	{
		return 0;
	}
	int res = 0;
	HelpKthLa(root, res, k);
	return res;
}
bool helpBal(TreeNode * root,int &depth)
{
	if (root==nullptr)
	{
		return true;
	}
	int leftDepth = 0, rightDepth = 0;
	if (!helpBal(root->left,leftDepth)||!helpBal(root->right,rightDepth))
	{
		return false;
	}
	if (abs(leftDepth-rightDepth)>1)
	{
		return false;
	}
	depth = max(leftDepth, rightDepth)+1;
	return true;
}

bool SolutionMediumNew::isBalanced(TreeNode* root)
{
	if (nullptr==root)
	{
		return true;
	}
	int depth = 0;
	return helpBal(root, depth);
}

std::vector<int> SolutionMediumNew::singleNumbers(vector<int>& data)
{
	if (data.empty())
	{
		return{};
	}
	int res = 0;
	for (auto&it:data)
	{
		res ^= it;
	}
	int key = 1;
	while ((res&key)==0)
	{
		key <<= 1;
	}
	int leftNumber = 0, rightNumber = 0;
	for (auto &it:data)
	{
		if ((it&key)==0)
		{
			leftNumber ^= it;
		}
		else
		{
			rightNumber ^= it;
		}
	}
	return{ leftNumber, rightNumber };
}

int SolutionMediumNew::singleNumber(vector<int>& nums)
{
	if (nums.empty())
	{
		return 0;
	}
	int res = 0;
	int i = 0;
	while (i<32)
	{
		int count = 0;
		int key = 1 << i;
		for (auto &it:nums)
		{
			if (it&key)
			{
				++count;
			}
		}
		if (count%3)
		{
			res |= key;
		}
		++i;
	}
	return res;
}

std::vector<int> SolutionMediumNew::twoSum(vector<int>& nums, int target)
{
	if (nums.size()<2)
	{
		return{};
	}
	int left = 0, right = nums.size()-1;
	while (left<right)
	{
		int addSum = nums[left] + nums[right];
		if (addSum>target)
		{
			--right;
		}
		else if (addSum<target)
		{
			++left;
		}
		else
		{
			return{ nums[left], nums[right] };
		}
	}
	return{};
}

std::vector<std::vector<int>> SolutionMediumNew::findContinuousSequence(int sum)
{
	if (sum<3)
	{
		return{};
	}
	vector<vector<int>> res;
	int left = 1, right = 2;
	int addSum = 3;
	while (right<=(sum+1)/2)
	{
		if (addSum==sum)
		{
			vector<int> path;
			for (int i = left; i <= right;++i)
			{
				path.push_back(i);
			}
			res.push_back(path);
		}
		else if (addSum>sum)
		{
			while (addSum>sum)
			{
				addSum -= left++;
			}
			if (addSum==sum)
			{
				vector<int> path;
				for (int i = left; i <= right; ++i)
				{
					path.push_back(i);
				}
				res.push_back(path);
			}
		}
		++right;
		addSum += right;
	}
	return res;
}

std::string SolutionMediumNew::reverseWords(string s)
{
	if (s.empty())
	{
		return s;
	}
	reverse(s.begin(), s.end());
	s.push_back(' ');
	int ipos = 0;
	string res;
	while (ipos<s.size())
	{
		if (s[ipos]==' ')
		{
			++ipos;
			continue;
		}
		int newIndex = ipos;
		while (s[newIndex]!=' ')
		{
			++newIndex;
		}
		reverse(s.begin() + ipos, s.begin() + newIndex);
		res += s.substr(ipos, newIndex - ipos+1);
		ipos = newIndex + 1;
	}
	res.pop_back();
	return res;
}

std::string SolutionMediumNew::reverseLeftWords(string str, int n)
{
	if (str.empty()||n<=0)
	{
		return str;
	}
	const int length = str.size();
	n %= length;
	reverse(str.begin(), str.end());
	reverse(str.begin(), str.begin() + length - n);
	reverse(str.begin() + length - n, str.end());
	return str;
}

std::vector<int> SolutionMediumNew::maxSlidingWindow(vector<int>& nums, int k)
{
	if (nums.empty()||nums.size()<k)
	{
		return{};
	}
	deque<int> deHelp;
	for (int i = 0; i < k;++i)
	{
		while (!deHelp.empty()&&nums[i]>nums[deHelp.back()])
		{
			deHelp.pop_back();
		}
		deHelp.push_back(i);
	}
	vector<int> res;
	for (int i = k; i < nums.size();++i)
	{
		res.push_back(nums[deHelp.front()]);
		while (!deHelp.empty() && nums[i]>nums[deHelp.back()])
		{
			deHelp.pop_back();
		}
		if (!deHelp.empty()&&i-deHelp.front()==k)
		{
			deHelp.pop_front();
		}
		deHelp.push_back(i);
	}
	res.push_back(nums[deHelp.front()]);
	return res;
}

std::vector<double> SolutionMediumNew::twoSumNew(int n)
{
	if (n<1)
	{
		return{};
	}
	vector<vector<int>> dp(n + 1, vector<int>(6 * n+1));
	for (int i = 1; i <= 6; ++i)
	{
		dp[1][i] = 1;
	}
	for (int i = 2; i <= n;++i)
	{
		for (int j = i; j <= 6 * i;++j)
		{
			for (int k = 1; k <= 6&&k<j;++k)
			{
				dp[i][j] += dp[i - 1][j - k];
			}
		}
	}
	const int number = pow(6, n);
	vector<double> res;
	for (int i = n; i <= 6 * n;++i)
	{
		res.push_back(dp.back()[i] * 1.0 / number);
	}
	return res;
}

bool SolutionMediumNew::isStraight(vector<int>& nums)
{
	if (nums.empty())
	{
		return false;
	}
	sort(nums.begin(), nums.end());
	int countZero = 0;
	for (auto&it:nums)
	{
		if (it!=0)
		{
			break;
		}
		++countZero;
	}
	int iNum = 0;
	for (int i = countZero+1; i < nums.size();++i)
	{
		if (nums[i]==nums[i-1])
		{
			return false;
		}
		iNum += nums[i] - nums[i - 1] -1;
	}
	return countZero >= iNum;
}

int SolutionMediumNew::lastRemaining(int n, int m)
{
	if (n<1||m<1)
	{
		return - 1;
	}
	int last = 0;
	//f(N,M)=(f(N−1,M)+M)%N
	for (int i = 2; i <= n;++i)
	{
		last = (last + m) % i;
	}
	return last;
}

int SolutionMediumNew::maxProfit(vector<int>& prices)
{
	if (prices.size()<2)
	{
		return 0;
	}
	int res = 0, minPri = prices[0];
	for (auto &it:prices)
	{
		res = max(res, it - minPri);
		minPri = min(it, minPri);
	}
	return res;
}

int SolutionMediumNew::add(int a, int b)
{
	int cur = 0;
	int res = 0;
	do 
	{
		res = a^b;
		cur = (unsigned int)(a&b)<<1;
		a = res;
		b = cur;
	} while (cur);
	return res;
}

std::vector<int> SolutionMediumNew::constructArr(vector<int>& a)
{
	if (a.size()<2)
	{
		return a;
	}
	int left = 1,right=1;
	vector<int> res(a.size(),1);
  	for (int i = 0; i < a.size();++i)
	{
		res[i] *= left;
		left *= a[i];
		res[a.size() - i - 1] *= right;
		right *= a[a.size() - i - 1];
	}
	return res;
}

TreeNode* SolutionMediumNew::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
	if (root==nullptr||p==nullptr||q==nullptr||p->val==q->val)
	{
		return nullptr;
	}
	if (root->val>p->val&&root->val>q->val)
	{
		return lowestCommonAncestor(root->left, p, q);
	}
	if (root->val<p->val&&root->val<q->val)
	{
		return lowestCommonAncestor(root->right, p, q);
	}
	return root;
}

TreeNode* SolutionMediumNew::lowestCommonAncestorNew(TreeNode* root, TreeNode* p, TreeNode* q)
{
	if (nullptr==root||root==p||root==q)
	{
		return root;
	}
	TreeNode * leftNode = lowestCommonAncestorNew(root->left, p, q);
	TreeNode * rightNode = lowestCommonAncestorNew(root->right, p, q);
	if (leftNode&&rightNode)
	{
		return root;
	}
	return leftNode ? leftNode : rightNode;
}

int SolutionMediumNew::trap(vector<int>& height)
{
	if (height.empty())
	{
		return 0;
	}
	const int len = height.size();
	vector<int> leftHeight(len + 1), rightHeight(len + 1);
	for (int i = 1; i <=len;++i)
	{
		leftHeight[i] = max(leftHeight[i - 1], height[i - 1]);
	}
	for (int i = len - 2; i >= 0;--i)
	{
		rightHeight[i] = max(rightHeight[i + 1], height[i + 1]);
	}
	int water = 0;
	for (int i = 0; i < len;++i)
	{
		int level = min(leftHeight[i], rightHeight[i]);
		water += max(0, level - height[i]);
	}
	return water;
}

int SolutionMediumNew::GetMaxGcd(const int a, const int b)
{
	if (a<1||b<1)
	{
		return 0;
	}
	return b == 0 ? a : GetMaxGcd(b, a%b);
}

int SolutionMediumNew::GetMinLcm(const int a, const int b)
{
	if (a<1||b<1)
	{
		return 0;
	}
	return a*b / GetMaxGcd(a, b);
}

void DFSNode(const string &beginWord, string &cur,vector<string> path,unordered_map<string, vector<string>> &hashNeighbor, vector<vector<string>> &res)
{
	if (cur==beginWord)
	{
		path.push_back(cur);
		reverse(path.begin(), path.end());
		res.push_back(path);
		return;
	}
	path.push_back(cur);
	for (auto &it:hashNeighbor[cur])
	{
		DFSNode(beginWord, it, path, hashNeighbor, res);
	}
}


std::vector<std::vector<std::string>> SolutionMediumNew::findLadders(string beginWord, string endWord, vector<string>& wordList)
{
	//两个关键的数据结构，每个节点的邻接节点（每个节点的下一层节点，只相差一个字符的）
	//存储每个节点所在的深度
	//使用BFS建图
	vector<vector<string>> resVec;
	vector<string> path;
	if (std::find(wordList.begin(),wordList.end(),endWord)==wordList.end())
	{
		return resVec;
	}
	unordered_map<string, int> hashMHigh;  //纪录每个节点的层数
	unordered_map<string, vector<string>> hashNeighbor;         //纪录每个节点的相邻节点
	unordered_set<string> hashSet(wordList.begin(), wordList.end());
	hashMHigh[beginWord] = 1;
	queue<string> queHelp;
	queHelp.push(beginWord);
	while (!queHelp.empty())
	{
		string qNode = queHelp.front();
		queHelp.pop();
		for (int i = 0; i < qNode.size();++i)
		{
			for (char ch = 'a'; ch <= 'z'; ++ch)
			{
				string strTemp = qNode;
				strTemp[i] = ch;
				if (hashSet.count(strTemp))
				{
					if (hashMHigh.count(strTemp)==0)
					{
						//高度set还没有
						queHelp.push(strTemp);
						hashMHigh[strTemp] = hashMHigh[qNode] + 1;
						hashNeighbor[strTemp].push_back(qNode);
					}
					else if (hashMHigh[strTemp]==hashMHigh[qNode]+1)
					{
						//将相邻的节点也放进去
						hashNeighbor[strTemp].push_back(qNode);
					}
				}
			}
		}
	}
	DFSNode(beginWord, endWord, path, hashNeighbor, resVec);
	return resVec;
}

bool SolutionMediumNew::equationsPossible(vector<string>& equations)
{
	if (equations.empty())
	{
		return true;
	}
	UnionFind uniF;
	uniF.Initialize(26);
	for (auto &it : equations)
	{
		if (it[1] == '=')
		{
			uniF.Union(it[0] - 'a', it[3] - 'a');
		}
	}
	for (auto &it : equations)
	{
		if (it[1] == '!')
		{
			if (uniF.find(it[0] - 'a') == uniF.find(it[3] - 'a'))
				return false;
		}
	}
	return true;
}

int GetSum(const vector<int> &arr, const vector<int> &preSum, const int &value)
{
	auto iter = lower_bound(arr.begin(), arr.end(), value);
	return preSum[iter - arr.begin()] + (arr.end() - iter)*value;
}

int SolutionMediumNew::findBestValue(vector<int>& arr, int target)
{
	if (arr.empty())
		return 0;
	sort(arr.begin(), arr.end());
	const int n = arr.size();
	vector<int> preSum(n + 1);
	for (int i = 1; i <= arr.size(); ++i)
		preSum[i] = arr[i - 1] + preSum[i - 1];
	int left = 1, right = arr.back();
	while (left < right)
	{
		int mid = left + (right - left) / 2;
		int sum = GetSum(arr, preSum, mid);
		if (sum<target)
		{
			left = mid + 1;
		}
		else if (sum>target)
		{
			right = mid;
		}
		else
		{
			return mid;
		}
	}
	return abs(GetSum(arr, preSum, left) - target) < abs(GetSum(arr, preSum, left - 1) - target) ? left : left - 1;
}

//处理一个数字，并将begin移动到下一个 '-' 或者end
int next_int(string::iterator &begin, string::iterator &end) {
	int res = 0;
	while (begin < end && *begin != '-') res = res * 10 + (*(begin++) - '0');
	return res;
}
//返回当前 '-' 个数，并将begin移动到下一个数字或者end
int cnt_line(string::iterator &begin, string::iterator &end) {
	auto last = begin;
	while (begin < end && *begin == '-') begin++;
	return begin - last;
}

TreeNode* SolutionMediumNew::recoverFromPreorder(string S)
{
	auto begin = S.begin(), end = S.end();
	TreeNode* node = new TreeNode(next_int(begin, end));
	stack<TreeNode*> stk({ node });                       //栈初始化加入头节点
	while (begin < end)
	{
		int depth = cnt_line(begin, end);                 //得到深度
		node = new TreeNode(next_int(begin, end));        //新节点创建
		while (stk.size() > depth) stk.pop();             //大于等于新节点深度的出栈
		if (stk.top()->left) stk.top()->right = node;     //左节点已经占用了
		else stk.top()->left = node;
		stk.push(node);                                   //新节点入栈
	}
	while (stk.size() > 1) stk.pop();                    //得到头节点
	return stk.top();
}

int col[9][10] = { 0 };
int row[9][10] = { 0 };
int squ[9][10] = { 0 };
vector<pair<int, int>> blank;

bool backtrack(int n, vector<vector<char>>& board) 
{
	if (n >= blank.size()) return true;
	int r = blank[n].first;
	int c = blank[n].second;
	for (int i = 1; i <= 9; i++)
	{
		if (col[c][i] == 0 && row[r][i] == 0 && squ[r / 3 * 3 + c / 3][i] == 0)
		{
			board[r][c] = '0' + i;
			col[c][i] = 1;
			row[r][i] = 1;
			squ[r / 3 * 3 + c / 3][i] = 1;
			if (backtrack(n + 1, board)) return true;
			else 
			{
				board[r][c] = '.';
				col[c][i] = 0;
				row[r][i] = 0;
				squ[r / 3 * 3 + c / 3][i] = 0;
			}
		}
	}
	return false;
}

void SolutionMediumNew::solveSudoku(vector<vector<char>>& board)
{
	if (board.empty())
	{
		return;
	}
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			if (board[i][j] == '.')
			{
				blank.push_back(pair<int, int>(i, j));
				continue;
			}
			int val = board[i][j] - '0';
			col[j][val] = 1;
			row[i][val] = 1;
			squ[i / 3 * 3 + j / 3][val] = 1;
		}
	}
	backtrack(0, board);
}
