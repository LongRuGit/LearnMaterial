#include "AutumnMove.h"

int AutumnMove::maxProfit1(vector<int>& prices)
{
    if (prices.empty())
    {
        return 0;
    }
    int ret = 0;
    int preMin = INT_MAX;
    for (auto &it:prices)
    {
        ret = max(ret, it - preMin);
        preMin = min(preMin, it);
    }
    return ret;
}

int AutumnMove::maxProfit2(vector<int>& prices)
{
    if (prices.empty())
    {
        return 0;
    }
    vector<vector<int>> dp(prices.size() + 1, vector<int>(2));
    dp[0][0] = 0;
    dp[0][1] = INT_MIN;
    for (int i=1;i<=prices.size();++i)
    {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i-1]);
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1]);
    }
    return dp.back()[0];
}

int AutumnMove::maxProfit3(vector<int>& prices)
{
    if (prices.empty())
    {
        return 0;
    }
    vector<vector<vector<int>>> dp(prices.size(), vector<vector<int>>(3, vector<int>(2)));
    dp[0][1][0] = 0;
    dp[0][1][1] = -prices[0];
    dp[0][2][0] = 0;
    dp[0][2][1] = -prices[0];
    for (int i = 1; i < prices.size(); ++i)
    {
        dp[i][1][0] = max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
        dp[i][1][1] = max(dp[i - 1][1][1], dp[i - 1][0][0] - prices[i]);
        dp[i][2][0] = max(dp[i - 1][2][0], dp[i - 1][2][1] + prices[i]);
        dp[i][2][1] = max(dp[i - 1][2][1], dp[i - 1][1][0] - prices[i]);
    }
    return dp[prices.size() - 1][2][0];
}

int AutumnMove::maxProfit4(int k, vector<int>& prices)
{
    if (k<=0)
    {
        return 0;
    }
    if (k>=prices.size()/2)
    {
        vector<vector<int>> dp(prices.size() + 1, vector<int>(2));
        dp[0][0] = 0;
        dp[0][1] = INT_MIN;
        for (int i = 1; i <= prices.size(); ++i)
        {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1]);
        }
        return dp.back()[0];
    }
    vector<vector<vector<int>>> dp(prices.size() + 1, vector<vector<int>>(k + 1, vector<int>(2)));
    for (int i=0;i<=k;++i)
    {
        dp[0][i][1] = INT_MIN;
    }
    for (int i=1;i<=prices.size();++i)
    {
        for (int j=1;j<=k;++j)
        {
            dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i-1]);
            dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j-1][0] - prices[i-1]);
        }
    }
    return dp[prices.size()][k][0];
}

int AutumnMove::maxProfit5(vector<int>& prices)
{
    if (prices.size()<2)
    {
        return 0;
    }
    vector<vector<int>> dp(prices.size() + 1, vector<int>(2));
    dp[0][1] = INT_MIN;
    for (int i=1;i<=prices.size();++i)
    {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
        if (i>=2)
        {
            dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i - 1]);
        }
        else
        {
            dp[i][1] = max(dp[i - 1][1], -prices[i - 1]);
        }
    }
    return dp[prices.size()][0];
}

int AutumnMove::maxProfit6(vector<int>& prices, int fee)
{
    if (prices.empty())
    {
        return 0;
    }
    vector<vector<int>> dp(prices.size() + 1, vector<int>(2));
    dp[0][1] = INT_MIN;
    for (int i=1;i<=prices.size();++i)
    {
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i - 1]);
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i - 1] - fee);
    }
    return dp[prices.size()][0];
}

std::vector<int> AutumnMove::smallestRange(vector<vector<int>>& nums)
{
    multimap<int, int> numstomap;//元素值-所属组序号，默认按值升序
    for (int i = 0; i < nums.size(); i++)
    {
        for (int j = 0; j < nums[i].size(); j++)
        {
            numstomap.insert(pair<int, int>(nums[i][j], i));
        }
    }
    multimap<int, int>::iterator left = numstomap.begin();//左指针，左右指针初始都位于左侧
    multimap<int, int>::iterator right = numstomap.begin();//右指针
    int res = INT_MAX;//设置一个绝对最大初始值
    int leftv = 0;//返回结果的两个元素值
    int rightv = 0;//
    int k = nums.size();//组个数
    unordered_map<int, int> curmap;//组序号-个数
    while (right != numstomap.end())//终止条件
    {
        curmap[right->second] ++;//窗口扩张纳入的新元素
        while (curmap.size() == k)//已经找到了一个可行解、优化它
        {
            if (right->first - left->first < res)//记录可行解
            {
                res = right->first - left->first;
                leftv = left->first;
                rightv = right->first;
            }
            curmap[left->second] --;//收缩窗口
            if (curmap[left->second] == 0)
            {
                curmap.erase(left->second);
            }
            left++;
        }
        right++;
    }
    if (res == INT_MAX) 
        return {};
    else 
        return { leftv,rightv };
}

std::string AutumnMove::addStrings(string num1, string num2)
{
    if (num1.empty())
    {
        return num2;
    }
    if (num2.empty())
    {
        return num1;
    }
    string ret(max(num1.size(), num2.size()) + 1,0);
    int left = num1.size()-1, right = num2.size()-1;
    int index = 0,add=0,sum=0;
    while (left>=0&&right>=0)
    {
        sum = num1[left--] + num2[right--] + add-2*'0';
        ret[index++] = sum % 10+'0';
        add = sum / 10;
    }
    while (left>=0)
    {
        sum = num1[left--]  + add - '0';
        ret[index++] = sum % 10+'0';
        add = sum / 10;
    }
    while (right >= 0)
    {
        sum = num2[right--] + add - '0';
        ret[index++] = sum % 10+'0';
        add = sum / 10;
    }
    if (add!=0)
    {
        ret[index] = add+'0';
    }
    else
    {
        ret.pop_back();
    }
    reverse(ret.begin(), ret.end());
    return ret;
}

bool AutumnMove::canJump(vector<int>& nums)
{
    if (nums.empty())
    {
        return false;
    }
    int nextIndex = 0;
    for (int i=0;i<nums.size();++i)
    {
        if (nextIndex<i)
        {
            return false;
        }
        nextIndex = max(nextIndex, i + nums[i]);
        if (nextIndex>=nums.size()-1)
        {
            return true;
        }
    }
    return false;
}

void AutumnMove::mergeVector(vector<int>& A, int m, vector<int>& B, int n)
{
    if (A.empty()||B.empty())
    {
        return;
    }
    int index = A.size()-1;
    --m;
    --n;
    while (m>=0&&n>=0)
    {
        if (A[m]>B[n])
        {
            A[index--] = A[m--];
        }
        else
        {
            A[index--] = B[n--];
        }
    }
    while (n>=0)
    {
        A[index--] = B[n--];
    }
}

int AutumnMove::cuttingRope(int n)
{
    if (n<=2)
    {
        return 1;
    }
    if (n==3)
    {
        return 2;
    }
    vector<int> dp(n + 1);
    dp[1] = 1;
    dp[2] = 2;
    dp[3] = 3;
    for (int i=4;i<=n;++i)
    {
        for (int j=1;2*j<=i;++j)
        {
            dp[i] = max(dp[i], dp[j] * dp[i - j]);
        }
    }
    return dp.back();
}

ListNode* AutumnMove::getIntersectionNode(ListNode* headA, ListNode* headB)
{
    if (nullptr ==headA||nullptr==headB)
    {
        return nullptr;
    }
    ListNode* newHeadA = headA;
    ListNode* newHeadB = headB;
    int countA = 0, countB = 0;
    while (newHeadA)
    {
        ++countA;
        newHeadA = newHeadA->next;
    }
    while (newHeadB)
    {
        ++countB;
        newHeadB = newHeadB->next;
    }
    int diff = countA - countB;
    newHeadA = headA;
    newHeadB = headB;
    if (diff<0)
    {
        while (diff)
        {
            ++diff;
            newHeadB = newHeadB->next;
        }
    }
    else
    {
        while (diff)
        {
            --diff;
            newHeadA = newHeadA->next;
        }
    }
    while (NULL != newHeadA&&NULL!=newHeadB)
    {
        if (newHeadA==newHeadB)
        {
            return newHeadA;
        }
        newHeadA = newHeadA->next;
        newHeadB = newHeadB->next;
    }
    return NULL;
}

double AutumnMove::Sqrt(double target, double diff)
{
    if (target<=0)
    {
        return 0;
    }
    double C = target, x0 = target;
    while (true) 
    {
        double xi = 0.5 * (x0 + C / x0);
        if (fabs(x0 - xi) < diff)
        {
            break;
        }
        x0 = xi;
    }
    return x0;
}

bool AutumnMove::canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
    if (prerequisites.empty())
    {
        return true;
    }
    vector<int> inVec(numCourses);
    vector<vector<int>> ouVec(numCourses);
    for (int i=0;i<prerequisites.size();++i)
    {
        inVec[prerequisites[i][0]] += prerequisites[i].size()-1;
        for (int j=1;j<prerequisites[i].size();++j)
        {
            ouVec[prerequisites[i][j]].emplace_back(prerequisites[i][0]);
        }
    }
    queue<int> que;
    for (int i=0;i<inVec.size();++i)
    {
        if (inVec[i]==0)
        {
            que.push(i);
        }
    }
    int count = 0;
    while (!que.empty())
    {
        int node = que.front();
        que.pop();
        ++count;
        for (auto &it:ouVec[node])
        {
            --inVec[it];
            if (inVec[it]==0)
            {
                que.push(it);
            }
        }
    }
    return count == numCourses;
}

ListNode* AutumnMove::addTwoNumbers(ListNode* l1, ListNode* l2)
{
    if (nullptr==l1)
    {
        return l2;
    }
    if (nullptr==l2)
    {
        return l1;
    }
    ListNode* newHead = new ListNode(0);
    ListNode* curNode = newHead;
    int add = 0,sum=0;
    while (l1&&l2)
    {
        sum = l1->val + l2->val + add;
        curNode->next = new ListNode(sum % 10);
        add = sum / 10;
        l1 = l1->next;
        l2 = l2->next;
        curNode = curNode->next;
    }
    while (l1)
    {
        sum = l1->val + add;
        curNode->next = new ListNode(sum % 10);
        add = sum / 10;
        l1 = l1->next;
        curNode = curNode->next;
    }
    while (l2)
    {
        sum = l2->val + add;
        curNode->next = new ListNode(sum % 10);
        add = sum / 10;
        l2 = l2->next;
        curNode = curNode->next;
    }
    if (add!=0)
    {
        curNode->next= new ListNode(add);
    }
    curNode = newHead->next;
    delete newHead;
    return curNode;
}

TreeNode* AutumnMove::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    //LCA问题
    if (nullptr==root|| p == root || q == root)
    {
        return root;
    }
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    if (nullptr!=left&&nullptr!=right)
    {
        return root;
    }
    else if (nullptr==left&&nullptr==right)
    {
        return nullptr;
    }
    return nullptr == left ? right : left;
}

int dfsPathSum(TreeNode* root, int& ret)
{
    if (nullptr==root)
    {
        return 0;
    }
    int leftMax = max(dfsPathSum(root->left, ret), 0);
    int rightMax = max(dfsPathSum(root->right, ret), 0);
    ret = max(leftMax + rightMax + root->val, ret);
    return max(root->val+max(leftMax, rightMax),0);
}

int AutumnMove::maxPathSum(TreeNode* root)
{
    if (nullptr==root)
    {
        return 0;
    }
    int ret = INT_MIN;
    dfsPathSum(root, ret);
    return ret;
}

void partion(vector<int>& nums, int left, int right)
{
    if (left>=right)
    {
        return;
    }
    int leftCur = left;
    int rightCur = right + 1;
    int prio = nums[leftCur];
    while (leftCur < rightCur)
    {
        do 
        {
            ++leftCur;
        } while (leftCur<rightCur&&nums[leftCur]<prio);
        do 
        {
            --rightCur;
        } while (nums[rightCur]>prio);
        if (leftCur>=rightCur)
        {
            //要填写>=否则会出错
            break;
        }
        swap(nums[leftCur], nums[rightCur]);
    }
    swap(nums[left], nums[rightCur]);
    partion(nums, left, rightCur - 1);
    partion(nums, rightCur+1, right);
}

void AutumnMove::QuickSort(vector<int>& nums)
{
    if (nums.size()<2)
    {
        return;
    }
    partion(nums,0,nums.size()-1);
}

std::vector<int> AutumnMove::preorderTraversal(TreeNode* root)
{
    if (nullptr==root)
    {
        return {};
    }
    stack<TreeNode*> stac;
    vector<int> ret;
    stac.push(root);
    while (!stac.empty())
    {
        TreeNode* node = stac.top();
        stac.pop();
        ret.emplace_back(node->val);
        if (node->right)
        {
            stac.push(node->right);
        }
        if (node->left)
        {
            stac.push(node->left);
        }
    }
    return ret;
}

std::vector<int> AutumnMove::postorderTraversal(TreeNode* root)
{
    if (nullptr==root)
    {
        return {};
    }
    stack<TreeNode*> stac;
    vector<int> ret;
    stac.push(root);
    while (!stac.empty())
    {
        TreeNode* node = stac.top();
        stac.pop();
        ret.emplace_back(node->val);
        if (node->left)
        {
            stac.push(node->left);
        }
        if (node->right)
        {
            stac.push(node->right);
        }
    }
    reverse(ret.begin(), ret.end());
    return ret;
}

int AutumnMove::longestConsecutive(vector<int>& nums)
{
    if (nums.empty())
    {
        return 0;
    }
    unordered_set<int> hashSet(nums.begin(), nums.end());
    int ret = 0;
    for (auto &it:hashSet)
    {
        if (hashSet.count(it+1)==0)
        {
            int temp = it;
            int count = 0;
            while (hashSet.count(temp))
            {
                --temp;
                ++count;
            }
            ret = max(count, ret);
        }
    }
    return ret;
}

int AutumnMove::trap(vector<int>& height)
{
    if (height.empty())
    {
        return 0;
    }
    vector<int> leftVec(height.size()+1);
    vector<int> rightVec(height.size()+1);
    for (int i=1;i<=height.size();++i)
    {
        leftVec[i] = max(leftVec[i - 1], height[i-1]);
    }
    for (int i=height.size()-2;i>=0;--i)
    {
        rightVec[i] = max(height[i+1], rightVec[i + 1]);
    }
    int ret = 0;
    for (int i=0;i<height.size();++i)
    {
        ret += max(min(leftVec[i], rightVec[i]) - height[i], 0);
    }
    return ret;
}

unordered_map<TreeNode*,int> hashM;
int dfsRob3(TreeNode* root)
{
    if (nullptr==root)
    {
        return 0;
    }
    if (hashM.count(root))
    {
        return hashM[root];
    }
    int ret1 = root->val;
    if (root->left)
    {
        ret1 += dfsRob3(root->left->left) + dfsRob3(root->left->right);
    }
    if (root->right)
    {
        ret1+= dfsRob3(root->right->left) + dfsRob3(root->right->right);
    }
    int ret2 = dfsRob3(root->left) + dfsRob3(root->right);
    hashM[root] = max(ret1, ret2);
    return hashM[root];
}

int AutumnMove::rob(TreeNode* root)
{
    if (NULL==root)
    {
        return 0;
    }
    return dfsRob3(root);
}

void Merge(vector<int>& nums, vector<int>& helpNums, int start, int end)
{
    if (start>=end)
    {
        helpNums[end] = nums[end];
        return;
    }
    const int len = (end-start)/2;
    Merge(nums, helpNums, start, start + len);
    Merge(nums, helpNums, start + len + 1, end);
    int startIndex = start + len;
    int endIndex = end;
    int curIndex = end;
    while (startIndex >=start&&endIndex>=start+len+1)
    {
        if (nums[startIndex]<nums[endIndex])
        {
            helpNums[curIndex--] = nums[endIndex--];
        }
        else
        {
            helpNums[curIndex--] = nums[startIndex--];
        }
    }
    while (startIndex >= start)
    {
        helpNums[curIndex--] = nums[startIndex--];
    }
    while (endIndex >= start + len + 1)
    {
        helpNums[curIndex--] = nums[endIndex--];
    }
    for (int i=start;i<=end;++i)
    {
        nums[i] = helpNums[i];
    }
}

void AutumnMove::MergeSort(vector<int>& nums)
{
    if (nums.size()<2)
    {
        return;
    }
    vector<int> helpNums(nums.size());
    Merge(nums, helpNums, 0, nums.size() - 1);
}

bool AutumnMove::isSameTree(TreeNode* p, TreeNode* q)
{
    if (nullptr==p&&nullptr==q)
    {
        return true;
    }
    if (nullptr==p||nullptr==q)
    {
        return false;
    }
    if (p->val!=q->val)
    {
        return false;
    }
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

ListNode* AutumnMove::reverseList(ListNode* head)
{
    if (nullptr==head)
    {
        return head;
    }
    ListNode* pre = nullptr;
    ListNode* pTemp = nullptr;
    while (head)
    {
        pTemp = head->next;
        head->next = pre;
        pre = head;
        head = pTemp;
    }
    return pre;
}

std::pair<ListNode*, ListNode*> myReverse(ListNode* head, ListNode* tail)
{
    ListNode* pre = tail->next;
    ListNode* p = head;
    while (pre !=tail)
    {
        ListNode* nex = p->next;
        p->next = pre;
        pre = p;
        p = nex;
    }
    return { tail,head };
}

ListNode* AutumnMove::reverseKGroup(ListNode* head, int k)
{
    if (nullptr==head||k<2)
    {
        return head;
    }
    ListNode* newHead = new ListNode(0);
    newHead->next = head;
    ListNode* pre = newHead;
    while (head)
    {
        ListNode* tail = pre;
        for (int i=0;i<k;++i)
        {
            tail = tail->next;
            if (!tail)
            {
                return newHead->next;
            }
        }
        ListNode* nex = tail->next;
        std::pair<ListNode*, ListNode*> res = myReverse(head, tail);
        head = res.first;
        tail = res.second;
        // 把子链表重新接回原链表
        pre->next = head;
        tail->next = nex;
        pre = tail;
        head = tail->next;
    }
    return newHead->next;
}

std::vector<std::vector<int>> AutumnMove::generateMatrix(int n)
{
    if (n<1)
    {
        return {};
    }
    vector<vector<int>> ret(n, vector<int>(n));
    int leftNum = 0, rightNum = n-1, top = 0, low = n-1;
    int number = 1;
    while (true)
    {
        for (int i=leftNum;i<= rightNum;++i)
        {
            ret[top][i] = number++;
        }
        ++top;
        if (top>low)
        {
            break;
        }
        for (int i=top;i<=low;++i)
        {
            ret[i][rightNum] = number++;
        }
        --rightNum;
        if (rightNum<leftNum)
        {
            break;
        }
        for (int i=rightNum;i>=leftNum;--i)
        {
            ret[low][i] = number++;
        }
        --low;
        if (low<top)
        {
            break;
        }
        for (int i=low;i>=top;--i)
        {
            ret[i][leftNum] = number++;
        }
        ++leftNum;
        if (leftNum>rightNum)
        {
            break;
        }
    }
    return ret;
}

int AutumnMove::countBinarySubstrings(string s)
{
    if (s.size()<2)
    {
        return 0;
    }
    int preCount = 0;
    int ret = 0;
    int left = 0;
    while (left<s.size())
    {
        int curCount = 0;
        int preIndex = left;
        while (left<s.size()&&s[left]==s[preIndex])
        {
            ++left;
            ++curCount;
        }
        ret += min(curCount, preCount);
        preCount = curCount;
    }
    return ret;
}

const int block = 4;

void BackIpAddresses(const string& str, int piece, int start, vector<int>& path, vector<string>& ret)
{
    if (start>str.size())
    {
        return;
    }
    if (piece==block)
    {
        if (start==str.size())
        {
            string strTemp;
            for (int i = 0; i < path.size(); ++i)
            {
                strTemp += to_string(path[i]);
                if (i != path.size() - 1)
                {
                    strTemp.push_back(',');
                }
            }
            ret.emplace_back(strTemp);
        }
        return;
    }
    if (str[start]=='0')
    {
        path.emplace_back(0);
        BackIpAddresses(str, piece + 1, start + 1, path, ret);
        path.pop_back();
        return;
    }
    int num=0;
    for (int i=0;i<3;++i)
    {
        num = str[start+i] - '0' + num * 10;
        if (num>255)
        {
            break;
        }
        path.emplace_back(num);
        BackIpAddresses(str, piece + 1, start + i+1, path, ret);
        path.pop_back();
    }
}

std::vector<std::string> AutumnMove::restoreIpAddresses(string s)
{
    if (s.size()<2)
    {
        return {};
    }
    vector<int> path;
    vector<string> ret;
    BackIpAddresses(s, 0, 0, path, ret);
    return ret;
}

void HeapHelp(vector<int>& nums, int start,const int len)
{
    int nextIndex = 2 * start + 1;
    int tempNode = nums[start];
    while (nextIndex<len)
    {
        if (nextIndex+1<len&&nums[nextIndex]<nums[nextIndex+1])
        {
            ++nextIndex;
        }
        if (tempNode>nums[nextIndex])
        {
            break;
        }
        nums[start] = nums[nextIndex];
        start =nextIndex;
        nextIndex = 2 * start + 1;
    }
    nums[start] = tempNode;
}

void AutumnMove::HeapSort(vector<int>& nums)
{
    if (nums.size()<2)
    {
        return;
    }
    const int len = nums.size();
    for (int i=len/2-1;i>=0;--i)
    {
        HeapHelp(nums, i, len);
    }
    for (int i=len-1;i>=0;--i)
    {
        std::swap(nums[i], nums[0]);
        HeapHelp(nums, 0, i);
    }
}

TreeNode* AutumnMove::buildTree(vector<int>& preorder, vector<int>& inorder)
{
    if (preorder.size()!=inorder.size())
    {
        return nullptr;
    }
    if (preorder.empty())
    {
        return nullptr;
    }
    if (preorder.size()==1)
    {
        return new TreeNode(preorder.front());
    }
    TreeNode* root = new TreeNode(preorder.front());
    auto node = find(inorder.begin(), inorder.end(), root->val);
    int len = node - inorder.begin();
    vector<int> preLeft(preorder.begin() + 1, preorder.begin() + len+1);
    vector<int> preRight(preorder.begin() + len + 1, preorder.end());
    vector<int> inorLeft(inorder.begin(), node);
    vector<int> inorRight(node + 1, inorder.end());
    root->left = buildTree(preLeft, inorLeft);
    root->right = buildTree(preRight, inorRight);
    return root;
}

int dirSolve[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };
void dfsSolve(vector<vector<char>>& board, int curX, int curY)
{
    if (board[curX][curY]!='O')
    {
        return;
    }
    board[curX][curY] = 'F';
    for (int i=0;i<4;++i)
    {
        int newX = curX + dirSolve[i][0];
        int newY = curY + dirSolve[i][1];
        if (newX>=0&&newX<board.size()&&newY>=0&&newY<board[0].size()&&board[newX][newY]=='O')
        {
            dfsSolve(board, newX, newY);
        }
    }
}

void AutumnMove::solve(vector<vector<char>>& board)
{
    if (board.empty())
    {
        return;
    }
    const int raw = board.size();
    const int col = board[0].size();
    for (int i=0;i< raw;++i)
    {
        if (board[i][0]=='O')
        {
            dfsSolve(board,i, 0);
        }
        if (board[i][col -1]=='O')
        {
            dfsSolve(board, i, col-1);
        }
    }
    for (int i=0;i<col;++i)
    {
        if (board[0][i] == 'O')
        {
            dfsSolve(board, 0, i);
        }
        if (board[raw - 1][i] == 'O')
        {
            dfsSolve(board, raw-1,i);
        }
    }
    for (auto &it:board)
    {
        for (auto &itsec:it)
        {
            if (itsec=='F')
            {
                itsec = 'O';
            }
            else if(itsec=='O')
            {
                itsec = 'X';
            }
        }
    }
}

bool dfsHasPathSum(TreeNode* root, int sum, const int& target)
{
    if (nullptr==root)
    {
        return false;
    }
    sum += root->val;
    if (nullptr==root->left&&nullptr==root->right)
    {
        if (target==sum)
        {
            return true;
        }
    }
    if (dfsHasPathSum(root->left,sum,target))
    {
        return true;
    }
    if (dfsHasPathSum(root->right,sum,target))
    {
        return true;
    }
    return false;
}

bool AutumnMove::hasPathSum(TreeNode* root, int sum)
{
    if (nullptr==root)
    {
        return false;
    }
    return dfsHasPathSum(root, 0, sum);
}

void dfsPathSumVec(TreeNode* root, const int& target, vector<int>& path, vector<vector<int>>& ret, int add)
{
    if (nullptr==root)
    {
        return;
    }
    add += root->val;
    path.emplace_back(root->val);
    if (nullptr==root->left&&nullptr==root->right)
    {
        if (target==add)
        {
            ret.emplace_back(path);
        }
        path.pop_back();
        return;
    }
    dfsPathSumVec(root->left, target, path, ret, add);
    dfsPathSumVec(root->right, target, path, ret, add);
    path.pop_back();
}

std::vector<std::vector<int>> AutumnMove::pathSum(TreeNode* root, int sum)
{
    if (nullptr==root)
    {
        return {};
    }
    vector<int> path;
    vector<vector<int>> ret;
    dfsPathSumVec(root, sum,path,ret, 0);
    return ret;
}

std::string AutumnMove::multiply(string num1, string num2)
{
    if (num1.empty()||num2.empty())
    {
        return "0";
    }
    vector<int> nums(num1.size() + num2.size() + 1);
    int index = 0;
    for (int i=num1.size()-1;i>=0;--i)
    {
        int curIndex = index++;
        for (int j= num2.size()-1;j>=0;--j)
        {
            nums[curIndex++] += (num1[i] - '0') * (num2[j] - '0');
        }
    }
    string ret(nums.size(),'0');
    int add = 0;
    for (int i=0;i<nums.size();++i)
    {
        int sum = nums[i] + add;
        add = sum / 10;
        ret[i] = sum % 10 + '0';
    }
    for (int j=ret.size()-1;j>=1;--j)
    {
        if (ret[j]=='0')
        {
            ret.pop_back();
        }
        else
        {
            break;
        }
    }
    reverse(ret.begin(), ret.end());
    return ret;
}

std::vector<std::vector<int>> AutumnMove::merge(vector<vector<int>>& intervals)
{
    if (intervals.empty())
    {
        return {};
    }
    sort(intervals.begin(), intervals.end(), [](const vector<int>& leftVec, const vector<int>& rightVec)
        {
            if (leftVec.front()!=rightVec.front())
            {
                return leftVec.front() < rightVec.front();
            }
            return leftVec.back() < rightVec.back();
        });
    int left = intervals[0].front(), right = intervals[0].back();
    vector<vector<int>> ret;
    for (int i=1;i<intervals.size();++i)
    {
        if (right<intervals[i].front())
        {
            vector<int> temp = { left,right };
            ret.emplace_back(temp);
            left = intervals[i].front();
            right = intervals[i].back();
        }
        else
        {
            right = max(right, intervals[i].back());
        }
    }
    vector<int> temp = { left,right };
    ret.emplace_back(temp);
    return ret;
}

std::string AutumnMove::reverseWords(string s)
{
    if (s.empty())
    {
        return s;
    }
    reverse(s.begin(),s.end());
    int index = 0;
    string ret;
    while (index<s.size()&& s[index] == ' ')
    {
        ++index;
    }
    while (index<s.size())
    {
        int curIndex = index;
        while (curIndex<s.size()&&s[curIndex]!=' ')
        {
            ++curIndex;
        }
        for (int j=curIndex-1;j>=index;--j)
        {
            ret.push_back(s[j]);
        }
        if (curIndex!=index)
        {
            ret += " ";
        }
        index = curIndex + 1;
    }
    if (!ret.empty())
    {
        ret.pop_back();
    }
    return ret;
}

std::unordered_map<TreeNode*, int> depthMap;
int GetTreeDepth(TreeNode* root)
{
    if (nullptr==root)
    {
        return 0;
    }
    if (depthMap.count(root))
    {
        return depthMap[root];
    }
    int leftDepth = GetTreeDepth(root->left);
    int rightDepth = GetTreeDepth(root->right);
    depthMap[root] = 1 + max(leftDepth, rightDepth);
    return depthMap[root];
}

bool AutumnMove::isBalanced(TreeNode* root)
{
    if (nullptr==root)
    {
        return true;
    }
    if (abs(GetTreeDepth(root->left)-GetTreeDepth(root->right))>1)
    {
        return false;
    }
    return isBalanced(root->left) && isBalanced(root->right);
}

TreeNode* AutumnMove::sortedListToBST(ListNode* head)
{
    if (nullptr==head)
    {
        return nullptr;
    }
    if (nullptr==head->next)
    {
        return new TreeNode(head->val);
    }
    ListNode* pre = head;
    //找到中点
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast&&fast->next)
    {
        slow = slow->next;
        fast = fast->next->next;
    }
    while (pre->next!=slow)
    {
        pre = pre->next;
    }
    fast = slow->next;
    pre->next = nullptr;
    TreeNode* root = new TreeNode(slow->val);
    root->left = sortedListToBST(head);
    root->right = sortedListToBST(fast);
    return root;
}

int GetSubString(const string& str, int left, int right)
{
    int ret = 0;
    while (left>=0&&right<str.size()&&str[left]==str[right])
    {
        --left;
        ++right;
        ++ret;
    }
    return ret;
}

int AutumnMove::countSubstrings(string s)
{
    if (s.empty())
    {
        return 0;
    }
    int ret = 0;
    for (int i=0;i<s.size();++i)
    {
        ret += GetSubString(s, i, i);
        ret += GetSubString(s, i, i + 1);
    }
    return ret;
}

int dir_x[8] = { 0, 1, 0, -1, 1, 1, -1, -1 };
int dir_y[8] = { 1, 0, -1, 0, 1, -1, 1, -1 };
void dfsBoard(vector<vector<char>>& board, int x, int y)
{
    int cnt = 0;
    for (int i = 0; i < 8; ++i) 
    {
        int tx = x + dir_x[i];
        int ty = y + dir_y[i];
        if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size())
        {
            continue;
        }
        // 不用判断 M，因为如果有 M 的话游戏已经结束了
        cnt += board[tx][ty] == 'M';
    }
    if (cnt > 0) 
    {
        // 规则 3
        board[x][y] = cnt + '0';
    }
    else 
    {
        // 规则 2
        board[x][y] = 'B';
        for (int i = 0; i < 8; ++i) 
        {
            int tx = x + dir_x[i];
            int ty = y + dir_y[i];
            // 这里不需要在存在 B 的时候继续扩展，因为 B 之前被点击的时候已经被扩展过了
            if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() || board[tx][ty] != 'E') 
            {
                continue;
            }
            dfsBoard(board, tx, ty);
        }
    }
}
std::vector<std::vector<char>> AutumnMove::updateBoard(vector<vector<char>>& board, vector<int>& click)
{
    int x = click[0], y = click[1];
    if (board[x][y] == 'M')
    {
        // 规则 1
        board[x][y] = 'X';
    }
    else
    {
        dfsBoard(board, x, y);
    }
    return board;
}

std::vector<std::vector<int>> AutumnMove::findContinuousSequence(int target)
{
    if (target<=1)
    {
        return {};
    }
    vector<vector<int>> ret;
    int left = 1,right=2,sum=3;
    while (right<=(target+1)/2)
    {
        while (sum > target)
        {
            sum -= left++;
        }
        if (sum==target)
        {
            vector<int> temp;
            for (int i=left;i<=right;++i)
            {
                temp.emplace_back(i);
            }
            ret.emplace_back(temp);
        }
        sum += ++right;
    }
    return ret;
}

void AutumnMove::nextPermutation(vector<int>& nums)
{
    if (nums.size()<2)
    {
        return;
    }
    int nextIndex = -1;
    for (int i=0;i<nums.size()-1;++i)
    {
        if (nums[i]<nums[i+1])
        {
            nextIndex = i;
        }
    }
    if (-1==nextIndex)
    {
        reverse(nums.begin(), nums.end());
    }
    else
    {
        int curIndex = nextIndex + 1;
        for (int i=curIndex+1;i<nums.size();++i)
        {
            if (nums[i]>nums[nextIndex]&&nums[i]<nums[curIndex])
            {
                curIndex = i;
            }
        }
        swap(nums[curIndex], nums[nextIndex]);
        sort(nums.begin() + nextIndex + 1, nums.end());
    }
}

int AutumnMove::firstMissingPositive(vector<int>& nums)
{
    if (nums.empty())
    {
        return 1;
    }
    const int length = nums.size();
    int ret = length;
    for (int i = 0; i < length; ++i)
    {
        while (nums[i] > 0 && nums[i] <= length && nums[i] != nums[nums[i] - 1])
        {
            swap(nums[i], nums[nums[i] - 1]);
        }
    }
    for (int i = 0; i < nums.size(); ++i)
    {
        if (nums[i] != i + 1)
        {
            return i + 1;
        }
    }
    return length + 1;
}

void rightSide(TreeNode* root, vector<int>& ret,int depth)
{
    if (nullptr==root)
    {
        return;
    }
    if (ret.size()==depth)
    {
        ret.emplace_back(root->val);
    }
    rightSide(root->right, ret, depth + 1);
    rightSide(root->left, ret, depth + 1);
}

std::vector<int> AutumnMove::rightSideView(TreeNode* root)
{
    vector<int> ret;
    rightSide(root, ret, 0);
    return ret;
}

bool AutumnMove::PredictTheWinner(vector<int>& nums)
{
	if (nums.empty())
	{
		return true;
	}
	if (nums.size()%2==0)
	{
		return true;
	}
	//i j 代表 i到j两玩家分数差值的最大值
	vector<vector<int>> dp(nums.size(), vector<int>(nums.size()));
	for (int i = 0; i < nums.size();++i)
	{
		dp[i][i] = nums[i];
	}
	for (int i = nums.size() - 2; i >= 0;--i)
	{
		for (int j = i+1; j <nums.size();++j)
		{
			dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
		}
	}
	return dp[0][nums.size() - 1]>=0;
}

bool AutumnMove::isNumber(string s)
{
	if (s.empty())
	{
		return false;
	}
	int ipos = 0;
	while (s[ipos] == ' ')
	{
		++ipos;
	}
	if (ipos != 0)
	{
		return isNumber(s.substr(ipos));
	}
	while (s.back() == ' ')
	{
		s.pop_back();
	}
	int pointNumber = 0;
	int number = 0;
	if (s[ipos] == '+' || s[ipos] == '-')
	{
		++ipos;
	}
	while (ipos < s.size() && s[ipos] != 'E'&&s[ipos] != 'e')
	{
		if (s[ipos] == '.')
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
	if (pointNumber > 1 || number == 0)
	{
		return false;
	}
	if (ipos == s.size())
	{
		return true;
	}
	++ipos;
	if (ipos == s.size())
	{
		return false;
	}
	if (s[ipos] == '+' || s[ipos] == '-')
	{
		++ipos;
	}
	if (ipos == s.size())
	{
		return false;
	}
	while (ipos < s.size() && s[ipos] >= '0'&&s[ipos] <= '9')
	{
		++ipos;
	}
	return ipos == s.size();
}

void BackPermute(vector<vector<int>>& ret, vector<int>& nums, vector<int>&path, int start)
{
	if (start==nums.size())
	{
		ret.push_back(path);
		return;
	}
	for (int i = start; i < nums.size();++i)
	{
		swap(nums[i], nums[start]);
		path.emplace_back(nums[start]);
		BackPermute(ret, nums, path, start + 1);
		path.pop_back();
		swap(nums[i], nums[start]);
	}
}

std::vector<std::vector<int>> AutumnMove::permute(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	vector<int> path;
	vector<vector<int>> ret;
	BackPermute(ret, nums, path, 0);
	return ret;
}

void BackPermute2(vector<vector<int>>& ret, vector<int>& nums, vector<int>&path, int start)
{
	if (start == nums.size())
	{
		ret.push_back(path);
		return;
	}
	unordered_map<int, int> hashM;
	for (int i = start; i < nums.size(); ++i)
	{
		if (hashM[nums[i]]) continue;
		hashM[nums[i]] = 1;
		swap(nums[i], nums[start]);
		path.emplace_back(nums[start]);
		BackPermute2(ret, nums, path, start + 1);
		path.pop_back();
		swap(nums[i], nums[start]);
	}
}

std::vector<std::vector<int>> AutumnMove::permute2(vector<int>& nums)
{
	if (nums.empty())
	{
		return{};
	}
	sort(nums.begin(), nums.end());
	vector<vector<int>> ret;
	vector<int> path;
	BackPermute2(ret, nums, path, 0);
	return ret;
}

bool CheckNQueens(vector<int>& path, int start)
{
	if (path.empty())
	{
		return true;
	}
	for (int i = 0; i < path.size();++i)
	{
		if (abs(path[i] - start) == abs((int)path.size() - i))
		{
			return false;
		}
	}
	return true;
}

void BackNQueens(vector<vector<int>>& ret, vector<int>& nums, vector<int>&path, int start)
{
	if (start == nums.size())
	{
		ret.push_back(path);
		return;
	}
	for (int i = start; i < nums.size(); ++i)
	{
		if (!CheckNQueens(path,nums[i]))
		{
			continue;
		}
		swap(nums[i], nums[start]);
		path.emplace_back(nums[start]);
		BackNQueens(ret, nums, path, start + 1);
		path.pop_back();
		swap(nums[i], nums[start]);
	}
}

std::vector<std::vector<std::string>> AutumnMove::solveNQueens(int n)
{
	vector<int> nums;
	for (int  i = 0; i < n; ++i)
	{
		nums.emplace_back(i);
	}
	vector<vector<int>> ret;
	vector<int> path;
	BackNQueens(ret, nums, path, 0);
	vector<vector<string>> resStr;
	for (auto &it : ret)
	{
		vector<string> strTempVec;
		for (auto & itVec : it)
		{
			string strTemp(it.size(), '.');
			strTemp[itVec] = 'Q';
			strTempVec.push_back(strTemp);
		}
		resStr.push_back(strTempVec);
	}
	return resStr;
}

void DFSTree(TreeNode* root, vector<string>&ret, vector<int>& path)
{
	if (nullptr==root)
	{
		return;
	}
	if (nullptr==root->left&&nullptr==root->right)
	{
		std::string str;
		for (int i = 0; i < path.size(); ++i)
		{
			str += to_string(path[i]) + "->";
		}
		str += to_string(root->val);
		ret.emplace_back(str);
		return;
	}
	path.push_back(root->val);
	DFSTree(root->left, ret, path);
	DFSTree(root->right, ret, path);
	path.pop_back();
}

std::vector<std::string> AutumnMove::binaryTreePaths(TreeNode* root)
{
	if (nullptr==root)
	{
		return{};
	}
	vector<string> ret;
	vector<int> path;
	DFSTree(root, ret, path);
	return ret;
}

struct  ListNodeCom
{
	ListNodeCom(ListNode* node) :m_pNode(node) {}
	ListNode* m_pNode;
	//一定要加上const
	bool operator < (const ListNodeCom &right) const
	{
		return m_pNode->val > right.m_pNode->val;
	}
};

ListNode* AutumnMove::mergeKLists(vector<ListNode*>& lists)
{
	if (lists.empty())
	{
		return nullptr;
	}
	priority_queue<ListNodeCom> prioHelp;
	for (auto &it:lists)
	{
		if (nullptr==it)
		{
			continue;
		}
		prioHelp.push(ListNodeCom(it));
	}
	ListNode* newHead = new ListNode(0);
	ListNode* curNode = newHead;
	while (!prioHelp.empty())
	{
		auto node = prioHelp.top();
		prioHelp.pop();
		curNode->next = new ListNode(node.m_pNode->val);
		if (nullptr!=node.m_pNode->next)
		{
			prioHelp.push(ListNodeCom(node.m_pNode->next));
		}
		curNode = curNode->next;
	}
	curNode = newHead->next;
	delete newHead;
	return curNode;
}

std::string AutumnMove::getPermutation(int n, int k)
{
	string ret;
	string num = "123456789";
	vector<int> f(n, 1);
	for (int i = 1; i < n;++i)
	{
		f[i] = i*f[i - 1];
	}
	--k;
	for (int i = n; i >= 1; --i)
	{
		int key = k / f[i - 1];
		k %= f[i - 1];
		ret.push_back(num[key]);
		num.erase(key,1);
	}
	return ret;
}

std::vector<int> AutumnMove::topKFrequent(vector<int>& nums, int k)
{
	if (nums.empty())
	{
		return{};
	}
	unordered_map<int, int> hashM;
	for (auto &it:nums)
	{
		++hashM[it];
	}
	vector<pair<int,int>> numsVec;
	for (auto &it:hashM)
	{
		numsVec.emplace_back(std::make_pair(it.first, it.second));
	}
	sort(numsVec.begin(), numsVec.end(), [](const pair<int, int>& left, const pair<int, int>& right)
	{
		return left.second > right.second;
	});
	vector<int> ret;
	for (int i = 0; i < k;++i)
	{
		ret.emplace_back(numsVec[i].first);
	}
	return ret;
}

void BackComBina(vector<int>& candidates, vector<vector<int>>&ret, vector<int>& path, const int target, int sum,int start)
{
	if (sum>target)
	{
		return;
	}
	if (sum==target)
	{
		ret.emplace_back(path);
		return;
	}
	for (int i = start; i < candidates.size();++i)
	{
		path.emplace_back(candidates[i]);
		BackComBina(candidates, ret, path, target, sum + candidates[i], i);
		path.pop_back();
	}
}

std::vector<std::vector<int>> AutumnMove::combinationSum(vector<int>& candidates, int target)
{
	if (candidates.empty())
	{
		return{};
	}
	sort(candidates.begin(), candidates.end());
	vector<vector<int>> ret;
	vector<int> path;
	BackComBina(candidates, ret, path, target, 0, 0);
	return ret;
}

void BackCombine(vector<int>& nums, vector<vector<int>>& ret, vector<int>& path, int start,const int k)
{
	if (path.size()==k)
	{
		ret.emplace_back(path);
		return;
	}
	for (int i = start; i < nums.size(); ++i)
	{
		swap(nums[start], nums[i]);
		path.emplace_back(nums[start]);
		BackCombine(nums, ret, path, i+1, k);
		swap(nums[start], nums[i]);
		path.pop_back();
	}
}

std::vector<std::vector<int>> AutumnMove::combine(int n, int k)
{
	vector<int> nums(n);
	for (int i = 1; i <= n;++i)
	{
		nums[i - 1] = i;
	}
	vector<vector<int>> ret;
	vector<int> path;
	BackCombine(nums, ret, path, 0, k);
	return ret;
}

void BackCombineSum2(vector<vector<int>>& ret,vector<int>& candidates,vector<int>& path, int start,int target)
{
	if (target < 0)
	{
		return;
	}
	if (target==0)
	{
		ret.emplace_back(path);
		return;
	}
	for (int i = start; i < candidates.size(); ++i)
	{
		if (i!=start&&candidates[i]==candidates[i-1])
		{
			continue;
		}
		path.emplace_back(candidates[i]);
		BackCombineSum2(ret, candidates, path, i + 1, target - candidates[i]);
		path.pop_back();
	}
}

std::vector<std::vector<int>> AutumnMove::combinationSum2(vector<int>& candidates, int target)
{
	if (candidates.empty())
	{
		return{};
	}
	sort(candidates.begin(), candidates.end());
	vector<vector<int>> ret;
	vector<int> path;
	BackCombineSum2(ret, candidates, path, 0, target);
	return ret;
}

int AutumnMove::findNthDigit(int n)
{
	if (n<10)
	{
		return n;
	}
	int count = 1;
	while (n>9*pow(10,count-1)*count)
	{
		n -= 9 * pow(10, count-1)*count;
		++count;
	}
	string strNum=to_string(pow(10, count - 1) + (n - 1) / count);
	if (n%count)
	{
		return strNum[n%count - 1] - '0';
	}
	return strNum[(n - 1) % count] - '0';
}

std::string AutumnMove::simplifyPath(string path)
{
	if (path.empty())
	{
		return "";
	}
	stack<string> stacHelp;
	int index = 0;
	string prePath;
	path += "/";
	while (index<path.size())
	{
		if (path[index]=='/')
		{
			if (prePath=="..")
			{
				if (!stacHelp.empty())
				{
					stacHelp.pop();
				}
			}
			else if (prePath!=".")
			{
				if (prePath!="")
				{
					stacHelp.push(prePath);
				}
			}
			prePath = "";
		}
		else
		{
			prePath.push_back(path[index]);
		}
		++index;
	}
	if (stacHelp.empty())
	{
		return "/";
	}
	string ret;
	while (!stacHelp.empty())
	{
		ret = "/" + stacHelp.top()+ret;
		stacHelp.pop();
	}
	return ret;
}

TreeNode* AutumnMove::invertTree(TreeNode* root)
{
	if (nullptr == root)
	{
		return root;
	}
	TreeNode* node = root->left;
	root->left = root->right;
	root->right = node;
	invertTree(root->left);
	invertTree(root->right);
	return root;
}

void SwapList(ListNode* left, ListNode* right)
{
	if (nullptr==left||nullptr==right)
	{
		return;
	}
	if (left->val==right->val)
	{
		return;
	}
	left->val ^= right->val;
	right->val ^= left->val;
	left->val ^= right->val;
}

void QuickListNodeSort(ListNode* left, ListNode* right)
{
	if (nullptr==left||nullptr==right||left==right)
	{
		return;
	}
	int prio = right->val;
	ListNode* preNode = nullptr;
	ListNode* leftCur = left;
	ListNode* p = left;
	while (p!=right)
	{
		if (p->val<prio)
		{
			SwapList(p, leftCur);
			preNode = leftCur;
			leftCur = leftCur->next;
		}
		p = p->next;
	}
	if (leftCur!=right)
	{
		SwapList(leftCur, right);
		preNode = leftCur;
		leftCur = leftCur->next;
	}
	QuickListNodeSort(left, preNode);
	QuickListNodeSort(leftCur, right);
}

ListNode* AutumnMove::sortList(ListNode* head)
{
	if (nullptr==head)
	{
		return nullptr;
	}
	ListNode* tail = head;
	while (tail&&tail->next)
	{
		tail = tail->next;
	}
	QuickListNodeSort(head, tail);
	return head;
}

void ConvertTree(TreeNode* root, int &ret)
{
	if (nullptr==root)
	{
		return;
	}
	ConvertTree(root->right,ret);
	int temp = ret;
	ret += root->val;
	root->val += temp;
	ConvertTree(root->left, ret);
}

TreeNode* AutumnMove::convertBST(TreeNode* root)
{
	if (nullptr==root)
	{
		return root;
	}
	int ret = 0;
	ConvertTree(root, ret);
	return root;
}

int ans = 0;
int dfs(TreeNode* root)
{
	if (root == NULL)
		return 2;
	int left_state = dfs(root->left);
	int right_state = dfs(root->right);
	if (left_state == 0 || right_state == 0) 
	{
		ans++;
		return 1;
	}
	else if (left_state == 1 || right_state == 1)
		return 2;
	return 0;
}

int AutumnMove::minCameraCover(TreeNode* root)
{
	if (nullptr==root)
	{
		return 0;
	}
	return dfs(root) == 0 ? ans + 1 : ans;
}

TreeNode* AutumnMove::mergeTrees(TreeNode* t1, TreeNode* t2)
{
	if (NULL==t1&&NULL==t2)
	{
		return NULL;
	}
	if (NULL==t1)
	{
		return t2;
	}
	if (NULL==t2)
	{
		return t1;
	}
	TreeNode* root = new TreeNode(0);
	root->val = t1->val + t2->val;
	root->left = mergeTrees(t1->left, t2->left);
	root->right = mergeTrees(t1->right, t2->right);
	return root;
}

bool CheckIsBack(const string & str,int left,int right)
{
	while (left<right)
	{
		if (str[left]!=str[right])
		{
			return false;
		}
		++left;
		--right;
	}
	return true;
}

void BackCheck(vector<vector<string>>& ret, vector<string>& path, int start, const string& s)
{
	if (start==s.size())
	{
		ret.emplace_back(path);
		return;
	}
	for (int i = start + 1; i <= s.size();++i)
	{
		if (CheckIsBack(s,start,i-1))
		{
			path.emplace_back(s.substr(start, i - start));
			BackCheck(ret, path, i, s);
			path.pop_back();
		}
	}
}

std::vector<std::vector<std::string>> AutumnMove::partitionString(string s)
{
	if (s.empty())
	{
		return{};
	}
	vector<vector<string>> ret;
	vector<string> path;
	BackCheck(ret, path, 0, s);
	return ret;
}

void VisitedNode(TreeNode* root, TreeNode* &pre, int &curMaxTimes, int &maxTimes, vector<int>&res)
{	
	if (nullptr==root)
	{
		return;
	}
	VisitedNode(root->left, pre, curMaxTimes, maxTimes, res);
	if (pre)
	{
		curMaxTimes = root->val == pre->val ? ++curMaxTimes : 1;
	}
	if (maxTimes == curMaxTimes)
	{
		res.emplace_back(root->val);
	}
	else if (curMaxTimes>maxTimes)
	{
		res.clear();
		res.emplace_back(root->val);
		maxTimes = curMaxTimes;
	}
	pre = root;
	VisitedNode(root->right, pre, curMaxTimes, maxTimes, res);
}

std::vector<int> AutumnMove::findMode(TreeNode* root)
{
	if (nullptr==root)
	{
		return{};
	}
	vector<int> ret;
	int curTimes = 1, maxTimes = 0;
	TreeNode* pre = nullptr;
	VisitedNode(root, pre, curTimes, maxTimes, ret);
	return ret;
}

std::vector<std::vector<int>> AutumnMove::insertVec(vector<vector<int>>& intervals, vector<int>& newInterval)
{
	int left = newInterval[0];
	int right = newInterval[1];
	bool placed = false;
	vector<vector<int>> ans;
	for (const auto& interval : intervals) {
		if (interval[0] > right) {
			// 在插入区间的右侧且无交集
			if (!placed) {
				ans.push_back({ left, right });
				placed = true;
			}
			ans.push_back(interval);
		}
		else if (interval[1] < left) {
			// 在插入区间的左侧且无交集
			ans.push_back(interval);
		}
		else {
			// 与插入区间有交集，计算它们的并集
			left = min(left, interval[0]);
			right = max(right, interval[1]);
		}
	}
	if (!placed) {
		ans.push_back({ left, right });
	}
	return ans;
}

int AutumnMove::ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
	if (beginWord==endWord)
	{
		return 0;
	}
	unordered_set<string> hashSet(wordList.begin(), wordList.end());
	queue<string> queHelp;
	unordered_set<string> hashUsed;
	queHelp.push(beginWord);
	hashUsed.insert(beginWord);
	int ret = 0;
	while (!queHelp.empty())
	{
		++ret;
		int len = queHelp.size();
		for (int i = 0; i < len;++i)
		{
			auto str = queHelp.front();
			queHelp.pop();
			for (int j = 0; j < str.size();++j)
			{
				char ch = str[j];
				for (int k = 0; k < 26; ++k)
				{
					str[j] = 'a' + k;
					if (hashUsed.count(str)==0&&hashSet.count(str))
					{
						if (str==endWord)
						{
							return ret + 1;
						}
						hashUsed.insert(str);
						queHelp.push(str);
					}
				}
				str[j] = ch;
			}
		}
	}
	return 0;
}

int AutumnMove::findRotateSteps(string ring, string key)
{
	int n = ring.size(), m = key.size();
	vector<int> pos[26];
	for (int i = 0; i < n; ++i) {
		pos[ring[i] - 'a'].push_back(i);
	}
	vector<vector<int>> dp(m, vector<int>(n, 0x3f3f3f3f));
	for (auto& i : pos[key[0] - 'a']) {
		dp[0][i] = std::min(i, n - i) + 1;
	}
	// 更新公式， 在前一个字符匹配的情况下， 求下一个字符匹配所需要的最小步数
	// dp[i][j] = min(dp[i][j], dp[i-1][k] + j与k之间的最小距离 + 1(打印);
	for (int i = 1; i < m; ++i) {
		for (auto& j : pos[key[i] - 'a']) {
			for (auto& k : pos[key[i - 1] - 'a']) {
				dp[i][j] = std::min(dp[i][j], dp[i - 1][k] + std::min(abs(j - k), n - abs(j - k)) + 1);
			}
		}
	}
	int res = INT_MAX;
	// 可能有多个字符和key的最后一个字符匹配， 取最小那个
	for (int i = 0; i < m; i++) {
		if (ring[i] == key[n - 1]) {
			res = min(res, dp[n - 1][i]);
		}
	}
	return res;
}

ListNode* AutumnMove::oddEvenList(ListNode* head)
{
	if (!head)
	{
		return head;
	}
	ListNode *headOdd = new ListNode(0);
	ListNode *headEven = new ListNode(0);
	ListNode *preOdd = headOdd;
	ListNode *preEven = headEven;
	int iNumber = 1;
	while (head)
	{
		if (iNumber % 2)
		{
			preOdd->next = head;
			preOdd = head;
		}
		else
		{
			preEven->next = head;
			preEven = head;
		}
		++iNumber;
		head = head->next;
	}
	preEven->next = NULL;
	preOdd->next = headEven->next;
	return headOdd->next;
}

std::string AutumnMove::removeKdigits(string num, int k)
{
	if (num.size()<=k)
	{
		return "0";
	}
	std::string ret;
	int len = num.size() - k;
	for (auto &it:num)
	{
		while (k&&!ret.empty() && ret.back() > it)
		{
			ret.pop_back();
		}
		--k;
		ret.push_back(it);
	}
	ret.resize(len);
	while (!ret.empty()&&ret.front()=='0')
	{
		ret.erase(ret.begin());
	}
	return ret.empty() ? "0" : ret;
}

std::vector<std::vector<int>> AutumnMove::reconstructQueue(vector<vector<int>>& people)
{
	if (people.empty())
	{
		return people;
	}
	sort(people.begin(), people.end(), [](const vector<int>& left,const vector<int>& right)
	{
		if (left[1]!=right[1])
		{
			return left[1] < right[1];
		}
		return left[0] > right[0];
	});
	std::vector<vector<int>> ret;
	for (auto& it:people)
	{
		ret.insert(ret.begin() + it[1], it);
	}
	return ret;
}

std::vector<std::vector<int>> AutumnMove::allCellsDistOrder(int R, int C, int r0, int c0)
{
	queue<pair<int, int>> queHelp;
	vector<vector<int>> ret;
	int dir[4][2] = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
	vector<vector<bool>> visit(R, vector<bool>(C, true));
	visit[r0][c0] = false;
	queHelp.push(std::make_pair(r0, c0));
	while (!queHelp.empty())
	{
		int len = queHelp.size();
		for (int i = 0; i < len;++i)
		{
			int curX = queHelp.front().first;
			int curY = queHelp.front().second;
			ret.push_back({ curX, curY });
			queHelp.pop();
			for (int j = 0; j < 4;++j)
			{
				int newX = curX + dir[j][0];
				int newY = curY + dir[j][1];
				if (newX>=0&&newX<R&&newY>=0&&newY<C&&visit[newX][newY])
				{
					queHelp.push(std::make_pair(newX, newY));
					visit[newX][newY] = false;
				}
			}
		}
	}
	return ret;
}

int AutumnMove::canCompleteCircuit(vector<int>& gas, vector<int>& cost)
{
	if (gas.empty()||cost.empty())
	{
		return -1;
	}
	int i = 0;
	while (i<gas.size())
	{
		int sumOfGas = 0, sumOfCos = 0;
		int cnt = 0;
		while (cnt<gas.size())
		{
			int j = (i + cnt) % gas.size();
			sumOfGas += gas[j];
			sumOfCos += cost[j];
			if (sumOfCos>sumOfGas)
			{
				break;
			}
			++cnt;
		}
		if (cnt==gas.size())
		{
			return i;
		}
		else
		{
			i = i + cnt + 1;
		}
	}
	return -1;
}

std::string AutumnMove::sortString(string s)
{
	vector<int> num(26);
	for (char &ch : s) 
	{
		num[ch - 'a']++;
	}

	string ret;
	while (ret.length() < s.length()) 
	{
		for (int i = 0; i < 26; i++)
		{
			if (num[i])
			{
				ret.push_back(i + 'a');
				num[i]--;
			}
		}
		for (int i = 25; i >= 0; i--) 
		{
			if (num[i]) 
			{
				ret.push_back(i + 'a');
				num[i]--;
			}
		}
	}
	return ret;
}

bool AutumnMove::isPossible(vector<int>& nums)
{
	if (nums.size()<3)
	{
		return false;
	}
	unordered_map<int, std::priority_queue<int, vector<int>, greater<int>>> mH;
	for (auto &it:nums)
	{
		if (mH.count(it)==0)
		{
			mH[it] = std::priority_queue<int, vector<int>, greater<int>>();
		}
		if (mH.count(it-1))
		{
			int preLen = mH[it - 1].top();
			mH[it - 1].pop();
			if (mH[it - 1].empty())
			{
				mH.erase(it - 1);
			}
			mH[it].push(preLen + 1);
		}
		else
		{
			mH[it].push(1);
		}
	}
	for (auto &it:mH)
	{
		if (it.second.top()<3)
		{
			return false;
		}
	}
	return true;
}
