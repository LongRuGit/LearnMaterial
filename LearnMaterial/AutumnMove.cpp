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
