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
