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
