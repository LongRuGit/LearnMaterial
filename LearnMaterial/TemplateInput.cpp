#include <iostream>
#include <vector>
#include <string>

using namespace std;

//将输入的字符按照符号进行分割
void SplitStringBySymbol(const string& iStr, const string& symbol, vector<string>& ioVecStr)
{
    if (iStr.empty())
    {
        return;
    }
    size_t iFindIndex = iStr.find(symbol, 0);
    if (iFindIndex == string::npos)
    {
        ioVecStr.emplace_back(iStr);
        return;
    }
    if (iFindIndex==0)
    {
        return;
    }
    ioVecStr.emplace_back(iStr.substr(0, iFindIndex));
    return SplitStringBySymbol(iStr.substr(iFindIndex + 1), symbol, ioVecStr);
}

//https://ac.nowcoder.com/acm/contest/320#question

//输入包括两个正整数a, b(1 <= a, b <= 10 ^ 9), 输入数据包括多组。
// int a, b;
//while (cin>>a>>b)
//{
//    cout << a + b;
//}

//输入第一行包括一个数据组数t(1 <= t <= 100)
//接下来每行包括两个正整数a, b(1 <= a, b <= 10 ^ 9)
//int length = 0;
//cin >> length;
//for (int i = 0; i < length; ++i)
//{
//    int a, b;
//    cin >> a >> b;
//    cout << a + b << "\n";
//}

//输入包括两个正整数a, b(1 <= a, b <= 10 ^ 9), 输入数据有多组, 如果输入为0 0则结束输入
//while (true)
//{
//    int a, b;
//    cin >> a >> b;
//    if (a == 0 && b == 0)
//    {
//        break;
//    }
//    cout << a + b << "\n";
//}

//输入数据包括多组。
//每组数据一行, 每行的第一个整数为整数的个数n(1 <= n <= 100), n为0的时候结束输入。
//接下来n个正整数, 即需要求和的每个正整数。
//while (true)
//{
//    int length;
//    cin >> length;
//    if (length == 0)
//    {
//        break;
//    }
//    int b = 0, sum = 0;
//    for (int i = 0; i < length; ++i)
//    {
//        cin >> b;
//        sum += b;
//    }
//    cout << sum << "\n";
//}


//输入的第一行包括一个正整数t(1 <= t <= 100), 表示数据组数。
//接下来t行, 每行一组数据。
//每行的第一个整数为整数的个数n(1 <= n <= 100)。
//接下来n个正整数, 即需要求和的每个正整数。
//int tLine = 0;
//cin >> tLine;
//for (int i = 0; i < tLine; ++i)
//{
//    int n = 0;
//    cin >> n;
//    int sum = 0, b = 0;
//    for (int i = 0; i < n; ++i)
//    {
//        cin >> b;
//        sum += b;
//    }
//    cout << sum << "\n";
//}

//输入数据有多组, 每行表示一组输入数据。
//每行的第一个整数为整数的个数n(1 <= n <= 100)。
//接下来n个正整数, 即需要求和的每个正整数。
//int n = 0;
//while (cin >> n)
//{
//    int sum = 0, b = 0;
//    for (int i = 0; i < n; ++i)
//    {
//        cin >> b;
//        sum += b;
//    }
//    cout << sum << "\n";
//}

//输入数据有多组, 每行表示一组输入数据。
//每行不定有n个整数，空格隔开。(1 <= n <= 100)。
//int sum = 0;
//int nu = 0;
//while (cin >> nu)
//{
//    sum += nu;
//    if (getchar() == '\n')
//    {
//        cout << sum << "\n";
//        sum = 0;
//    }
//}

//输入有两行，第一行n
//第二行是n个空格隔开的字符串
//int line = 0;
//cin >> line;
//string str;
//vector<string> vecOu;
//for (int i = 0; i < line; ++i)
//{
//    cin >> str;
//    vecOu.push_back(str);
//}

//多个测试用例，每个测试用例一行。
//每行通过空格隔开，有n个字符，n＜100
//string str;
//vector<string> vecOu;
//while (cin >> str)
//{
//    vecOu.emplace_back(str);
//    if (getchar() == '\n')
//    {
//        sort(vecOu.begin(), vecOu.end());
//        for (auto& it : vecOu)
//        {
//            cout << it << " ";
//        }
//        cout << "\n";
//        vecOu.clear();
//        continue;
//    }
//}

//多个测试用例，每个测试用例一行。
//每行通过','隔开，有n个字符，n＜100   a, c, bb
//string str;
//while (cin >> str)
//{
//    if (getchar() == '\n')
//    {
//        vector<string> vecOu;
//        SplitStringBySymbol(str, ",", vecOu);
//        cout << vecOu.back();
//        cout << "\n";
//    }
//}