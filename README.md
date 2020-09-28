# LearnMaterial
使用VS编译出错
如果是windows程序：
1.菜单中选择 Project->Properties, 弹出Property Pages窗口
2.在左边栏中依次选择：Configuration Properties->C/C++->Preprocessor,然后在右边栏的Preprocessor Definitions对应的项中删除_CONSOLE, 添加_WINDOWS.
3.在左边栏中依次选择：Configuration Properties->Linker->System,然后在右边栏的SubSystem对应的项改为Windows(/SUBSYSTEM:WINDOWS)
如果是控制台程序：
1.菜单中选择 Project->Properties, 弹出Property Pages窗口
2.在左边栏中依次选择：Configuration Properties->C/C++->Preprocessor,然后在右边栏的Preprocessor Definitions对应的项中删除_WINDOWS, 添加_CONSOLE.
3.在左边栏中依次选择：Configuration Properties->Linker->System,然后在右边栏的SubSystem对应的项改为CONSOLE(/SUBSYSTEM:CONSOLE)


# 教程
1. 首先去茹龙仓库clone ： git clone https://github.com/RL-PCGit/LearnMaterial.git
2. github去fork茹龙仓库
3. git remote add xxx 自己的仓库地址，xxx是自己远程仓库的别名
4. 如果原仓库有修改
    4.1 git pull origin master
	4.2 git push xxx master
5. 如果自己有更改
    5.1 git push xxx master
	5.2 去自己的仓库做pull request
