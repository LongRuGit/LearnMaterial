# LearnMaterial
编译出错
如果是windows程序：
1.菜单中选择 Project->Properties, 弹出Property Pages窗口
2.在左边栏中依次选择：Configuration Properties->C/C++->Preprocessor,然后在右边栏的Preprocessor Definitions对应的项中删除_CONSOLE, 添加_WINDOWS.
3.在左边栏中依次选择：Configuration Properties->Linker->System,然后在右边栏的SubSystem对应的项改为Windows(/SUBSYSTEM:WINDOWS)
如果是控制台程序：
1.菜单中选择 Project->Properties, 弹出Property Pages窗口
2.在左边栏中依次选择：Configuration Properties->C/C++->Preprocessor,然后在右边栏的Preprocessor Definitions对应的项中删除_WINDOWS, 添加_CONSOLE.
3.在左边栏中依次选择：Configuration Properties->Linker->System,然后在右边栏的SubSystem对应的项改为CONSOLE(/SUBSYSTEM:CONSOLE)