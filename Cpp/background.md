# python 和 C++ 联合调试
> https://zhuanlan.zhihu.com/p/495551122

用 python 写前端，调用了 c/c++ 编写的库，想 debug 追踪进 c 的代码：[实例代码](./python_c_debug/)

## 步骤
* `g++ --std=c++11 -g --shared myadd.cpp -o myadd.so`
* 创建 `launch.json` 
    * `"additionalSOLibSearchPath": "/home/alex/Desktop/python_c_debug",` 需要写绝对路径
    * "program": "你使用的python环境对应的python"
    * python的调试配置里cwd要设置对，否则python代码会找不到so文件
* 调试
    * 先配置 python 的调试，断点放在调用 lib 的位置
    * 运行 gdb 调试，选择进程。在跳出来的框框中搜索运行的 python 文件名，选择带 access token的那个进程，加断点。然后点调试步进，c++ 的调试器就会在断点卡住。

* Debug
    * `Authentication is needed to run `/usr/bin/gdb' as the super user`: https://zhuanlan.zhihu.com/p/450589971 
    * `How can I solve "Unable to open 'raise.c' " Error?(VSCODE , LINUX)`: https://stackoverflow.com/questions/59126945/how-can-i-solve-unable-to-open-raise-c-errorvscode-linux  
    在 CMakeList.txt 中加：
        ```cmake
        SET(CMAKE_BUILD_TYPE "Debug")
        # SET(CMAKE_BUILD_TYPE "Release")
        SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
        SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
        ```
        * 其中 `O0` `O3` 是代码优化级别，越高优化的越多（循环展开, 函数内联等），编译越耗时
        * `Wall`：显示所有 warning
        * `ggdb`：

        