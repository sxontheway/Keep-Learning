
# CmakeLists.txt
> 见 [SLAM14](../SLAM14/Chapter2.md)  

* 写 CmakeLists 的 6 步：
    * cmake版本
    * 项目名称 
    * 引入库
    * 定义环境变量；  
    * 添加可执行文件
    * 添加可执行文件需要的库

```cmake
#-----
# 1. cmake version
#-----
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


#-----
# 2. 项目名称一般和项目的文件夹名称对应，
#-----
project(myexe)


#-----
# 3. 引入库，有find，include 和 link
# find_package(PythonLibs 3.6 REQUIRED)：cmake自动找包；找到后就可以用 include_directories(${PYTHON_LIBRARY_DIRS})，和 link_directories(${PYTHON_LIBRARY_DIRS})
# include_directories()：添加头文件目录
# link_directories()：添加库文件目录
#-----
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
include_directories(/usr/include/python3.6)                 # include_directories(${PYTHON_LIBRARY_DIRS})
link_directories(/usr/local/lib/python3.6/dist-packages)    # link_directories(${PYTHON_LIBRARY_DIRS})
    

#-----
# 4. set environment variable，设置环境变量
编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
#-----
#for server
# set(CMAKE_PREFIX_PATH "~/libtorch-1.7.0/libtorch")
# set(CMAKE_PREFIX_PATH "./libtorch-1.7.0/libtorch")
#for edge node
set(Torch_DIR /usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch)
set(DCMAKE_BUILD_TYPE "Release")    # 设置编译模式


#-----
# 5. add executable file，添加要编译的可执行文件
#-----
aux_source_directory(. DIR_SRCS)    在目录中查找所有源文件（我们写的代码），并将名称保存到 DIR_SRCS 变量
add_executable(myexe ${DIR_SRCS})
add_executable(exec_reader evaluation/exec_reader.cpp)


#-----
# 6. add link library，添加可执行文件所需要的库
#-----
target_link_libraries(myexe "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
set_property(TARGET myexe PROPERTY CXX_STANDARD 14)
```

## 其他
> https://zhuanlan.zhihu.com/p/409257749 
* 语法使用
    ```cmake
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")   # 追加
    set(CMAKE_CXX_FLAGS "-O3")                      # 覆盖
    ```
* 常用变量  

    | 变量名 | 含义  | 
    | :------------ |:---------------:| 
    | PROJECT_NAME | 工程名 | 
    | CMAKE_BUILD_TYPE | 编译模式 | 
    | CMAKE_CXX_STANDARD | 使用的C++标准 | 
    | CMAKE_CXX_STANDARD_REQUIRED  | 是否强制使用指定的C++标准 | 
    | CMAKE_CXX_FLAGS | 编译参数 | 
    | CMAKE_CXX_FLAGS_DEBUG | ebug模式下的编译参数 | 
    | CMAKE_CXX_FLAGS_RELEASE | Release模式下的编译参数 | 

