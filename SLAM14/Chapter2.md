# 1. cmake和make
## 1.1 生成可执行文件基本流程
> 参见：https://blog.csdn.net/cxsydjn/article/details/79548984
* 编写.cpp、 CMakeLists.txt文件
* mkdir build (建build文件夹，将所有中间代码放到一个文件夹中，方便删除)
* cd build
* cmake .. （对`/CMakeLists.txt`文件进行cmake操作， 得到`/build/MakeFile`文件）
* make (对`/build/MakeFile`文件进行make操作，生成可执行文件)

CMakeList.txt文件如下：
```cmake
# 声明要求的 cmake 最低版本
cmake_minimum_required( VERSION 2.8 )

# 声明一个 cmake 工程
project( HelloSLAM )

# 设置编译模式
set( CMAKE_BUILD_TYPE "Debug" )

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
add_executable( helloSLAM helloSLAM.cpp )
```
## 1.2 使用库
在一个 C++ 工程中，并不是所有代码都会编译成可执行文件。**只有带有 main 函数的文件才会生成可执行程序**。而另一些代码，我们只想把它们打包成一个东西，供其他程序调用。这个东西叫做库。<br>
此时CMakeList.txt文件如下（多了add_library 和 target_link_libraries命令）：
```cmake
# 若cmake版本低于此处写的版本，cmake时会报错
cmake_minimum_required( VERSION 2.8 )

# 不能为空，否则cmake时会报错
project( HelloSLAM )

set( CMAKE_BUILD_TYPE "Debug" )

# 添加一个静态库
add_library( hello libHelloSLAM.cpp )
# 或选择用共享库
add_library( hello_shared SHARED libHelloSLAM.cpp )

# useHello.cpp中是main函数
add_executable( useHello useHello.cpp )

# 将库文件链接到可执行程序上
target_link_libraries( useHello hello_shared )
```
使用cmake编译整个工程命令同上：
```bash
mkdir build  
cd build
cmake ..  
make
```
Linux 中，库文件分成静态库和共享库（类比于Windows中的动态库）两种。静态库以`.a`作为后缀名，共享库以`.so`结尾。所有库都是一些函数打包后的集合，差别在于**静态库每次被调用都会生成一个副本，而共享库被不同的程序调用也只在内存生成一个副本**，更高效。  
<br>
头文件说明了这些库里都有些什么，接口是怎样。**对于库的使用者（写main函数的人），只要拿到了头文件和库文件，就可以调用这个库。**
## 1.3 自动寻找库
CMakeList.txt的一个例子：
```cmake
# 添加 c++ 11 标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 如果cmake能够找到它，就会提供头文件和库文件所在目录的变量
find_package(Caffe REQUIRED)

if (NOT Caffe_FOUND)
    message(FATAL_ERROR "Caffe Not Found!")
endif (NOT Caffe_FOUND)

include_directories(${Caffe_INCLUDE_DIRS})

add_executable(useSSD ssd_detect.cpp)
target_link_libraries(useSSD ${Caffe_LIBS})
```
find_package原理详见：https://www.jianshu.com/p/46e9b8a6cb6a  
CMakeLists.txt编写要点：https://blog.csdn.net/u011728480/article/details/81480668
## 1.4 关于删除源码
自己make install 之后，安装程序就会把需要的执行文件和库文件添加到必须的地方，所以很多make install需要root权限。安装之后当然可以删除这个目录，但之后如果你想删除这个软件的话，你需要自己去寻找都安装了那些文件，安装到了哪个目录里。这会很麻烦。  
一般来说，软件包里也会提供make uninstall，帮助你删除有关联的文件。  

------
<br><br>
