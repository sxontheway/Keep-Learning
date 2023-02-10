# 1. 程序的编译，链接和执行
> https://www.cnblogs.com/skynet/p/3372855.html  
* 为什么要有编译和链接？  
因为大型程序从头编译一次需要很长时间，而我们往往需要在某一个文件上做一些改动。有了编译、链接的分布处理，只需要编译我们更改的那部分就好。  
经常从源码编译安装会有：`./configure`，`make`，`make install`三个命令
 
* 动态库和静态库的区别？   
Windows中静态、动态库后缀分别为：`.lib`，`.dll`；Linux中分别为：`.a`，`.so`。一个静态库可以简单看成是一组目标文件（`.o/.obj`文件）的集合，即很多目标文件经过压缩打包后形成的一个文件。  
静态库被使用目标代码最终和可执行文件在一起，而动态库与它相反，它并不在链接时将需要的二进制代码都“拷贝”到可执行文件中，而是仅仅“拷贝”一些重定位和符号表信息，这些信息可以在程序运行时完成真正的链接过程。  
所以通常，链接了静态库的可执行文件：所占磁盘空间大、拓展性和兼容性较差、但加载时更快。



* make，make install
    > https://www.cnblogs.com/tinywan/p/7230039.html    
    
    * configure 可配置安装路径等
    * make 对应编译，通常一个个输入命令太麻烦，会用 Makefile 控制代码
    * "undefined reference to" 问题 debug：https://blog.csdn.net/lovemysea/article/details/79520516



# 2. 指针的运用
## 2.1 函数指针 
首先回顾一下函数的作用：完成某一特定功能的代码块。  
再回忆一下指针的作用：一种特殊的变量，用来保存地址值，某类型的指针指向某类型的地址。  
下面定义了一个求两个数最大值的函数:
```C
int maxValue (int a, int b) {
    return a > b ? a : b;
}     
```
而这段代码编译后生成的CPU指令存储在代码区，而这段代码其实是可以获取其地址的，而其地址就是函数名，我们可以使用指针存储这个函数的地址——函数指针。
```C
int (*p)(int, int) = NULL;   // p就是一个函数指针
p = maxValue;
// 或p = &maxValue，用函数名加上取地址符也可以
p(20, 45);
```
### 2.1.1 进阶1
* `int (*pf(char *))(char *)`  
  * `pf`是一个函数名  
  * `*pf(char *)` pf函数接受char \*类型的参数，返回值为一个指针
  * `(*pf(char* ))()` 返回的指针指向一个函数，也即pf函数返回值为一个函数指针  
  * `int (*pf(char *))(char *)` 这个被pf返回值指向的函数，接受参数为char \*，返回值为int  
```C
// FUNC是一个函数指针。指向的函数的参数为两个int，返回值为int
typedef int (*FUNC)(int, int);

// 求最大值函数
int maxValue(int a, int b) {
    return a > b ? a : b;
}

// 求最小值函数
int minValue(int a, int b) {
    return a < b ? a : b;
}
// findFunction函数定义
FUNC findFunction(char *name) {
    if (0 == strcmp(name, "max")) {
        return maxValue;
    } else if (0 == strcmp(name, "min")) {
        return minValue;
    }

    printf("Function name error");
    return NULL;
}   

int main() {

    int (*p)(int, int) = findFunction("max");
    printf("%d\n", p(3, 5));

    int (*p1)(int, int) = findFunction("min");
    printf("min = %d\n", p1(3, 5));

    return 0;
}
```
> * 为什么要以函数pf去获取函数呢？直接使用maxValue和minValue不就好了么？  
其实在以后的编程过程中，很有可能maxValue和minValue被封装了起来，类的外部是不能直接使用的，我们需要findFunction这个返回值为函数指针的公有成员函数去调用私有成员函数。  

### 2.1.2 进阶2
* 函数指针数组的指针：`int (*(*pf)[3]) (int, int)`  
  * `(*pf)[3]` 是数组，包含有3个函数指针  
  * `pf` 则是函数指针数组的指针  
> `int p[4]` p是数组名  
> `int *p[4]`p是一个指针，指向一个包含4个int元素的数组
``
