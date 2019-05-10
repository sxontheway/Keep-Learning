# 1. 指针的运用
## 1.1 函数指针 
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
### 1.1.1 进阶1
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

### 1.1.2 进阶2
* 函数指针数组的指针：`int (*(*pf)[3]) (int, int)`  
  * `(*pf)[3]` 是数组，包含有3个函数指针  
  * `pf` 则是函数指针数组的指针  
> `int p[4]` p是数组名  
> `int *p[4]`p是一个指针，指向一个包含4个int元素的数组
``
