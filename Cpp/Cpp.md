Cpp手册： https://zh.cppreference.com/w/
   * [1. 面向对象的特性](#1-面向对象的特性)
      * [1.1 继承和组合](#11-继承和组合)
      * [1.2 虚函数](#12-虚函数)
   * [2. C++的一些细节](#2-c的一些细节)
      * [2.1 struct和class](#21-struct和class)
      * [2.2 冒号与双冒号](#22-冒号与双冒号)
      * [2.3 &lt;iostream.h&gt;和&lt;iostream&gt;](#23-iostreamh和iostream)
      * [2.4 #include中""和&lt;&gt;的区别](#24-include中和的区别)
      * [2.5 一条语句分多行写](#25-一条语句分多行写)
   * [3. 动态内存管理](#3-动态内存管理)
      * [3.1 智能指针](#31-智能指针)
         * [3.1.1 shared_ptr](#311-shared_ptr)
            * [3.1.1.1 概述](#3111-概述)
            * [3.1.1.2 初始化方式](#3112-初始化方式)
            * [3.1.1.3 make_shared 和 shared_ptr 的区别](#3113-make_shared-和-shared_ptr-的区别)
            * [3.1.1.4 使用注意事项](#3114-使用注意事项)
            * [3.1.1.5 常用成员函数](#3115-常用成员函数)
      * [3.2 分配器](#32-分配器)
   * [4. 标准模板库(STL)](#4-标准模板库stl)
      * [4.1 容器](#41-容器)
         * [4.1.1 vectors顺序容器](#411-vectors顺序容器)
---
<br>

# 1. 面向对象的特性
## 1.1 继承和组合
组合：一个类里面的数据成员是另一个类的对象。创建对象时既要对自己本身的成员进行初始化，又要对内嵌对象初始化。
```C++
class A
{
    int a, b;		    // class构造的成员默认为private
public:
    A(int i, int ii) : a(i), b(ii) { }	// 构造函数中用冒号赋值，i赋给a，ii赋给b
    ~A() { }
    void f() const { }	   // 成员函数(只能修改const成员变量，只能调用const成员函数）
};

class B 
{
    int i;
public:
    B(int ii) : i(ii) { }
    ~B() { }
    void f() const { }
};

class C : public B      // C继承了B
{	
    A aa;		        // C组合了A
public:
    C(A a, int i, int ii) : aa(a), B(ii) { }	// 用ii构造B，a赋值给aa
    ~C() { } 		    // 会调用~A(), ~B()
    void f() const {  	
        aa.f();		    // 组合用一个句点
        B::f();		    // 继承用双冒号
     }
};

int main() 
{
    A a(1, 2);
    C c(a, 1, 2);	    // 对象c将类C实例化了
} 
```
## 1.2 虚函数
Without "virtual" you get "early binding". Which implementation of the method is used gets decided at compile time based on the type of the pointer that you call through.

With "virtual" you get "late binding". Which implementation of the method is used gets decided at run time based on the type of the pointed-to object - what it was originally constructed as. This is not necessarily what you'd think based on the type of the pointer that points to that object.
```C++
class Base
{
  public:
            void Method1 ()  {  std::cout << "Base::Method1" << std::endl;  }
    virtual void Method2 ()  {  std::cout << "Base::Method2" << std::endl;  }
};

class Derived : public Base
{
  public:
    void Method1 ()  {  std::cout << "Derived::Method1" << std::endl;  }
    void Method2 ()  {  std::cout << "Derived::Method2" << std::endl;  }
};

Base* obj = new Derived ();
  //  Note - constructed as Derived, but pointer stored as Base*

obj->Method1 ();  //  Prints "Base::Method1"
obj->Method2 ();  //  Prints "Derived::Method2"
```
------
<br><br>

# 2. C++的一些细节
## 2.1 struct和class
C++的struct和C中的struct差别很大:    
  * C中，struct只能包含成员变量
  * 在C++中，struct类似于class，既可以包含成员变量，又可以包含成员函数
  
C++中，struct和class有以下区别：
  * struct默认所有成员为public，class默认所有成员为private
  * class 可以使用模板，而 struct 不能。
## 2.2 冒号与双冒号
```C++
class TrackingNodelet : public nodelet :: Nodelet{...}    // TrackingNodelet类继承了nodelet命名空间中的Nodelet类
::a	           // 双冒号时全局作用域符号，调用全局变量a
```
## 2.3 <iostream.h>和\<iostream>
  * `#include <iostream.h>`属于C遗留下来的东西。当代码中用`#include <iostream.h>`时，输出可直接引用`cout<<x`，因为C中未引入namespace的概念。
  * `#include <iostream>`是C++中标准输入输出流。当使用`#include<iostream>`时，输出需要引用`std::cout<<x`。
  * 当调用`std::cout<<"abc";`时，实际上调用了`ostream& operator<<(ostream &temp, char *ps);`这个运算符重载函数。
    * 其中，cout一个是ostream类型的对象。由于返回的是流对象的引用，引用可以作为左值使用，所以当程序中有类似`cout<<"abc"<<"abc";`这样的语句出现的时候，就能够构成连续输出。
    * `>>a`表示将数据放入a对象中；`<<a`表示将a对象中存储的数据拿出。 
    * 例如`cout<< s << "abc"`先将"abc"拿出，放入s中。再从s中拿出，放入cout中。cout是标准输出流对象，对应设备是屏幕，故最终得到输出。
  * cerr：输出到标准错误的ostream对象，**常用于程序错误信息**，和cout作用类似，有点不同就是cout通常是传到显示器输出，但可以被重定向输出到文件，而cerr流中的信息只能在显示器输出。此外，cerr 不被缓冲，也就说错误消息可以直接发送到显示器，而无需等到缓冲区或者新的换行符时，才被显示。
  
## 2.4 #include中""和<>的区别
* #include "" 首先在当前目录下寻找，若查找成功，则遮蔽 #include <> 所能找到的同名文件；如果找不到，再到系统目录中寻找。#include "" 一般用于自定义的头文件。
* #include <> 先去系统目录中找头文件，如果没有在到当前目录下找，所以像标准的头文件 stdio.h、stdlib.h 等用这个方法。

## 2.5 一条语句分多行写
“/”是续行符，用续行符之后，下一行不能有空格或缩进。一些情况下，可以不用续行符直接换行，因为换行在编译器看来只不过是一种空白字符，在做语法解析时所有空白字符都被丢弃了。
```C++
//一个用续行符的极端例子
std::cou\
t<<"hello world"<<std::endl;

//不用续行符直接换行的例子
A::A(int x, double y):
    X(x),
    Y(y)
{
    //类A的构造函数的具体内容
}

```
------
<br><br>

# 3. 动态内存管理
## 3.1 智能指针
### 3.1.1 shared_ptr
#### 3.1.1.1 概述  
>当编译器为C++11及之后的版本时，程序开头 `#include\<memory>`，中间类似`std::shared_ptr<int>`即可  
>当编译器为C++11之前版本,要用Boost库。程序开头`#include<boost/smart_ptr.hpp>`，中间用`boost::shared_ptr`
    
智能指针主要为了方便指针的内存管理，其基本原理是记录资源所有者个数（引用计数）。  
当个数为 0 的时候，也即最后一个指向某对象的共享指针析构时（生命周期结束或用reset函数析构），对象所占的内存就会被自动释放掉。  
一个例子：
```C++
int* p = new int(233);	
boost::shared_ptr<int> sp(p); //在程序结束时, p所指的内存会自动释放
```
#### 3.1.1.2 初始化方式
  ```C++
  // 方式1：用指向在堆上分配的内存空间的指针初始化
  int *p = new int(30);
  std::shared_ptr<int> p1(p);
  
  // 方式2：用已有的智能指针初始化，拷贝构造函数或运算符重载
  std::shared_ptr<T> p2(p1);            // 使用拷贝构造函数创建p2    
  std::shared_ptr<T> p3;  p3 = p2;      // 使用重载的"="创建p3    
 
  // 方式3：最推荐最安全的方式，用模板函数 std::make_shared，可以返回一个指定类型的std::shared_ptr
  std::shared_ptr<int> p4 = std::make_shared<int>(20);
  ```
#### 3.1.1.3 make_shared 和 shared_ptr 的区别
>std::shared_ptr<StructA> pA1(new StructA());  
>shared_ptr 分两步：1.执行数据体的内存申请  2.执行控制块的内存申请  
    
>std::shared_ptr<StructA> pA2 = std::make_shared<StructA>();   
>make_shared 数据体和控制块的内存一块申请  

总的来说，make_shared方式比shared_ptr更安全和高效，  
参见 http://bitdewy.github.io/blog/2014/01/12/why-make-shared/

#### 3.1.1.4 使用注意事项
  * shared_ptr多次引用同一数据，会导致多次释放同一内存。
  ```C++
  // 错误做法
  int* p = new int(100);
  shared_ptr<int> sp1(p);
  shared_ptr<int> sp2(p);
  
  // 正确做法
  int* p = new int[100];
  shared_ptr<int> sp1(p);
  shared_ptr<int> sp2(sp1);
  ```
  * shared_ptr循环引用导致内存泄露(此处用到 weak_ptr)  
  >https://blog.csdn.net/Gykimo/article/details/8728735  
  >https://stackoverflow.com/questions/12030650/when-is-stdweak-ptr-useful#
  ```C++
#include <iostream>
#include <memory>

int main()
{
    /**** 
        OLD, problem with dangling pointer
        PROBLEM: ref will point to undefined data!
    ****/
    int* ptr = new int(10);
    int* ref = ptr;
    delete ptr;

    /**** 
        NEW SOLUTION: 
        check expired() or lock() to determine if pointer is valid
    ****/
    std::shared_ptr<int> sptr;
    sptr.reset(new int);
    *sptr = 10;

    // get pointer to data without taking ownership
    std::weak_ptr<int> weak1 = sptr;
    
    // deletes managed object, acquires new pointer
    sptr.reset(new int);
    *sptr = 5;

    // get pointer to new data without taking ownership
    std::weak_ptr<int> weak2 = sptr;
    
    // weak1 is expired due to the reset.
    if(auto tmp = weak1.lock())
        std::cout << *tmp << '\n';
    else
        std::cout << "weak1 is expired\n";

    // weak2 points to new data (5)   
    if(auto tmp = weak2.lock())
        std::cout << *tmp << '\n';
    else
        std::cout << "weak2 is expired\n";
} 
```
- weak_ptr必须从一个share_ptr或另一个weak_ptr转换而来。  
- 这也说明，进行该对象的内存管理的是那个强引用的share_ptr，weak_ptr只是提供了对管理对象的一个访问手段。weak_ptr除了对所管理对象的基本访问功能（通过get()函数）外，还有两个常用的功能函数：
```
expired()       // 用于检测所管理的对象是否已经释放  
lock()          // 用于获取所管理的对象的强引用指针但是这个东西在Mac和Linux很友好，windows似乎不那么友好。不过这里也给了windows的解决方案。
```
#### 3.1.1.5 常用成员函数
  
|执行操作|成员函数|
|:----------:|:-----------:|
|查看资源的所有者个数|ptr.use_count()|
|销毁对象|ptr.reset()|
|重新指向new_ptr|ptr.reset(new_ptr)|

<br>

## 3.2 分配器
* new 的工作步骤分为两步： 1.申请原始内存 2.执行构造函数。`std::allocator<T>`提供了一种更高级的内存的分配和构造方法，能够将new的两部拆分：
```cpp
int *p = new int(10);   // 包含了memory allocation和construction两步
delete [] p;   // 包含了memory deallocation和destruction两步

std::allocator<int> a1;   // int 的默认分配器
int* a = a1.allocate(1);  // 一个 int 的空间
a1.construct(a, 7);       // 构造 int
```
  
* Allocation和Construction的分离在构造容器时很有用。  
> If vector use `new X()` to create the object, it would have to reallocate every time you wanted to add or 
remove an element, which would be terrible for performance.
```cpp
std::vector<X> v;
v.reserve(4);        // (1)
v.push_back( X{} );  // (2)
v.push_back( X{} );  // (3)
v.clear();           // (4)  
```

  
------
<br><br>

# 4. 标准模板库(STL)
## 4.1 容器
### 4.1.1 vectors顺序容器
  * 使用: 
```C++
vector <int> v1;                // 定义vector，int可换成其他类型，如string或自定义结构体等
vector <int>::iterator it;      // 定义一个vector迭代器
```
  * 常用操作：
```C++但是这个东西在Mac和Linux很友好，windows似乎不那么友好。不过这里也给了windows的解决方案。
v1.push_back    // 在数组的最后添加一个数据		
v1.pop_back()   // 去掉数组的最后一个数据 
v1.front()      // 返回第一个元素(栈顶元素)		
v1.begin()      // 得到数组头的指针，用迭代器接受
v1.end()        // 得到数组的最后一个单元+1的指针，用迭代器接受
v1.clear()      // 移除容器中所有数据		
v1.empty()      // 判断容器是否为空
v1.erase(pos)   // 删除pos位置的数据			
v1.erase(beg,end)	    // 删除[beg,end)区间的数据
v1.size()       // 回容器中实际数据的个数		
v1.insert(pos, data)    // 在pos处插入数据
```
  * 二维容器: 	 
```C++
vector< vector<int> > a;        // 在 C++98 中，两个>>之间的空格是必不可少的，构建了一个的二维vector对象；C++98之后可以不用隔开
vector< vector<int> > a(3, vector<int>(4)); 	// 知道维度时，可以指定维度，如3*4
```
