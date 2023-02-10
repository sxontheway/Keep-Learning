# 1. 基础
## 1.1 python包安装的路径
`sudo apt-get install`，`pip install`，`sudo pip install`，`pip install –user`，`pip3 install`，`sudo pip3 install`，`pip3 install –user`几种安装方式的安装路径不同。  
* 其中pip和pip3主要是python版本的区别，下表只以`pip`为例。   
  * `pip3`时，将路径中的`python2.7`换成`python3.5`即可。   
  * `dist-packages`是从包管理器安装的python包的路径； `site-packages`是从第三方库安装的的python包的路径
  * 可用 `pip show` 查看

   | 命令 | python包安装路径 | 说明 |
   | :------------ |:---------------|:---|
   | sudo apt-get install | /usr/lib/python2.7/dist-packages |apt-get install必须要sudo|
   | sudo pip install | /usr/local/lib/python2.7/site-packages | 用sudo可能会报错：cannot import name 'main'|
   | pip install | ～/.local/lib/python2.7/site-packages |作用同下，但可能会报错： Permission denied|
   | pip install --user| ～/.local/lib/python2.7/site-packages |--user 使包安装在当前用户文件夹下|

* 批注：
  * `usr` 很多人都认为是 `user` 缩写，其实不是，是 `unix system resource` 缩写
  * `/lib` 是内核级的；`/usr/lib` 是系统级的；`/usr/local/lib` 是用户级的，主要存放一些用户自己安装的软件
  * `sudo pip install` 是为所有用户安装，`pip install --user` 是只为本用户安装

* 安装有多个python版本时，每个版本可能都对应一个pip库，导致环境混乱
  > 此时可以手动指定使用哪个python版本进行包安装：`python3.6 -m pip install XXXX` 
  
  * 常见命令：
    * `pip list -V`：查看所有pip包安装路径
  * 如果确实有多个版本 python，例如同时有 3.6 和 3.8，可以用如下方法指定 `pip3 install XXX` 对应的 python 版本
    * 查看 `which pip3`，`which python3`；将 `which pip3` 的输出 `/home/xxx/.local/bin/pip3` 文件中第一行改为想要的 python 版本路径，例如 `/usr/bin/python3.6`，即指定了 pip3 默认的 python 版本。见 https://blog.csdn.net/toopoo/article/details/99956326  

* 在.py中输入
  ```python
  import sys,pprint;  
  pprint.pprint(sys.path)   # 输出一个换一行
  ```
  可查看环境变量，环境变量顺序对应于包的导入顺序
* 假设虚拟环境路径为`~/venv`， python3虚拟环境中安装的包，可以在`~/venv/lib/python3.5/`中找到，但这些包只能在该虚拟环境下调用
  * 例如要想在`jupytor-notebook`中能import在venv中安装的包，`jupytor-notebook`本身也需要安装在venv内

## 1.2 python虚拟环境
> 参见： [虚拟环境是什么?](https://ningyu1.github.io/site/post/63-python-virtualenv/)  

* 虚拟环境(venv)工具，可以避免不同项目所需的python版本不同而引起的问题  

    ___在默认情况下，所有安装在系统范围内的包对于venv都是可见的，在venv中安装的包，就只能在venv中用。 当没有在venv找到对应软件包时，还是会全局环境找。___ 例如对于Ubuntu16.04， python3的venv只代表优先使用python3， 在python3的包没找到的时候还是会找python2的包， 用venv可实现python2、 3的的混合编程。    
    以上这种行为可以被更改：在创建venv时增加`--no-site-packages`选项，venv就不会读取系统包，会一个完全独立和隔离的Python环境。  
    ```bash
    # Install
    sudo pip3 install -U virtualenv

    # Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it:
    virtualenv --system-site-packages -p python3 ./venv

    # Activate the virtual environment using a shell-specific command:
    source ./venv/bin/activate  

    # Check the installation
    pip install --upgrade pip
    pip list
    python --version        # 显示3.x， 也即虚拟环境下只安装了python3
    python3 --version       # 显示3.x
    python2 --version       # 还是会显示2.x。 Python3虚拟环境没找到包，还是会去Python2的环境里面找

    # Quit the vitual env
    deactivate              
    python --version        # 显示2.x
    ```

* 对于python环境而言，`pip + venv` 比 `Anaconda` (https://blog.csdn.net/lwgkzl/article/details/89329383) 稳定，Anaconda出现一些奇奇怪怪的毛病

## 1.3 慎用相对路径
* 在.py文件中，相对路径（./等）始终是相对于终端位置本身的，例如：
    ```bash
    cd /home/Desktop
    python3 /home/alex/t.py
    ```
    在`/home/alex/t.py`中用相对路径，那么系统会在`/home/Desktop`中找，而不是`/home/alex`中找。这和我们的所设想的不同。

* 解决方案：获取该脚本的绝对路径->得到文件夹->join文件夹和目标文件相对路径，得到绝对路径，见 https://www.jianshu.com/p/76a3d317722c
    ```python
    import os, sys

    # 从命令行的输入参数 (/home/alex/t.py) 获取目录的绝对路劲
    print('realpath', os.path.realpath(sys.argv[0]))  
    # 分割得到文件夹名
    print(os.path.split(os.path.realpath(sys.argv[0]))[0])

    # 用__file__得到绝对路径
    print(os.path.realpath(__file__))
    # 得到相对于命令行目录的文件夹路径    
    print(os.path.dirname(__file__))
    ```
* `os.path.expanduser(~/Desktop/a.txt)` 可将用户目录下的路径转换为根目录下的路径

## 1.4 import
### 1.4.1 相对导入和绝对导入  
> https://medium.com/@alan81920/python-import-%E7%B0%A1%E6%98%93%E6%95%99%E5%AD%B8-c98e8e2553d3  
> https://blog.csdn.net/u010138758/article/details/80152151

相对导入和绝对导入：

| 导入方式 | 相对导入 | 绝对导入 |
| :------------ |:---------------|:---|
| 格式 | `from .A import B` 或 `from ..A import B`。其中`.`代表当前模块，`..`代表上层模块，依次类推（或需要配合`python -m`使用)|`import A.B` 或 `from A import B`|
| 优点 | 相对导入可以避免硬编码，更改包名后，代码仍然可以运行，可维护好 | 可读性好。绝对导入可以避免与标准库命名的冲突，实际上也不推荐自定义模块与标准库命令相同。|
| 以哪个目录为参考目录 | import语句所在的文件，其所属package的目录 |当前工作目录（例如在命令行输入 `python3 a.py`时，文件`a.py`所在的目录）|
|要求| 必须要在package内，所谓的package，就是包含 `__init__.py` 文件的目录 | 有没有package都可以使用|

* package 内 `__init__.py` 文件的作用  
  通常`__init__.py`文件为空，但是我们还可以为它增加其他的功能。我们在导入一个包时(例如`import numpy`)，实际上是执行了它的`__init__.py文件`。这样我们可以在`__init__.py`文件中批量导入我们所需要的模块，而不再需要一个一个的导入。例如有如下文件结构：
  ```
  |-- pack_a
      |-- __init__.py：from .functions import f_a, f_b
      |-- functions.py：定义了 f_a, f_b
  ```
  那么想要使用`f_a`和`f_b`，只用 `import pack_a`，而省去了`from pack_a.functions import f_a, f_b`

### 1.4.2 python import 的规则
python在import时，可以按三种方式找：  
* 按绝对导入方式查找包
* 有 package 时，按相对导入方式查找包
* 到 `sys.path` 变量给出的目录列表中查找，例如`/usr/lib/python3.X`等

### 1.4.3 python import 的例子

```
-- src
    |-- train.py: from utils.dataset import * 
    |-- utils
        |-- __init__.py
        |-- dataset.py
        |-- utils.py
```
当前工作目录： `/src`,  
命令行输入：`python3 train.py`,  
想要在`dataset.py`中 `import utils.py`的函数，有两种方法： 
* 使用绝对导入：`from utils.utils import *`，使用`from utils import *` 是行不通的，因为没有`/src/utils.py`) 
* 使用相对导入：`from .utils import *`

### 1.4.4 其他
* import 其他文件夹下的module  
  > http://www.361way.com/python-import-dif-dir-module/4064.html  
  > https://medium.com/@alan81920/python-import-%E7%B0%A1%E6%98%93%E6%95%99%E5%AD%B8-c98e8e2553d3 

  程序结构：  
    ```
    -- src
        |-- __init__.py
        |-- main.py: from a.a import all
        |-- a
            |-- __init__.py
            |-- a.py
        |-- test
            |-- __init__.py
            |-- test.py
    ```
    当前工作目录： `/src`,  
    命令行输入：`python3 main.py`  
    如果`a.py`中想要引入`test.py`，需要的做法如下：  
    ```python
    import sys
    sys.path.append("..")
    from test.test import *
    ```
    直接使用 `from ..test.test import *` 会导致报错：`attempted relative import beyond top-level package`。是因为`src`本身这个 package 并没有被记录下来，见：https://stackoverflow.com/questions/30669474/beyond-top-level-package-error-in-relative-import

* python -m 参数  
像运行脚本一样运行模块，有以下两种用途：https://a7744hsc.github.io/python/2018/05/03/Run-python-script.html  
    * 更方便地运行库模块  
    例如： `python -m http.server` 等效于运行 `python3 /usr/lib64/python3.6/http/server.py`   
    因为 `sys.path` 中包含 `/usr/lib64/python3.6`，所以 python 可以定位到 `python3 /usr/lib64/python3.6/http/server.py` 这个文件（见1.4.2中 python import 的顺序）
    * 解决上面的从 `a.py` import `test.py` 的问题  
    通过 `python -m` 执行一个脚本时，会将当前路径加入到系统路径中；而使用 `python xxx.py` 执行脚本，则会将脚本所在文件夹加入到系统路径中。
    继续上面一节的例子，若要想执行 `a.py`（假如a.py中有 main 函数），只用 `python -m a.a` 即可，因为用了 `-m` 会自动把 `''`（也即 `./src`）加入 `sys.path`

* \_\_all__ 和 import * 
  ```python
  ### 文件foo.py ###
  __all__ = ['x', 'test']
  x = 2
  y = 3
  def test():
      print('test')
  ```
  ```python
  ### 文件main.py ###
  from foo import *
  print('x: ', x)
  print('y: ', y)     #此句会报错，因为没有导入y
  test()

  # `import *`表示导入import all；但`__all__`指定了能够被导入的部分（例如 y 就没有被包含）
  ```

* from \_\_future__ import  

    用于导入之后版本的一些功能。  
    例如，在开头加上`from __future__ import print_function`这句之后，即使在python2.X，
    print就可以像python3.X那样加括号使用（python2.X中print不需要括号，而在python3.X中则需要） 


## 1.5 \*args和\*\*kwargs
```python
def test(a,*args,**kwargs):
    print a
    print args
    print kwargs

test(1,2,3,d='4',e=5)
# 输出结果：
# 1
# (2, 3)
# {'e': 5, 'd': '4'}
 ```
意思就是1还是参数a的值，args表示剩余变量的值，kwargs在args之后表示成对键值对。

## 1.6 if \_\_name__ == '\_\_main__':
每个python模块（python文件）都包含一些内置的变量，其中：
* `'__main__'`等于当前执行文件的名称（包含后缀.py)  
* `__name__`
    * 当.py文件被直接运行的时候，`__name__`等于该文件名（包含后缀.py）
    * 当作为模块被import到其他文件中时，则`__name__`等于模块名称（不包含后缀.py）
    
所以，当.py文件被直接运行时，`if __name__ == '__main__'`为真；当.py文件被作为模块导入其他文件时，`if __name__ == '__main__'`为假。


## 1.7 扁平结构比嵌套结构好
 ```python
import numpy as np
a = np.array([-1,2,3,-4,5])
a = [x+3 if x<0 else x for x in a]  # 得到[2, 2, 3, -1, 5]
```

## 1.8 继承object类
见： https://www.zhihu.com/question/19754936  

属于历史遗留问题，继承 object 类的是新式类，不继承 object 类的是经典类。在Python3中已经不存在这个问题，object已经作为所有东西的基类了。

## 1.9 异常处理
> 见： https://blog.csdn.net/u012609509/article/details/72911564  

`with` 和 `try except`功能不同，with语句只是帮忙关闭没有释放的资源，并且抛出异常，但是后面的语句是不能执行的。  
`with` 语句通常只使用欧冠语与系统资源或执行环境相关的对象。为了即能够输出我们自定义的错误信息，又能不影响后面代码的执行，还得必须使用`try except`语句

* with
    ```python
    with clause as B:
        do_something()
    ```

    * 首先，clause 会被求值，返回一个对象，称它为A。该对象的`__enter__()`方法被调用，`__enter__()`方法的返回值将被赋值给as后面的变量 B
    * 当 `do_something()` 全部被执行完之后，将调用 A 的`__exit__()`方法，其会自动释放资源。
    * 在 with 后面的代码块 (clause 或 do_something())抛出异常时，`__exit()__`方法也被执行，帮忙释放资源 

* try except
    > https://stackoverflow.com/questions/18675863/load-data-from-python-pickle-file-in-a-loop
    
    如果 try 中出现 Error，控制权将被传递给相应的 except 代码块中的代码（如果有的话）。异常处理完毕后，程序将继续执行紧跟在最后一个 except 代码块后面的语句，而不会返回到发生异常的位置。例如下面的代码中，如果 try 中出现 EOFError，将会执行 except 中的代码。 
    ```python
    import pickle as pkl

    # 不用担心 pkl.load() 已经读取完毕报错
    def pickleLoader(pklFile):
        try:
            while True:
                yield pkl.load(pklFile)
        except EOFError as e:
            print(e)

    with open(filename) as f:
        for event in pickleLoader(f):
            do_something()
    ```

* raise  
raise 用于手工引发异常
    ```python
    try:
        raise EOFError("no more content")
    except EOFError as e:
        print(type(e), e)

    # 输出 <class 'EOFError'> no more content
    ```


---
<br>

# 2. 方法
> 用`dir()`函数可以查看对象上可用的方法
## 2.1 super()和__call__()方法
> https://www.zhihu.com/question/20040039
* super(): 使用继承时，基类的函数不会自动被调用。需要手动调用，例如调用基类的构造函数：`super().__init__()`  
    * Python3.x 和 Python2.x 语法有区别: Python2 中用`super(Class, self).xxx`，Python3 中简化了，可以用`super().xxx`  
    * 多继承时， `super(A, self).func` 执行的是MRO中的下一个类的 func；MRO 全称是 Method Resolution Order，它代表了类继承的顺序。老老实实地用类名去掉就不会出现这种问题。
        ```python
        class C(A,B):
        def __init__(self):
            A.__init__(self)
            B.__init__(self)
        ```
    * `self` 是首先调用自身的方法，如果自身没有再去父类中，`super` 是直接从父类中找方法
* \_\_call__()： 如果在创建class的时候写了`__call__()`方法，那么该class实例化出实例后，`实例名()`就是调用`__call__()`方法。`__call__()`方法使实例能够像函数一样被调用。
    ```python
    # Python2 代码
    class Person(object):
        def __init__(self, name, gender):
            self.name = name
            self.gender = gender
        def __call__(self, friend):  
            print 'My name is %s' % self.name
            print 'My friend is %s' % friend

    # 定义Student类时，只需要把额外的属性加上，例如score：
    class Student(Person):
        def __init__(self, name, gender, score):
            super(Student, self).__init__(name, gender)  # 若不写这一句，子类将缺失name和gender属性
            self.score = score

    p = Person('Bob', 'male')
    p('Tim')        # 调用了__call__方法

    # 输出结果：
    # My name is Bob
    # My friend is Tim
    ```

## 2.2 enumerate()方法
对于一个可遍历的对象，numerate()方法将其组成一个索引序列，利用它可以同时获得索引和值。
```python
list1 = list("abc")
for index, item in enumerate(list1, 3):    # 第二个参数表明从index从3开始
print index, item

# 输出结果
# 3 a
# 4 b
# 5 c
```
对于字典，获取键值对：
```python
for key,values in  dict.items():
    print key,values
```

## 2.3 @staticmethod，@classmethod
> https://www.zhihu.com/question/49660420  

### 普通方法，实例方法，类方法，静态方法
* 普通方法：只能用类名调用，传参时无 self
* 实例方法：只能用实例名调用，传递的参数第一个是self
* 静态方法：@staticmethod，可同时使用类名或实例名调用
* 类方法：@classmethod，类方法只与类本身有关而与实例无关，一般用于继承的情况，可以将类作为对象传入函数：https://www.zhihu.com/question/20021164/answer/676780051 

```python
class Num:
    def one():      # 普通方法：能用Num调用而不能用实例化对象调用    
        print ('1')

    def two(self):  # 实例方法：能用实例化对象调用，而不能用Num调用
        print ('2')

    @staticmethod   # 静态方法：能同时用Num和实例化对象调用
    def three():   
        print ('3')
        # Num.two()   # 报错
  
    # 类方法：第一个参数cls是什么不重要，都是指Num类本身，调用时将类作为对象传入方法 
    @classmethod  
    def go(cls):     
        cls().two() 
        cls.two(cls)  

Num.one()      # 1
# Num.two()    # Error
Num.three()    # 3
Num.go()       # 2 2

i = Num()                 
# i.one()          # Error         
i.two()            # 2       
i.three()          # 3
i.go()             # 2 2
```

### 总结：
* `cls` 对应的是类本身，`self` 对应的是类实例化后的对象
* `@staticmethod` 和普通的方法区别不太大，跟它所属的类没有必然联系，将一个 `staticmethod` 放到类的外部也是可以实现一样的功能，之所以放到类里面，是便于管理
* `staticmethod`，`classmethod` 都不需要实例化，直接用`类名.方法名()`来调用，但区别是：`classmethod` 因为传入了一个 `cls`，可以调用类中定义的其他方法（例如`two()`），但 `staticmethod` 不行





## 2.4 一些处理文本的方法
|方法|用途|
| :------------ | :-----|
|split()|将字符串按空格分割，返回分割后的列表|
|lstrip(), rstrip()|去掉左右空格|
|join()|`str = "-"; seq = ("a", "b", "c"); print(str.join(seq))`，得到`a-b-c`|
|repr()|输入为对象，返回一个对象的 string 格式，见 https://www.jianshu.com/p/2a41315ca47e|
|str.format()|使字符串格式化，例如`"{1}, {0}, {1}".format("hello", "world")`>>>`'world, hello, world'`|
|f"{ }"|f-string，在 python3.6 中才引入，见 https://blog.csdn.net/sunxb10/article/details/81036693|

---
<br>

# 3. 其他特性
## 3.1 多变量赋值
```python
a, b = b, a        # 交换a，b

# 慎用连等，这会使得a,b,c都是同一个对象的引用。改变其中一个，其他两个也会被改变
a = b = c = 1    
a, b, c = 1, 2, "john"
```  

## 3.2 生成器和迭代器
### 3.2.1 生成器
* 为什么要有生成器？  

    固然很多时候，我们可以直接创建一个列表，但受到内存限制，列表容量肯定是有限的。而且创建一个包含100万个元素的列表，不仅占用很大的存储空间，而且如果我们仅仅需要访问前面几个元素，那后面绝大多数元素占用的空间都白白浪费了。所以，如果列表元素可以按照某种算法推算出来，那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list，从而节省大量的空间，在Python中，这种一边循环一边计算的机制，称为生成器：generator。  
    
    生成器一次只能产生一个值，这样消耗的内存数量将大大减小，而且允许调用函数可以很快的处理前几个返回值，因此生成器看起来像是一个函数，但是表现得却像是迭代器。

* 生成器是什么？  

    **任何使用了 yield 的函数都称为生成器，调用生成器函数将创建一个对象，该对象有一个 `__next__()` 方法** (python3 中是 __next__()，python2 中是 next())   
    
    例如下面的函数 `countdown()` 是一个生成器，对 `countdown()` 的调用不会执行函数里面的语句，而只能获得一个生成器对象 (也即代码中的 gen)。这个对象包含了函数的原始代码和函数调用的状态，其中函数状态包括函数中变量值以及当前的执行点。  
    ```python
    def countdown(n):   # 生成器
        print("count down!")
        while n > 0:
            yield n   # 函数暂停、返回当前值、存储函数调用状态
            n -= 1
            
    gen = countdown(5)  # 生成器对象
    print(gen)   # 输出<generator object countdown at 0x.......>

    # 用__next__()方法
    gen.__next__()  # or next(gen)
    gen.__next__()    

    # 用for循环
    for i in countdown(5):
        print(i)

    # 会打印
    # count down!
    # 5 4 3 2 1
    ```    
    **`__next__()` 调用使得生成器函数一直运行，知道遇见 yield 语句后会暂停、返回当前的值、并储存函数的调用状态。当再次调用 `_next__()` 时，函数将从上次停止的状态开始继续执行，直到再次遇见yield语句**。但一般不直接调用`__next__()`方法，而是用for循环进行迭代

* 协程：用于编写生产者-消费者并发程序  
包含将 yield 语句作为输入的函数叫做协程
    ```python
    def print_line():   # 协程
        while True:
            line = yield
            print(line)
    
    printer = print_line()
    printer.__next__()  # 向前执行到第一个yield语句，使协程准备好
    printer.send("Hello world")
    ```

* 和列表之间的转换
  ```python
  # 列表
  [x ** 3 for x in range(5)]
  >>> [0, 1, 8, 27, 64]

  # 生成器
  (x ** 3 for x in range(5))
  >>>  <generator object <genexpr> at 0x000000000315F678>

  # 两者之间转换
  list((x ** 3 for x in range(5)))
  ```

### 3.2.2 迭代器与可迭代对象 (Iterator vs. Iterable)
> https://www.cnblogs.com/wj-1314/p/8490822.html
* `可迭代对象` 包含 `迭代器` 包含 `生成器`
    * Iterable: 含有`__iter__()`方法的对象都是可迭代对象。可直接作用于 for 循环，包含迭代器，列表，字典，字符串等
    * Iterator：含有`__iter__()`和`__next__()`方法的对象都是迭代器，所以上文说的**生成器对象是一种特殊的迭代器**
* list, tuple, dict, str 等都是Iterable对象, 但不是Iterator
* 可用`isinstance()`可判断对象是否Iterable或是否是Iterator
    ```python
    from collections import Iterable
    from collections import Iterator

    isinstance((x for x in range(10)), Iterable) # True
    isinstance([], Iterable) # True

    isinstance((x for x in range(10)), Iterator) # True 
    isinstance([], Iterator) # False
    ```
* 可用`iter()`函数把list、dict、str等Iterable对象变成Iterator
    ```python
    isinstance(iter([]), Iterator)
    isinstance(iter('abc'), Iterator)
    ```
* 一个对象实现了`__getitem__()`方法也可以通过`iter()`函数转成`Iterator`，即也可以在for循环中使用，但它本身不是一个可迭代对象：https://juejin.im/post/5ccafbf5e51d453a3a0acb42  
* Pytorch 中的 dataloader   
dataloader 是一个 `torch.utils.data.dataloader.DataLoader` 对象，是 iterable 的，但不是一个 iterator。要获取单一batch，可用 `next(iter())`，`iter()`的作用是将 dataloader 先变成一个迭代器。


## 3.3 parser
test.py文件的书写：
```python
#!/usr/bin/env python3
import argparse

if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="data/model_epoch46.chkpt",
        help="directory to model")
    parser.add_argument("--dataset_dir", type=str, default="/data/test.p",
        help="directory to dataset?")     

    opt = parser.parse_args()

    model_dir = opt.model_dir   
    dataset_dir = opt.dataset_dir
```
命令行中用法： `python3 test.py --model_dir=data/model_epoch01.chkpt`  

<br>

## 3.4 变量作用域
* 每个变量都有作用域，Python中除了`def/class/lambda`外，其他如`if/elif/else/， try/except， for/while`等都不能改变变量作用域  
* 因为函数(def)会改变变量作用域，在函数内部定义的变量，是局部变量，函数外外面看不到。***局部变量会和同名全局变量冲突***
* 带`self.`的 类的成员变量作用域只在类内部，***不会与同名全局变量/局部变量都冲突***
    ```python
    class test:
        def __init__(self):
            self.global_var = 3;    # 类的成员变量不会和同名全局变量冲突
        def main(self):
            local_var = 2           # 类的成员变量不会和同名局部变量冲突
            self.local_var = 4;    

            global global_var   # 在函数中使用全局变量，没有global关键词会print报错
            print(global_var, local_var, self.global_var, self.local_var)
            global_var = 5 

    if __name__ == '__main__':
        global_var = 1              # if不改变作用域，所以是全局变量
        test().main()               # 或 s = test(); s.main() 但test.main()会报错，因为并没有先初始化类
        print(global_var)               # 此处是全局变量
        # print(local_var) 会报错

    # 输出结果分别为 1 2 3 4 5
    ```
* global和nonlocal关键字的使用: 
  * python 变量查找法则：LEGB，见 https://zhuanlan.zhihu.com/p/111284408 
  * python 可以遵循 LEGB 法则自动在函数之外查找变量，变量的修改就严格得多

    * 如果函数内部有引用同名的全局变量，并且对其做了修改，那么python会认为它是一个局部变量。要在函数内修改全局变量，必须用 global 关键字 
    * 有意思的是，在函数中也可以随时定义 global 变量，使得它可以被其他函数使用
  
        ```python
        def a():
            global q # 若去掉这一行，显然会报错
            q=7
        def b():
            print(q)

        a(); b()
        print(q)    # 输出 7 7 
        ```
  * nonlocal关键字用来在函数或其他作用域中使用外层 **(非全局)** 变量
    ```python
    def funx():
        x=5
        def funy():
            nonlocal x
            x += 1
            return x
        return funy

    if __name__ == "__main__":
        a = funx()
        print(a())
    ```



## 3.5 闭包，装饰器，语法糖
参见 https://www.zhihu.com/question/25950466/answer/31731502
### 3.5.1 闭包
> 所谓闭包，就是将组成函数的语句和这些语句的执行环境打包在一起时，得到的对象
```python
# foo.py
filename = "foo.py"

def call_fun(f):
    return f()
```
```python
# func.py
import foo

filename = "func.py"
def show_filename():
    return "filename: %s" % filemame

if __name__ == "__main__":
    print(foo.call_fun(show_filename))  # 返回 filename:func.py
```
这个例子说明了尽管 show_filename 函数的调用发生在 foo.py 文件内，但 show_filename 函数包含了其执行所需的整个环境（也即filename变量的值），构成了一个闭包，其中 filename 变量的查找遵循`Local-> Enclosing-> Global-> Builtin`顺序查找:  
* 本地函数(show_filename内部)：通过任何方式赋值的，而且没有被global关键字声明为全局变量的filename变量
* 直接外围空间(show_filename外部)：如果有多层嵌套，则由内而外逐层查找，直至最外层的函数
* 全局空间(func.py和其import部分)
* 内置模块(\_\_builtin\_\_)  

闭包在其捕捉的执行环境(def语句块所在上下文)中，也遵循LEGB规则逐层查找，直至找到符合要求的变量，或者抛出异常。  
这说明函数内可以查找到函数外定义的变量，这是显而易见的，例如全局变量理所当然可以在任何位置被使用。***这与前面讲的变量作用域不冲突，就变量本身而言，函数内定义的变量生命周期只在函数内部。但函数被作为闭包返回，为函数内变量续了命***

### 3.5.2 装饰器
> 装饰器本质上是一个Python 函数或类。运用闭包能封存上下文的特性，它可以让其他函数或类在不做任何代码修改的前提下增加额外功能，装饰器的返回值也是一个函数/类对象

下面的例子，用闭包的性质为add函数添加了新的功能：
```python
def checkParams(fn):
    def my_wrapper(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return fn(a,b)  # 解释器按照LEGB法则找到fn，也即add函数对象的一个引用
        print("type incompatible")
        return
    return my_wrapper

def add(a, b):
    return a+b

if __name__ == "__main__":
    add = checkParams(add)  # 返回的是一个 my_wrapper 这个函数的闭包
    add(3, "hello")
```

### 3.5.3 装饰器的语法糖
> 装饰器的语法糖：在写法上简化上面的代码，参见：https://www.jianshu.com/p/fd746acbdf1e
```python
def checkParams(fn):
    # 这个函数的实现同上，不变

@checkParams
def add(a, b):
    return a+b

if __name__ == "__main__":
    add(3, "hello")
```
简单来说，就是将`@checkParams`写在`add(a,b)`定义上面，等效于`add = checkParams(add)`，也即在不改变 add 函数本身的情况下，为它加了额外的功能

### 3.5.4 `@functools.wraps()`
在这里就有一个问题，被修饰器修饰过的函数还是以前的函数吗？答案是不是的，它会带有wrapper属性：
```python
print(add)  # 输出 <function checkParams.<locals>.my_wrapper at 0x7fca8c79c8c8>
print(add.__name__) # 输出 my_wrapper
```
为了消除装饰器对原函数造成的影响，即对原函数的相关属性进行拷贝，达到装饰器不修改原函数的目的，可以用 `@functools.wraps(fn)`，此时再 `print(add.__name__)` 就会输出 `add` 而不是 `my_wrapper`了
```python
def checkParams(fn):
    @functools.wraps(fn)
    def my_wrapper(a, b):
        ...
    return my_wrapper
```


### 3.5.5 用类写一个多重的，带参数的装饰器
```python
class add_prefix(object):
    def __init__(self, word):
        self.word = word
    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            return self.word + fn(*args, **kwargs)
        return wrapper

@add_prefix("dear ")
@add_prefix("Prof.")
def hello(name):
    return name

if __name__ == "__main__":
    print(hello("Chen"))
```
上面的两个装饰器等价于:
```python
def hello(name):
    return name
    
if __name__ == "__main__":
    hello = add_prefix("dear ")(add_prefix("Prof.")(hello))
    print(hello("Chen"))
```

### 3.5.5 其他
[函数内部的变量在函数执行完后就销毁，为什么可变对象却能保存上次调用时的结果呢？](https://www.zhihu.com/question/264533969)



