# 1.多进程
> 官方文档： https://docs.python.org/zh-cn/3/library/multiprocessing.html
## 基础
* `p.daemon = True`：进程 p 作为守护进程，也即当主进程结束时，无论子进程是否还在运行，都伴随主进程一起退出（主死子死）
* `p.start()`：主进程开启进程，不会产生阻塞
* `p.join()`：主进程会阻塞，会一直等到子进程返回，和 `daemon=True` 正好相反


## Process 类
一个类用一个进程（需要继承 Process）。使用 event，主进程可以选择性激活某些类，未激活的类处于 wait 状态（在联邦学习的多进程实现中会有用）
```python
import multiprocessing
from multiprocessing import Process

class Local(Process): 
    def __init__(self, client_num, event):
        super(Local, self).__init__()
        self.num = client_num
        self.event = event

    def run(self):
        while 1:
            print(f"{self.num}--before")
            self.event.wait()
            print(f"{self.num}--after")
            self.event.clear()


if __name__ == '__main__':

    process_list, event_list = [], []
    for i in range(5):  
        e = multiprocessing.Event()
        p = Local(i, e) 
        process_list.append(p)
        event_list.append(e)

    for p in process_list:
        p.daemon = True
        p.start()

    for i in range(5):
        p = process_list[0]
        p.event.set()
        time.sleep(1)

    import time
    time.sleep(10)

    for p in process_list:
        p.terminate()

    print("End")
```

多进程运行函数：
```python
from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
```
## Pool 类  
> `Pool.apply_async` 和 `Pool.map` 区别：https://zhuanlan.zhihu.com/p/33971568  
* Pool.apply_async  
`Pool.apply_async`: the Pool of worker processe to perform many function calls asynchronously    
不保证输入输出顺序对应，下面的代码可能输出：[1, 0, 4, 9, 25, 16, 49, 36, 81, 64]

    ```python
    from multiprocessing import Pool

    def foo_pool(x):
        return x*x

    result_list = []
    def log_result(result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    if __name__ == '__main__':
        pool = Pool()
        for i in range(10):
            pool.apply_async(foo_pool, args = (i, ), callback = log_result)
        pool.close()
        pool.join()
        print(result_list)
    ```
* Pool.map    
    > 多进程运行含有任意个参数的函数：https://blog.csdn.net/qq_15969343/article/details/84672527  

    `Pool.map`: apply the same function to many arguments，尽管多个线程执行的顺序不同，但是输出顺序和输入顺序一致
    ```python
    from multiprocessing import Pool

    def my_print(x, y):
        print(x + y)    # sometimes 2 4 3 6 5
        return(x + y) 

    if __name__ == '__main__':
        x = [1, 2, 3, 4, 5]
        y = [1, 1, 1, 1, 1]

        zip_args = list(zip(x, y))
        pool = Pool(5)
        output = pool.starmap(my_print, zip_args)
        pool.close()
        pool.join()
        
        print(output)   # always [2, 3, 4, 5, 6]
    ```


## 在进程之间交换对象: 
* Queue 类   
`q.get()`会阻塞直到队列中有新的数据
    ```python
    from multiprocessing import Process, Queue

    def f(q):
        q.put([42, None, 'hello'])

    if __name__ == '__main__':
        q = Queue()
        p = Process(target=f, args=(q,))
        p.start()
        print(q.get())    # prints "[42, None, 'hello']"
        p.join()
    ```
* Pipe 类  
`Pipe()` 返回的两个对象表示管道的两端。每个连接对象都有 send() 和 recv() 方法，是双向的。

    ```python
    from multiprocessing import Process, Pipe

    def f(conn):
        conn.send([42, None, 'hello'])
        conn.close()

    if __name__ == '__main__':
        parent_conn, child_conn = Pipe()
        p = Process(target=f, args=(child_conn,))
        p.start()
        print(parent_conn.recv())   # prints "[42, None, 'hello']"
        p.join()
    ```

* Manager 类  
    > Pool + starmap + Manager: https://blog.csdn.net/HappyRocking/article/details/83856489

    * 以下代码用 Pool 实现了用进程池管理多个进程，用 Manager 实现了多个进程之间的数据共享，用 starmap 将函数分配给多个进程执行（并支持传任意多参数）
    * 以下代码每次运行可能产生不同的输出，因为没有对共享区做互斥处理。例如两个进程，一个进程加3，一个进程加4，如果他们同时读取到`d['count']=3`。考虑到 `dic['count'] += c` 操作是要先算出 `dic['count'] + c` 再赋值给 `dic['count']`，如果加3的进程晚于加4的进程完成，这时就会出现少加4的情况
    * 需要有处理代码段处理**孤儿进程**：孤儿进程指的是在其父进程执行完成或被终止 后仍继续运行的一类进程。例如主进程被`ctrl+c`终止了，但子进程还在运行，可能会造成内存消耗。下面代码用 `try except` 处理了这个问题。

    ```python
    from multiprocessing import Pool, Manager
    import time

    def func(dic, c):
        dic['count'] += c

    if __name__=="__main__":
        try:
            d = Manager().dict()    # 进程间共享的字典
            d['count'] = 0

            # 参数是一个可迭代对象
            args = [(d,1), (d,2), (d,3), (d,4), (d,5)]  
            pool = Pool(5)
            pool.starmap(func, args)
            
            pool.close()    # 关闭进程池，使其不再接受新的任务
            pool.join()     # 主进程阻塞直到所有子进程执行完毕，需要在pool.close()之后调用
            print(f'dic={d}')
        except:
            pool.terminate()
    ```
## 随机数
* 多进程的各个子进程中，可能会产生相同的随机数：https://blog.csdn.net/largetalk/article/details/7910400 

<br>

# 2.多线程  
## 多线程之间可以贡献全局变量：  
https://zhuanlan.zhihu.com/p/258716555  

## GIL锁
Python虽然有多线程，但python解释器执行代码时，有一个GIL锁：`Global Interpreter Lock`。任何Python线程执行前，必须先获得GIL锁，然后每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核。不过，Python虽然不能利用多线程实现多核任务，但可以通过多进程实现多核任务。多个Python进程有各自独立的GIL锁，互不影响。  

简单来说，python多线程能提高IO密集型任务的效率；对于计算密集型，需要用多进程。  
PS：但有了协程之后，python的多线程就显得鸡肋了。在 2.x 系列里我们可以使用 gevent，在 3.x 系列的标准库里又有了 asyncio 。IO bound 的问题完全可以用协程解决。而且我们可以自主的控制协程的调度了。为什么还要使用由 OS 调度的不太可控的线程呢？


<br>

# 3. 协程
## 理解协程
早期的多任务操作系统是没有抢占式任务调度机制的，采用的是协作式多任务机制，也就是任务需要自己主动让出 CPU，让其进入 idle 状态，这实际上就是协程。比如早期的 Windows(1.0-3.0)，就是协作式多任务，直到 windowsNT 问世，微软才宣称 Windows 实现了抢占式多任务调度系统。所以协程就是线程的前身。以现在的眼光看，线程好像是个理所当然的概念，但在硬件还很简陋的时代，比如8位/16位芯片时代，要在操作底层实现一套线程机制是不可能的，利用中断驻留程序，通过定时器以及各种软硬件中断实现协作式多任务，在当时是唯一可行的做法。STM32 上的中断本质上和协程是一个原理：单线程下的协作式多任务，一切都是由程序员控制的

协程的目的是在 **单线程** 下实现并发（任务切换+保存状态），主要应用场景是在单线程下加速IO密集的多任务执行。  
优点是切换的代价比多线程、多进程小。缺点是由于只用了单线程，切换完全依据程序员写的代码逻辑，且需要检测到单线程下所有的IO行为，少一个都不行，因为一旦一个任务阻塞了，整个线程就阻塞了，而多线程是由 OS 调度的，不需要手动检测 IO。

## 同步和异步IO
同步： 代码调用 IO 操作时，必须等待 IO 操作完成才返回的调用方式  
异步： 代码调用 IO 操作时，不必等 IO 操作完成就返回的调用方式  
因为 IO 过程中所需要的 CPU 资源非常少，大部分工作是分派给 DMA 完成的，所以通过异步 IO，可以在 CPU 等待 IO 完成的时间内让 CPU 去执行其他操作去提高效率 
所以本质上协程提高效率还是要依赖于异步 IO，也即检测到 IO 操作就自动切换到另外一个任务，例如使用 `gevent` 模块，可以自动检测I/O操作，自动进行切换（实现协程）

## 协程怎样提高性能？
> https://blog.csdn.net/u010841296/article/details/89608492   

如上文所说，利用协程提高效率的关键在于：在“I/O阻塞”时让出 CPU 时间切片。  
所以，协程实现库需要实现一些底层的监听，例如发现当前协程即将要进行“网络I/O”时（假设当前操作系统的网络IO模型是epoll），协程库能劫持该系统调用，把这些IO操作注册到系统网络IO的epoll事件中（包括IO完成后的回调函数），注册完成则把当前协程挂起，让出cpu资源给其他协程。直到该IO操作完成，通过异步回调的方式通知该协程库去恢复之前挂起的协程，协程继续执行。


## 语法
> https://www.liujiangblog.com/course/python/83  
### yield 和 send
协程其实就是一个可以暂停执行的函数，并且可以恢复继续执行。yield 已经可以暂停执行了，如果在暂停后有办法把一些 value 发回到暂停执行的函数中，那么就有了协程。“把东西发送到已经暂停的生成器中” 的方法，这个方法就是 `send()`  
`send()` 的功能：1. 传值进入协程；2. 获取返回值；3. `next()`
```python
def customer():     # 本质上是个 generator
    r = ""
    while True:
        n = yield r     # 接收生产者的消息，并向消费者发送r
        print("customer receive", n)
        r = "ok"

def produce(c):
    # 第一次启动协程使其执行到 yield 位置，否则报错：can't send non-None value to a just-started generator
    c.send(None)    # 等同于 next(c)
    
    for i in range(3):
        print("start send to customer", i)
        r = c.send(i)   #向消费者发送值 i，并收到消费者返回的 “ok”
        print("receive customer", r)

c = customer()
produce(c)
```
输出  
```python
start send to customer 0
customer receive 0
receive customer ok
start send to customer 1
customer receive 1
receive customer ok
start send to customer 2
customer receive 2
receive customer ok
```

### 从 yield/send 到 yield from 再到 async/await
> https://blog.csdn.net/soonfly/article/details/78361819

<br>

# 4. 生产者/消费者模型
> https://blog.csdn.net/qq_32252957/article/details/79433679

用队列优化Pytorch框架的数据加载过程：https://blog.csdn.net/winycg/article/details/92443146 

