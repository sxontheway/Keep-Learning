# 1.多进程
> 官方文档： https://docs.python.org/zh-cn/3/library/multiprocessing.html
## Process 类
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

<br>

# 2.多线程  
Python的线程虽然是真正的线程，但解释器执行代码时，有一个GIL锁：`Global Interpreter Lock`，任何Python线程执行前，必须先获得GIL锁，然后，每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核。不过，Python虽然不能利用多线程实现多核任务，但可以通过多进程实现多核任务。多个Python进程有各自独立的GIL锁，互不影响。  

简单来说，python多线程适能提高IO密集型任务的效率；对于计算密集型，需要用多进程。