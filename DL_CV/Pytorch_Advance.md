# Deterministic 的问题
> 其实没必要严格要求 deterministic，从实际角度出发，只要每次跑出来结果差距都不大就行了：https://pytorch.org/docs/stable/notes/randomness.html 
* `torch.backends.cudnn.deterministic=True` 是能让卷积操作 deterministic，其他操作如 `torch.nn.MaxPool3d` 基本没办法确保deterministic： https://stackoverflow.com/a/66647424
* 初始化也不行：https://github.com/pytorch/pytorch/issues/19013
* Dataloader 的 worker 数量不同也会使得采样结果 non-deterministic: https://pytorch.org/docs/stable/data.html#data-loading-randomness 


<br>
<br>

# DDP
* DP: 单机多卡，所有设备都负责计算和训练网络，除此之外， device[0] (并非 GPU 真实标号而是输入参数 device_ids 首位) 负责整合梯度，更新参数，是 parameter server 的构架
* DDP：https://zhuanlan.zhihu.com/p/187610959, https://zhuanlan.zhihu.com/p/358974461 

    * 单进程单卡，推荐
    * 多进程多卡，每个进程都是 DP，对于 python 的 PIL 会有 CPU bound
    * 单进程多卡，并行（model 太大放不下 batchsize=1）

    ```python
    import os
    import torch
    import torchvision
    import torch.distributed as dist
    import torch.utils.data.distributed
    from torchvision import transforms
    from torch.multiprocessing import Process

    os.environ['MASTER_ADDR'] = 'localhost'     # New added
    os.environ['MASTER_PORT'] = '12355'         # New added


    def main(rank):

        dist.init_process_group("ncll", rank=rank, world_size=3)    # New added
        torch.cuda.set_device(rank)                                 # New added

        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)   # New added
        data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=256, sampler=train_sampler)    # use train_sampler to split the original batch size

        net = torchvision.models.resnet101(num_classes=10)
        net.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        net = net.cuda()

        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank]) # New added
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(10):
            for i, data in enumerate(data_loader_train):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                opt.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()
                if i % 10 == 0:
                    print("loss: {}".format(loss.item()))
        if rank == 0:       # only the main process saves the model 
            torch.save(net, "my_net.pth")   


    if __name__ == "__main__":
        size = 3
        processes = []
        for rank in range(size):
            p = Process(target=main, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # or use spawn
        # mp.spawn(main, args=(size,), nprocs=size, join=True)
    ```
    
* torch.distributed.init_process_group() 默认 `env://` 的初始方法，也可以使用 tcp 和 file
* 多机分布式启动方式： 用 torch.distributed.launch。假设一共有两台机器（节点1和节点2），每个节点上有8张卡，节点1的IP地址为192.168.1.1，占用的端口12355（端口可以更换），启动的方式如下：

    ```bash
    # 节点1
    python -m torch.distributed.launch --nproc_per_node=8
            --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
            --master_port=12355 MNIST.py
    # 节点2
    python -m torch.distributed.launch --nproc_per_node=8
            --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
            --master_port=12355 MNIST.py
    ```
    * 其中 torch.distributed.launch 将会被 torchrun 代替：https://pytorch.org/docs/stable/elastic/run.html#launcher-api
<br>
<br>


# 多GPU多进程模拟联邦学习，进程初始化报错
> https://discuss.pytorch.org/t/understanding-minimum-example-for-torch-multiprocessing/101010 

## 背景
* 由于要调用 CUDA，多进程用的是 `spawn` 而不是 `fork`
* 多进程之间是要共享一个 list of tensors，但不同进程用的是 list 里面不同的 slices    

一个错误复现的代码如下：
```python
import torch, os
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    print(f'rank {rank} of {size}')

def init_process(rank, size, data, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def processes(data):
    size = len(data)
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, data, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    size_vector = 133
    part = int(size_vector/8)
    indices = torch.arange(size_vector)
    split_data = torch.split(indices, part)
    print(split_data)
    processes(split_data)
```

## 解决方案
* 以上代码在 Pytorch1.9 的Linux版本下会报错：` ValueError: bad value(s) in fds_to_keep`，在windows下则正常运行
* 解决方案有两种：
    * 将 `mp.Process()` 那行改成：`data_i = data[rank]; p = mp.Process(target=init_process, args=(rank, size, data_i, run))`
    * 在 `mp.Process()` 那行之前加：`data = [i.clone() for i in data]`，相当于将 data 的每个元素重新在内存里面复制了一遍

## 原因分析
> You cannot pass a tensor to the mp.Process that has data shared with other processes

* 报错的的原因应该是来源于 Pytorch 内部的内存管理机制，tensor 和 tensor 之间是自动共享内存的。试过 `copy.deepcopy(data)` 不起作用，必须要用 `.clone()`
* 为什么 windows 不报错而 linux 报错，可能和两个平台的多进程实现机制有关，windows没有fork，所以为新进程强行开了新的存储空间

<br>
<br>

# pytorch多gpu并行训练  
> https://zhuanlan.zhihu.com/p/105755472
> https://zhuanlan.zhihu.com/p/86441879  
> https://zhuanlan.zhihu.com/p/95700549  
> https://zhuanlan.zhihu.com/p/68717029   

Use single-machine multi-GPU `DataParallel`, if there are multiple GPUs on the server, and you would like to speed up training with the minimum code change.

Use single-machine multi-GPU `DistributedDataParallel`, if you would like to further speed up training and are willing to write a little more code to set it up.

Use multi-machine `DistributedDataParallel` and the launching script, if the application needs to scale across machine boundaries.

* `python -m torch.distributed.launch main.py`
    > 按模块启动：https://www.cnblogs.com/xueweihan/p/5118222.html
    
    * 直接启动 `python xxx.py` 是把 `xxx.py` 文件所在的目录放到了sys.path属性中
    * 按模块启动 `python -m xxx.py` 是把你输入命令的目录（也就是当前路径），放到了sys.path属性中  

<br>
<br>

# Pytorch Internals
## Folders
* `torch/`：包含导入和使用的实际Python模块。Python代码，很容易上手调试。

* `torch/csrc/`：它实现了在Python和C++之间进行转换的绑定代码，以及一些非常重要的PyTorch功能，如autograd引擎和JIT编译器。它还包含C++前台代码。
    * `torch._C` 模块在 `torch/csrc/Module.cpp` 中定义。这个模块被称为是一个扩展模块（一个用C实现的Python模块），它允许我们定义新的内建对象类型（例如：Tensor）并调用 C/C++ 函数。

* `aten/`：“A Tensor Library”的缩写（由Zachary DeVito创造），是一个实现Tensors操作的C++库。存放一些内核代码存在的地方，尽量不要在那里花太多时间。

* `c10/`：这是一个双关语。C代表Caffe，10既是二级制的2 (Caffe2)，也是十进制的10（英文Ten，同时也是Tensor的前半部分）。包含PyTorch的核心抽象，包括Tensor和Storage数据结构的实际实现。

<br>
<br>

# Pytorch Features
> https://speakerdeck.com/perone/pytorch-under-the-hood?slide=21

## Tensors
* Although PyTorch has an elegant python first design, all PyTorch heavy work is actually implemented in C++. The integration of C++ code is usually done using what is called an **extension**.

* zero-copy tensors
    ```python
    # a copy is made
    np_array = np.ones((1,2))
    torch_array = torch.tensor(np_array)    # This make a copy
    torch_array.add_(1.0)   # underline after an operation means an in-place operation
    print(np_array)     # array([[1., 1.]])

    # zero-copy
    np_array = np.ones((1,2))
    torch_array = torch.from_numpy(np_array)    # This make a copy
    torch_array.add_(1.0) # or torch_array += 1.0 (in place operation)
    print(np_array)     # array([[2., 2.]])

    # zero-copy
    np_array = np.ones((1,2))
    torch_array = torch.from_numpy(np_array)    # This make a copy
    torch_array = torch_array + 1.0     # not an in-place operatio on torch_array 
    print(np_array)     # array([[1., 1.]])
    ```
    The tensor **FloatTensor** did a copy of the **numpy array data pointer** instead of the contents. The reference is kept safe by Python reference counting mechanism.

* The abstraction responsible for holding the data isn't actually the Tensor, but the Storage. We can have multiple tensors sharing the same storage, but with different interpretations, also called views, but without duplicating memory.

    ```python 
    t_a = torch.ones((2,2))
    t_b = t_a.view(4)
    t_a_data = t_a.storage().data_ptr()
    t_b_data = t_b.storage().data_ptr()
    t_a_data == t_b_data

    # True
    ```


## JIT: just-in-time compiler 
> https://zhuanlan.zhihu.com/p/52154049  
> https://zhpmatrix.github.io/2019/03/01/c++-with-pytorch/

早期的PyTorch只有Python前端，但对于工业界的实际部署问题，Python语言太慢，可移植性和适用性根本无法和C++相比。当时的一个想法是，PyTorch训练模型，然后前向推断时将结构和参数灌入到C++代码中，这估计也是早些年的一些做法。但是调研之后，将PyTorch的C++后端拉出来并不容易，而且如果从C++原生代码来写起，工作量也很大。因此，希望有一个C++前端方便做推断部署。

千呼万唤始出来。PyTorch1.0发布了，这样业界部署的工作流程可以变成这样：

> 论文发布->PyTorch开源代码(或者自己实现)->训练模型->导出模型->载入模型(C++/Python/其他框架/其他硬件平台)

PyTorch1.0后，可以通过两种方式，分别是Tracing和Script，将一个Python代码转化为TorchScript代码，继而导出相应的模型可以继续被优化，同时被C++所调用，最终实现对生产环境下的支持（考虑到多线程执行和性能原因，一般Python代码并不适合做部署）

* Tracing  
Tracing方式对于含有if和for-loop的场景失效，需要用script方式

* Script  
https://zhpmatrix.github.io/2019/03/09/torch-jit-pytorch/   



## Dataloader 加速
* 仅从使用者的角度考虑,DataLoader做了下面的事情：
    * 开启多个子进程worker
    * 每个 worker 通过主进程获得自己需要采集的idx。idx的顺序由采样器（sampler）或 shuffle 得到。每个 worker 开始采集一个batch的数据。因此增大 num_workers 的数量，内存（不是显存）占用也会增加。因为每个 worker 都需要缓存一个 batch 的数据
    * 第一个 worker 数据采集完成后，会卡在这里，等着主进程取走数据。主进程处理完这个 batch 之后，这个 worker 开始采集下一个 batch    
    * 主进程采集完最后一个 worker 的batch。此时需要回去采集第一个 worker 产生的第二个 batch。如果第一个 worker 此时没有采集完，主线程会卡在这里等（这也是为什么在数据加载比较耗时的情况下，每隔 num_workers 个 batch，主进程都会在这里卡一下）

* Dataloader 数据装载阻塞的问题: https://zhuanlan.zhihu.com/p/91521705  
    <p align="center" >
        <img src="pictures/dataloader.jpg", width='800'>
    </p>

    * Pytorch Dataloader 的实现是多进程
    * 一个 worker 独立的处理一个 batch，而不是多个 worker 同时处理一个 batch
    * dataloader **不是** 等所有worker数据取完才进行下一批次的数据读取，worker 之间并没有同步
    * 输出的数据保持顺序性：主线程（进行front/back propagation）按照`idx=0, 1, 2, 3...`依次处理 worker 产生的 batch
    * worker 会等待主进程处理完（主要即GPU time）上个 batch，才采样下一个 batch
* 用 GPU 来完成 dataloader 中的 transform:   
https://zhuanlan.zhihu.com/p/77633542  
https://github.com/pytorch/pytorch/issues/31359  

* 进一步加速：  
    > https://www.cnblogs.com/pprp/p/14199865.html 

    * Prefetch next batch / 新开的cuda stream拷贝tensor到gpu：https://zhuanlan.zhihu.com/p/97190313  
    * 生产者消费者模型：https://blog.csdn.net/winycg/article/details/92443146    

## APEX
