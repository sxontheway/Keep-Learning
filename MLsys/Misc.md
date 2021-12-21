# TVM 安装
TVM 支持交叉编译，例如在 server 上编译好，直接 flash 到 TX2。但下面都是在 server (host) 上编译安装的过程。TVM 需要先安装 LLVM。我的 server 配置：CentOS Linux release 7.7.1908；GPU：2080

## LLVM 源码编译安装
> [Centos7上源码编译安装llvm 11.0.0](https://zhuanlan.zhihu.com/p/350595463)   
> [官方文档](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)

* 下载 llvm-11.0  
    llvm官网从9.0版本开始打包了一个包含所有组件源码的大项目 llvm-project，这里也直接下载这个大的项目包，而不是各个组件分别安装
    ```bash
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.1.0/llvm-project-11.1.0.src.tar.xz
    ```

* 解压，make  
    ```bash
    mkdir build 
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi" -G "Unix Makefiles" ../llvm
    make install
    ```

* 期间升级了 cmake，gcc
    * 升级 cmake（源码编译安装）：https://www.jianshu.com/p/55153a29facf
    * 升级 gcc：https://zhuanlan.zhihu.com/p/350595463 
        ```bash
        yum install centos-release-scl
        yum install devtoolset-7
        scl enable devtoolset-7 bash
        source /opt/rh/devtoolset-7/enable

        echo "source /opt/rh/devtoolset-7/enable" >> ~/.bash_profile 
        source /opt/rh/devtoolset-7/enable

        gcc --version
        ```
* 检查
    ```bash
    llvm-config --version
    clang --version
    whereis llvm-config
    ```
    
## TVM 源码编译安装
> https://tvm.apache.org/docs/install/from_source.html#install-from-source 
## Build from source
* 源码编译三个步骤：
    * 配置(configure)：可以指定安装目录等，https://blog.csdn.net/finded/article/details/51889588
    * 编译(make)
    * 安装(make install)

* `/tvm/build/config.cmake` 文件要根据 document enable 一些选项，比如 `set(USE_CUDA ON)`，`set(USE_LLVM /usr/local/bin/llvm-config --link-static)`，`set(HIDE_PRIVATE_SYMBOLS ON)`, `set(USE_RPC ON)` 等 
* `cd build`，`cmake ..`，`make -j16`，显示 `[100%] build target tvm`

## Python Package Installation
将以下代码加入 `~/.bashrc`，其中 `/path/to/tvm` 是上面 git clone 的目录 
```bash
export TVM_HOME=/path/to/tvm 
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
之后 `import tvm; print(dir(tvm)); tvm.gpu().exist` 验证 python 包是否安装成功

## Debug
* 以下代码可以成功运行 
    ```
    import tvm
    from tvm import te
    import numpy as np

    n = te.var("n")
    m = te.var("m")

    A = te.placeholder((m, n), name="A")
    B = te.placeholder((m, n), name="B")
    C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

    s = te.create_schedule([C.op])

    print(tvm.lower(s, [A, B, C], simple_mode=True))
    ```
    但一旦代码涉及 GPU，见 [How to optimize convolution on GPU](https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_cuda.html#sphx-glr-how-to-optimize-operators-opt-conv-cuda-py)，就会跑不通，报错：`Check failed: (bf != nullptr) is false: target.build.cuda is not enabled`。提示 TVM 编译的时候没用 CUDA
    
* TVM 需要 LLVM 来进行 CPU codegen，验证 LLVM 的代码见 [An Example of Schedule](./Background.md#an-example-of-schedule)。当报错 `target.build.llvm is not enabled`，表示编译时没 LLVM
* 解决方案：
    * 原因可能是 cuda toolkit 没安装在 cmake 会找的默认路径 `/usr/local/cuda/`，所以 cmake 找不到，见 https://github.com/apache/tvm/issues/535 ，可以建立一个软连接：`ln -s /home/xian/cuda-10.1 /usr/local/cuda`；找不到 LLVM 的情况同理可以解决
    * 确认 `config.make` 中 `set(USE_CUDA ON)`，然后看 `CMakeCache.txt` 中搜 `cuda` 和 `llvm`，发现是 off，**手动改成 on**
    * `cmake ..`，`make -j16`