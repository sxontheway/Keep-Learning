# CUDA 的内部工作原理:为什么能加速
> https://www.jianshu.com/p/34a504af8d51


<br>

# Installation
> https://blog.csdn.net/wanzhen4330/article/details/81699769  

用Nvidia显卡配置Deep Learning环境分三步：
* 安装Nvidia Driver和CUDA Toolkit：Driver/CUDA toolkit  
* 安装Library： cuDNN/TensorRT等 
* 安装Framework：tensorflow/pytorch等

## Driver & CUDA toolkit
* 查看显卡型号：`lspci | grep -i vga`
* 查是显卡否支持CUDA: http://developer.nvidia.com/cuda-gpus
* 下载CUDA Installer并运行: https://developer.nvidia.com/cuda-toolkit-archive
    ```bash
    wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
    sudo sh cuda_10.1.243_418.87.00_linux.run
    ```
    Installer中可选择是否安装driver，如果已经安装过了那就不安装driver

## cuDNN 
* 下载需要的版本 https://developer.nvidia.com/rdp/cudnn-archive
* 解压下载的文件，可以看到cuda文件夹。做两件事：（1）将解压所得的cuda内的文件复制到cuda安装目录。（2）改权限
    ```bash
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/ 
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    ```

## Conda 创建环境安装 Torch
* tensorflow 和 CUDA/cuDNN 版本兼容性：https://www.tensorflow.org/install/source_windows#gpu 
    ```bash
    # conda 安装 https://www.jianshu.com/p/edaa744ea47d
    mkdir miniconda3
    cd miniconda3
    wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod 777 Miniconda3-latest-Linux-x86_64.sh #给执行权限
    bash Miniconda3-latest-Linux-x86_64.sh -u #运行

    # conda 内安装
    conda create -n ml python=3.7
    conda config --set auto_activate_base false
    # 在 ~/.bashrc 最后一行加 conda activate ml

    # 进入 (ml) 后
    pip3 install matplotlib numpy scipy scikit-learn pandas pyyaml 

    # 安装 nvidia driver 对应版本的 cuda/torch
    nvidia-smi
    nvcc --version  # 发现是 9.0 版本，只能安pytocrh 1.1
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
    pip3 install tensorboard
    ```
* conda 常用命令，查询/退出/激活/删除
    * `conda env list`, `conda deactivate`, `conda activate $ENV_NAME`, `conda env remove -n $ENV_NAME`
<br>


# 检查是否安装成功
```python
# torch-gpu, cudnn
import torch
from torch.backends import cudnn
print(torch.cuda.is_available())
print(cudnn.is_available())
a = torch.tensor(1.)
print(cudnn.is_acceptable(a.cuda()))

# tensorflow-gpu
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
print(tf.__version__)
print(tf.__path__)
print(tf.test.is_gpu_available())

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(C)
```
Tensorflow可能会提示warning，说TensorRT未安装，如果不用的话不用管：
```
2020-04-18 13:35:50.278790: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/home/xian/cuda-10.1/lib64
2020-04-18 13:35:50.278888: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/home/xian/cuda-10.1/lib64
2020-04-18 13:35:50.278903: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
```



<br>


# Others
## Non-root下安装
在多个用户共用一台server的情况下，每个user可以安装不同的 CUDA Toolkit / cuDNN / tensorflow / pytorch 版本：
* Nvidia显卡driver必须要用root权限安装，多用户共用一个driver版本
* 其他的都用非root权限安装，并安装在user目录（CUDA Installer包的运行还是要用root权限，否则报错segmentation fault)。进入后，不勾选install driver，并且更改安装路径为user内部）
* 提前改变user目录的所有者： 
    ```bash
    su root
    chown -R alex /home/alex
    ls -l
    ```

## 查看CUDA，cuDNN安装路径、版本等
* 如果安装在root目录：
    * CUDA 版本：`cat /usr/local/cuda/version.txt`
    * cuDNN 版本：`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`
* 安装在non-root目录，手动搜索安装路径：
    * `find -name "*cuda*"`
    * `find -name "*cudnn*"`

## Driver/CUDA/cuDNN等的关系
* Driver 安装之后，可用`nvidia-smi`查看。`nvidia-smi`中显示的`CUDA version`仅代表该 driver 兼容的 CUDA 版本，并不意味着 CUDA 已经安装。(The CUDA VERSION display within nvidia-smi was not added until driver 410.72. CUDA VERSION displayed by nvidia-smi associated with newer drivers is the DRIVER API COMPATIBILITY VERSION.)

* CUDA Installer内含特定版本Nvidia显卡驱动的，所以只选择下载CUDA Installer就足够了。如果想安装其他版本的显卡驱动就下载相应版本即可：https://www.nvidia.com/Download/index.aspx?lang=en-us

* NVIDIA显卡驱动和CUDA Toolkit本身不具有捆绑关系的，也不是一一对应。只不过是离线安装的CUDA Toolkit会默认携带与之匹配的最新的驱动程序。**CUDA本质上只是一个工具包而已**，所以我可以在同一个设备上安装很多个不同版本的CUDA工具包，例如 CUDA 9.0、CUDA 9.2、CUDA 10.0三个版本。一般情况下，我们直接安装最新版的显卡驱动，然后在线安装不同版本的CUDA即可。

* cuDNN是一个SDK，是一个专门用于神经网络的加速包。它跟我们的CUDA没有一一对应的关系，即每一个版本的CUDA可能有好几个版本的cuDNN与之对应，但一般有一个最新版本的cuDNN版本与CUDA对应更好。

## Using Docker
* Install CUDA driver-> Install docker -> Install nvidia-docker -> pull images of CUDA tools:  
https://devblogs.nvidia.com/nvidia-docker-gpu-server-application-deployment-made-easy/  
* If meet problem in driver installation:  
https://blog.csdn.net/u014561933/article/details/79958130
