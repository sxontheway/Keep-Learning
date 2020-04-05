# 1.安装
## 1.1 后端Tensorflow的安装
按照官网用pip安装： https://www.tensorflow.org/install/pip?hl=zh-cn  
* Warning： Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
  * 原因：  
  当用GPU时，可忽略它。 这个警告简而言之就是CPU支持AVX（高级向量扩展指令集），但是安装的tensorflow并没有运用这个功能，不能实现CPU上的加速，参见： 
  https://blog.csdn.net/hq86937375/article/details/79696023
  * 解决方案：  
  从源码安装可解决这个问题，参见：https://github.com/lakshayg/tensorflow-build
  * 验证是否安装成功:  
  查看 cuDNN version：`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`  
  查看 CUDA version: `cat /usr/local/cuda/version.txt` 
    ```python
    import tensorflow as tf 
    import os 

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 屏蔽掉 info 等级的log
    tf.test.is_gpu_available()
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)
    ```

## 1.2 前端Kares的安装
* 为配合tensorflow， 使用python3: 
  * 进入python3的虚拟环境(venv为创建虚拟环境时自定义的文件夹名称):` source ./venv/bin/activate`
  * `pip install keras`

<br><br>

# 2. 模型的格式
* Tensorflow训练后的模型可以保存`ckpt`文件或`pb`文件。
    * `ckpt`导出网络结构与权值分离的四个文件，一般用于继续训练，但是需要原来的代码支持，所以不能在多平台、编程语言中迁移，在恢复模型之前需要再定义一遍网络结构
    * `.pb`文件则是`GraphDef`或`FrozenGraphDef`的序列化文件，训练好的模型可在多平台、多语言中迁移。`.pb` 有两种：
        * FrozenGraphDef 类：包含所有的variable，但是所有的variable都已经变成了tf.constant和graph一起frozen到一个文件，可用于作为预训练模型或推理
        * GraphDef 类：不包含variable的值，因此只能从中恢复计算图，但一些训练的权值和参数需要从ckpt文件中恢复。

* 官方提供`freeze_graph.py`脚本可将`ckpt`文件转为`pb`文件
## 2.1 名词定义
> https://tensorflow.juejin.im/extend/tool_developers/index.html
> https://zhuanlan.zhihu.com/p/31308381

`Protocol Buffers (简称protobufs)`：  
是Google开源的一个与语言无关、平台无关的序列化协议。所谓序列化就是，在Tensorflow上写好也训练好一个神经网络之后，怎样把这个模型保存起来，以方便抑制到其他平台上（例如嵌入式系统）。**protobufs支持二进制格式（.pb文件）、文本格式（.pbtxt文件，方便人阅读）两种格式。** 文本文件结构跟 XML，Json等文件结构类似。所有的 TensorFlow 文件格式都是基于 Protocol Buffers的

`Graph`：  
是一个抽象概念，一些 Operation 和 Tensor 的集合就叫做 Graph

`MetaGraph`：  
一个Meta Graph 由一个计算图和其相关的元数据构成。其包含了用于继续训练，实施评估和（在已训练好的的图上）做前向推断的信息

`GraphDef`：  
 简单理解就是 Graph 按 protobufs 协议序列化之后的对象，用`.pb`文件存储。这种格式文件包含了计算图，可以从中得到所有运算符的细节，也包含张量（tensors）和 Variables 定义，**但不包含 Variable 的值，因此只能从中恢复计算图，训练需要的的权值仍需要从 checkpoint 中恢复**

`FrozenGraphDef`：  
 TensorFlow 一些例程中用到 `frozen_inference_graph.pb` 文件作为预训练模型，这和上面 GraphDef 不同，属于冻结（Frozen）后的 GraphDef 文件，简称 FrozenGraphDef 格式。GraphDef 虽然不能保存 Variable，但可以保存 Constant，通过 tf.constant 将 weight 直接存储在 NodeDef 中

`NodeDef`：  
 NodeDef是GraphDef的组成部分。一个Graph由很多个Node构成，每个Node都定义了一个运算操作和输入连接

`MetaGraphDef`： 
一个类的名称，是MetaGraph的具体实现。




## 2.2 Checkpoint (.ckpt文件) ---四件套
> https://www.zhihu.com/question/61946760

* API
    ```python
    saver = tf.train.Saver()
    saver.save(session, checkpoint_path)        # 保存checkpoint
    saver.restore(session, checkpoint_path)     # 载入checkpoint
    ```

* CheckPoint (*.ckpt)
    ```python
    import tensorflow as tf
    
    if __name__ == "__main__":
        #定义两个变量
        a = tf.Variable(tf.constant(1.0,shape=[1],name="a"))
        b = tf.Variable(tf.constant(2.0,shape=[1],name="b"))
        c = a + b
    
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
    
        #声明一个保存
        saver = tf.train.Saver()
        saver.save(sess,"./model.ckpt")
    ```
    运行完上述代码，会产生四个文件：  
    `checkpoint`: 保存了一个目录下所有断点模型文件列表，可以用来迅速查找最近一次的断点文件  
    `model.ckpt.meta`: 是MetaGraphDef序列化的二进制文件，保存了网络结构相关的数据，包括graph_def和saver_def等  
    `model.ckpt.data-00000-of-00001`: 保存所有变量的值：网络权值, 梯度, 超参数等  
    `model.ckpt.index`: 文件为数据文件提供索引

## 2.3 Protocol Buffer (.pb文件)
> https://zhuanlan.zhihu.com/p/60069860  

从.pb文件中构建图 (GraphDef)
```python
import tensorflow as tf

graph = tf.GraphDef()
with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)
```

