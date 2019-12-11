
## 各框架模型保存的格式汇总
> https://blog.csdn.net/charlotte_android/article/details/93889130 
### Tensorflow
* CheckPoint (.ckpt)  
在训练 TensorFlow 模型时，每迭代若干轮需要保存一次权值到磁盘，称为“checkpoint”。这种格式文件是由 tf.train.Saver() 对象调用 saver.save() 生成的，只包含若干 Variables 对象序列化后的数据，不包含图结构，**所以只给 checkpoint 模型不提供代码是无法重新构建计算图的。**  

* GraphDef (.pb)  
.pb 为二进制文件。这种格式文件包含 protobuf 对象序列化后的数据，包含了计算图，可以从中得到所有运算符（operators）的细节，也包含张量（tensors）和 Variables 定义，但不包含 Variable 的值。`.pb` 有两种类型：
    * FrozenGraphDef 类：尽管`.pb`不能包含variable的只，但FrozenGraphDef将所有variable都变成了tf.constant，和graph一起frozen到一个文件，可用于作为预训练模型或推理
    * GraphDef 类：不包含variable的值，因此只能从中恢复计算图，但一些训练的权值和参数需要从ckpt文件中恢复。

---
### Keras
* 保存整个模型（.h5），可用于继续训练  
`model.save(filepath)`将Keras模型和权重保存在一个HDF5文件中，该文件将包含：
模型的结构，模型的权重，训练配置（损失函数，优化器，准确率等），优化器的状态

* 保存模型结构  
model.to_json()将模型序列化保存为json文件，里面记录了网络的整体结构, 各个层的参数设置等信息. 将json字符串保存到文件.
除了json格式,还可以保存为yaml格式的字符串，形式与JSON一样

* 保存模型权重（.h5）  
经过调参后网络的输出精度比较满意后,可以将训练好的网络权重参数保存下来，可通过下面的代码利用HDF5进行保存：
`model.save_weights(‘model_weights.h5’)`

---
### Pytorch
> https://zhuanlan.zhihu.com/p/38056115 
* 保存和加载整个模型( `.pth`或`.pkl`都可以，没区别，都是以二进制文件存储)。类同tensorflow中的`.ckpt`，不能进行平台迁移。 
    ```python
    checkpoint = {
        "model": model.state_dict(),      # model是一个继承了torch.nn.Module的类的对象
        "optimizer": optimizer.state_dict(),    # 例如 optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        'best_loss': lossMIN,
        'epoch': epochID + 1
    }
     
    # checkpoint 是一个字典 
    torch.save(checkpoint, os.path.join( path, "epoch%d.pth" % epoch)))   #
    checkpoint = torch.load('XXX.pth')
    ```

* 仅保存和加载模型参数(推荐使用)，可进行平台见的迁移。  
    > state_ditc 详见：https://zhuanlan.zhihu.com/p/38056115   

    PS：平台迁移的意思是，例如在pytorch和Tensorflow中分别构建了一个相同的网络，并且每一层的命名这些也都是一样的，那么在一个框架下训练的模型文件可以在另外一个框架中直接拿来用。checkpoint文件就没有这种功能。

    ```python
    torch.save(model.state_dict(), 'params.pkl')
    model.load_state_dict(torch.load('params.pkl'))
    ```

* 其他保存数据的格式：`.t7`文件，`.pth`文件，`.pkl`格式，`.h5`文件等。

<br><br>

## 从tensorflow模型到pytorch模型
> https://blog.csdn.net/weixin_42699651/article/details/88932670  
> 注意这种方法，只在两个框架中网络保存时，参数名一致时才能用

例如从 [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 下载权重。下载后，文件夹中有 `checkpoint`， `model.ckpt.data-00000-of-00001`， `model.ckpt.index`， `model.ckpt.meta` 几个文件    

执行命令：`python3 ckpt2h5.py ./model.ckpt`
会生成: `model.h5`文件
 ```python
 # File ckpt2h5.py
 import tensorflow as tf
import deepdish as dd
import argparse
import os
import numpy as np

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to deepdish hdf5")
    parser.add_argument("infile", type=str,
                        help="Path to the ckpt.")  
    parser.add_argument("outfile", type=str, nargs='?', default='',
                        help="Output file (inferred if missing).")
    args = parser.parse_args()
    if args.outfile == '':
        args.outfile = os.path.splitext(args.infile)[0] + '.h5'
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    weights = read_ckpt(args.infile)
    dd.io.save(args.outfile, weights)
 ```
 以上代码将.ckpt文件转换成了一个.h5文件，文件中包含网络权重


```python
# 在pytorch中将权重读出来
net = ...
import torch
import deepdish as dd
net = resnet50(..)
model_dict = net.state_dict()

# 将字典内部元素从 numpy 转换为 tensor 类型
weights_dict =  = dd.io.load('./model.h5')  # 将.h5读入成 OrderedDict
tensor_dict = {}
for k,v in weights_dict.items():
    tensor_dict[k] = torch.Tensor(v)
    print(k)    # 可以看到网络每一层的名字是什么

model_dict.update(tensor_dict)
net.load_state_dict(model_dict)
```