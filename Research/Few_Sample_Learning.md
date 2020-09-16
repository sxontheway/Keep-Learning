# Few Sample Learning
> Learning from Very Few Samples: A Survey  
> https://arxiv.org/abs/2009.02653

## Definition
<p align="center" >
<img src="./pictures/fsl_def.png" width="800">
</p>

* `D_A` 包含很多已标注数据，`D_T` 只包含少量标注数据，两者类别不重合，测试时在 `D_T` 上
* 方法上主要可分为 Generative 和 Deterministic

<br>

## Generative Methods
<p align="center" >
<img src="./pictures/gen1.png" width="400">
</p>

<p align="center" >
<img src="./pictures/gen2.png" width="400">
</p>

* 一般都用了一个 latent variable `z` 来求条件分布 `p(x|y)`。例如对于数字、字母识别等，每个类（也即`y`）都可以对应一个latent image（也即`z`），所有属于该类的图像（也即`x`）都可以由这个 latent image 经过变换之后得到；并且对于不同类，这个变换的分布可能是相同的
* 生成式的方法普遍只适用于特定一些场景，现在更多的基于深度学习的方式是判别式的

<br>

## Deterministic Methods
### Data Augmentation
* 利用 auxiliary dataset `D_A` 使得 Gan 能够扩充 `D_T` 的数据 
### Metric Learning
* 利用 auxiliary dataset `D_A` 训练一个网络，输入是 `x_i`，`x_j` （例如两张图），输出是他们之间的 distance。被称作 metric learning 的原因是因为这个网络要学习的是一个metric，来判断两个输入之间的distance。
* 用 `D_A` 训练后，可在 `D_T` 上 finetune。inference 阶段：将输入数据和 `D_T` 中同类别的数据进行比较，计算 distance，从而完成分类。
### Meta-Learning
<p align="center" >
<img src="./pictures/meta.jpg" width="600">
</p>

* meta-learning 是一种训练策略，可以和其他方法一起使用
* 普通的learning，上图所有的方框步骤都是人指定的。Meta-learning就是把某些红框中的步骤留给模型自己学。任何一个红框改变，那么得到的learning Algorithm都是不同的，例如使用不同的初始化参数 `θ^0`，就得到了不同的learning algorithm。一种meta-learning的方法就是让机器自己找到一个最好的初始化权重（其他红框不变情况下的一个最好的learning algorithm），例如 `MAML，Reptile` 两篇
* MAML：设初始模型为 M0，先在 support set 上迭代一次，进行一次BP，得到 M1。M1再在 query set 迭代一次，算出梯度，最终用梯度对 M0 进行更新得到 M'，这就完成了在一个 training task 上的学习。MAML解决的问题，本质上是使得网络在所有 training task 的 query set 上的loss之和最小化。参考: https://zhuanlan.zhihu.com/p/72920138 

* 为什么 few-shot learning 经常和 meta learning 一起被提起？  
Meta-learning 是 Few-shot learning 的一种训练策略。 Meta-learning 的目标是使得网络能够快速有效地学习到新东西，也及使网络有很好的学习能力，而这种能力正是 few-shot 中所需要的。


<br>

# Papers

## Imprinting
> Low-Shot Learning with Imprinted Weights: https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf

## Few-Shot Attention RPN
> Few-Shot Attention RPN: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Few-Shot_Object_Detection_With_Attention-RPN_and_Multi-Relation_Detector_CVPR_2020_paper.pdf

<p align="center" >
<img src="./pictures/attention_rpn.png" width="600">
</p>

* 在 training task 上训练时，有两个loss，一个是match loss，另一个box regression loss （training task 和 test task 类别无交集）
* 在 test 之前可在 support set 上进行 finetine（因为根据定义：test task 中，query set 和 support set 类别是相同的，且具有相似分布），并有两种方式进行inference：
    * Support image 和 query image 一起输入网络（需要运行两个branch），最后 match 的分数可作为置信度
    * 将 support images 对应的绿色 feature maps 存成离线的（提供大量prior），inference时就只用运行下面的branch即可