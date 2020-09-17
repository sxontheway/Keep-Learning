# Few Sample Learning
> Learning from Very Few Samples: A Survey  
> https://arxiv.org/abs/2009.02653

## Definition
<p align="center" >
<img src="./pictures/fsl_def.png" width="600">
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
<img src="./pictures/meta.jpg" width="500">
</p>

<p align="center" >
<img src="./pictures/meta_set.jpg" width="500">
</p>

* 普通的learning，上图所有的方框步骤都是人指定的。Meta-learning就是把某些红框中的步骤留给模型自己学。任何一个红框改变，那么得到的learning Algorithm都是不同的，例如使用不同的初始化参数 `θ^0`，就得到了不同的learning algorithm。

* meta-learning 的目的是使得网络具有 learn-to-learn 能力的一类方法的统称。广义上来讲，NAS（对象是网络结构），AutoML（对象是超参数）都属于 meta-learning。在 few-shot 中，meta-learning 一般用于寻找一个好的初始化权重，有了这个权重，用少量数据就能训练得比较好，例如 `MAML，Reptile` 两篇

* MAML：
    > https://zhuanlan.zhihu.com/p/72920138
    > https://zhuanlan.zhihu.com/p/66926599

    MAML 要解决的问题，本质上是使得网络在所有 training task 的 query set 上的loss之和最小化。经过一些数学近似（忽略高阶项），训练策略可简化为：

    * 对于每一个training task：设初始模型为 M0，先在 support set 上迭代一次，进行一次BP，得到 M1。M1再在 query set 迭代一次，算出梯度，最终用该梯度对 M0 进行更新得到 M'，这就完成了在一个 training task 上的学习  
    <p align="center" >
    <img src="./pictures/maml.jpg" width="400">
    </p>

* Reptile
    > https://zhuanlan.zhihu.com/p/239929601

    * MAML 中 training tasks 是分为 support set 和 query set，Reptile 不用分了
    * 对于每个 training task，MAML 和 Reptile 都只走一步，但是方向不一样

    <p align="center" >
    <img src="./pictures/reptile.jpg" width="400">
    </p>

* 为什么 few-shot learning 经常和 meta learning 一起被提起？  
Meta-learning 是解决 Few-shot 问题的一种训练策略，可以和其他方法（例如data augmentation）一起使用来解决 few-shot 问题。 Meta-learning 的目标是使网络 learn-to-learn，而这种能力正是 few-shot 中所需要的


<br>

# Papers

## Imprinting
> Low-Shot Learning with Imprinted Weights: https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf
<p align="center" >
<img src="./pictures/imprinting.png" width="600">
</p>

* 文中解释了：（FC layer + softmax classifier） 和 （triplet-based embedding training + Nearest Neighbor）两种方法原理上是相通的
* 灵魂性的句子： Intuitively, one can think of the imprinting operation as
remembering the semantic embeddings of low-shot examples as the templates for new classes
* 实验表明，这种方法甚至无需在 low-shot examples 上 finetune，即可达到较好效果

## Few-Shot Attention RPN
> Few-Shot Attention RPN: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Few-Shot_Object_Detection_With_Attention-RPN_and_Multi-Relation_Detector_CVPR_2020_paper.pdf

<p align="center" >
<img src="./pictures/attention_rpn.png" width="600">
</p>

* 在 training task 上训练时，有两个loss，一个是match loss，另一个box regression loss （training task 和 test task 类别无交集）
* 在 test 之前可在 support set 上进行 finetine（因为根据定义：test task 中，query set 和 support set 类别是相同的，且具有相似分布），并有两种方式进行inference：
    * Support image 和 query image 一起输入网络（需要运行两个branch），最后 match 的分数可作为置信度
    * 将 support images 对应的绿色 feature maps 存成离线的（提供大量prior），inference时就只用运行下面的branch即可
