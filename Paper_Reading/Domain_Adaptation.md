# Domain Adaptive Object Detection
Let's divide the methods in domain adaptation using number of networks:

|Number of Networks Used in Total|Number of Networks to be Trained|Description|
| :------------: |:---------------:|:-----:|
| 1 | 1 | Self-Learning, Pseudo-Labeling, Regularized Tranfer Learning|
| 2 | 1 | Knowledge Distillation, Feature Learning |
| 2 | 2 | Co-Teaching |

<br>

* Few-shot domain adaptation  
    * `LSTD: A Low-Shot Transfer Detector for Object Detection_AAAI18` : Faster-RCNN + SSD + Extra Modules
    * `Few-shot Adaptive Faster_RCNN_CVPR19` :  Extra Modules

* Zero-shot domain adaptation
    * Feature Learning Based 
        * `Domain Adaptive Faster R-CNN for Object Detection in the Wild_CVPR18` : Add extra modeles
    * Pseudo-Labeling Based 
        * `Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation_CVPR18` : GAN-based domain transfer + pseudo labeling  
        * `Automatic adaptation of object detectors to new domains using self-training_CVPR19` : Pseudo-Labeling + Score mapping, detect only one class 
        * `A Robust Learning Approach to Domain Adaptive Object Detection_ICCV19` : How to train with noisy labels (robust learning)
        * `Training Object Detectors With Noisy Data_IV19` : co-teaching  

* Other Related Topics
    * Training on the Edge
        * `Training on the Edge: The why and the how_IPDPSW19`
        * `Performance Analysis and Characterization of Training Deep Learning Models on Mobile Devices_ArXiv19` : Deep learning model training on TX2
    * Incremental Learning on the Edge   
        * `RILOD: Near Real-Time Incremental Learning for Object Detection at the Edge_SEC19` : Pseudo labeling + Incremental learning
        * `In-situ AI: Towards Autonomous and Incremental Deep Learning for IoT Systems_HPCA18`
    * Automatically Labeling
        * `Mark Yourself: Road Marking Segmentation via Weakly-Supervised Annotations from Multimodal Data_ICRA18`

<br><br>


# Transfer Learning, Few-shot learning, Meta-learning 辨析
> https://juejin.im/entry/5bd2b5a16fb9a05cda77ab7e  
* Meta-learning 和 Transfer learning：
    * `Meta-SSD: Towards Fast Adaptation for Few-Shot Object Detection With Meta-Learning`：Meta-learning 多了一个 hypernetwork (也即 meta-learner)
* Few-shot learning 有很多方法，meta-learning 只是其中一种方法  
    * 可以采用 meta-learning 那种 `N-way K-shot` 的训练方式；也可以就用 dataloader 中普通的 shuffle 选项
    * 可以用上  meta-learner，也可以不用
* Meta-learning 也有很多用途，few-shot learning 只是其中一个应用  
    * Meta-learning 解决的目标是 Learning to learn，输入和输出数据可以完全是异构的。例如人即便没有见过斑马，仅仅借助文字信号输入就可以识别斑马：身上有类似斑马线的纹路，并且长相很像马的动物是斑马。
* 一个典型的 Few-shot detection 问题：  
    * 给定COCO数据集为source domain，PASCAL VOC数据集为target domain

<br><br>

# 正负样本均衡 
## Hard Example Mining
* 这么做的intuition：
    * 由于one-stage detector没有专门生成候选框的子网络，无法将候选框的数量减小到一个比较小的数量级（例如几百，但SSD300是8732），导致了绝大多数候选框都是负样本（背景类）
    * 负样本远多于正样本将导致梯度被简单负样本。尽管单个负样本造成的loss很小，但是由于它们的数量极其巨大，对loss的总体贡献还是占优的。而真正应该主导loss的正样本由于数量较少（因为只有正样本产生位置误差），无法真正发挥作用。这样就导致收敛不到一个好的结果。
* 在SSD中：https://www.cnblogs.com/xuanyuyt/p/7447111.html  
    * 训练用正样本 = 与某一个GT bbox的IOU最大的 + IOU大于0.5的(若存在)
    * 训练用负样本： 对所有负样本进行抽样，抽样时按照`confidence loss` 进行降序排列，选取 top-k 作为训练的负样本，这里就可以利用 k 来控制最后正、负样本的比例为 1 : 3。其中 `confidence loss` 是分类置信度误差，置信度越低，误差越高，说明越是hard examples

## Online Hard Example Mining
> https://zhuanlan.zhihu.com/p/78837273   
* Hard Example Mining：保留所有正例，只注意难负例  
* Online Hard Example Mining：注意所有难例（loss大的），不论正负。相当用了一个小网络来判断哪些样本是难例，原文是用在Fast-RCNN。

## Focal Loss: 
> https://www.cnblogs.com/xuanyuyt/p/7444468.html  

In this work, we identify class imbalance as the primary obstacle preventing one-stage object detectors from surpassing top-performing, two-stage methods, such as Faster R-CNN variants. To address this, we propose the focal loss which applies a modulating term to the cross entropy loss in order to focus learning on hard examples and down-weight the numerous easy negatives.

## Soft Samping:    
> `Soft Sampling for Robust Object Detection_BMCV19`

解决图像标注数据集中有漏标的情况，有两种解释：

* 训练数据的漏标会导致正确的检测结果被误判为假阳。对于有漏标的数据集，既然我们无法确定假阳是否真的是假阳，那就减少假阳的惩罚------减少与 GT 的 IOU 小于一定值（比如0.2）的预测在损失中的权重，也即减少了白框的权重
* 因为白色框这种easy negative样本很多，所以在BP的时候会占据很大梯度，而掩盖了hard negative samples（绿色框）的梯度。本文的方法就是使得减少白框的权重，使得他们贡献相对较小的总梯度

    <p align="center" >
    <img src="./pictures/soft_labeling.png", width='800'>
    </p>

