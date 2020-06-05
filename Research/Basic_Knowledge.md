## 关于Loss Function
> https://zhuanlan.zhihu.com/p/36670444   
> https://www.cnblogs.com/yinheyi/p/6131262.html

* `Softmax Function + Cross Entropy Loss` 和 `Logistic Function + Logistic Loss` 关系？
    > https://www.zhihu.com/question/36981158 
    * 前者是后者从二分到多分类的推广
    * softmax 和 logistic 函数都将负无穷到正无穷映射到 [0, 1] 
    * softmax强调互斥性，多个类的概率加起来为1；logistic没有互斥性。
        * 例如people包含male，那么应该对people和male两类分别用求logistic loss，再相加；而不是用softmax
        * 例如要判断一张图是笑脸还是哭脸，推荐做法：  
        网络输出`1*2`的向量-> softmax -> focal loss/BCE loss -> 看两个数哪个大，得到结果   
        如果只输出一个标量，用sigmoid，那么只最后得到一个score，还需要手动选取阈值确定是笑脸还是哭脸，而这个阈值可能和训练数据分布有很大关系。

* 为什么 Loss Function 要用对数？  
    * 对于 Logistic Loss Function: 最小化Loss Function = 最大化二项分布的对数似然函数 = 最大化二项分布的似然函数
    * 同理，对于Cross Entropy: 最小化Loss Function = 最大化多项分布的对数似然函数

<br>


## 关于 One-hot Coding
> https://www.zhihu.com/question/53802526/answer/515535985

为了使得各种结果之间距离相等

<br>


## LSTM
> https://www.zhihu.com/question/64470274

<p align="center" >
	<img src="./pictures/lstm.jpg">
</p>

* A被称作cell，LSTM的`cell`在每个`time_step`是复用的
* `num_units` = `h_t` 和 `s_t` 的维度大小（两者相同维度）= 最后一个time step输出维度大小（输出为h） = 黄色框的输出维度大小 = `hidden_size`
* LSTM参数量：`(hidden_size * (hidden_size + x_dim ) + hidden_size) *4 `，因为 `f = sigma(W[h, x] + b)`，相当于将维度 `(hidden_size + x_dim)` 变到了 `hidden_size`