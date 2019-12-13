## 关于Loss Function
> https://zhuanlan.zhihu.com/p/36670444   
> https://www.cnblogs.com/yinheyi/p/6131262.html
* `Softmax Function + Cross Entropy` 和 `Logistic Function + Logistic Loss` 关系？
    * 前者是后者从二分到多分类的推广
* Softmax, Logistic的主要作用？
    * 将负无穷到正无穷映射到 [0, 1]
* 为什么Loss Function要用对数？  
    * 对于Logistic Loss Function: 最小化Loss Function = 最大化二项分布的似然函数 = 最大化二项分布的对数似然函数
    * 同理，对于Cross Entropy: 最小化Loss Function = 最大化多项分布的对数似然函数

## 关于 One-hot Coding
> https://www.zhihu.com/question/53802526/answer/515535985 

