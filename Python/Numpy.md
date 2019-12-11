# 1. 安装
* pip 是 Python 包管理工具，该工具提供了对Python 包的查找、下载、安装、卸载的功能。  
* Ubuntu 16.04 自带 python2.7，python3.5.2。两个版本的 python 都自带 pip，版本为 8.1。  
* 输入`python -V`，`python3 -V`查看python版本，输入`pip -version`，`pip3 -version`查看pip版本

python3 下numpy等的安装：（python2 下把命令中的 3 去掉即可）
```bash
# 安装numpy等一系列包
python3 -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose     
```
打开 jupyter：`jupyter notebook`

---
<br>

# 2. NumPy 的使用
## 2.1 数据结构： list, tuple, array, mat, dictionary
> python中提供： list, tuple, dictionary中就有的  
> numpy中提供： array, mat  
> tuple, list, array 三者有很多相似之处 
* list 和 tuple 及 zip
    * 列表和元组的区别：创建元组后无法修改（替换、删除已有元素，加入新元素）； 而列表比较灵活
    * 列表和元组内的元素可以为任意对象： 列表内可以放元组，元组内可以装列表
    ```python
    # 用[]表示列表，用()表示元组
    a = [1,2,3]; b = (1,2,3)； 
    c = [a,b]; d = [a,b]
    ```
    > zip的用法： http://www.runoob.com/python3/python3-func-zip.html
* list 和 array  
    简单来说，np.array支持比list更多的索引方式，在numpy中使用更方便
    ```python
    # 创建长度为定值的空二维列表
    empty_list = [[]]*10

    # 用python创建二维列表，type(a1)得到list
    a1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    b = a1[:, 1]  # 报错
    c = np.array(a1)[；， 1] # 正确

    # 用numpy创建二维数组，type(a2)得到numpy.ndarray
    a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # 列表与数组的相互转换
    a3 = a2.tolist()
    a3 = np.array(a1)
    ```
* array 和 mat  
    Numpy中,mat必须是2维的，但是array可以是多维的（1D，2D，3D····ND），mat相对于array的一个优势是可用 \* 直接进行矩阵乘法运算
    ```python
    a1 = np.mat([[1,2], [3,4]])  # 正确
    a2 = np.mat([ [[1,2],[3,4]], [[1,2],[3,4]] ])  # 会报错
    a3 = np.array([ [[1,2],[3,4]], [[1,2],[3,4]] ])  # 正确

    # 用mat，*代表矩阵乘法； 用array，*表示按元素相乘，矩阵乘法要用np.dot（a, b） 或 a.dot(b)
    print(a1*a1)
    print(a3*a3)
    print(np.dot(a3,a3))
    ```
* 其他
    * 获取大小（最好记的方法）：字典用`len(A)`，其他四个用`np.shape(A)[0]， np.size(A)`，见 https://blog.csdn.net/zenghaitao0128/article/details/78574131
    * 构建空矩阵`np.zeros(), np.empty()`， 例如`np.zeros((1, 2, 3)).shape`输出(1,2,3)
    * 将array或mat压成1维向量`A.flatten（）`，用法`np.shape(a.flatten())`
    * 合并`np.append(a,b,axis=1)`，见 https://docs.scipy.org/doc/numpy/reference/generated/numpy.append.html
    * 删除某些行或列`np.delete()`，见 https://docs.scipy.org/doc/numpy/reference/generated/numpy.delete.html
    * 获取指定元素位置：
        ```python
        A = np.array([ [[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]] ])
        loc = np.where(A==6)    # np.where() 只能用于 array
        # 返回 (array([0, 1]), array([1, 1]), array([2, 2]))，这是个tuple，由3个array构成，每个array是一个1*2向量
        # 3 = array的个数 = A的维数；  2 = 每个array包含的元素数量 = 值为6的元素的数量；  参考3.3.3 
        ```

---
        
## 2.2 几个细节
### 2.2.1 \[3, ]和\[3, 1]是不同的shape
```python
a = [[1, 2, 3]]; b = [1, 2, 3]; c = [[1], [2], [3]]
np.shape(a)    # 得到(1, 3)
np.shape(b)    # 得到(3, )
np.shape(c)    # 得到(3, 1)
```
### 2.2.2 直接赋值，浅拷贝，深拷贝
* 直接赋值，会采用shared memory方式，也即只是创建了一个对象的引用
* .copy()创建浅拷贝：深拷贝父对象（一级目录），子对象（二级目录）还是引用
* copy.deepcopy()深拷贝:完全拷贝了父对象及其子对象。 深拷贝需要import copy模块
   ```python
   import copy

   dict1 = {'user':'a','num':[1,2,3]} 
   dict2 = dict1              # 对象的引用
   dict3 = dict1.copy()       # 浅拷贝：深拷贝父对象（一级目录），子对象（二级目录）不拷贝，还是引用
   dict4 = copy.deepcopy(dict1)   # 深拷贝，完全拷贝了父对象及其子对象

   # 修改 data 数据
   dict2['user']='b'
   dict2['num'].remove(1)

   # 输出结果
   print(dict1)         # {'user': 'b', 'num': [2, 3]}
   print(dict2)         # {'user': 'b', 'num': [2, 3]}
   print(dict3)         # {'user': 'a', 'num': [2, 3]}
   print(dict4)         # {'user': 'a', 'num': [1, 2, 3]}
   ```
   > 实例中对dict2的修改会影响dict1。 dict3对dict1的父对象进行了深拷贝，不会随dict1修改而修改，子对象是浅拷贝所以随dict1的修改而修改。 dict4是dict1的深拷贝。
   

### 2.2.3 index索引会导致将降维
 ```python
# a[1, :]中1是indexing，：是slicing
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]    # 混用 index 和 slice，用了 index 的维度消失（也即少了一对[]）
row_r2 = a[1:2, :]  # 只用 slice，输出和输入维度相同
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"
```

### 2.2.4 numpy矩阵的rescale,resize,downsampling
安装： `pip3 install -U scikit-image`
```python
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

tmp = np.array(range(100))
test = tmp.reshape(10, 10, 1)   
test_new = resize(test, (50,50), anti_aliasing=True)

plt.figure(figsize=(5, 5))
plt.imshow(test_new)
plt.ion()
plt.pause(3)
plt.close()
```
参见： https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image

---

## 2.3 索引
### 2.3.3 多维array作为索引
```python
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"，[0, 1, 2]是row indices，[0, 1, 0]是column indices
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"
```
### 2.3.2 用array作索引的应用
```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"
a[np.arange(4), b] += 10   # a = [[11, 2, 3], [4, 5, 16], [17, 8, 9], [10, 21, 12]]
```
### 2.3.3 Broadcast的应用
* 当对两个array进行element-wise运算时，NumPy会逐个比较它们的shape。只有在下述情况下才能够 broadcasting：
  * 相等
  * 其中一个为1，（进而可进行拷贝拓展，直到shape匹配）
```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y) # Prints "[[ 2  2  4]，[ 5  5  7],  [ 8  8 10], [11 11 13]]"
```
> 见 https://blog.csdn.net/lanchunhui/article/details/50158975

### 2.3.4 花式索引
如果目标是一维数组，那么索引的结果就是对应位置的元素；如果目标是二维数组，那么就是对应下标的行,尝试以下代码：
```python
a = np.array([1, 2, 3, 4, 5, 6, 7 ,8])
b = np.array([0, 1, 2, 3])
print(a[b])

c = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
d = np.array([[1, 0], [2, 3]])  
print(c[d], np.shape(c[d]))
```
> 见 http://www.runoob.com/numpy/numpy-advanced-indexing.html 
