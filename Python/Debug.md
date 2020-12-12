> 本文记录代码实现过程中遇到的各种奇怪的 bug 及解决方案

---
<br>

* PIL image shape 是 `W*H`，numpy 存图片是 `H*W`，输入CNN的tensor是 `C*H*W`

<br>

* 画 2d histogram
  * 用 `np.histgram2d` + `plt.show()`
  * 固定 colorbar 的 scale
    ```python
    imshow(my_array, vmin=0, vmax=1)
    plt.colorbar()
    ```
  * `plt` 中画图是以左下角为零点，img 一般以左上角为零点，可用 `plt.gca().invert_yaxis()` 解决

<br>

* `cv2.rectangle(), cv2.circle()` 等代码不显示：https://blog.csdn.net/qq_36679208/article/details/103006091#_1  
opencv读入的图像是BGR，要转化为RGB，可以有如下两种实现，但第二种会导致 `cv2.rectangle()` 等命令不显示：

  ```python
  # 用这个
  image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

  # 不要用这个
  inp = img[:, :, [2, 1, 0]]  # BGR2RGB
  ```
<br>

* Yolov3中，当 gradient_accumulations=1时，会出现网络没有进行训练的情况
  ```python
  if batches_done % opt.gradient_accumulations:
     optimizer.step()
     optimizer.zero_grad()
  ```
  将 if 语句其改为`batches_done % opt.gradient_accumulations==0`

<br>

* `invalid syntax (<unknown>, line 55)[55,1]`  
可能是上一行的错误， 例如少一个 ")"  

<br>

* numpy.int_()  
在本地，一些时候`np.int_(a)`，其中`a=np.array([ 0.])`时， 会返回一个`numpy.int64`类型，而非`numpy.ndarray`，导致后续操作报错，进行了如下尝试  
  * 在本地，将`np.int_()`去掉后就不会报错，于是推测为`np.int_()`自身的问题  
  * 但在`jupyter-notebook`中输入`np.int_(np.array([ 0.]))`，没有问题，于是推测该错误是可能因为本地numpy版本过老
  * 经查证，`jupyter-notebook`的numpy版本为1.16.2， 本地为1.11.0。但将本地numpy升级后，问题仍未解决
  * __弃用`np.int_()`，转而使用`.astype(int)`，问题解决__   

<br>

* python中的bool类型  
  * `[tensor(False), tensor(True)]` 和 `[False, True]` 不一样， 不能作为mask
  * https://stackoverflow.com/questions/32192163/python-and-operator-on-two-boolean-lists-how  
    x and y: 如果 x 为 False，返回 False，否则它返回 y 的计算值；  
    只有`'', (), []`这种 empty sequence才是False，例如下面：`x and y`：因为 x 其实为 True， 所以直接输出 y
    ```python
    x = [True, True, False, False]
    y = [False, True, True, False]
    print(x and y)
    print(y and x)

    >>> [False, True, True, False]
    >>> [True, True, False, False]
    ```
    要想按元素与运算（两种方法）：
    * `[a and b for a, b in zip(x, y)]`
    * 将 list 转换为numpy array 然后用 `&` 或者 `np.logical_and(x,y)`
    
      ```python
      a = np.array([True, False])
      b = np.array([False, True])
      print(a & b)
      print(np.logical_and(a, b))

      >>> array(False, False)
      ```
  * 在numpy, torch中查找属于一个区间的元素：
    ```python
    import numpy as np
    a = np.array([[1,1],[2,2],[3,3],[4,4]])
    a = a[a[:, 1]<3]  # 正确
    a = a[1<a[:, 1]<3]  # 报错 The truth value of an array with more than one element is ambiguous.
    a = a[(1<a[:, 1]) & (a[:, 1]<3)] # 正确，原因见上
    a = a[np.where((1<a[:, 1]) & (a[:, 1]<3))]  # 正确

    import torch
    b = torch.from_numpy(a)
    b = b[b[:, 1]<3]  # 正确
    b = b[1<b[:, 1]<3]  # 报错 The truth value of an array with more than one element is ambiguous.
    b = b[(1<b[:, 1]) & (b[:, 1]<3)] # 正确，原因见上
    b = b[torch.where((1<b[:, 1]) & (b[:, 1]<3))]  # 正确
    ```

<br>