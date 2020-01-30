* `invalid syntax (<unknown>, line 55)[55,1]`  
可能是上一行的错误， 例如少一个 ")"  

* numpy.int_()  
在本地，一些时候`np.int_(a)`，其中`a=np.array([ 0.])`时， 会返回一个`numpy.int64`类型，而非`numpy.ndarray`，导致后续操作报错，进行了如下尝试  
  * 在本地，将`np.int_()`去掉后就不会报错，于是推测为`np.int_()`自身的问题  
  * 但在`jupyter-notebook`中输入`np.int_(np.array([ 0.]))`，没有问题，于是推测该错误是可能因为本地numpy版本过老
  * 经查证，`jupyter-notebook`的numpy版本为1.16.2， 本地为1.11.0。但将本地numpy升级后，问题仍未解决
  * __弃用`np.int_()`，转而使用`.astype(int)`，问题解决__   


* python的and的返回值  
  https://stackoverflow.com/questions/32192163/python-and-operator-on-two-boolean-lists-how  
  x and y: 如果 x 为 False，返回 False，否则它返回 y 的计算值；  
  只有`'', (), []`这种 empty sequence才是False，例如下面：`x and y`：因为 x 其实为 True， 所以直接输出 y
  ```python
  x = [True, True, False, False]
  y = [False, True, True, Fasle]
  print(x and y)
  print(y and x)

  >>> [False, True, False, True] 
  >>> [True, False, False, True]
  ```
  要想按元素与运算，用 `np.logical_and(x,y)`或`[a and b for a, b in zip(x, y)]`
