# 1. Pytorch的安装
> 可考虑是否在虚拟环境中安装， 安装步骤见 https://pytorch.org/
### 1.1 CUDA，cuDNN 的区别？
> 见 [CUDA Installation](../Hardware_Related/CUDA_Installation)
* CUDA is a general purpose parallel computing **PLATFORM** and programming model that leverages the parallel compute engine in NVIDIA GPUs in a more efficient way on a CPU
* cuDNN(CUDA Deep Neural Network library) is a **LIBRARY**


---
<br><br>
# 2. Pytorch 的使用
## 2.1 常用命令
### 2.1.1 torch.nn.Sequential() 和 torch.nn.ModuleList()
> 见 https://zhuanlan.zhihu.com/p/64990232  
* ModuleList 就是一个储存各种模块的 list，这些模块之间没有联系，没有实现 forward 功能。相比于普通的 Python list，ModuleList 可以把添加到其中的模块和参数自动注册到网络上。
* Sequential 内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部 forward 功能已经实现，可以使代码更加整洁。

### 2.1.2 torch.nn 和 torch.nn.functional 的区别
> torch.nn API: https://pytorch.org/docs/stable/nn.html 
* torch.nn 是里面包含的是，torch.nn.functional 里面包含的是函数。  
* 如果我们只保留nn.functional下的函数的话，在训练或者使用时，我们就要手动去维护weight, bias, stride这些中间量的值，这显然是给用户带来了不便。  
* 而如果我们只保留nn下的类的话，其实就牺牲了一部分灵活性，因为做一些简单的计算都需要创造一个类，这也与PyTorch的风格不符。
  > 见 https://www.zhihu.com/question/66782101/answer/246341271

### 2.1.3  torch.no_grad(),  torch.set_grad_enabled(), torch.enable_grad() 和 model.eval()
>见 https://www.cnblogs.com/guoyaohua/p/8724433.html

* `model.eval()`: changes the forward() behaviour of the module it is called upon. It disables certain layers exclusive for training stage. BN层一般放在conv层后面，激活函数之前；Dropout对于conv层和FC层都可以适用   
  * 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新  
  * 在val模式下，dropout层会让所有的激活单元都通过；而batchnorm层会停止计算和更新mean和var，用从所有训练实例中获得的统计量来代替Mini-Batch里面m个训练实例获得的mean和var的统计量

* `torch.no_grad()` or `torch.set_grad_enabled(False)`: Disable the gradient computation. In this mode, the result of every computation will have `requires_grad=False`, even when the inputs have `requires_grad=True`.  

  ```python
  # There is no different bewteen: 

  with torch.no_grad():
      <code>
      
  # and
  
  torch.set_grad_enabled(False)
      <code>
  torch.set_grad_enabled(True)   

  # 只是torch.set_grad_enabled()可以选择是开还是关梯度计算，
  # torch.no_grad()只能选择关
  ```


* `torch.enable_grad()`: Enables gradient calculation, if it has been disabled via `no_grad()` or `set_grad_enabled(False)`.
  ```python
  x = torch.tensor([1], requires_grad=True)
  with torch.no_grad():
      with torch.enable_grad():
      y = x * 2
  y.requires_grad
  # 输出 True
  ```

* `model.eval()` 和 `torch.no_grad()` 可以一起使用：
  ```python
  model = CNN()
  for e in num_epochs:
      # do training
      model.train()

  # evaluate model:
  model = model.eval()
  with torch.set_grad_enabled(False): 
      logits, probas = model(testset_features)
  ```

### 2.1.4 model.zero_grad() 和 optimizer.zero_grad()
> https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html

* optimizer.zero_grad() 有什么用 ?  
  一般的训练方式是进来一个batch更新一次梯度，所以每次计算梯度前都需要用 optimizer.zero_grad() 手动将梯度清零。如果不手动清零，pytorch会自动对梯度进行累加。
  * 梯度累加可以模拟更大的batch size，在内存不够大的时候，是一种用更大batch size训练的trick，见 https://www.zhihu.com/question/303070254/answer/573037166  
  * 梯度累加可以减少multi-task时的内存消耗问题。因为当调用了.backward()后，computation graph就从内存释放了。这样进行multi-task时，在任意时刻，在内存中最少只存储一个graph。 见 https://www.zhihu.com/question/303070254/answer/608153308
    ```python
    for idx, data in enumerate(train_loader):
      xs, ys = data
      
      optmizer.zero_grad()
      # 计算d(l1)/d(x)
      pred1 = model1(xs) #生成graph1
      loss = loss_fn1(pred1, ys)
      loss.backward()  #释放graph1

      # 计算d(l2)/d(x)
      pred2 = model2(xs) #生成graph2
      loss2 = loss_fn2(pred2, ys)
      loss.backward()  #释放graph2

      # 使用d(l1)/d(x)+d(l2)/d(x)进行优化
      optmizer.step()
    ```
* model.zero_grad() 和 optimizer.zero_grad() 的区别  
当`optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)`时，二者等效，其中SGD也可以换成其他优化器例如Adam。当一个model中用了多个optimizer时，model.zero_grad() 是将所有梯度清零，optimizer.zero_grad() 是清零一个optimizer

### 2.1.5 learning rate decay
  > https://www.cnblogs.com/wanghui-garcia/p/10895397.html  

  一般会用到`torch.optim.lr_scheduler.LambdaLR`， `torch.optim.lr_scheduler.StepLR`， `torch.optim.lr_scheduler.MultiStepLR`，使用格式为：
  ```python
  import torch.optim.lr_scheduler.StepLR
  scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
  for epoch in range(100):
      scheduler.step()
      optimier.step()
      optimizer.zero_grad()
      (other training steps...)
      (other validate steps...)
  ```
  查看 `learning rate`: `print(optimizer.param_groups[0]['lr'])`


### 2.1.6 .detach(), .detach_() 和 .data 区别
> https://www.cnblogs.com/wanghui-garcia/p/10677071.html  
> https://zhuanlan.zhihu.com/p/83329768  

* .detach() 和 .detach_()  
detach_() 是对 Variable 本身的更改，detach() 则是生成了一个新的 Variable
* .detach() 和 .data 相同点：  
  * requires_grad 都为 false，即使之后重新将它的requires_grad置为true，它也不会具有梯度grad
  * 都返回一个从当前计算图中分离下来的，新的Variable。但是仍指向原变量的存放位置，也即和原变量共享一块内存

* .detach() 和 .data 不同点：  
  * `.detach()`之后修改会被autograd追踪，保证了只要在backward过程中没有报错，那么梯度的计算就是正确的
  * `.data`之后的修改不会被autograd追踪，可能会产生错误的梯度，**所以 .data 不够安全**，用 `x.detach()` 更好

    .detach() 之后不进行修改：
    ```python
    import torch

    a = torch.tensor([1, 2, 3.], requires_grad=True)
    out = a.sigmoid()
    c = out.detach()
    
    # 这时候没有对c进行更改，所以并不会影响backward()
    out.sum().backward()
    ```

    .detach() 之后进行in-place修改：
    ```python
    import torch

    a = torch.tensor([1, 2, 3.], requires_grad=True)
    out = a.sigmoid()
    c = out.detach()

    c.zero_()
    print(c)  # tensor([0., 0., 0.])
    print(out)  # tensor([0., 0., 0.], grad_fn=<SigmoidBackward>)
    out.sum().backward()  # 报错
    ```

    .data 之后进行修改
    ```python
    import torch

    a = torch.tensor([1, 2, 3.], requires_grad=True)
    out = a.sigmoid()
    c = out.data

    # 会发现c的修改同时也会影响out的值
    c.zero_()
    print(c, out)

    # 不同之处在于.data的修改不会被autograd追踪，这样当进行backward()时它不会报错，会得到一个错误的backward值
    out.sum().backward()
    print(a.grad) # tensor([0., 0., 0.])
    ```

### 2.1.7 hook
> https://zhuanlan.zhihu.com/p/75054200  

pytorch 中，对于中间变量（由别的变量计算得到的变量），一旦完成了反向传播，它就会被释放掉以节约内存。利用hook，我们不必改变网络输入输出的结构，就可方便地获取、改变网络中间层变量的梯度  
使用方式： `y.register_hook(fn)`，其中自定义函数`fn(grad)`返回Tensor或没有返回值
```python
def save_grad():
    def hook(grad):
        print(grad)
    return hook

# register gradient hook for tensor y
if y.requires_grad == True:
    y.register_hook(save_grad())
```

### 2.1.8 其他
* torchvision 由以下四部分组成：  
  torchvision.datasets， torchvision.models， torchvision.transforms， torchvision.utils  
    > 见 https://pytorch.org/docs/master/torchvision/transforms.html?highlight=torchvision%20transforms  

  * torchvision.transforms 包含很多类，其中 torchvision.transforms.Compose() 可以把多个步骤合在一起  
  例如 torchvision.transforms.Compose(\[transforms.CenterCrop(10), transforms.ToTensor()])

* In PyTorch, every method that ends with an underscore (_) makes changes in-place, meaning, they will modify the underlying variable.

<br>

## 2.2 Tensor是什么
### 2.2.1 一个例子
```python
import torch

x = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])
x.requires_grad = True
y = x + 1
z = y * y
out = z.mean()
loss = 20 - out
loss.backward()

print(x.data, '\n', x.dtype, '\n', x.device, 
      '\n', x.grad, '\n', x.grad_fn, '\n', x.requires_grad, '\n')
print(x)
print(y)
print(z)
print(out)
```
运行得到:
```
tensor([[1., 2., 3.],
        [4., 5., 6.]]) 
torch.float32 
cpu 
tensor([[-0.6667, -1.0000, -1.3333],
      [-1.6667, -2.0000, -2.3333]]) 
None 
True 

tensor([[1., 2., 3.],
        [4., 5., 6.]], requires_grad=True)
tensor([[2., 3., 4.],
        [5., 6., 7.]], grad_fn=<AddBackward0>)
tensor([[ 4.,  9., 16.],
        [25., 36., 49.]], grad_fn=<MulBackward0>)
tensor(23.1667, grad_fn=<MeanBackward1>)
```
* x是一个tensor， x的核心部分 x.data 可以理解成一个n-dimensional array。 此外，tensor还有其他几个属性： `x.dtpe, x.device, x.grad, x.grad_fn`等， 其中： 
  * x.grad 是求得的梯度
  * x.requires_grad 表示该变量是否需要autograd
  * y.grad_fn 记录了该变量求导应该用的function。 例如y由加法得到， y.grad_fn = \<AddBackward0\>; z由乘法得到， y.grad_fn = \<MulBackward0\> 

### 2.2.2 tensor的操作
tensor 能像 numpy array 一样进行索引
* max 操作  
  `.max(k)`表示求第k维的最大值，对于二维tensor，求列最大 k = 0，行最大 k = 1
  ```python
  import torch
  a = torch.tensor([[1, 2], [3, 4], [5, 6]])
  print(a.max(1))     # 默认keepdim为false
  print(a.max(1)[0])  # 最大值
  print(a.max(1)[1])  # 最大值对应的index
  print(a.max(1, keepdim = True))
  ```
  输出
  ```
  torch.return_types.max(values=tensor([2, 4, 6]), indices=tensor([1, 1, 1])) 
  tensor([2, 4, 6]) 
  tensor([1, 1, 1])
  torch.return_types.max(values=tensor([[2], [4], [6]]), indices=tensor([[1], [1], [1]])) 
  ```

* 矩阵操作

  |用途|命令|
  | :------------ | :-----|
  |创建随机数矩阵|x = torch.rand(5, 3)|
  |创建正态分布随机数矩阵|x = torch.randn(2,4)| 
  |创建空矩阵|x = torch.empty(5, 3)|
  |创建零矩阵并指定类型|x = torch.zeros(5, 3, dtype=torch.long)|
  |直接指定元素值|x = torch.tensor([5.5, 0.02])|
  |维度变换|x = y.view(-1,10)|
  |去掉个数为1的维度|x = y.squeeze()|
  |one-hot encoding/decoding|pytorch.scatter_(), pytorch.gather()|

### 2.2.3 Tensor 和 Numpy 转换 
Tensor与numpy对象共享内存，但numpy只支持CPU，所以他们在CPU之间切换很快。但也意味着其中一个变化了，另外一个也会变。
* tensor 和 python 对象转换：  
`tensor.tolist()`：多个元素的tensor  
`tensor.item()`：只含一个元素的tensor

* Torch -> NumPy:
  ```python
  a = torch.ones(5) # Torch Tensor
  b = a.numpy() # NumPy Array
  ```
* Numpy -> Torch:
  ```python
  import numpy as np
  a = np.ones(5) # NumPy Array
  b = torch.from_numpy(a) # Torch Tensor
  ```
* 图像从`cv2.imread()`的`numpy`转换为可输入网络的`tensor`：  
cv2 numpy默认格式：`H*W*C, BGR`; torch tensor默认格式：`C*H*W, RGB`
  ```python
  import cv2
  import torchvision.transforms
  image_np = cv2.imread("1.jpg")

  # Method 1
  image_tensor = transforms.Totensor()(image_np)

  # Method 2
  image_np = image_np[:, :, ::-1]
  image_tensor = image_np.transpose((1,2,0))
  ```

### 2.2.4 在 CPU 和 GPU 之间移动数据
```python
# move the tensor to GPU
x = x.to("cuda")  # or x = x.cuda()

# directly create a tensor on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

# move the tensor to CPU
x = x.to("cpu") # or x = x.cpu()
```

---
<br><br>

# 3. 代码分析
> 参考:  
> https://github.com/pytorch/tutorials  
> https://github.com/pytorch/examples  
> http://pytorch.org/docs/  
> https://discuss.pytorch.org/

## 3.1 A Toy Example of Back Propagation
Only need to define the forward function, and the backward function is automatically defined.
* Define the network (step 1)
  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 6, 5)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)
      
    def forward(self, x):
      x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
      x = F.max_pool2d(F.relu(self.conv2(x)), 2)
      x = x.view(-1, 16 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
  return x

  net = Net()
  ```

* Process inputs (step 2)
  ```python
  input = torch.randn(1, 1, 32, 32)
  out = net(input)
  ```

* Compute the loss (step 3)
  ```python
  output = net(input)
  target = torch.randn(10) # a dummy target, for example
  target = target.view(1, -1) # reshape the dimension with -1 inferred from other dimensions (10/1=10, the same shape with input)
  criterion = nn.MSELoss()
  loss = criterion(output, target)
  ```

* Backprop and update the weights (step 4)
  ```python
  import torch.optim as optim
  optimizer = optim.SGD(net.parameters(), lr=0.01)    # optimizer obtains the references of parameters

  optimizer.zero_grad()     # zero the gradient buffers
  loss.backward()       # calculate the gradients of parameters
  optimizer.step()    # Does the update
  ```

<br>

## 3.2 Steps to Train a Classifier
* Load the dataset (step 1)
  ```python
  import torch
  import torchvision
  import torchvision.transforms as transforms

  transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
  ```

* Define the network. Same as before. (step 2)
  ```python
  import torch.nn as nn
  import torch.nn.functional as F

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 6, 5)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)
      
    def forward(self, x):
      x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
      x = F.max_pool2d(F.relu(self.conv2(x)), 2)
      x = x.view(-1, 16 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
      
  net = Net()
  ```
* Define the loss function and optimizer. Same as before. (step 3)
  ```python
  import torch.optim as optim

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  ```

* Train the network (step 4)
  ```python
  for epoch in range(2): # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
    
    # get the inputs
    inputs, labels = data
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999: # print every 2000 mini-batches
      print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0
  ```

* Test the network (step 5)
  ```python
  correct = 0
  total = 0
  with torch.no_grad():          # 因为Pytorch会自动计算梯度，但这里明确告诉它不用计算梯度了
    for data in testloader:
      images, labels = data
      outputs = net(images)      # 这里output是torch.autograd.Variable的类型
      _, predicted = torch.max(outputs.data, 1)     # output.data才是tensor格式，1代表在哪个维度求最大值
      total += labels.size(0)    # labels.size()=1
      correct += (predicted == labels).sum().item()    
    
  print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
  ```

* Options: Train on GPU/GPUs
  > 用多个GPU：`net = nn.DataParallel(net)`
  ```python
  # training on the first cuda device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  net.to(device)
  inputs, labels = inputs.to(device), labels.to(device)
  ```

<br>

## 3.3 Yolov3的实现摘要
```python
import torch.nn as nn
import torch.nn.functional as F 

def get_test_input():
    # 读取图片，得到torch变量(略)#
    return img_

def parse_cfg(cfgfile):
    #block格式是字典，代表网络中的一个Module (一个Module可能有多层)。blocks是block组成的列表，代表整个cfg文件。
    block = {}
    blocks = []
    # 此处要进行一些cfg文件读取操作(略)#
    return blocks

def create_modules(blocks):
    net_info = blocks[0]	#读取cfg文件中的[net]信息
    module_list = nn.ModuleList()	#module_list存储了用blocks构建的整个网络，module_list对应于blocks[1:]
    output_filters = []		#记录之前每个层的卷积核数量
    prev_filters = 3		#初始输入数据3通道。每次卷积都将prev_filters个通道变为filters个通道
    for index, x in enumerate(blocks[1:]): 	#x是一个字典，与block类似
        module = nn.Sequential()		#A module could have many layers (Conv, BN, ReLU...)
        if (x["type"] == "convolutional"):	#卷积层
            # 添加卷基层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias)
            module.add_module("conv_{0}".format(index), conv)		#Adds a child module to the current module
            #添加BN层
            bn = nn.BatchNorm2d(filters)
            module.add_module("batch_norm_{0}".format(index), bn)
            #其他层以此类推(略)#   
        elif (x["type"] == "upsample"):	#上采样层
            #写法同卷基层(略)#     
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)
```
