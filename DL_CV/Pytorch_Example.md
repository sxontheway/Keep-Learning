# Tensor Processing
```python
import torch
x = torch.zeros((16, 10, 30, 30), dtype=torch.float)
print(x.shape)

a = torch.stack((x,x,x),1)  # stack是建立一个新的维度
print(a.shape)

b = torch.cat((x,x,x), dim=1)  # cat是在已有的维度上拼接
print(b.shape)

c = x.repeat(1,1,2,1,1)
print(c.shape)

d = torch.unsqueeze(x, 2)
print(d.shape)

e = torch.unsqueeze(x, 2).repeat(1,1,3,1,1)
print(e.shape)
```
得到  
torch.Size([16, 10, 30, 30])  
torch.Size([16, 3, 10, 30, 30])  
torch.Size([16, 30, 30, 30])  
torch.Size([1, 16, 20, 30, 30])  
torch.Size([16, 10, 1, 30, 30])  
torch.Size([16, 10, 3, 30, 30])  

<br>

# Custom DataLoader Example
见： https://pytorch.org/tutorials/beginner/data_loading_tutorial.html  
https://zhuanlan.zhihu.com/p/30934236  
* 要点在于在 `CustomDataset(Dataset)`的`__init__`中不直接读入图片，而只读入csv文件，包含图片路径等；在`__getitem__`中才读入index所对应图片。这样可以节省内存。  
* Pytorch的数据读取主要包含三个类，这三者大致是一个依次封装的关系: 1被装进2, 2被装进3
    * Dataset: 提供了自定义数据集的方法，可在`__getitem__`中使用`transform`
      * `class torchvision.transforms.Compose(transforms)`  
      见: https://pytorch.org/docs/stable/torchvision/transforms.html  
      https://blog.csdn.net/Hansry/article/details/84071316
    * DataLoader: 在`Dataset`的基础上，加上了mini-batch, shuffle, multi-threading 的功能
    * DataLoaderIter
      ```python
      from torch.utils.data import Dataset, DataLoader
      from torchvision import transforms, utils

      class CustomDataset(Dataset):
         def __init__(self, transform = None):
             XXX
             self.transform = transform

         def __len__(self):
             XXX

         def __getitem__(self, idx):
             sample = XXX
             if self.transform:
                 sample = self.transform(sample)
             return sample

      my_dataset = CustomDataset(transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
      dataloader = Dataloader(my_dataset, batch_size=4, shuffle=True, num_workers=4)

      for index, sample in enumerate(dataloader):
         # training...
      ```
    * torchvision: torchvision package provides some common datasets and transforms  
    见： https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
* collate_fn的使用:
   > You can use your own collate_fn to process the list of samples to form a batch. The batch argument is a list with all your samples. E.g. if you would like to return variable-sized data.  
   > https://zhuanlan.zhihu.com/p/30385675  
   > https://blog.csdn.net/weixin_42028364/article/details/81675021  
   
   `collate_fn`中可以定义怎样将 从`__getitem__`获取的长度为`batch_size`的数据 组成`a batch of training data`，输入训练网络。比如文字识别，label是一个单词，每个label不一样长，需要先把他们统一成相同长度。或者`multi-scale training: selects new image size every tenth batch`:
   ```python
   def collate_fn(self, batch):
   
       # Selects new image size every tenth batch
       if self.multiscale and self.batch_count % 10 == 0:
           self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
     
       # Resize images to input shape
       imgs = torch.stack([resize(img, self.img_size) for img in imgs])
       self.batch_count += 1
  
       return paths, imgs, targets
   ```

<br>

# Linear Regression Example
> 这个例子是把所有训练数据一次性读到内存中了的  

见: https://gist.github.com/dvgodoy/1d818d86a6a0dc6e7c07610835b46fe4 
* Only load the batch training data instead of the whole data into GPU because graphics card’s RAM is precious.
* We need to send our model to the same device where the data is. If our data is made of GPU tensors, our model must “live” inside the GPU as well.
* During validation, it's better to use `with torch.no_grad() ` and `model.eval()` together.

```python
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############## Genreate dataset, dataloader ################
np.random.seed(42)
x = np.random.rand(100, 1)
true_a, true_b = 1, 2
y = true_a + true_b*x + 0.1*np.random.randn(100, 1)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

dataset = TensorDataset(x_tensor, y_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)

train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)     # it is on CPU
val_loader = DataLoader(dataset=val_dataset, batch_size=20)         # it is on CPU


##################### Generate Model ########################
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


################### Define train step ######################
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step


################### Train the model ######################
# Estimate a and b
torch.manual_seed(42)

model = ManualLinearRegression().to(device) # model = nn.Sequential(nn.Linear(1, 1)).to(device)
loss_fn = nn.MSELoss(reduction='mean')  # output the mean of several MSEloss
optimizer = optim.SGD(model.parameters(), lr=1e-1)
train_step = make_train_step(model, loss_fn, optimizer) # return of make_train_step is a function 

n_epochs = 100
training_losses = []
validation_losses = []
print(model.state_dict())

for epoch in range(n_epochs):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)    # Transfer the batch data from CPU to device
        y_batch = y_batch.to(device)
        loss = train_step(x_batch, y_batch)
        batch_losses.append(loss)
    training_loss = np.mean(batch_losses)   # report one training_loss per epoch
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            model.eval()
            yhat = model(x_val)
            val_loss = loss_fn(y_val, yhat).item()  # change tensor to python type
            val_losses.append(val_loss)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

print(model.state_dict())   # get the current value for all parameters
```
