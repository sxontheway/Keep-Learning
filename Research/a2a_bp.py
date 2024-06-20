import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import os, math
os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', '6009'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
find_unused_parameters = False


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(5, 1)
        self._initialize_weights()
        self.sigmoid = nn.Sigmoid()

    def _initialize_weights(self):
        init.zeros_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def forward(self, x, rank):
        x = self.fc1(x) # (5,2) -> (5,2)

        "Consecutive all2alls, while kepping x unchanged"
        input_splits = [3,2]
        if rank == 0:
            output_splits = [3,3]
        else:
            output_splits = [2,2]
        x = _AllToAll.apply(x, input_splits, output_splits)   # rank0: [5,2] -> [6,2]; rank1: [5,2] -> [4,2]

        if rank == 0:
            input_splits = [3,3]
        else:
            input_splits = [2,2]
        output_splits = [3,2]
        x = _AllToAll.apply(x, input_splits, output_splits)  # rank0: [6,2] -> [5,2]; rank1: [4,2] -> [5,2]
        
        x = x.T         # (5,2) -> (2,5)
        x = self.sigmoid(x)
        x = self.fc2(x) # (2,5) -> (2,1)
        return x


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, input_splits, output_splits):  
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        output = torch.empty([sum(output_splits), *input.shape[1:]], dtype=input.dtype).to(input.device)
        dist.all_to_all_single(output, input, output_splits, input_splits)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (_AllToAll.apply(grad_output, ctx.output_splits, ctx.input_splits), None, None)
    

def generate_tensor(shape, rank):
    "generate tensor like tensor([[[0., 1.], [2., 3.], [4., 5.]], [[6., 7.],  ...]"
    input = torch.arange(math.prod(shape)) + rank*math.prod(shape)*1.0
    input = input.reshape(*shape)
    input = input.to(f"cuda:{rank}")
    return input


def run(rank, size):
    # 初始化进程组
    dist.init_process_group('nccl', rank=rank, world_size=size)
    
    # 创建一个简单的模型
    model = SimpleModel().cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)   # 同步模型参数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 生成输入和目标张量，每个进程的输入不同
    input = generate_tensor([5,2], rank)            # (5,1)
    target = torch.tensor([[1.0],[2.0]]).cuda(rank) # (2,1)

    for iteration in range(50):
        # 前向
        output = model(input, rank)
        loss = criterion(output, target)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM) # 同步 loss
        loss /= size
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            # print(model.module.fc2.weight)


if __name__ == "__main__":
    size = 2
    mp.spawn(run, args=(size,), nprocs=size, join=True)
