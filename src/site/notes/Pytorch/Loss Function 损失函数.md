---
{"dg-publish":true,"permalink":"/pytorch/loss-function/","dgPassFrontmatter":true,"created":"2025-08-22T15:29:30.545+08:00","updated":"2025-12-26T14:58:46.050+08:00"}
---


## L1Loss  适合回归问题

![Pasted image 20250822153813.png](/img/user/Pasted%20image%2020250822153813.png)

### 例子

```python
inputs = torch.tensor([1,2,3],dtype=torch.float32)  
targets = torch.tensor([1,2,5],dtype=torch.float32)  
  
inputs = torch.reshape(inputs,(1,1,1,3))  
targets = torch.reshape(targets,(1,1,1,3))  
  
loss = L1Loss()  # (5-3)/3 = 0.667
# loss = MSELoss()  
result = loss(inputs,targets)  
print(result)
```

```python
loss = L1Loss()  
input = torch.randint(0,10,(3,5),dtype=torch.float32)  
print(input)  
target = torch.randint(0,10,(3,5),dtype=torch.float32)  
print(target)  
output = loss(input, target)  
print(output)

tensor([[9., 9., 3., 9., 8.],
        [5., 0., 2., 9., 3.],
        [3., 7., 9., 3., 9.]])
tensor([[6., 6., 6., 2., 9.],
        [7., 1., 4., 0., 1.],
        [5., 2., 1., 6., 1.]])
tensor(3.9333)  # (3+3+3+7+1+2+1+2+9+2+2+5+8+3+8)/15=59/15=3.933
```

## MSELoss 适合回归问题

![Pasted image 20250822155418.png](/img/user/Pasted%20image%2020250822155418.png)

### 例子

```python
import torch  
from torch.nn import *  
  
inputs = torch.tensor([1,2,3],dtype=torch.float32)  
targets = torch.tensor([1,2,5],dtype=torch.float32)  
  
inputs = torch.reshape(inputs,(1,1,1,3))  
targets = torch.reshape(targets,(1,1,1,3))  
  
# loss = L1Loss()  
loss = MSELoss()  # 2^2/3 = 1.3333
result = loss(inputs,targets)  
print(result)
```

```python
loss = MSELoss()  
input = torch.randint(0,10,(3,5),dtype=torch.float32)  
print(input)  
target = torch.randint(0,10,(3,5),dtype=torch.float32)  
print(target)  
output = loss(input, target)  
print(output)

tensor([[3., 2., 5., 3., 9.],
        [6., 4., 9., 5., 4.],
        [1., 1., 3., 6., 0.]])
tensor([[5., 6., 8., 1., 3.],
        [1., 4., 6., 7., 5.],
        [3., 6., 4., 3., 8.]])
tensor(14.0667) 
```

## CrossEntropyLoss 适合分类问题

[CrossEntropyLoss — PyTorch 2.8 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

![Pasted image 20250823110028.png](/img/user/Pasted%20image%2020250823110028.png)

### 适用范围
当使用 C 类训练分类问题时，它很有用。 如果提供`weight`，可选参数应该是为每个类分配权重的一维张量。 当您的训练集不平衡时，这特别有用。

### 例子

```python
output = torch.tensor([0.1,0.2,0.3])  
y = torch.tensor([1])  
output = torch.reshape(output,(1,3))  
loss = CrossEntropyLoss()  
result = loss(output,y)  
print(result) 

exp_all = math.exp(0.1)+math.exp(0.2)+math.exp(0.3)  
res = -(math.log(math.exp(0.2)/exp_all))  
print(res)

```

```python
# Example of target with class indices  带有类索引的目标示例
loss =CrossEntropyLoss()  
input = torch.randn(3, 5, requires_grad=True)  
print(input)  
target = torch.empty(3, dtype=torch.long).random_(5)  
print(target)  
output = loss(input, target)  
print(output)  
output.backward()

tensor([[ 1.3622,  0.3561, -1.2396,  1.0952,  0.5529],
        [-0.4801,  0.1979,  0.6740, -1.2265, -0.1212],
        [ 2.3869, -0.8293,  2.1457,  1.0697, -1.1881]], requires_grad=True)
tensor([4, 3, 4])
tensor(2.9810, grad_fn=<NllLossBackward0>)


result = 0  
for id,list in enumerate(input):  
    res = 0  
    for j in list:  
        res += math.exp(j)  
    result += -math.log(math.exp(list[target[id]])/res)  
print(result/len(target))
```

```python
# Example of target with class probabilities  具有类概率的目标示例
loss =CrossEntropyLoss()  
input = torch.randn(3, 5, requires_grad=True)  
print(input)  
target = torch.randn(3, 5).softmax(dim=1)  
print(target)  
output = loss(input, target)  
print(output)  
output.backward()

tensor([[ 0.0460, -0.3625, -0.7095,  0.4335, -1.1691],
        [-1.3188, -0.8861, -1.0856,  0.5278, -2.1268],
        [ 1.0059,  0.5506,  2.0493,  1.0854, -0.0606]], requires_grad=True)
tensor([[0.0239, 0.0120, 0.0174, 0.1207, 0.8260],
        [0.6462, 0.0495, 0.1033, 0.0303, 0.1707],
        [0.1570, 0.0966, 0.5652, 0.0385, 0.1428]])
tensor(2.0372, grad_fn=<DivBackward1>)
```

## KLDivLoss 适合模型蒸馏

[KLDivLoss — PyTorch 2.9 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)

![Pasted image 20251025161657.png](/img/user/Pasted%20image%2020251025161657.png)

### 例子

```python
import torch.nn as nn  
import torch.nn.functional as F  
  
kl_loss = nn.KLDivLoss(reduction="batchmean")  
# input should be a distribution in the log space  
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)  
print(input)  
   tensor([[-1.7911, -1.9003, -2.3099, -1.3681, -1.1092],
        [-3.2043, -0.7602, -0.9078, -3.5380, -2.8242],
        [-1.5738, -1.6282, -0.7104, -2.5855, -3.5186]],
       grad_fn=<LogSoftmaxBackward0>)
# Sample a batch of distributions. Usually this would come from the dataset  
target = F.softmax(torch.rand(3, 5), dim=1)  
print(target)  
  tensor([[0.1765, 0.2404, 0.1431, 0.3004, 0.1395],  sum=1
        [0.1633, 0.3357, 0.1744, 0.1475, 0.1792],
        [0.3069, 0.1434, 0.1330, 0.1319, 0.2847]])
# print(target.sum())  
output = kl_loss(input, target)  
print(output)  
  tensor(0.3777, grad_fn=<DivBackward0>)
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)  
log_target = F.log_softmax(torch.rand(3, 5), dim=1)  
print(log_target)  
   tensor([[-1.6082, -1.6829, -1.8801, -1.4060, -1.5315],
        [-1.7730, -1.9375, -1.5741, -1.4506, -1.4085],
        [-1.5591, -2.0299, -1.4864, -1.6360, -1.4381]])
output = kl_loss(input, log_target)  
print(output)
   tensor(0.4222, grad_fn=<DivBackward0>)
```