---
{"dg-publish":true,"permalink":"/pytorch/embedding-function/","dgPassFrontmatter":true,"created":"2025-09-15T16:39:11.895+08:00","updated":"2025-12-26T14:58:39.335+08:00"}
---


## nn.Embedding

### 用途 

将数字转换成向量形式 

### 参数

num_embeddings=10,  # 有多少个数字 比如氨基酸有20个 加上开头结尾,mask和padding一共24个 这里就应该是24
embedding_dim=512,  # 嵌入的维度 
padding_idx=9  # 哪个index是padding  作为padding的话就会将权重置为 0 


```python
from torch import nn  
from torch.nn import *  
import torch  
  
input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9],[0, 3, 2, 9]])  
print(input1.shape)  # 1,3,4?  实际上为 3,4
embedding = Embedding(num_embeddings=10,embedding_dim=512,padding_idx=9)  # 这里的意思是只能为 0-9之间 不能为10或者11  需要注意的是 这里embedding后相同的数字是相同的表达形式  
print(embedding.weight.shape) # 10 512  
# torch.Size([10, 512])
print(embedding.weight)  
tensor([[-1.9155, -1.5182, -0.0818,  ...,  0.8255, -1.0510,  0.0435],
        [-0.0065, -0.3752, -0.7622,  ..., -1.2615, -1.6029,  1.1051],
        [-0.0755, -1.6070, -0.6997,  ..., -0.0902, -2.0161,  1.0061],
        ...,
        [-0.1598, -2.1448,  0.1906,  ...,  1.6189, -0.6125,  0.2567],
        [ 0.1029,  0.8511, -1.5063,  ...,  0.7246, -0.0559, -0.5365],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       requires_grad=True)
# 由于9是padding  没意义 所以就用0来表示  
res = embedding(input1)  
print(res)  
  
# pretrain  
  
# FloatTensor containing pretrained weights  包含预训练权重的FloatTensor  
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])  
embedding = Embedding.from_pretrained(weight)  
# Get embeddings for index 1  获取索引1的嵌入  
input2 = torch.LongTensor([1])  
print(embedding(input2))  
  
# input1 = torch.LongTensor([[0,1,2], [2,0,0]])  # C D G  #  G C C  
# print(input1.shape)  # 1,3,4?  实际上为 3,4# embedding = Embedding(num_embeddings=3,embedding_dim=4)  # 这里的意思是只能为 0-9之间 不能为10或者11  需要注意的是 这里embedding后相同的数字是相同的表达形式  
# linear = Linear(4,10)  
# # print(embedding.weight.shape) # 10 512  
# # print(embedding.weight)  
# # 由于9是padding  没意义 所以就用0来表示  
# res = embedding(input1)  
# print(res)  
# res = linear(res)  
# print(res)
```

