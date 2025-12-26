---
{"dg-publish":true,"permalink":"/pytorch/data-loader/","dgPassFrontmatter":true,"created":"2025-08-22T14:51:09.095+08:00","updated":"2025-12-26T14:58:31.799+08:00"}
---

# 功能 

将 dataset 分成一定的 batch_size，并确定是否打乱等

## Docs
help(DataLoader)

## 用法

```python
test_dataloader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)  
```

## 一个例子

```python 
'''  
Description: Author: Chenglin Cai  
Date: 2025-07-16 17:07:16  
LastEditTime: 2025-07-17 15:32:33  
LastEditors:  '''  
import torchvision  
from torch.utils.data import DataLoader  
from torch.utils.tensorboard import SummaryWriter  
  
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())  # ,download=True  
  
test_dataloader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)  
  
img,target = test_data[0]  # 查看第一张图像  
  
print(img.shape)  # torch.Size([3, 32, 32])  
print(target)  # 3  
  
writer = SummaryWriter("./dataloader")  
step = 0  
for epoch in range(2):  
    for data in test_dataloader:  
        imgs,targets = data  
        # print(imgs.shape)  # torch.Size([4, 3, 32, 32])  
        # print(targets)  # tensor([3, .....])        writer.add_images(f"test_{epoch}", imgs, global_step=step)   
        step += 1  
writer.close()
```

