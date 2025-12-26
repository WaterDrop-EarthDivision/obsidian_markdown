---
{"dg-publish":true,"permalink":"/pytorch/tqdm/","dgPassFrontmatter":true,"created":"2025-08-25T13:52:06.572+08:00","updated":"2025-12-26T14:58:53.080+08:00"}
---


## 例子

```python
import torch.optim  
import torchvision  
from torch.utils.data import DataLoader  
from torch.utils.tensorboard import SummaryWriter  
  
from model import *  
import tqdm  
  
train_data = torchvision.datasets.CIFAR10("./dataset",train=True,  
                                          transform=torchvision.transforms.ToTensor(),download=True)  

  
train_data_size = len(train_data)  
  
train_loader = DataLoader(train_data,batch_size=64)

for epoch in range(epoch):
	train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/Train", leave=True)
		for data in train_pbar:
			......
			train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})   # 设置描述
		
```

```python
bar = trange(10)  
for i in bar:  
    time.sleep(0.1)  
    if not (i % 3):  
        tqdm.write("Done task %i" % i)
```