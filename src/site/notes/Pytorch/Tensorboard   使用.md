---
{"dg-publish":true,"permalink":"/pytorch/tensorboard/","dgPassFrontmatter":true,"created":"2025-07-04T13:25:10.799+08:00","updated":"2025-12-26T14:58:50.690+08:00"}
---

## 功能

提供可视化的图像进行模型的评估
writer = SummaryWriter("logs")

## 用法

### writer.add_scalar(tag,scalar_value,global_step)

tag 为图标名
scalar_value 为y轴 拟合的值
global_step 为x轴 为当前的步骤

### writer.add_image(tag,img_tensor,global_step)

tag 为图表名
img_tensor 为图标的array  (torch.Tensor, numpy.ndarray, or string/blobname)
global_step 为当前的步骤

### writer.add_graph(model,input)

model 为模型
input 为输入数据

```python
img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

```

## 一个栗子

```python

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("logs")

# y=x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()

# tensorboard --logdir=logs --port=6007 m
from PIL import Image

img_path = "./back.png"
img = Image.open(img_path)
img = np.array(img)
print(img.shape)  # (491, 800, 3)  当通道在最后时，要显式设置

img = img.transpose(2, 0, 1)  # (3,491,800)  使用方法改变顺序 

writer.add_image("test",img,global_step=1) # 也可以在这里添加dataformats="HWC"来指定通道在最后
```


```python
import torch  
from torch import nn  
from torch.nn import *  
from torch.utils.tensorboard import SummaryWriter  
  
  
class Model(nn.Module):  
    def __init__(self,input_channel,out_feature):  
        super(Model, self).__init__()  
        self.hidden = 32  
        self.model1 = nn.Sequential(  
            Conv2d(input_channel,self.hidden,5,padding=2),  
            MaxPool2d(2),  
            Conv2d(self.hidden, self.hidden, 5, padding=2),  
            MaxPool2d(2),  
            Conv2d(self.hidden, self.hidden * 2, 5, padding=2),  
            MaxPool2d(2),  
            Flatten(),  
            Linear(64 * 4 * 4, 64),  
            Linear(64, out_feature),  
            Softmax()  
        )  
  
    def forward(self,x):  
        x = self.model1(x)  
        idx = torch.argmax(x,dim=1)  
        return idx,x  
  
  
model = Model(3,10)  
print(model)  
input = torch.ones((64,3,32,32))  
idx,output = model(input)  
print(idx)  
print(output)  
print(output.shape)  
  
writer = SummaryWriter("logs_seq")  
writer.add_graph(model,input)  
writer.close()
```

