---
{"dg-publish":true,"permalink":"/pytorch/save-load/","dgPassFrontmatter":true,"created":"2025-08-22T13:25:31.821+08:00","updated":"2025-12-26T14:58:21.622+08:00"}
---


### 第一种 保存模型的所有数据

```python
import torch  
import torchvision  
  
vgg16 = torchvision.models.vgg16(pretrained=False)  
  
# 保存方式1  
torch.save(vgg16,"vgg16_f1.pth")

vgg16 = torch.load("vgg16_f1.pth")  
print(vgg16)

```

### 第二种 只保存模型的参数

```python
import torch  
import torchvision  
  
vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式2  只保存参数(官方推荐)  
torch.save(vgg16.state_dict(),"vgg16_f2.pth")

vgg16 = torchvision.models.vgg16()  
vgg16.load_state_dict(torch.load("vgg16_f2.pth"))  
print(vgg16)
```