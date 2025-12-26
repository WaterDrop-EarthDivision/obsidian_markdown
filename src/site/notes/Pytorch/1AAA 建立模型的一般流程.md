---
{"dg-publish":true,"permalink":"/pytorch/1-aaa/","dgPassFrontmatter":true,"created":"2025-08-24T12:25:41.674+08:00","updated":"2025-12-26T14:58:11.747+08:00"}
---


## 构建测试模型

```python
import torch  
from torch.nn import *  
from torch import nn  
  
  
class Model(nn.Module):  
    def __init__(self):  
        super(Model, self).__init__()  
        self.model = nn.Sequential(  
            Conv2d(3,32,5,padding=2),   # 32 = (32+2p-5)/1+1  
            MaxPool2d(2),  
            Conv2d(32,32,5,padding=2),  # 16 = (16+2p-5)/1+1  
            MaxPool2d(2),  
            Conv2d(32,64,5,padding=2),  
            MaxPool2d(2),  
            Flatten(),  
            Linear(64*4*4,64),  
            Linear(64,10)  
        )  
  
    def forward(self,x):  
        result = self.model(x)  
        return result  
  
if __name__ == '__main__':  
    data = torch.zeros([64,3,32,32])  
    # data = torch.reshape(data,[-1,3,32,32])  
    model = Model()  
    res = model(data)  
    print(res.shape)
```

## Train其余流程
```python
import torch.optim  
import torchvision  
from torch.utils.data import DataLoader  
from torch.utils.tensorboard import SummaryWriter  
  
from model import *  
import tqdm  
  # 加载数据集
train_data = torchvision.datasets.CIFAR10("./dataset",train=True,  
                                          transform=torchvision.transforms.ToTensor(),download=True)  
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),  
                                         download=True)  
  
train_data_size = len(train_data)  
test_data_size = len(test_data)  
  # DataLoader
train_loader = DataLoader(train_data,batch_size=64)  
test_loader = DataLoader(test_data,batch_size=1)  
  # 实例化模型
model = Model()  
  # 定义损失函数
loss_fun = nn.CrossEntropyLoss()  
  # 定义优化器
learning_rate = 0.01  
optim = torch.optim.SGD(model.parameters(),lr=learning_rate,)  
  # 定义超参数
epoch = 10  
train_step = 0  
test_step = 0  
  # 定义tensorboard
writer = SummaryWriter("./logs_train")  
  
for epoch in range(epoch):  
    print(f"---------------第{epoch+1}次训练---------------")  
    model.train()   # 模型训练
    train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/Train", leave=True)  
    for data in train_pbar:  
    # for i,data in enumerate(train_loader):  
        imgs,target = data  
  
        optim.zero_grad()  # 梯度清零
        res = model(imgs)  # 送入模型计算
        loss = loss_fun(res,target)  # 计算损失
  
        loss.backward()  # 根据损失计算梯度
        optim.step()  # 优化器根据计算的梯度进行参数优化
  
        train_step += 1  
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})  
  
        writer.add_scalar("train_loss",loss.item(),global_step=train_step)  
        # print(f"训练轮数:{train_step}  Loss为{loss.item()}")  
  
    model.eval()  # 模型验证
    total_loss = 0  
    total_acc = 0  
    with torch.no_grad():  
        test_tqdm = tqdm.tqdm(test_loader,desc=f"Epoch {epoch + 1}/Test",leave=True)  
        # for i,data in enumerate(test_loader):  
        for data in test_tqdm:  
            imgs, target = data  
            res = model(imgs)  
            loss = loss_fun(res,target)  
  
            idx = torch.argmax(res)  
            true = (idx == target).sum()  
            total_acc += true  
            # all_len = len(target)  
            # acc = true/all_len  
            test_step += 1  
            total_loss += loss  
  
    print(f"Total Loss为{total_loss}")  
    print(f"Total Acc 为{total_acc/test_data_size}")  
    writer.add_scalar("total acc",total_acc/test_data_size,global_step=epoch)  
    writer.add_scalar("test loss",total_loss,global_step=epoch)  
    torch.save(model,f"./model{epoch}.pth")  # 模型保存
    torch.save(model.state_dict(),f"./model{epoch}.pth")  
    print("保存了模型")  
  
writer.close()
```

## Test 其余流程

```python
import torch.cuda  
import torchvision  
from PIL import Image  
  
from model import *  
  
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"  
print(device)  
  
test_data = torchvision.datasets.CIFAR10("./dataset",train=False)  
dict_idx = test_data.class_to_idx  
  
model = Model()  
model.load_state_dict(torch.load("./model9.pth"))  
model.to(device)  
  
img_path = "./dog.png"  
image = Image.open(img_path)  
print(image)  
  
transform = torchvision.transforms.Compose([  
    torchvision.transforms.Resize((32,32)),  
    torchvision.transforms.ToTensor()  
])  
  
image = transform(image)  
print(image)  
  
image = torch.reshape(image,(-1,3,32,32))  
image = image.to(device)  
model.eval()  
with torch.no_grad():  
    output = model(image)  
  
print(output)  
  
keys = [key for key, value in dict_idx.items() if value == torch.argmax(output)]  
print(keys)
```