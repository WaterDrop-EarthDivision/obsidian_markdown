---
{"dg-publish":true,"permalink":"/pytorch/gpu/","dgPassFrontmatter":true,"created":"2025-08-22T15:13:29.307+08:00","updated":"2025-12-26T14:58:41.626+08:00"}
---

F:\Pytorch_LearingProject\Pytorch_learing_xiaotudui\GPU_model.py
F:\Pytorch_LearingProject\Pytorch_learing_xiaotudui\GPU_model2.py
# 用途

将模型和数据送入GPU以提升训练速度
可以将模型model、损失函数loss_fun、数据 data target 送入GPU，但是损失函数可以不送

# 用法

## 用法1 使用.cuda()送入

```python
model = Model()  
if torch.cuda.is_available():  
    model = model.cuda()

loss_fun = nn.CrossEntropyLoss().cuda()

for epoch in range(epoch):  
    print(f"---------------第{epoch+1}次训练---------------")  
    model.train()  
    train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/Train", leave=True)  
    for data in train_pbar:  
    # for i,data in enumerate(train_loader):  
        imgs,target = data  
        imgs = imgs.cuda()  
        target = target.cuda()
```

### 用法2 使用.to()

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model()  
model.to(device)

for data in train_pbar:  
    imgs,target = data  
    imgs = imgs.to(device)  
    target = target.to(device)
```