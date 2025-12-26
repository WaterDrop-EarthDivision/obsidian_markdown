---
{"dg-publish":true,"permalink":"/pytorch/optimizer-method-torch-optim/","dgPassFrontmatter":true,"created":"2025-08-24T12:31:34.873+08:00","updated":"2025-12-26T14:58:48.287+08:00"}
---


## SGD 

![Pasted image 20250824123318.png](/img/user/Pasted%20image%2020250824123318.png)

### 传入参数
```python
model = Model()  
torch.optim.SGD(model.parameters(),lr=0.0001)
```

### 例子 
#### 动态调整学习率

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau  
  
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)  
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)  
# mode='min' 表示监控的指标越小越好（如损失）  
# factor: 学习率乘以此因子  
# patience: 等待多少个epoch没有改善才降低学习率  
  
# 训练循环  
for epoch in range(num_epochs):  
    train_loss = train(...)  
    val_loss = validate(...)  # 验证损失  
    scheduler.step(val_loss)  # 根据验证损失调整学习率
```

#### 断点保存与回复

```python
model = torch.nn.Linear(10, 10)
optim = torch.optim.SGD(model.parameters(), lr=3e-4)
scheduler1 = torch.optim.lr_scheduler.LinearLR(
    optim,
    start_factor=0.1,
    end_factor=1,
    total_iters=20,
)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim,
    T_max=80,
    eta_min=3e-5,
)
lr = torch.optim.lr_scheduler.SequentialLR(
    optim,
    schedulers=[scheduler1, scheduler2],
    milestones=[20],
)

# torch.save(optim.state_dict(), "save_optim.pt") 
# torch.save(lr.state_dict(), "save_seq.pt")

lr.load_state_dict(torch.load("./save_seq.pt"))
# now load the optimizer checkpoint after loading the LRScheduler
optim.load_state_dict(torch.load("./save_optim.pt"))
```

## Adam

### 传入参数

```python
model = Model()  
torch.optim.Adam(model.parameters(),lr=0.001)
```

## AdamW

### 传入参数

```python
model = Model()  
torch.optim.AdamW(model.parameters(),lr=0.001)
```