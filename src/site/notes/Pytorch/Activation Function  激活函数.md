---
{"dg-publish":true,"permalink":"/pytorch/activation-function/","dgPassFrontmatter":true,"created":"2025-08-24T12:09:35.086+08:00","updated":"2025-12-26T14:58:26.851+08:00"}
---


## ReLu

![Pasted image 20250824121037.png](/img/user/Pasted%20image%2020250824121037.png)

### 示例

```python
m = nn.ReLU()  
input = torch.randn(2).unsqueeze(0)  
print(input)  
output = torch.cat((m(input), m(-input)))  
print(output)

tensor([[-0.6228,  0.4439]])
tensor([[0.0000, 0.4439],
        [0.6228, 0.0000]])
```

## Sigmoid

![Pasted image 20250824121322.png](/img/user/Pasted%20image%2020250824121322.png)
将目标缩放到0~1之间
### 示例 

```python
m = nn.Sigmoid()  
input = torch.randn(2)  
print(input)  
output = m(input)  
print(output)

tensor([ 1.3445, -1.1299])
tensor([0.7932, 0.2442])
```