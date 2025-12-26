---
{"dg-publish":true,"permalink":"/pytorch/layers-function/","dgPassFrontmatter":true,"created":"2025-08-24T12:15:51.381+08:00","updated":"2025-12-26T14:58:43.886+08:00"}
---


## Softmax

![Pasted image 20250824121631.png](/img/user/Pasted%20image%2020250824121631.png)

### 示例

```python
m = Softmax(dim=1)  
input = torch.randn(2, 3)  
print(input)  
output = m(input)  
print(output)

tensor([[-0.1436,  1.7089, -0.3768],
        [-1.0266,  0.0463,  1.6492]])
tensor([[0.1224, 0.7806, 0.0970],
        [0.0542, 0.1585, 0.7873]])
```