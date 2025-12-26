---
{"dg-publish":true,"permalink":"/pytorch/argparse-argument-parser/","dgPassFrontmatter":true,"created":"2025-10-24T20:30:11.732+08:00","updated":"2025-12-26T14:58:29.612+08:00"}
---



```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--alpha', default=1, type=int, required=False, help='soft loss')
    # | --alpha | 1 | int | 软损失权重系数，用于加权蒸馏损失（KL 散度）和硬损失（如 CE）。
    # 但在你的代码中：
    # loss = soft_loss * args.alpha
    # 因为 alpha=1，所以只用了软损失（KL 散度），没有硬任务损失。
    # 📌 如果你想做混合损失，应改为 float 类型，比如 alpha=0.7 |
    # parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--top_k', default=10, type=int, required=False, help='top k sampling')  # 生成文本时，只从概率最高的 K 个词中采样
    parser.add_argument('--top_p', default=1.0, type=float, required=False, help='top p sampling')  # 生成文本时，只从累计概率超过 P 的词中采样 常与 top_k 二选一使用，典型值：0.9, 0.95 |  相当于不限制
    parser.add_argument('--sigma', default=15, type=float, required=False, help='the weight of score')  #
    parser.add_argument('--n_steps', default=10000, type=float, required=False, help='the weight of score')  # 训练总步数
    parser.add_argument('--n_tokens', default=10, type=int, required=False, help='soft embedding') # 软提示（Soft Prompt）的长度，即在输入前添加多少个可学习的连续向量。
    
	args =parser.parse_args()
	print("Args:", args)
```

```bash
python train.py --batch_size 32 --lr 5e-4 --alpha 0.7 --top_k 50 --top_p 0.9 --sigma 20 --n_steps 20000 --n_tokens 20
```

