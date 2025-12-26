---
{"dg-publish":true,"permalink":"/pytorch/early-stopping/","dgPassFrontmatter":true,"created":"2025-10-24T20:01:53.704+08:00","updated":"2025-12-26T14:58:37.101+08:00"}
---


## 示例
```python
early_stopping = EarlyStopping(patience=20, verbose=False)  # 设置早停  如果验证损失在连续 20 轮内没有下降，则停止训练。

# 当前累计所有 step 的平均训练损失 要保存的模型 模型保存路径/文件名前缀
early_stopping(np.mean(total_loss), Student.rnn, 'AMP_rnn_topk')  # 这里有问题  需要传入验证集的损失 或者使用最近一段区间的平均损失

if early_stopping.early_stop:
	print("Early stopping")
	break
```


## 完整代码

```python
import gzip
import pickle
import time
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as Ffrom AMP_distiilation import RNN
from utils import calculate_likelihood_loss, top_k_top_p_filtering, unique, decode, prompt_model_loader, \
    Variable
from early_stop.pytorchtools import EarlyStopping
from soft_prompt_embedding import SoftEmbedding

def setup_args():
    parser = argparse.ArgumentParser(
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
    return parser.parse_args()
    
def sample(args, model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)
    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 34
    input_ids = list(range(100, 100 + 10))
    input_ids.extend(tokenizer.encode(text))
    input_ids = input_ids[:11]
    input_tensor = torch.zeros(batch_size, 11).long()
    for index, i in enumerate(input_ids):
        input_tensor[:, index] = input_ids[index]
    Seq_list = []
    finished = torch.zeros(batch_size, 1).byte().to(device)
    log_probs = Variable(torch.zeros(batch_size))
    log_probs_list = []
    sequences = []
    for i in range(max_length):
        # input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        try:
            outputs = model(**inputs)
        except Exception as e:
            print(e)
        logits = outputs.logits
        logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        prob = F.softmax(logits[:, -1, :])
        log_prob = F.log_softmax(logits[:, -1, :], dim=1)
        last_token_id = torch.multinomial(prob, 1)
        sequences.append(last_token_id.view(-1, 1))
        # Flatten the tokens
        log_probs_list.append(log_prob)
        # .detach().to('cpu').numpy()
        EOS_sampled = (last_token_id == tokenizer.sep_token_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            # print('End')
            break
        # last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
        # Seq_list.append(last_token)
    sequences = torch.cat(sequences, 1)
    # sequences = sequences.detach()
    # Seq_list = np.array(Seq_list).T
    return sequences.data, log_probs_list


if __name__ == '__main__':
    # set_seed(42)
    start_time = time.time()
    train_losses = []
    total_step = 0
    total_loss = []
    early_stopping = EarlyStopping(patience=20, verbose=False)  # 设置早停  如果验证损失在连续 20 轮内没有下降，则停止训练。
    args = setup_args()
    n_steps = args.n_steps
    args.model_bin_path, args.vocab_path = './model_ckpt/pytorch_model.bin', '../voc/vocab.txt'
    args.model_path = './model_ckpt/'
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    Teacher = prompt_model_loader(args.model_bin_path, args.model_path)
    Student = RNN(tokenizer)
    # We dont need gradients with respect to Prior
    for param in Teacher.parameters():
        param.requires_grad = False
    # Agent.transformer.wte.learned_embedding.requires_grad = False
    optimizer = torch.optim.Adam(Student.rnn.parameters(), lr=args.lr)
    soft_criterion = torch.nn.KLDivLoss()
    # window_loss = []
    for step in range(n_steps):
        # Sample from Agent
        seqs, agent_likelihood = sample(args, Teacher, tokenizer, batch_size=args.batch_size)
        # Remove duplicates, ie only consider unique seqs
        # unique_idxs = unique(seqs)
        # seqs = seqs[unique_idxs]
        # agent_likelihood = agent_likelihood[unique_idxs]
        # Get prior likelihood and score
        prior_likelihood, hard_loss = Student.likelihood(seqs)
        soft_loss = soft_criterion(torch.cat(agent_likelihood, 1), torch.cat(prior_likelihood, 1))
        # Calculate loss
        # loss = soft_loss * args.alpha + hard_loss * (1 - args.alpha)
        loss = soft_loss
        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        if step % 100 == 0 and step != 0:
            # print(optim.state_dict()['param_groups'][0]['lr'])
            # decrease_learning_rate(optim, decrease_by=0.03)
            tqdm.write("*" * 50)
            tqdm.write("step {:3d}    loss: {:5.2f}\n".format(step, loss.item()))
            # mean_loss = np.mean(window_loss)   # 计算窗口内的平均损失会更加合理 不然会出现这种情况
            # window_loss = []  # 清空窗口
            # early_stopping(mean_loss, Student.rnn, 'AMP_rnn_topk')
            # if early_stopping.early_stop:
            #     print("Early stopping triggered!")
            #     break
        # 当前累计所有 step 的平均训练损失 要保存的模型 模型保存路径/文件名前缀
        early_stopping(np.mean(total_loss), Student.rnn, 'AMP_rnn_topk')  # 这里有问题  需要传入验证集的损失 或者使用最近一段区间的平均损失
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Step  当前 Loss 累计平均 Loss
# 0     5.0         5.0
# 1     4.8         (5.0 + 4.8)/2 = 4.9
# 2     4.6         (5.0+4.8+4.6)/3 ≈ 4.8
# ...   ... ...
# 99    2.0         ≈ 3.5（还在下降）
# 100   1.5         ≈ 3.45（缓慢下降）
# 200   1.2         ≈ 2.8
# 500   1.0         ≈ 2.0
# 1000  1.0         ≈ 1.5
# 2000  1.0         ≈ 1.25
```