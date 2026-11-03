import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 64 # 并行处理多少段独立序列？
block_size = 256 # 预测时的最大上下文长度是多少？
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 获取这段文本中出现的所有不重复字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 创建字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 编码器：输入字符串，输出整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解码器：输入整数列表，输出字符串

# 划分训练集和测试集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 前 90% 用于训练，其余用于验证
train_data = data[:n]
val_data = data[n:]

# 数据加载
def get_batch(split):
    # 生成一小批输入 x 和目标 y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ 一个自注意力头 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入形状为（批次大小，时间步数，通道数）
        # 输出形状为（批次大小，时间步数，注意力头维度）
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # 计算注意力分数（“亲和度”）
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 对值向量进行加权汇总
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ 并行运行的多个自注意力头 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 一个简单的线性层，后接非线性变换 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer 块：先交换信息，再进行计算 """

    def __init__(self, n_embd, n_head):
        # n_embd：嵌入维度，n_head：所需的注意力头数量
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 每个 token 直接从查找表中读取下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # 最后的层归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 更好的初始化方式，原 GPT 视频中未介绍，但很重要，将在后续视频中讲解
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx 和 targets 都是形状为 (B,T) 的整数张量
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文中形状为 (B, T) 的索引数组
        for _ in range(max_new_tokens):
            # 将 idx 截取为最后 block_size 个 token
            idx_cond = idx[:, -block_size:]
            # 获取预测结果
            logits, loss = self(idx_cond)
            # 只关注最后一个时间步
            logits = logits[:, -1, :] # 变为 (B, C)
            # 应用 softmax 得到概率
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将采样得到的索引追加到当前序列
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# 打印模型的参数数量
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# 创建 PyTorch 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # 每隔一段时间评估训练集和验证集的损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 抽取一批数据
    xb, yb = get_batch('train')

    # 计算损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 切换到评估模式，关闭 Dropout
m.eval()

# 起始输入：1 段序列，其中只有一个编号为 0 的 token
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# 生成时不记录梯度
with torch.no_grad():
    result = m.generate(context, max_new_tokens=10000)

# 取出第一段序列，转为编号列表，再解码成文字
token_ids = result[0].tolist()
generated_text = decode(token_ids)

print(generated_text)

with open('more-261027.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)