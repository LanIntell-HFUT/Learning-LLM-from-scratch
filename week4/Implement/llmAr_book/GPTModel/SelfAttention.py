"""
一个简单的自注意力模块实现。
其中包括了单头注意力机制的实现以及由单头注意力扩展到多头注意力机制的实现；其中的多头注意力机制包括了两种实现方案；
前者使用cat函数将多个头的输出进行拼接，后者就是使用矩阵的变换将单头分解多多头。
后者的形状变化为[batch_size, seq_len, d_in] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_out]
"""
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self,d_in, d_out, bias = False):
        super().__init__()

        # 初始化三个权重矩阵，使用Linear层实现
        self.query = nn.Linear(d_in, d_out, bias=bias)
        self.key = nn.Linear(d_in, d_out, bias=bias)
        self.value = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        """
        x.shape = (batch_size, seq_len, d_in)
        """
        # 计算三个矩阵
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout,bias = False):
        super().__init__()
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out
        # 初始化三个权重矩阵，使用Linear层实现
        self.query = nn.Linear(d_in, d_out, bias=bias)
        self.key = nn.Linear(d_in, d_out, bias=bias)
        self.value = nn.Linear(d_in, d_out, bias=bias)

        # 注册一个上三角矩阵的掩码，防止未来信息泄露
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        """
        x.shape = (batch_size, seq_len, d_in)
        """
        batch_size, seq_len, d_in = x.size()

        # 计算三个矩阵
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

        attention_scores.masked_fill_(
            self.mask[:seq_len, :seq_len], -torch.inf
        )
        
        # print("attention_scores:", attention_scores)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output


# 由上面的CausalSelfAttention给出多头注意力的实现
# 第一种方法可以使用堆叠的方式实现

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self,d_in, d_out, context_length,
                 dropout, num_heads, bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.heads = nn.ModuleList(
            [CausalSelfAttention(
                d_in, d_out // num_heads, context_length, dropout, bias=bias
            ) for _ in range(num_heads)]
        )

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_in)
        head_outputs = [head(x) for head in self.heads]
        # 将多个头的输出拼接在一起
        output = torch.cat(head_outputs, dim=-1)
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # 计算多头的输出维度
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 初始化矩阵
        self.query = nn.Linear(d_in, d_out, bias=bias)
        self.key = nn.Linear(d_in, d_out, bias=bias)
        self.value = nn.Linear(d_in, d_out, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_out, d_out, bias=bias)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1).bool()
        )


    def forward(self, x):
        batch_size, seq_len, d_in = x.size()

        # 计算Q, K, V  形状为 (batch_size, seq_len, d_out)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 将Q, K, V 分割成多个头 形状为 (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 此时最后两维度就和上述的单头注意力一致
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        attention_scores.masked_fill_(
            self.mask[:seq_len, :seq_len], -torch.inf
        )

        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, V).transpose(1, 2) # 此时的形状为(batch_size, seq_len, num_heads, head_dim)

        # 将多个头的输出拼接在一起
        output = output.contiguous().view(batch_size, seq_len, self.d_out)
        return self.out(output)
    



# 测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_in = 8
    d_out = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_in)
    self_attention = MultiHeadAttention(d_in, d_out, seq_len, dropout=0.1, num_heads=num_heads)
    output = self_attention(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print(output)

