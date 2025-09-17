import torch.nn as nn
from GPTModel.GELU import GELU

class FeedForward(nn.Module):
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_ff)
        # 使用自己实现的GELU
        self.activation = GELU()
        # self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_in)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x