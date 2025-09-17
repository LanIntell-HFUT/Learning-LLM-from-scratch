from GPTModel.SelfAttention import MultiHeadAttention
from GPTModel.LayerNorm import LayerNorm
from GPTModel.RMSNorm import RMSNorm
from GPTModel.FeedForward import FeedForward
import torch


class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in = config["embed_dim"],
            d_out = config["embed_dim"],
            context_length = config["context_size"],
            dropout = config["dropout"],
            num_heads = config["num_heads"]
        )

        # 使用torch自带的LayerNorm
        # self.layer_norm1 = torch.nn.LayerNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # self.layer_norm2 = torch.nn.LayerNorm(config["embed_dim"], eps=config["layer_norm_eps"])

        # 使用自己实现的LayerNorm
        self.layer_norm1 = LayerNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        self.layer_norm2 = LayerNorm(config["embed_dim"], eps=config["layer_norm_eps"])

        # 使用torch自带的RMSNorm
        # self.layer_norm1 = torch.nn.RMSNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # self.layer_norm2 = torch.nn.RMSNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # 使用自己实现的RMSNorm
        # self.layer_norm1 = RMSNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # self.layer_norm2 = RMSNorm(config["embed_dim"], eps=config["layer_norm_eps"])

        self.ffn = FeedForward(
            d_in = config["embed_dim"],
            d_ff = config["ffn_dim"],
        )

        self.dropout = torch.nn.Dropout(config["dropout"])

    def forward(self, x):
        # x.shape = (batch_size, seq_len, embed_dim)
        # 注意力子层
        norm_x = self.layer_norm1(x)
        attention_output = self.attention(norm_x)
        x = x + self.dropout(attention_output)

        # 前馈神经网络子层
        norm_x = self.layer_norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + self.dropout(ffn_output)

        return x




# 测试代码
if __name__ == "__main__":
    batch_size = 2
    from config import GPT_CONFIG_124M
    config = GPT_CONFIG_124M
    x = torch.randn(batch_size, config["context_size"], config["embed_dim"])
    print("Input shape:", x.shape)
    transformer_block = TransformerBlock(config)
    output = transformer_block(x)
    print("Output shape:", output.shape)
    print(output)
