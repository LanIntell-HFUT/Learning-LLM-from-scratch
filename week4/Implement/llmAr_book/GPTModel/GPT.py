from GPTModel.LayerNorm import LayerNorm
from GPTModel.config import GPT_CONFIG_124M
from GPTModel.Transformer import TransformerBlock
import torch


class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.position_embedding = torch.nn.Embedding(config["context_size"], config["embed_dim"])
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.layers = torch.nn.ModuleList([
            TransformerBlock(config) for _ in range(config["num_layers"])
        ])
        
        # 可以选择不同的Norm层
        self.final_norm = LayerNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # self.final_norm = torch.nn.LayerNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # self.final_norm = RMSNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        # self.final_norm = torch.nn.RMSNorm(config["embed_dim"], eps=config["layer_norm_eps"])
        
        self.out_head = torch.nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)
        
        self.context_size = config["context_size"]
        self.embed_dim = config["embed_dim"]
        

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        token_embeds = self.token_embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        position_embeds = self.position_embedding(
            torch.arange(seq_len, device=input_ids.device)
        )

        x = token_embeds + position_embeds  # (batch_size, seq_len, embed_dim)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)  # (batch_size, seq_len, embed_dim)
        logits = self.out_head(x)
        return logits  # (batch_size, seq_len, vocab_size)
    

# 带有温度的TOPK生成函数
def generate(model, input_ids, max_length, context_size,
             temperature=1.0, tok_k=None, eos_id=None):
    model.eval() # 设置模型eval模式
    for _ in range(max_length):
        input_ids = input_ids[:, -context_size:]  # 截断输入序列以适应上下文窗口
        # 计算下一个概率分布
        with torch.no_grad():
            logits = model(input_ids)

        # 获取下一个token的id
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # topk采样
        if tok_k is not None:
            topk_logits, topk_indexes = torch.topk(next_token_logits, tok_k, dim=-1)
            min_values = topk_logits[:, -1]
            # 使用topk过滤不需要的token
            next_token_logits = torch.where(
                next_token_logits < min_values,
                torch.tensor(float('-inf')).to(next_token_logits.device),
                next_token_logits
            )
        if temperature != 1.0:
            # 当温度为1时和argmax效果一样
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            # 采用随机采样
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            # 当温度为一时，直接取最大值
            next_token_logits = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        if next_tokens == eos_id:
            break

        input_ids = torch.cat((input_ids, next_tokens), dim=1)

    return input_ids


# 测试代码
if __name__ == "__main__":
    # 计算模型的参数量
    total_params = sum(p.numel() for p in GPTModel(GPT_CONFIG_124M).parameters())
    print("Total parameters:", total_params)

    # 测试模型输入输出shape
    batch_size = 2
    seq_len = 10
    model = GPTModel(GPT_CONFIG_124M)
    input_ids = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))
    logits = model(input_ids)
    print("Logits shape:", logits.shape)  # 应该是 (batch_size, seq_len, vocab_size)

    # 测试生成函数
    generated_ids = generate(model, input_ids, max_length=20, context_size=GPT_CONFIG_124M["context_size"])
    print("Generated IDs:", generated_ids)
    print("Generated IDs shape:", generated_ids.shape)