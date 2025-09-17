import tiktoken
import GPTModel.config as config
import torch
from GPTModel.GPT import GPTModel, generate





def text2tokenids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def tokenids2text(token_ids, tokenizer):
    flattened_ids = token_ids.squeeze(0)
    text = tokenizer.decode(flattened_ids.tolist())
    return text





if __name__ == "__main__":

    # 测试代码，将GPTModel包装为一个包
    config = config.GPT_CONFIG_124M
    print(config)

    total_params = sum(p.numel() for p in GPTModel(config).parameters())
    print("Total parameters:", total_params)

    # 测试模型输入输出shape
    batch_size = 2
    seq_len = 10
    model = GPTModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    logits = model(input_ids)
    print("Logits shape:", logits.shape)  # 应该是 (batch_size, seq_len, vocab_size)

    # 测试生成函数
    generated_ids = generate(model, input_ids, max_length=20, context_size=config["context_size"])
    print("Generated IDs:", generated_ids)
    print("Generated IDs shape:", generated_ids.shape)


    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2") # 使用GPT2的分词器

    token_ids = generate(
        model=model,
        input_ids=text2tokenids(start_context, tokenizer=tokenizer),
        max_length=10,
        context_size=config["context_size"]
    )

    print("Token IDs:", tokenids2text(token_ids, tokenizer=tokenizer))