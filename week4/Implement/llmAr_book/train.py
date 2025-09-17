from sklearn.model_selection import train_test_split
import torch
from GPTModel.GPT import GPTModel as GM
from GPTModel.GPT import generate
from dataDownload import create_dataloader
from GPTModel.config import GPT_CONFIG_124M
import torch.nn as nn
import GPTutils
import tiktoken
import matplotlib.pyplot as plt
import tqdm

# 计算一个batche的loss
def calc_loss_batch(inputs, targets, model, device):
    input_batch = inputs.to(device)
    target_batch = targets.to(device)
    logits = model(input_batch)

    # 其中logits的shape是(batch_size, seq_len, vocab_size)
    # target_batch的shape是(batch_size, seq_len)
    loss_fn = nn.functional.cross_entropy
    loss = loss_fn(logits.flatten(0, 1), target_batch.flatten())
    # 计算单个batch中的交叉熵损失
    return loss

# 计算一个dataloader的平均loss
def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0.0
    if len(dataloader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(dataloader)

    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(inputs, targets, model, device)
        total_loss += loss.item()
    return total_loss / num_batches if num_batches > 0 else float('nan')



# 训练循环中用于验证model的函数
def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        # 传入eval_iter参数，限制评估时使用的batch数量
        train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_dataloader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


# 用于在循环中生成文本的函数
def generate_and_print_samples(model, tokenizer, device, start_context):
    model.eval()
    # context_size = model.pos_embed.weight.shape[0]
    context_size = GPT_CONFIG_124M["context_size"]
    input_ids = GPTutils.text2tokenids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            input_ids=input_ids,
            max_length=50,
            context_size=context_size,
            temperature=1.5,
            tok_k=3
        )
    decoded_text = GPTutils.tokenids2text(token_ids, tokenizer)
    # print("Generated text:\n", decoded_text)
    print(decoded_text.replace("\n", " "))
    model.train()




def train_loop(
    model, train_dataloader, val_dataloader, optimizer, device,
    epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    参数：其中eval_freq是每多少步评估一次，eval_iter是评估时使用多少个batch
    返回：train_losses, val_losses, track_tokens_seen
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1 
    for _ in range(epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            # token_seen是累计的token数量
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(token_seen)
                print(f"Step {global_step}: Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Tokens seen = {token_seen}")

        generate_and_print_samples(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def show_losses(epochs, train_losses, val_losses, tokens_seen):
    plt.figure(figsize=(10, 6))
    plt.plot(tokens_seen, train_losses, label='Train Loss')
    plt.plot(tokens_seen, val_losses, label='Validation Loss')
    plt.xlabel('Tokens Seen')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss over {epochs} Epochs')
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)    
    model = GM(GPT_CONFIG_124M).to(device)

    tokenizer = tiktoken.get_encoding("gpt2")

    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    # print(raw_text[:120])

    # train_data, val_data = train_test_split(raw_text, test_size=0.1, random_state=42)
    split_idx = int(0.9 * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]


    print(f"Train data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")

    # 创建dataloaders
    train_dataloader = create_dataloader(
        raw_text=train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_size"],
        stride=GPT_CONFIG_124M["context_size"]
    )

    val_dataloader = create_dataloader(
        raw_text=val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_size"],
        stride=GPT_CONFIG_124M["context_size"]
    )


    # for x, y in train_dataloader:
    #     print(x.shape, y.shape)

    # for x, y in val_dataloader:
    #     print(x.shape, y.shape)
    # print(f"Number of batches in train dataloader: {len(train_dataloader)}")
    # print(f"Number of batches in val dataloader: {len(val_dataloader)}")


    # 验证batch_loss计算
    # with torch.no_grad():
    #     train_loss = calc_loss_loader(train_dataloader, model, device)
    #     val_loss = calc_loss_loader(val_dataloader, model, device)

    # print(f"Initial train loss: {train_loss:.4f}")
    # print(f"Initial val loss: {val_loss:.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    print("model:", model)
    train_losses, val_losses, tokens_seen = train_loop(
        model, train_dataloader, val_dataloader, optimizer, device,
        epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    # 保存模型
    torch.save(model.state_dict(), "model.pth")

    # torch.save({
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     },
    #     "model_and_optimizer.pth"
    # )

    show_losses(num_epochs, train_losses, val_losses, tokens_seen)

