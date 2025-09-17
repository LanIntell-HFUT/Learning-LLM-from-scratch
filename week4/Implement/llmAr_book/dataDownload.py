# import urllib.request
# url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)
# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")

# file_path = "the-verdict.txt"
# with open(file_path, "r", encoding="utf-8") as f:
#     raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print("The length of tokenized text:", len(tokenizer.encode(raw_text)))
# print(raw_text[:99])


from torch.utils.data import DataLoader, Dataset
import torch
import tiktoken


# def create_dataset(raw_text, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0):

class TextDataset(Dataset):
    def __init__(self, raw_text, tokenizer, max_length, stride):
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        self.token_ids = tokenizer.encode(raw_text)
        self.input_ids = []
        self.targets = []

        for i in range(0, len(self.token_ids) - max_length, stride):
            input_chunk = self.token_ids[i : i + max_length]
            target_chunk = self.token_ids[i + 1 : i + 1 + max_length]
            self.input_ids.append(input_chunk)
            self.targets.append(target_chunk)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.input_ids)


def create_dataloader(raw_text, batch_size, max_length, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(raw_text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader