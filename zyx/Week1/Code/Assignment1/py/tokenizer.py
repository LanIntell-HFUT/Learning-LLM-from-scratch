from typing import Iterable, Iterator
import regex as re

def split_token(token: str):
    """将字符串 token 拆成 utf-8 bytes 列表"""
    return [c.encode("utf-8") for c in token]

def perform_merge(token: str, merges: list[tuple[bytes, bytes]]):
    """执行 BPE 合并"""
    chars = split_token(token)
    i = 0
    while i < len(chars) - 1:
        if (chars[i], chars[i+1]) in merges:
            combined = chars[i] + chars[i+1]
            chars = chars[:i] + [combined] + chars[i+2:]
        else:
            i += 1
    return chars

def map_merged_tokens_to_ids(merged_tokens: list[bytes], vocab: dict[int, bytes]):
    """将合并后的 bytes token 映射到 ID"""
    token_ids = []
    for token in merged_tokens:
        for k, v in vocab.items():
            if token == v:
                token_ids.append(k)
                break
    return token_ids

def pre_tokenization_with_specials(raw_text: str, 
                                   special_tokens: list[str], 
                                   PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
    """先识别特殊符号，再正则分词"""
    if not special_tokens:
        special_tokens = []
    special_pattern = "|".join(re.escape(t) for t in special_tokens)
    if special_pattern:
        combined_pattern = f"({special_pattern})|({PAT})"
    else:
        combined_pattern = f"({PAT})"

    tokens_with_type = []
    for match in re.finditer(combined_pattern, raw_text):
        if special_tokens and match.group(1) is not None:  
            tokens_with_type.append((match.group(1), True))   # 特殊token
        else:
            tok = match.group(2) if special_tokens else match.group(1)
            if tok:
                tokens_with_type.append((tok, False))         # 普通token
    return tokens_with_type


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # 加载 vocab
        vocab = {}
        with open(vocab_filepath, "rb") as vf:
            for i, line in enumerate(vf):
                vocab[i] = line.strip()

        # 加载 merges
        merges = []
        with open(merges_filepath, "rb") as mf:
            for line in mf:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for token, is_special in pre_tokenization_with_specials(text, self.special_tokens):
            if is_special:
                # 特殊符号直接映射
                for k, v in self.vocab.items():
                    if v.decode() == token:
                        token_ids.append(k)
                        break
            else:
                merged_tokens = perform_merge(token, self.merges)
                ids = map_merged_tokens_to_ids(merged_tokens, self.vocab)
                token_ids.extend(ids)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: 
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[i].decode("utf-8") for i in ids]
        return "".join(tokens)