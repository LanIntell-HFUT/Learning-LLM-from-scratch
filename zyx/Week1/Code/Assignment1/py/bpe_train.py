import multiprocessing
from collections import Counter
from functools import partial
import os
from typing import BinaryIO

# 初始化词表
def init_vocabulary(vocab_size=256) -> dict:
    init_vocab = {chr(i):i for i in range(vocab_size)} # 初始词典有256种可能的字节值
    return init_vocab
    
# 去除特殊tokens，便于统计词频
def remove_special_tokens(raw_text,special_tokens = ["<|endoftext|>"]):
    for sp_token in special_tokens:
        raw_text = raw_text.replace(sp_token, "")
    return raw_text

import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 预编解码
def pre_tokenization(raw_text : str,
                     PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                     special_tokens=["<|endoftext|>"]) -> re.Scanner:
    removed_text = remove_special_tokens(raw_text,special_tokens)
    return re.finditer(PAT,removed_text)


# 统计经预编解码后的文本的词频
def count_word_freq(it : re.Scanner) -> dict[str : int]:
    word_freq = {}
    for i in it:
        word_freq[i.group()] = word_freq.get(i.group(),0) + 1
    return word_freq


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    把文件分割成可以被单独计数的块，如果边界重叠，可能返回比预期更少的块
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    # print(f"初步简单规划的区块边界为:{chunk_boundaries}")
    # print("第一个区块无需检查")
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # 目的是确保区块边界开始于特殊token
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        # print(f"第{bi+1}/{len(chunk_boundaries) - 1}个区块，初始边界猜测位置为：{initial_position}")
        file.seek(initial_position)  # 在猜测的边界开始循环检测(EOF/特殊token)
        
        while True: # 每一次循环的操作对象都是一个mini_chunk
            mini_chunk = file.read(mini_chunk_size)  # 读取一个 mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                # print("抵达文件末尾，结束")
                chunk_boundaries[bi] = file_size
                break

            # 在这个mini chunk里寻找特殊token，found_at返回的是特殊token在mini chunck中的相对位置
            found_at = mini_chunk.find(split_special_token)
            
            if found_at != -1: # 如果找到了特殊token
                # print(f"在{found_at}处发现特殊token, 将区块开始位置置于该特殊token之后")
                chunk_boundaries[bi] = initial_position + found_at # 把这个边界的实际初始位置后移found_at个单位，这样正好能把边界设在特殊token的后面。
                break

            # print("forward") # 如果在该mini chunk中未发现特殊token，则继续检查下一个mini chunk
            initial_position += mini_chunk_size # 始终保持边界指针位于当前mini chunk的开头，以便于用相对位置来移动指针

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 统计经预编解码后的文本的词频(多进程版本)
def process_chunk(start_end_pair, file_path):
    """
    处理单个文件块并计算词频。
    此函数由每个工作进程独立执行。

    Args:
        start_end_pair (tuple): 包含块的起始和结束字节位置的元组 (start, end)。
        file_path (str): 要读取的文件的路径。

    Returns:
        Counter: 该块内单词及其频率的 Counter 对象。
    """
    start, end = start_end_pair
    # 每个进程必须独立打开文件，因为文件对象不能在进程间共享。
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # 对你的数据块运行预分词并存储词频
        it = pre_tokenization(chunk)
        word_count = count_word_freq(it)
        return Counter(word_count)

def parallel_word_count(file_path, num_processes=4):
    """
    并行计算文件中单词的频率。

    Args:
        file_path (str): 目标文件的路径。
        num_processes (int): 要使用的进程数。

    Returns:
        dict: 包含所有单词及其总频率的字典。
    """
    # 1. 在主进程中确定所有数据块的边界
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 创建一个 (start, end) 元组的列表，供工作进程处理
    chunks = list(zip(boundaries[:-1], boundaries[1:]))

    # 2. 创建一个工作进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 functools.partial 创建一个新函数
        # 这样可以将 file_path 参数固定，pool.map 调用时只需传递变化的 chunk 参数
        worker_func = partial(process_chunk, file_path=file_path)

        # 3. 将任务分配给进程池并收集结果
        # pool.map 会将 chunks 列表中的每个元素作为参数传递给 worker_func，并并行执行
        list_of_counts = pool.map(worker_func, chunks)

    # 4. 合并所有进程返回的局部结果
    final_word_freq = Counter()
    for count_obj in list_of_counts:
        final_word_freq.update(count_obj)
        
    return dict(final_word_freq)
    
# 将词频字典的键（字符串类型）转换为元组，便于后续统计字节对词频
def word_freq_str2tuple(word_freq : dict[str:int]) -> dict[tuple:int]:
    tuple_word_freq = {}
    for k,v in word_freq.items():
        tuple_word_freq[tuple(k)] = v
    return tuple_word_freq

# 统计字节对词频
def count_bytes_pair_freq(tuple_word_freq : dict[tuple:int]) -> dict[tuple[str,str]:int]:
    # print(f"输入:{tuple_word_freq}")
    bytes_pair_freq = {}
    for k,v in tuple_word_freq.items():
        if len(k)==1: continue # 由于只统计字节对的词频，所以只有一个字节的元组可以不考虑
        # 计算字节对词频
        for i in range(len(k)-1):
            combined_tuple = (k[i],k[i+1])
            bytes_pair_freq[combined_tuple] = bytes_pair_freq.get(combined_tuple,0) + v
    return bytes_pair_freq

# 找出词频最高的字节对并加入词表
def find_max_bytes_pair(bytes_pair_freq : dict[tuple[str,str]:int]) -> tuple[str,str]:
    max_value = max(bytes_pair_freq.values()) # 最高的词频数
    # print(f"max_value:{max_value}")
    max_items = {key: value for key, value in bytes_pair_freq.items() if value == max_value}
    # print(f"max_items:{max_items}")
    max_items_list = list(max_items.keys())
    word_tuple_to_add = max(max_items_list) # 要加入词表的新词
    return word_tuple_to_add

def add_to_vocab(raw_vocab : dict[str:int], new_key:str) -> dict[str:int]:
    new_vocab = raw_vocab.copy()
    # print(len(raw_vocab))
    # print(new_key)
    new_vocab[new_key] = len(raw_vocab)
    # print(len(new_vocab))
    
    return new_vocab

# 修改tuple_word_freq，合并新加入的字节对
# def update_tuple_word_freq(raw_tuple_word_freq : dict[tuple:int], word_tuple_to_add:tuple[str,str],new_key:str) -> dict[tuple:int] :
#     new_tuple_word_freq = {}
#     for k,v in raw_tuple_word_freq.items():
#         if len(k)==1:
#             continue
#         # 寻找需要合并的字节对
#         for i in range(len(k)-1):
#             if k[i] == word_tuple_to_add[0] and k[i+1] == word_tuple_to_add[1]: # 合并
#                 ls = list(k)
#                 ls[i] = new_key
#                 del ls[i+1]
#                 new_tuple_word_freq[tuple(ls)]=v
#                 break
#             if i == len(k)-2:
#                 new_tuple_word_freq[k] = v    
#     return new_tuple_word_freq
# 修改tuple_word_freq，合并新加入的字节对 (修正版)
def update_tuple_word_freq(raw_tuple_word_freq: dict[tuple, int], word_tuple_to_add: tuple[str, str], new_key: str) -> dict[tuple, int]:
    new_tuple_word_freq = {}
    for word_tuple, freq in raw_tuple_word_freq.items():
        if len(word_tuple) < 2:
            # 如果词本身只有一个单元，不可能包含字节对，直接保留
            new_tuple_word_freq[word_tuple] = freq
            continue

        # 使用 while 循环来替换一个词中所有的目标字节对
        new_word_list = list(word_tuple)
        i = 0
        while i < len(new_word_list) - 1:
            if new_word_list[i] == word_tuple_to_add[0] and new_word_list[i+1] == word_tuple_to_add[1]:
                new_word_list[i] = new_key
                del new_word_list[i+1]
                # 合并后，索引 i 不需要增加，因为当前位置是新token，需要和下一个token再比较
                # 例如：合并 ('a','b','c') 中的 ('a','b')，得到 ('ab','c')，下一次应该从 'ab' 开始
            else:
                i += 1
        
        new_tuple_word_freq[tuple(new_word_list)] = freq
            
    return new_tuple_word_freq
def bpe_merge(init_vocab,file_path,vocab_size,special_tokens=["<|endoftext|>"]) -> dict[bytes:int]:
    # 统计经预分词处理后的词频
    # word_freq = count_word_freq(it)
    word_freq = parallel_word_count(file_path)
    # 将词频字典键的字符串转换为元组
    tuple_word_freq = word_freq_str2tuple(word_freq)

    new_vocab = init_vocab.copy()
    
    # 计算要合并的次数(由于最后还要加上特殊token，所以合并次数要减去特殊tokens的数量)
    t = vocab_size - len(init_vocab) - len(special_tokens)
    # print(f"将进行{t}次合并")
    merges = [] # 记录每次合并的tuple[bytes:bytes]
    # 合并t次
    for i in range(t):
        # print(f"第{i+1}次合并开始")
        # 统计字节对词频
        bytes_pair_freq = count_bytes_pair_freq(tuple_word_freq)
        # 找出词频最高的字节对
        word_tuple_to_add = find_max_bytes_pair(bytes_pair_freq)
        # print(word_tuple_to_add)
        # 转换成字节串，加入到merges中
        word_tuple_to_merge = tuple(s.encode('utf-8') for s in word_tuple_to_add)
        merges.append(word_tuple_to_merge)
        
        new_key = word_tuple_to_add[0] + word_tuple_to_add[1]
        # print(new_key)
        # 加入词表
        new_vocab = add_to_vocab(new_vocab, new_key)
        # 修改tuple_word_freq，合并新加入的字节对
        tuple_word_freq = update_tuple_word_freq(tuple_word_freq,word_tuple_to_add,new_key)
        # print(f"输出:{tuple_word_freq}")
    assert len(new_vocab) == vocab_size - len(special_tokens), f"length of new_vocab is {len(new_vocab)},while expected is {vocab_size - len(special_tokens)}"
    
    # 加上特殊tokens
    for sp_tok in special_tokens:
        new_vocab[sp_tok] = len(new_vocab)
    assert len(new_vocab) == vocab_size

    # 把字符串转换为字节串
    new_vocab_bytes = {key.encode('utf-8'): value for key, value in new_vocab.items()}
    reversed_dict = {v: k for k, v in new_vocab_bytes.items()}
    return reversed_dict,merges


def BPE_tokenizer_training(input_path="./data/TinyStoriesV2-GPT4-valid.txt", 
                           vocab_size=300, special_tokens=["<|endoftext|>"]) :
    init_vocab = init_vocabulary()
    return bpe_merge(init_vocab, input_path, vocab_size)

