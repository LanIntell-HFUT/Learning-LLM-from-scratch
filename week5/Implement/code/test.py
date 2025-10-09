from __future__ import annotations

import timeit
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
import yaml
from pathlib import Path
from cs336_basics.model import BasicsTransformerLM # type: ignore
from cs336_basics.optimizer import AdamW, get_cosine_lr
import torch.cuda.nvtx as nvtx
import os
from contextlib import nullcontext
import pandas as pd
from datetime import datetime

# 打印当前工作目录
print(os.getcwd())

# 读取配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 创建随机输入数据
def create_random_batch(batch_size: int, context_length: int, vocab_size: int, device: str) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)

def benchmark_model(
    model: nn.Module,
    batch: torch.Tensor,
    warmup_steps: int,
    benchmark_steps: int,
    forward_only: bool,
    device: str,
    autocast: bool
):
    """
    Benchmark the model's forward and backward passes.
    
    Args:
        model: The model to benchmark
        batch: Input batch tensor
        warmup_steps: Number of warmup steps
        benchmark_steps: Number of steps to benchmark
        forward_only: Whether to only benchmark forward pass
        device: Device to run on
        autocast: Whether to use autocast
        
    Returns:
        Tuple of (forward_time, backward_time, forward_memory, backward_memory) in seconds/GB
    """
    print("warm...")
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 设置上下文管理器
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()

    # 预热
    with ctx: 
        for _ in range(warmup_steps):
            lr = 0.1
            for group in optimizer.param_groups:
                group['lr'] = lr
            outputs = model(batch)
            if not forward_only:
                optimizer.zero_grad()
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
    
    # 基准测试
    print("benchmark...")
    forward_times = []
    backward_times = []
    with ctx: 
        for _ in range(benchmark_steps):
            lr = 0.1
            for group in optimizer.param_groups:
                group["lr"] = lr
            # Forward pass
            start_time = timeit.default_timer()
            outputs = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            forward_time = timeit.default_timer() - start_time
            forward_times.append(forward_time)
            
            if not forward_only:
                # Backward pass
                start_time = timeit.default_timer()
                optimizer.zero_grad()
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                backward_time = timeit.default_timer() - start_time
                backward_times.append(backward_time)

        # 内存测量
        torch.cuda.reset_peak_memory_stats()
        outputs = model(batch)
        torch.cuda.synchronize()
        memory_before_backward = torch.cuda.max_memory_allocated()/(1024**3)

        memory_backward = 0
        if not forward_only:
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad()
            loss = outputs.mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            memory_backward = torch.cuda.max_memory_allocated()/(1024**3)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times) if not forward_only else 0.0
    
    return avg_forward_time, avg_backward_time, memory_before_backward, memory_backward

def run_single_config(config_name: str, config_path: Path) -> List[Dict]:
    """运行单个配置的基准测试，返回结果数据"""
    print(f"当前测试: '{config_name}'")
    
    results = []
    
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 初始化模型
        model = BasicsTransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
        ).to(config["device"])

        model=torch.compile(model)
        # 创建随机批次数据
        batch = create_random_batch(
            config["batch_size"],
            config["context_length"],
            config["vocab_size"],
            config["device"]
        )
        
        # 运行基准测试 - 可以测试两种情况：False和True
        for autocast in [False, True]:  # 测试混合精度和非混合精度
            forward_time, backward_time, forward_memory, backward_memory = benchmark_model(
                model,
                batch,
                config["warmup_steps"],
                config["benchmark_steps"],
                config["forward_only"],
                config["device"],
                autocast
            )
            
            # 计算总时间
            total_time = forward_time + (backward_time if not config["forward_only"] else 0.0)
            
            # 收集结果数据
            result = {
                "模型名称": config_name,
                "混合精度(bf16)": "是" if autocast else "否",
                "平均前向传播时间(ms)": round(forward_time * 1000, 2),
                "前向传播峰值内存(GB)": round(forward_memory, 2),
                "平均反向传播时间(ms)": round(backward_time * 1000, 2) if not config["forward_only"] else 0.0,
                "反向传播内存峰值(GB)": round(backward_memory, 2) if not config["forward_only"] else 0.0,
                "总时间(ms)": round(total_time * 1000, 2),
                # 额外的配置信息
                "层数": config["num_layers"],
                "模型维度": config["d_model"],
                "注意力头数": config["num_heads"],
                "FFN维度": config["d_ff"],
                "上下文长度": config["context_length"],
                "批次大小": config["batch_size"]
            }
            
            results.append(result)
            
            # 打印结果
            print(f"\n{config_name} 基准测试结果 (混合精度: {'是' if autocast else '否'}):")
            print(f"模型配置:")
            print(f"  - 层数: {config['num_layers']}")
            print(f"  - 模型维度: {config['d_model']}")
            print(f"  - 注意力头数: {config['num_heads']}")
            print(f"  - FFN维度: {config['d_ff']}")
            print(f"  - 上下文长度: {config['context_length']}")
            print(f"  - 批次大小: {config['batch_size']}")
            print(f"  - 自动混合精度 (bf16): {autocast}")
            
            print(f"\n时间结果:")
            print(f"  - 平均前向传播时间: {forward_time*1000:.2f} ms")
            print(f"  - 前向传播峰值内存: {forward_memory:.2f} GB")
            
            if not config["forward_only"]:
                print(f"  - 平均反向传播时间: {backward_time*1000:.2f} ms")
                print(f"  - 反向传播内存峰值: {backward_memory:.2f} GB")
                print(f"  - 总时间: {total_time*1000:.2f} ms")
        
        # 清理GPU内存
        del model, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"配置 {config_name} 测试失败: {str(e)}")
    
    return results

def save_results_table(all_results: List[Dict]):
    """保存结果到表格文件"""
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存到CSV文件（支持中文）
    csv_filename = f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 CSV 文件: {csv_filename}")
    
    
    # 生成markdown格式的表格
    markdown_filename = f"benchmark_results_{timestamp}.md"
    with open(markdown_filename, 'w', encoding='utf-8') as f:
        f.write("# Transformer模型基准测试结果\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 主要性能指标表格
        f.write("## 性能指标总表\n\n")
        
        # 只显示主要指标
        main_cols = ["模型名称", "混合精度(bf16)", "平均前向传播时间(ms)", 
                     "前向传播峰值内存(GB)", "平均反向传播时间(ms)", 
                     "反向传播内存峰值(GB)", "总时间(ms)"]
        
        main_df = df[main_cols]
        f.write(main_df.to_markdown(index=False))
        f.write("\n\n")
        
        # 配置详情表格
        f.write("## 模型配置详情\n\n")
        config_cols = ["模型名称", "层数", "模型维度", "注意力头数", "FFN维度", "上下文长度", "批次大小"]
        config_df = df[config_cols].drop_duplicates(subset=["模型名称"])
        f.write(config_df.to_markdown(index=False))
        f.write("\n")
    
    print(f"结果已保存到 Markdown 文件: {markdown_filename}")
    

def main():
    # 定义所有配置文件
    config_files = {
        "Small": "small.yaml",
        "Medium": "medium.yaml", 
        "Large": "large.yaml",
        "xl":"xl.yaml"
    }
    
    config_dir = Path("../configures")
    
    print("开始基准测试...")
    
    # 存储所有结果
    all_results = []
    
    # 遍历所有配置文件
    for config_name, config_file in config_files.items():
        config_path = config_dir / config_file
        
        # 检查配置文件是否存在
        if not config_path.exists():
            print(f"警告: 配置文件 {config_path} 不存在，跳过...")
            continue
        
        # 运行单个配置的测试
        results = run_single_config(config_name, config_path)
        all_results.extend(results)
    
    print(f"\n")
    print("所有配置测试完成!")
    
    # 保存结果表格
    if all_results:
        save_results_table(all_results)
    else:
        print("没有有效的测试结果可保存")

if __name__ == "__main__":
    main()