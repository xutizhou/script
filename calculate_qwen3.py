#!/usr/bin/env python3
"""
模型FLOPs和参数量计算脚本
根据提供的模型架构信息计算每个模块的计算量和参数量
所有数值都以直观的单位显示

FLOPs计算说明：
- 对于矩阵乘法 A(m×k) × B(k×n)，包含 m×n×k 次乘法和 m×n×(k-1) 次加法
- 总FLOPs = m×n×(2k-1) ≈ 2×m×n×k
- 线性层的FLOPs都乘以2来包含乘法和加法运算
- 非矩阵乘法运算（如激活函数、softmax）不乘2
"""

def format_number(num):
    """将数字格式化为更直观的单位"""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"

def calculate_model_stats(batch_size=1, seq_len=2048, show_memory=True):
    """
    计算模型的FLOPs和参数量
    
    Args:
        batch_size (int): 批次大小
        seq_len (int): 序列长度
        show_memory (bool): 是否显示内存估算
    """
    
    # 模型基本参数
    vocab_size = 151936
    hidden_size = 4096
    num_layers = 94
    
    # Attention参数
    num_q_heads = 64
    num_kv_heads = 4  # GQA (Grouped Query Attention)
    head_dim = 128
    
    # MoE参数
    num_experts = 128
    num_activated_experts = 8
    intermediate_size = 1536  # up projection的一半
    
    print(f"模型配置:")
    print(f"  词汇表大小: {format_number(vocab_size)}")
    print(f"  隐藏维度: {format_number(hidden_size)}")
    print(f"  层数: {num_layers}")
    print(f"  Q头数: {num_q_heads}, KV头数: {num_kv_heads}")
    print(f"  专家总数: {num_experts}, 激活专家数: {num_activated_experts}")
    print(f"  批次大小: {batch_size}, 序列长度: {format_number(seq_len)}")
    print("\n" + "="*80)
    
    total_params = 0
    total_flops = 0
    
    # 1. Input Embedding
    print("\n1. Input Embedding Layer")
    embedding_params = vocab_size * hidden_size
    embedding_flops = batch_size * seq_len * hidden_size  # lookup操作，不涉及矩阵乘法
    
    print(f"  参数量: {format_number(embedding_params)}")
    print(f"  FLOPs: {format_number(embedding_flops)}")
    
    total_params += embedding_params
    total_flops += embedding_flops
    
    # 2. Attention Layers (x94)
    print(f"\n2. Attention Layers (x{num_layers})")
    
    # Q projection
    q_proj_params = hidden_size * (num_q_heads * head_dim)
    q_proj_flops = 2 * batch_size * seq_len * hidden_size * (num_q_heads * head_dim)  # 乘以2包含乘加运算
    
    # K, V projection  
    kv_proj_params = hidden_size * (num_kv_heads * head_dim * 2)
    kv_proj_flops = 2 * batch_size * seq_len * hidden_size * (num_kv_heads * head_dim * 2)  # 乘以2包含乘加运算
    
    # Attention computation
    # Q * K^T
    qk_flops = 2 * batch_size * num_q_heads * seq_len * seq_len * head_dim  # 乘以2包含乘加运算
    # Softmax (近似)
    softmax_flops = batch_size * num_q_heads * seq_len * seq_len * 3  # softmax不是矩阵乘法，不乘2
    # Attention weights * V
    av_flops = 2 * batch_size * num_q_heads * seq_len * seq_len * head_dim  # 乘以2包含乘加运算
    
    # O projection
    o_proj_params = (num_q_heads * head_dim) * hidden_size
    o_proj_flops = 2 * batch_size * seq_len * (num_q_heads * head_dim) * hidden_size  # 乘以2包含乘加运算
    
    # 每层attention的总计
    attn_params_per_layer = q_proj_params + kv_proj_params + o_proj_params
    attn_flops_per_layer = (q_proj_flops + kv_proj_flops + 
                           qk_flops + softmax_flops + av_flops + o_proj_flops)
    
    total_attn_params = attn_params_per_layer * num_layers
    total_attn_flops = attn_flops_per_layer * num_layers
    
    print(f"  每层参数量: {format_number(attn_params_per_layer)}")
    print(f"    - Q projection: {format_number(q_proj_params)}")
    print(f"    - KV projection: {format_number(kv_proj_params)}")
    print(f"    - O projection: {format_number(o_proj_params)}")
    print(f"  每层FLOPs: {format_number(attn_flops_per_layer)}")
    print(f"    - Q projection: {format_number(q_proj_flops)}")
    print(f"    - KV projection: {format_number(kv_proj_flops)}")
    print(f"    - QK computation: {format_number(qk_flops)}")
    print(f"    - Softmax: {format_number(softmax_flops)}")
    print(f"    - Attention×V: {format_number(av_flops)}")
    print(f"    - O projection: {format_number(o_proj_flops)}")
    print(f"  总参数量: {format_number(total_attn_params)}")
    print(f"  总FLOPs: {format_number(total_attn_flops)}")
    
    total_params += total_attn_params
    total_flops += total_attn_flops
    
    # 3. MoE Layers (x94)
    print(f"\n3. MoE Layers (x{num_layers})")
    
    # Up projection (gate + up)
    up_params_per_expert = hidden_size * (intermediate_size * 2)  # gate和up合并
    total_up_params = up_params_per_expert * num_experts
    activated_up_params = up_params_per_expert * num_activated_experts
    up_flops = 2 * batch_size * seq_len * hidden_size * (intermediate_size * 2) * num_activated_experts  # 乘以2包含乘加运算
    
    # Down projection
    down_params_per_expert = intermediate_size * hidden_size
    total_down_params = down_params_per_expert * num_experts
    activated_down_params = down_params_per_expert * num_activated_experts
    down_flops = 2 * batch_size * seq_len * intermediate_size * hidden_size * num_activated_experts  # 乘以2包含乘加运算
    
    # 激活函数 (SiLU + 乘法)
    activation_flops = batch_size * seq_len * intermediate_size * num_activated_experts * 3  # 激活函数不是矩阵乘法，不乘2
    
    # 路由网络
    router_params = hidden_size * num_experts
    router_flops = 2 * batch_size * seq_len * hidden_size * num_experts  # 乘以2包含乘加运算
    
    # 每层MoE的总计
    moe_params_per_layer = total_up_params + total_down_params + router_params
    moe_flops_per_layer = up_flops + down_flops + activation_flops + router_flops
    
    total_moe_params = moe_params_per_layer * num_layers
    total_moe_flops = moe_flops_per_layer * num_layers
    
    print(f"  每层参数量: {format_number(moe_params_per_layer)}")
    print(f"    - Up projection (总): {format_number(total_up_params)}")
    print(f"    - Down projection (总): {format_number(total_down_params)}")
    print(f"    - Router: {format_number(router_params)}")
    print(f"  每层激活参数量: {format_number(activated_up_params + activated_down_params)}")
    print(f"  每层FLOPs: {format_number(moe_flops_per_layer)}")
    print(f"    - Up projection: {format_number(up_flops)}")
    print(f"    - Down projection: {format_number(down_flops)}")
    print(f"    - Activation: {format_number(activation_flops)}")
    print(f"    - Router: {format_number(router_flops)}")
    print(f"  总参数量: {format_number(total_moe_params)}")
    print(f"  总FLOPs: {format_number(total_moe_flops)}")
    
    total_params += total_moe_params
    total_flops += total_moe_flops
    
    # 4. Output Embedding (通常与input embedding共享权重)
    print(f"\n4. Output Embedding Layer")
    output_embedding_params = 0  # 通常共享权重
    output_embedding_flops = 2 * batch_size * seq_len * hidden_size * vocab_size  # 乘以2包含乘加运算
    
    print(f"  参数量: {format_number(output_embedding_params)} (共享权重)")
    print(f"  FLOPs: {format_number(output_embedding_flops)}")
    
    total_flops += output_embedding_flops
    
    # 5. Layer Norm (估算)
    print(f"\n5. Layer Normalization")
    # 每层有2个LayerNorm: attention后和MoE后
    layernorm_params = hidden_size * 2 * num_layers * 2  # scale和bias
    layernorm_flops = batch_size * seq_len * hidden_size * 5 * 2 * num_layers  # 每个LN约5个操作
    
    print(f"  参数量: {format_number(layernorm_params)}")
    print(f"  FLOPs: {format_number(layernorm_flops)}")
    
    total_params += layernorm_params
    total_flops += layernorm_flops
    
    # 总计
    print("\n" + "="*80)
    print("📊 总计:")
    print(f"  总参数量: {format_number(total_params)}")
    print(f"  总FLOPs: {format_number(total_flops)}")
    
    # 激活参数量（实际使用的参数）
    activated_params = (embedding_params + 
                       total_attn_params + 
                       (activated_up_params + activated_down_params + router_params) * num_layers +
                       layernorm_params)
    
    print(f"  激活参数量: {format_number(activated_params)}")
    print(f"  参数利用率: {activated_params/total_params*100:.1f}%")
    
    # 详细参数分解
    print(f"\n📈 参数分解:")
    print(f"  Embedding: {format_number(embedding_params)} ({embedding_params/total_params*100:.1f}%)")
    print(f"  Attention: {format_number(total_attn_params)} ({total_attn_params/total_params*100:.1f}%)")
    print(f"  MoE: {format_number(total_moe_params)} ({total_moe_params/total_params*100:.1f}%)")
    print(f"  LayerNorm: {format_number(layernorm_params)} ({layernorm_params/total_params*100:.1f}%)")
    
    # FLOPs分解
    print(f"\n⚡ FLOPs分解:")
    print(f"  Embedding: {format_number(embedding_flops)} ({embedding_flops/total_flops*100:.1f}%)")
    print(f"  Attention: {format_number(total_attn_flops)} ({total_attn_flops/total_flops*100:.1f}%)")
    print(f"  MoE: {format_number(total_moe_flops)} ({total_moe_flops/total_flops*100:.1f}%)")
    print(f"  Output: {format_number(output_embedding_flops)} ({output_embedding_flops/total_flops*100:.1f}%)")
    print(f"  LayerNorm: {format_number(layernorm_flops)} ({layernorm_flops/total_flops*100:.1f}%)")
    
    # 内存估算
    if show_memory:
        print(f"\n💾 内存使用量估算:")
        print("-" * 40)
        
        # 模型权重内存 (fp16)
        model_memory = activated_params * 2  # fp16 = 2 bytes
        
        # 激活内存估算 (粗略估计)
        # 激活内存主要包括：attention中间结果、MoE中间结果
        attention_activation = batch_size * seq_len * hidden_size * 8  # 多个中间tensor
        moe_activation = batch_size * seq_len * intermediate_size * 2 * num_activated_experts  # MoE中间结果，激活8个专家
        activation_memory = (attention_activation + moe_activation) * num_layers * 2  # fp16
        
        # KV Cache (假设全部缓存)
        kv_cache_memory = batch_size * seq_len * num_kv_heads * head_dim * 2 * num_layers * 2  # k+v, fp16
        
        total_memory = model_memory + activation_memory + kv_cache_memory
        
        print(f"  模型权重内存: {format_number(model_memory)}B")
        print(f"  激活内存: {format_number(activation_memory)}B")
        print(f"  KV Cache内存: {format_number(kv_cache_memory)}B")
        print(f"  总内存估算: {format_number(total_memory)}B")
        print(f"  约 {total_memory/(1024**3):.1f} GB")
    
    return {
        'total_params': total_params,
        'activated_params': activated_params,
        'total_flops': total_flops,
        'params_breakdown': {
            'embedding': embedding_params,
            'attention': total_attn_params,
            'moe': total_moe_params,
            'layernorm': layernorm_params
        },
        'flops_breakdown': {
            'embedding': embedding_flops,
            'attention': total_attn_flops,
            'moe': total_moe_flops,
            'output': output_embedding_flops,
            'layernorm': layernorm_flops
        }
    }

def compare_sequence_lengths():
    """比较不同序列长度下的计算量"""
    print("\n" + "="*80)
    print("🔍 不同序列长度下的FLOPs对比:")
    print("="*80)
    
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    
    print(f"{'序列长度':<10} {'总FLOPs':<12} {'Attention FLOPs':<15} {'MoE FLOPs':<12}")
    print("-" * 55)
    
    for seq_len in seq_lengths:
        stats = calculate_model_stats(batch_size=1, seq_len=seq_len, show_memory=False)
        attn_flops = stats['flops_breakdown']['attention']
        moe_flops = stats['flops_breakdown']['moe']
        total_flops = stats['total_flops']
        
        print(f"{seq_len:<10} {format_number(total_flops):<12} {format_number(attn_flops):<15} {format_number(moe_flops):<12}")

if __name__ == "__main__":
    print("🚀 Qwen3-235B 模型分析")
    print("="*80)
    
    # 计算默认配置下的统计信息（包含内存估算）
    calculate_model_stats(batch_size=4, seq_len=2048, show_memory=True)
    
    # 比较不同序列长度（不显示内存，避免重复）
    # compare_sequence_lengths() 