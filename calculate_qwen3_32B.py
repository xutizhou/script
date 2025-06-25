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

def calculate_model_prefill_stats(batch_size=1, seq_len=2048, show_memory=True):
    """
    计算模型预填充阶段的FLOPs、参数量和访存量
    
    Args:
        batch_size (int): 批次大小
        seq_len (int): 序列长度
        show_memory (bool): 是否显示内存估算
    """
    
    # Qwen3-32B 模型参数
    vocab_size = 151936
    hidden_size = 5120
    num_layers = 64
    
    # Attention参数
    num_q_heads = 64
    num_kv_heads = 8  # GQA
    head_dim = 128
    
    # FFN (Feed-Forward Network) 参数
    intermediate_size = 25600
    
    print(f"模型配置:")
    print(f"  词汇表大小: {format_number(vocab_size)}")
    print(f"  隐藏维度: {format_number(hidden_size)}")
    print(f"  层数: {num_layers}")
    print(f"  Q头数: {num_q_heads}, KV头数: {num_kv_heads}, Head维度: {head_dim}")
    print(f"  FFN中间维度: {format_number(intermediate_size)}")
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
    
    # 2. Attention Layers (x64)
    print(f"\n2. Attention Layers (x{num_layers})")
    
    # Q, K, V projections
    q_proj_params = hidden_size * (num_q_heads * head_dim)
    kv_proj_params = hidden_size * (num_kv_heads * head_dim * 2)
    q_proj_flops = 2 * batch_size * seq_len * hidden_size * (num_q_heads * head_dim)
    kv_proj_flops = 2 * batch_size * seq_len * hidden_size * (num_kv_heads * head_dim * 2)
    
    # Attention computation
    qk_flops = 2 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    softmax_flops = batch_size * num_q_heads * seq_len * seq_len * 3  # 近似
    av_flops = 2 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    
    # O projection
    o_proj_params = (num_q_heads * head_dim) * hidden_size
    o_proj_flops = 2 * batch_size * seq_len * (num_q_heads * head_dim) * hidden_size
    
    attn_params_per_layer = q_proj_params + kv_proj_params + o_proj_params
    attn_flops_per_layer = (q_proj_flops + kv_proj_flops + qk_flops + softmax_flops + av_flops + o_proj_flops)
    
    total_attn_params = attn_params_per_layer * num_layers
    total_attn_flops = attn_flops_per_layer * num_layers
    
    print(f"  每层参数量: {format_number(attn_params_per_layer)}")
    print(f"  每层FLOPs: {format_number(attn_flops_per_layer)}")
    print(f"  总参数量: {format_number(total_attn_params)}")
    print(f"  总FLOPs: {format_number(total_attn_flops)}")
    
    total_params += total_attn_params
    total_flops += total_attn_flops
    
    # 3. FFN Layers (x64)
    print(f"\n3. FFN Layers (x{num_layers})")
    
    # Gate and Up projections
    gate_up_params = hidden_size * intermediate_size * 2
    gate_up_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size * 2
    
    # Activation (SiLU * element-wise multiply)
    activation_flops = batch_size * seq_len * intermediate_size * 2 # silu + mult
    
    # Down projection
    down_params = intermediate_size * hidden_size
    down_flops = 2 * batch_size * seq_len * intermediate_size * hidden_size
    
    ffn_params_per_layer = gate_up_params + down_params
    ffn_flops_per_layer = gate_up_flops + activation_flops + down_flops
    
    total_ffn_params = ffn_params_per_layer * num_layers
    total_ffn_flops = ffn_flops_per_layer * num_layers
    
    print(f"  每层参数量: {format_number(ffn_params_per_layer)}")
    print(f"  每层FLOPs: {format_number(ffn_flops_per_layer)}")
    print(f"  总参数量: {format_number(total_ffn_params)}")
    print(f"  总FLOPs: {format_number(total_ffn_flops)}")
    
    total_params += total_ffn_params
    total_flops += total_ffn_flops
    
    # 4. Output Layer
    print(f"\n4. Output Layer")
    # `tie_word_embeddings` is false, so this is a separate layer
    output_params = hidden_size * vocab_size
    output_flops = 2 * batch_size * seq_len * hidden_size * vocab_size
    
    print(f"  参数量: {format_number(output_params)}")
    print(f"  FLOPs: {format_number(output_flops)}")
    
    total_params += output_params
    total_flops += output_flops
    
    # 5. Layer Norm
    print(f"\n5. Layer Normalization")
    # RMSNorm per layer: before attention and before FFN
    layernorm_params = hidden_size * 2 * num_layers
    layernorm_flops = batch_size * seq_len * hidden_size * 4 * 2 * num_layers # RMSNorm is simpler than LayerNorm
    
    # Final RMSNorm after last layer
    final_norm_params = hidden_size
    final_norm_flops = batch_size * seq_len * hidden_size * 4
    
    total_ln_params = layernorm_params + final_norm_params
    total_ln_flops = layernorm_flops + final_norm_flops

    print(f"  参数量: {format_number(total_ln_params)}")
    print(f"  FLOPs: {format_number(total_ln_flops)}")
    
    total_params += total_ln_params
    total_flops += total_ln_flops
    
    # 总计
    print("\n" + "="*80)
    print("📊 总计:")
    print(f"  总参数量: {format_number(total_params)}")
    print(f"  总FLOPs: {format_number(total_flops)}")
    
    # 内存估算
    if show_memory:
        print(f"\n💾 内存访存量估算 (bf16):")
        
        # 模型权重内存 (bf16)
        model_memory = total_params * 2  # bf16 = 2 bytes
        
        # 激活内存估算
        attention_activation = batch_size * seq_len * hidden_size * 8  # Q, K, V, O, etc.
        ffn_activation = batch_size * seq_len * intermediate_size * 2 # Gate, Up outputs
        activation_memory = (attention_activation + ffn_activation) * num_layers * 2  # bf16
        
        # KV Cache (bf16)
        kv_cache_memory = batch_size * seq_len * num_kv_heads * head_dim * 2 * num_layers * 2  # k+v
        
        total_memory = model_memory + activation_memory + kv_cache_memory
        
        print(f"  模型权重内存: {format_number(model_memory/1024**3)} GB")
        print(f"  激活内存: {format_number(activation_memory/1024**3)} GB")
        print(f"  KV Cache内存: {format_number(kv_cache_memory/1024**3)} GB")
        print(f"  总内存估算: {format_number(total_memory/1024**3)} GB")

def estimate_time_us(flops, mem_bytes, tflops, gpu_mem_bandwidth_gbps):
    """根据FLOPs和访存量估算理论时间（us）"""
    # Time based on compute (FLOPs)
    compute_time_s = (flops / (tflops * 1e12)) if tflops > 0 else 0
    
    # Time based on memory access (Bytes)
    mem_time_s = (mem_bytes / (gpu_mem_bandwidth_gbps * 1e9)) if gpu_mem_bandwidth_gbps > 0 else 0
    
    # Convert to microseconds
    compute_time_us = compute_time_s * 1e6
    mem_time_us = mem_time_s * 1e6
    
    return compute_time_us, mem_time_us

def calculate_model_decode_stats(batch_size=1, prefix_length=2048, show_memory=True, gpu_mem_bandwidth_gbps=1398, gpu_tflops_bf16=148, gpu_tflops_fp8=296):
    """
    计算模型解码阶段的FLOPs、参数量和访存量，并估算理论执行时间。
    假设:
    - GEMM (矩阵乘法) 权重为 FP8 (1 byte)。
    - 其他所有张量 (激活、KV Cache、Norm权重) 为 BF16 (2 bytes)。
    """
    
    # Qwen3-32B 模型参数
    vocab_size = 151936
    hidden_size = 5120
    num_layers = 64
    
    # Attention参数
    num_q_heads = 64
    num_kv_heads = 8  # GQA
    head_dim = 128
    
    # FFN (Feed-Forward Network) 参数
    intermediate_size = 25600

    new_token_len = 1
    bf16_size = 2
    fp8_size = 1
    
    print(f"模型配置 (解码阶段):")
    print(f"  批次大小: {batch_size}, 前缀长度: {format_number(prefix_length)}")
    print(f"  GPU Specs: Mem BW={gpu_mem_bandwidth_gbps} GB/s, TFLOPS (BF16/FP8)={gpu_tflops_bf16}/{gpu_tflops_fp8}")
    print("\n" + "="*80)
    
    total_params = 0
    total_flops = 0
    total_mem_access = 0
    total_compute_time_us = 0
    total_mem_time_us = 0

    # 1. Input Embedding (只处理新token)
    print("\n1. Input Embedding Layer")
    embedding_params = vocab_size * hidden_size
    embedding_flops = 0 # Lookup is memory-bound
    embedding_mem_access = (embedding_params * bf16_size) + (batch_size * new_token_len * hidden_size * bf16_size) # read weights + write hidden_states
    
    compute_time, mem_time = estimate_time_us(embedding_flops, embedding_mem_access, gpu_tflops_bf16, gpu_mem_bandwidth_gbps)
    print(f"  参数量: {format_number(embedding_params)}")
    print(f"  FLOPs: {format_number(embedding_flops)}")
    print(f"  访存量: {format_number(embedding_mem_access)}B")
    print(f"  Time (Compute): {compute_time:.2f} us, Time (Memory): {mem_time:.2f} us")
    total_params += embedding_params; total_flops += embedding_flops; total_mem_access += embedding_mem_access
    total_compute_time_us += compute_time; total_mem_time_us += mem_time

    # 2. Attention Layers (x64)
    print(f"\n2. Attention Layers (x{num_layers})")
    q_proj_params = hidden_size * (num_q_heads * head_dim)
    kv_proj_params = hidden_size * (num_kv_heads * head_dim * 2)
    o_proj_params = (num_q_heads * head_dim) * hidden_size
    attn_params_per_layer = q_proj_params + kv_proj_params + o_proj_params

    # QKV Proj
    qkv_proj_flops = 2 * batch_size * new_token_len * hidden_size * (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv_proj_mem_access = (batch_size * new_token_len * hidden_size * bf16_size) + \
                          (batch_size * (num_q_heads + 2 * num_kv_heads) * head_dim * bf16_size) + \
                          (hidden_size * (num_q_heads + 2 * num_kv_heads) * head_dim * fp8_size)
    
    # Attention Score (BF16)
    attn_flops = 2 * batch_size * num_q_heads * new_token_len * prefix_length * head_dim * 2 # QK^T and Score*V
    attn_mem_access = (batch_size * num_q_heads * new_token_len * head_dim * bf16_size) + \
                      (2 * batch_size * prefix_length * num_kv_heads * head_dim * bf16_size) + \
                      (batch_size * num_q_heads * new_token_len * head_dim * bf16_size) # Q read, KV read, O write
    
    # O Proj
    o_proj_flops = 2 * batch_size * new_token_len * (num_q_heads * head_dim) * hidden_size
    o_proj_mem_access = (batch_size * new_token_len * num_q_heads * head_dim * fp8_size) + \
                        (batch_size * new_token_len * hidden_size * bf16_size) + \
                        (num_q_heads * head_dim * hidden_size * fp8_size)

    attn_flops_per_layer = qkv_proj_flops + attn_flops + o_proj_flops
    
    print(f"  -- Per Layer --")
    # QKV Proj Time
    qkv_compute_time, qkv_mem_time = estimate_time_us(qkv_proj_flops, qkv_proj_mem_access, gpu_tflops_fp8, gpu_mem_bandwidth_gbps)
    print(f"  QKV Proj (FP8): FLOPs={format_number(qkv_proj_flops)}, Mem={format_number(qkv_proj_mem_access)}B, Time(C/M)={qkv_compute_time:.2f}/{qkv_mem_time:.2f} us")
    # Attn Score Time
    attn_compute_time, attn_mem_time = estimate_time_us(attn_flops, attn_mem_access, gpu_tflops_bf16, gpu_mem_bandwidth_gbps)
    print(f"  Attn Score (BF16): FLOPs={format_number(attn_flops)}, Mem={format_number(attn_mem_access)}B, Time(C/M)={attn_compute_time:.2f}/{attn_mem_time:.2f} us")
    # O Proj Time
    o_proj_compute_time, o_proj_mem_time = estimate_time_us(o_proj_flops, o_proj_mem_access, gpu_tflops_fp8, gpu_mem_bandwidth_gbps)
    print(f"  O Proj (FP8): FLOPs={format_number(o_proj_flops)}, Mem={format_number(o_proj_mem_access)}B, Time(C/M)={o_proj_compute_time:.2f}/{o_proj_mem_time:.2f} us")
    
    total_attn_params = attn_params_per_layer * num_layers
    total_attn_flops = attn_flops_per_layer * num_layers
    total_attn_mem_access = (qkv_proj_mem_access + attn_mem_access + o_proj_mem_access) * num_layers
    
    total_attn_compute_time_us = (qkv_compute_time + attn_compute_time + o_proj_compute_time) * num_layers
    total_attn_mem_time_us = (qkv_mem_time + attn_mem_time + o_proj_mem_time) * num_layers
    print(f"  Total (x{num_layers}): FLOPs={format_number(total_attn_flops)}, Mem={format_number(total_attn_mem_access)}B, Time(C/M)={total_attn_compute_time_us:.2f}/{total_attn_mem_time_us:.2f} us")
    
    total_params += total_attn_params; total_flops += total_attn_flops; total_mem_access += total_attn_mem_access
    total_compute_time_us += total_attn_compute_time_us; total_mem_time_us += total_attn_mem_time_us

    # 3. FFN Layers (x64)
    print(f"\n3. FFN Layers (x{num_layers})")
    gate_up_params = hidden_size * intermediate_size * 2
    down_params = intermediate_size * hidden_size
    ffn_params_per_layer = gate_up_params + down_params

    # Gate/Up Proj
    gate_up_flops = 2 * batch_size * new_token_len * hidden_size * intermediate_size * 2
    gate_up_mem_access = (batch_size * new_token_len * hidden_size * fp8_size) + \
                         (batch_size * new_token_len * intermediate_size * 2 * bf16_size) + \
                         (hidden_size * intermediate_size * 2 * fp8_size)
    
    # Down Proj
    down_flops = 2 * batch_size * new_token_len * intermediate_size * hidden_size
    down_mem_access = (batch_size * new_token_len * intermediate_size * fp8_size) + \
                      (batch_size * new_token_len * hidden_size * bf16_size) + \
                      (intermediate_size * hidden_size * fp8_size)
    
    ffn_flops_per_layer = gate_up_flops + down_flops # Note: activation flops are negligible
    
    print(f"  -- Per Layer --")
    gate_up_compute_time, gate_up_mem_time = estimate_time_us(gate_up_flops, gate_up_mem_access, gpu_tflops_fp8, gpu_mem_bandwidth_gbps)
    print(f"  Gate/Up Proj (FP8): FLOPs={format_number(gate_up_flops)}, Mem={format_number(gate_up_mem_access)}B, Time(C/M)={gate_up_compute_time:.2f}/{gate_up_mem_time:.2f} us")
    down_compute_time, down_mem_time = estimate_time_us(down_flops, down_mem_access, gpu_tflops_fp8, gpu_mem_bandwidth_gbps)
    print(f"  Down Proj (FP8): FLOPs={format_number(down_flops)}, Mem={format_number(down_mem_access)}B, Time(C/M)={down_compute_time:.2f}/{down_mem_time:.2f} us")

    total_ffn_params = ffn_params_per_layer * num_layers
    total_ffn_flops = ffn_flops_per_layer * num_layers
    total_ffn_mem_access = (gate_up_mem_access + down_mem_access) * num_layers
    total_ffn_compute_time_us = (gate_up_compute_time + down_compute_time) * num_layers
    total_ffn_mem_time_us = (gate_up_mem_time + down_mem_time) * num_layers
    print(f"  Total (x{num_layers}): FLOPs={format_number(total_ffn_flops)}, Mem={format_number(total_ffn_mem_access)}B, Time(C/M)={total_ffn_compute_time_us:.2f}/{total_ffn_mem_time_us:.2f} us")
    
    total_params += total_ffn_params; total_flops += total_ffn_flops; total_mem_access += total_ffn_mem_access
    total_compute_time_us += total_ffn_compute_time_us; total_mem_time_us += total_ffn_mem_time_us

    # 4. Output Layer
    print("\n4. Output Layer")
    output_params = hidden_size * vocab_size
    output_flops = 2 * batch_size * new_token_len * hidden_size * vocab_size
    output_mem_access = (batch_size * new_token_len * hidden_size * bf16_size) + \
                        (batch_size * new_token_len * vocab_size * bf16_size) + \
                        (hidden_size * vocab_size * fp8_size)
    
    compute_time, mem_time = estimate_time_us(output_flops, output_mem_access, gpu_tflops_fp8, gpu_mem_bandwidth_gbps)
    print(f"  参数量: {format_number(output_params)}")
    print(f"  FLOPs: {format_number(output_flops)}")
    print(f"  访存量: {format_number(output_mem_access)}B")
    print(f"  Time (Compute): {compute_time:.2f} us, Time (Memory): {mem_time:.2f} us")

    total_params += output_params; total_flops += output_flops; total_mem_access += output_mem_access
    total_compute_time_us += compute_time; total_mem_time_us += mem_time

    # 5. Other Memory Access (Norm, Residual, KV Cache)
    print("\n5. Other Memory Access (No significant FLOPs)")
    # Norm and Residuals
    # Each layer has 2 norms and 2 residual connections. Read/Write for each.
    layernorm_params = (hidden_size * 2 * num_layers) + hidden_size # all norm weights
    norm_resid_mem_access = (batch_size * new_token_len * hidden_size * bf16_size * 2 * 2 * 2 * num_layers) + \
                            (layernorm_params * bf16_size)
    
    # KV Cache
    kv_cache_read_volume = batch_size * prefix_length * num_kv_heads * 2 * head_dim * bf16_size * num_layers
    kv_cache_write_volume = batch_size * new_token_len * num_kv_heads * 2 * head_dim * bf16_size * num_layers
    total_kv_cache_access = kv_cache_read_volume + kv_cache_write_volume
    
    _, mem_time_norm = estimate_time_us(0, norm_resid_mem_access, gpu_tflops_bf16, gpu_mem_bandwidth_gbps)
    _, mem_time_kv = estimate_time_us(0, total_kv_cache_access, gpu_tflops_bf16, gpu_mem_bandwidth_gbps)
    print(f"  Norm/Residuals: Mem={format_number(norm_resid_mem_access)}B, Time (Memory): {mem_time_norm:.2f} us")
    print(f"  KV Cache R/W: Mem={format_number(total_kv_cache_access)}B, Time (Memory): {mem_time_kv:.2f} us")

    total_mem_access += norm_resid_mem_access + total_kv_cache_access
    total_mem_time_us += mem_time_norm + mem_time_kv

    print("\n" + "="*80)
    print("📊 总计 (解码阶段 - 生成1个token):")
    print(f"  总参数量: {format_number(total_params)}")
    print(f"  总FLOPs: {format_number(total_flops)}")
    print(f"  总访存量: {format_number(total_mem_access)}B")
    print("-" * 40)
    print(f"  总估算时间 (Compute-bound): {total_compute_time_us:.2f} us")
    print(f"  总估算时间 (Memory-bound): {total_mem_time_us:.2f} us")
    print(f"  预估瓶颈: {'Memory' if total_mem_time_us > total_compute_time_us else 'Compute'}")
    
    # 内存占用估算
    if show_memory:
        print(f"\n💾 内存占用估算:")
        
        # 模型权重内存 (混合精度)
        gemm_params = total_attn_params + total_ffn_params + output_params
        non_gemm_params = embedding_params + layernorm_params
        model_memory = (gemm_params * fp8_size) + (non_gemm_params * bf16_size)

        # 激活内存 (BF16)
        attention_activation = batch_size * new_token_len * hidden_size * 8
        ffn_activation = batch_size * new_token_len * intermediate_size * 2
        activation_memory = (attention_activation + ffn_activation) * num_layers * bf16_size
        
        # KV Cache (BF16)
        kv_cache_memory = batch_size * prefix_length * num_kv_heads * head_dim * 2 * num_layers * bf16_size
        total_memory = model_memory + activation_memory + kv_cache_memory
        
        print(f"  模型权重内存 (FP8+BF16): {format_number(model_memory/1024**3)} GB")
        print(f"  激活内存 (BF16): {format_number(activation_memory/1024**3)} GB")
        print(f"  KV Cache内存 (BF16): {format_number(kv_cache_memory/1024**3)} GB")
        print(f"  总内存占用: {format_number(total_memory/1024**3)} GB")

if __name__ == "__main__":
    print("🚀 Qwen3-32B 模型分析")
    print("="*80)
    
    # H100 SXM5 specs as reference
    GPU_MEM_BANDWIDTH_GBPS = 1398 
    GPU_TFLOPS_BF16 = 148
    GPU_TFLOPS_FP8 = 296

    # 计算预填充阶段统计信息
    print("\n🔥 预填充阶段分析:")
    calculate_model_prefill_stats(batch_size=1, seq_len=4096, show_memory=True)
    
    # 计算解码阶段统计信息
    print("\n\n" + "="*80)
    print("\n🎯 解码阶段分析:")
    calculate_model_decode_stats(
        batch_size=128, 
        prefix_length=1024, 
        show_memory=True,
        gpu_mem_bandwidth_gbps=GPU_MEM_BANDWIDTH_GBPS,
        gpu_tflops_bf16=GPU_TFLOPS_BF16,
        gpu_tflops_fp8=GPU_TFLOPS_FP8
    )
