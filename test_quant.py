from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

import torch

# 1. Warmup
for _ in range(3):
    x = torch.randn(8192, 7168, dtype=torch.bfloat16, device="cuda")
    sglang_per_token_group_quant_fp8(x, 128)

# 2. Measurement
x = torch.randn(8192, 7168, dtype=torch.bfloat16, device="cuda")
group_size = 128

# Create CUDA events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
x_q, x_s = sglang_per_token_group_quant_fp8(x, group_size)
end_event.record()

# Wait for the events to complete
torch.cuda.synchronize()

# Calculate execution time in seconds
elapsed_time_ms = start_event.elapsed_time(end_event)
elapsed_time_s = elapsed_time_ms / 1000

# 3. Calculate memory access and bandwidth
# The kernel reads input x twice: once to find the max absolute value,
# and a second time to apply the quantization.
input_bytes = x.numel() * x.element_size()
output_q_bytes = x_q.numel() * x_q.element_size()
output_s_bytes = x_s.numel() * x_s.element_size()

total_bytes = input_bytes * 2 + output_q_bytes + output_s_bytes
bandwidth_gbps = total_bytes / elapsed_time_s / 1e9

print(f"Input shape: {x.shape}, dtype: {x.dtype}")
print(f"Quantized shape: {x_q.shape}, dtype: {x_q.dtype}")
print(f"Scale shape: {x_s.shape}, dtype: {x_s.dtype}")
print("-" * 20)
print(f"Elapsed time: {elapsed_time_ms:.4f} ms")
print(f"Total memory access: {total_bytes / 1e6:.2f} MB")
print(f"Effective Bandwidth: {bandwidth_gbps:.2f} GB/s")

