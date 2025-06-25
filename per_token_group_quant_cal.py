# void per_token_group_quant_8bit_kernel
block = 256
grid = 28672
groups_per_block = block / 16
num_groups = grid * groups_per_block
input_size = num_groups * 128
print(f"per_token_group_quant_8bit_kernel: {input_size/7168}") #7168 for deepseek-r1 and 4096 for qwen3

# _per_token_group_quant_fp8
grid = 5120
input_size = grid * 128
print(f"_per_token_group_quant_fp8: {input_size/5120}") #5120 for qwen3-32B