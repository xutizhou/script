block = 256
grid = 28672
groups_per_block = block / 16
num_groups = grid * groups_per_block
input_size = num_groups * 128
print(input_size/7168)