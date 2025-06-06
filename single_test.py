import torch
import psutil
import gc

def print_memory_info():
    """打印当前内存使用情况"""
    # GPU 内存
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        print(f"CUDA Memory - Allocated: {allocated / 1024**2:.2f} MB")
        print(f"CUDA Memory - Reserved: {reserved / 1024**2:.2f} MB") 
        print(f"CUDA Memory - Max Allocated: {max_allocated / 1024**2:.2f} MB")
    
    # CPU 内存
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"CPU Memory - RSS: {memory_info.rss / 1024**2:.2f} MB")
    print(f"CPU Memory - VMS: {memory_info.vms / 1024**2:.2f} MB")
    print("-" * 50)

print("Initial memory state:")
print_memory_info()

print("Creating tensors x1 and x2...")
x1 = torch.empty(
    (
        1024,
        1024,
    ),
    device="cuda",
    dtype=torch.float16,
)

x2 = torch.empty(
    (
        1024,
        1024,
    ),
    device="cuda",
    dtype=torch.float16,
)

print("After creating tensors:")
print_memory_info()

print("Creating tuple x3...")
x3 = (x1, x2)

print("After creating tuple:")
print_memory_info()

print("Deleting x1...")
del x3[0]
print("After deleting x1:")
print_memory_info()

print("Deleting x2...")
del x3[1]
print("After deleting x2:")
print_memory_info()

print("Deleting x3...")
del x3
print("After deleting x3:")
print_memory_info()

print("Running garbage collection...")
gc.collect()
print("After garbage collection:")
print_memory_info()

print("Clearing CUDA cache...")
torch.cuda.empty_cache()
print("After clearing CUDA cache:")
print_memory_info()

