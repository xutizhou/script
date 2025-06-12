#!/usr/bin/env python3
"""
Memory formula: b*0.92G + 235B/world_size + 8G = 141G
Solve for b: b = (141 - 8 - 235/world_size) / 0.92 = (133 - 235/world_size) / 0.92
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_batch_size(world_size):
    """Calculate batch size given world size"""
    if world_size <= 0:
        return 0
    
    # Memory formula: b*0.92 + 235/world_size + 8 = 141
    # Solve for b: b = (133 - 235/world_size) / 0.92
    batch_size = (133 - 235/world_size) / 0.92
    return max(0, batch_size)  # Ensure non-negative

def create_table():
    """Create a table showing the relationship between world_size and batch_size"""
    world_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    data = []
    for ws in world_sizes:
        bs = calculate_batch_size(ws)
        memory_per_gpu = 235/ws + 8  # Model memory + fixed memory
        batch_memory = bs * 0.92     # Batch memory
        total_memory = memory_per_gpu + batch_memory
        
        data.append({
            'World Size': ws,
            'Batch Size': f"{bs:.2f}",
            'Model Memory/GPU (GB)': f"{235/ws:.2f}",
            'Batch Memory (GB)': f"{batch_memory:.2f}",
            'Fixed Memory (GB)': "8.00",
            'Total Memory (GB)': f"{total_memory:.2f}"
        })
    
    df = pd.DataFrame(data)
    return df

def plot_relationship():
    """Plot the relationship between world_size and batch_size"""
    world_sizes = np.arange(1, 257)
    batch_sizes = [calculate_batch_size(ws) for ws in world_sizes]
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(world_sizes, batch_sizes, 'b-', linewidth=2)
    plt.xlabel('World Size (Number of GPUs)')
    plt.ylabel('Batch Size')
    plt.title('Batch Size vs World Size\n(Memory Constraint: 141GB per GPU)')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    # Log scale plot
    plt.subplot(2, 2, 2)
    plt.semilogx(world_sizes, batch_sizes, 'r-', linewidth=2)
    plt.xlabel('World Size (log scale)')
    plt.ylabel('Batch Size')
    plt.title('Batch Size vs World Size (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Memory breakdown
    plt.subplot(2, 2, 3)
    model_memory = [235/ws for ws in world_sizes]
    batch_memory = [bs * 0.92 for bs in batch_sizes]
    fixed_memory = [8] * len(world_sizes)
    
    plt.plot(world_sizes, model_memory, 'g-', label='Model Memory', linewidth=2)
    plt.plot(world_sizes, batch_memory, 'b-', label='Batch Memory', linewidth=2)
    plt.plot(world_sizes, fixed_memory, 'r-', label='Fixed Memory', linewidth=2)
    plt.xlabel('World Size')
    plt.ylabel('Memory (GB)')
    plt.title('Memory Breakdown per GPU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    # Efficiency plot (batch_size * world_size = total batch size)
    plt.subplot(2, 2, 4)
    total_batch_sizes = [bs * ws for bs, ws in zip(batch_sizes, world_sizes)]
    plt.plot(world_sizes, total_batch_sizes, 'm-', linewidth=2)
    plt.xlabel('World Size')
    plt.ylabel('Global Batch Size')
    plt.title('Global Batch Size vs World Size')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    plt.tight_layout()
    plt.savefig('batchsize_worldsize_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Memory Formula Analysis: b*0.92G + 235B/world_size + 8G = 141G")
    print("=" * 60)
    
    # Create and display table
    df = create_table()
    print("\nBatch Size vs World Size Table:")
    print(df.to_string(index=False))
    
    # Save table to CSV
    df.to_csv('batchsize_worldsize_table.csv', index=False)
    print(f"\nTable saved to: batchsize_worldsize_table.csv")
    
    # Create plots
    plot_relationship()
    print("Plot saved to: batchsize_worldsize_analysis.png")
    
    # Find optimal configurations
    print("\n" + "=" * 60)
    print("Optimal Configurations:")
    
    world_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    max_global_batch = 0
    optimal_config = None
    
    for ws in world_sizes:
        bs = calculate_batch_size(ws)
        if bs > 0:
            global_bs = bs * ws
            print(f"World Size: {ws:3d}, Batch Size per GPU: {bs:6.2f}, Global Batch Size: {global_bs:8.2f}")
            
            if global_bs > max_global_batch:
                max_global_batch = global_bs
                optimal_config = (ws, bs)
    
    if optimal_config:
        print(f"\nOptimal Configuration for Maximum Global Batch Size:")
        print(f"World Size: {optimal_config[0]}, Batch Size per GPU: {optimal_config[1]:.2f}")
        print(f"Global Batch Size: {max_global_batch:.2f}")

if __name__ == "__main__":
    main()
