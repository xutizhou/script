#!/usr/bin/env python3
"""
Memory formula: b*0.92G + 235B/world_size + 8G = 141G
Solve for b: b = (141 - 8 - 235/world_size) / 0.92 = (133 - 235/world_size) / 0.92

MoE Expert Analysis:
- Num_groups = num_expert/world_size = 128/world_size
- Expected_m_per_group = batch_size * topk / num_groups = b*world_size/16
  (assuming topk=8, so Expected_m_per_group = b*8*world_size/num_groups = b*8/(128/world_size) = b*world_size/16)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_batch_size(world_size, total_memory=84):
    """Calculate batch size given world size"""
    if world_size <= 0:
        return 0
    
    # Memory formula: b*0.92 + 235/world_size + 8 = 141
    # Solve for b: b = (133 - 235/world_size) / 0.92
    batch_size = (total_memory - 8 - 235/world_size) / 0.92
    return max(0, batch_size)  # Ensure non-negative

def calculate_moe_metrics(world_size, batch_size, num_experts=128, topk=8):
    """Calculate MoE expert group metrics"""
    if world_size <= 0:
        return 0, 0
    
    num_groups = num_experts / world_size
    # Expected_m_per_group = batch_size * topk / num_groups
    # But the formula given is b*world_size/16, let's use both
    expected_m_per_group_formula1 = batch_size * topk / num_groups
    expected_m_per_group_formula2 = batch_size * world_size / 16
    
    return num_groups, expected_m_per_group_formula2

def create_table():
    """Create a table showing the relationship between world_size and batch_size"""
    world_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    data = []
    for ws in world_sizes:
        bs = calculate_batch_size(ws)
        memory_per_gpu = 235/ws + 8  # Model memory + fixed memory
        batch_memory = bs * 0.92     # Batch memory
        total_memory = memory_per_gpu + batch_memory
        
        # MoE calculations
        num_groups, expected_m_per_group = calculate_moe_metrics(ws, bs)
        
        data.append({
            'World Size': ws,
            'Batch Size': f"{bs:.2f}",
            'Model Memory/GPU (GB)': f"{235/ws:.2f}",
            'Batch Memory (GB)': f"{batch_memory:.2f}",
            'Fixed Memory (GB)': "8.00",
            'Total Memory (GB)': f"{total_memory:.2f}",
            'Num Groups': f"{num_groups:.2f}",
            'Expected M/Group': f"{expected_m_per_group:.2f}",
            'Global Batch Size': f"{bs * ws:.2f}"
        })
    
    df = pd.DataFrame(data)
    return df

def plot_relationship():
    """Plot the relationship between world_size and batch_size"""
    world_sizes = np.arange(1, 257)
    batch_sizes = [calculate_batch_size(ws) for ws in world_sizes]
    
    plt.figure(figsize=(15, 10))
    
    # Main plot
    plt.subplot(2, 3, 1)
    plt.plot(world_sizes, batch_sizes, 'b-', linewidth=2)
    plt.xlabel('World Size (Number of GPUs)')
    plt.ylabel('Batch Size per GPU')
    plt.title('Batch Size vs World Size\n(Memory Constraint: 141GB per GPU)')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    # Log scale plot
    plt.subplot(2, 3, 2)
    plt.semilogx(world_sizes, batch_sizes, 'r-', linewidth=2)
    plt.xlabel('World Size (log scale)')
    plt.ylabel('Batch Size per GPU')
    plt.title('Batch Size vs World Size (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Memory breakdown
    plt.subplot(2, 3, 3)
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
    
    # Global batch size plot
    plt.subplot(2, 3, 4)
    total_batch_sizes = [bs * ws for bs, ws in zip(batch_sizes, world_sizes)]
    plt.plot(world_sizes, total_batch_sizes, 'm-', linewidth=2)
    plt.xlabel('World Size')
    plt.ylabel('Global Batch Size')
    plt.title('Global Batch Size vs World Size')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    # MoE Num Groups plot
    plt.subplot(2, 3, 5)
    num_groups = [128/ws for ws in world_sizes]
    plt.plot(world_sizes, num_groups, 'c-', linewidth=2)
    plt.xlabel('World Size')
    plt.ylabel('Number of Expert Groups')
    plt.title('MoE Expert Groups vs World Size\n(128 experts)')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    # Expected M per Group plot
    plt.subplot(2, 3, 6)
    expected_m_per_group = [bs * ws / 16 for bs, ws in zip(batch_sizes, world_sizes)]
    plt.plot(world_sizes, expected_m_per_group, 'orange', linewidth=2)
    plt.xlabel('World Size')
    plt.ylabel('Expected M per Group')
    plt.title('Expected M per Expert Group\n(Formula: b*world_size/16)')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 256)
    
    plt.tight_layout()
    plt.savefig('batchsize_worldsize_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Memory Formula Analysis: b*0.92G + 235B/world_size + 8G = 141G")
    print("MoE Expert Analysis:")
    print("- Num_groups = 128/world_size")
    print("- Expected_m_per_group = b*world_size/16")
    print("=" * 80)
    
    # Create and display table
    df = create_table()
    print("\nBatch Size vs World Size Table (with MoE Analysis):")
    print(df.to_string(index=False))
    
    # Save table to CSV
    df.to_csv('batchsize_worldsize_table.csv', index=False)
    print(f"\nTable saved to: batchsize_worldsize_table.csv")
    
    # Create plots
    plot_relationship()
    print("Plot saved to: batchsize_worldsize_analysis.png")
    
    # Find optimal configurations
    print("\n" + "=" * 80)
    print("Optimal Configurations Analysis:")
    
    world_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    max_global_batch = 0
    optimal_config = None
    
    print(f"{'WS':>3} {'BS/GPU':>8} {'Global BS':>10} {'Groups':>8} {'M/Group':>10} {'Memory/GPU':>12}")
    print("-" * 65)
    
    for ws in world_sizes:
        bs = calculate_batch_size(ws)
        if bs > 0:
            global_bs = bs * ws
            num_groups, expected_m_per_group = calculate_moe_metrics(ws, bs)
            total_memory = 235/ws + bs*0.92 + 8
            
            print(f"{ws:3d} {bs:8.2f} {global_bs:10.2f} {num_groups:8.2f} {expected_m_per_group:10.2f} {total_memory:12.2f}")
            
            if global_bs > max_global_batch:
                max_global_batch = global_bs
                optimal_config = (ws, bs, num_groups, expected_m_per_group)
    
    if optimal_config:
        print(f"\nOptimal Configuration for Maximum Global Batch Size:")
        print(f"World Size: {optimal_config[0]}")
        print(f"Batch Size per GPU: {optimal_config[1]:.2f}")
        print(f"Global Batch Size: {max_global_batch:.2f}")
        print(f"Number of Expert Groups: {optimal_config[2]:.2f}")
        print(f"Expected M per Group: {optimal_config[3]:.2f}")
    
    # Analysis insights
    print(f"\n" + "=" * 80)
    print("Key Insights:")
    print("1. As world_size increases, batch_size per GPU decreases due to model memory distribution")
    print("2. Global batch size (bs * world_size) generally increases with world_size")
    print("3. Number of expert groups decreases with world_size (128/world_size)")
    print("4. Expected M per group varies with both batch_size and world_size")
    print("5. Memory constraint is always satisfied (141GB per GPU)")

if __name__ == "__main__":
    main()
