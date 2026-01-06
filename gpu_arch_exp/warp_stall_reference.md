# NCU Warp State Statistics 参考指南

## Warp Stall 原因分类

### 1. Memory Related Stalls (内存相关)

| Stall Type | 中文名 | 原因 | 优化建议 |
|------------|--------|------|----------|
| **long_scoreboard** | 长记分板等待 | 等待L1TEX (全局/本地/纹理内存)操作完成 | 增加ILP、使用prefetch、优化内存访问模式 |
| **short_scoreboard** | 短记分板等待 | 等待MIO (共享内存/L1等)操作完成 | 减少bank冲突、优化shared memory访问 |
| **lg_throttle** | L/G throttle | 本地/全局内存请求队列满 | 减少outstanding内存请求 |
| **mio_throttle** | MIO throttle | MIO队列满 (shared memory相关) | 减少shared memory访问密度 |
| **tex_throttle** | 纹理throttle | 纹理单元队列满 | 减少纹理访问频率 |
| **drain** | 排空等待 | 等待之前的内存操作完成 | 通常发生在kernel末尾 |
| **membar** | 内存屏障 | 等待memory fence/barrier | 减少不必要的__threadfence |

### 2. Synchronization Stalls (同步相关)

| Stall Type | 中文名 | 原因 | 优化建议 |
|------------|--------|------|----------|
| **barrier** | 屏障同步 | 等待__syncthreads() | 减少同步点、确保warp均匀到达 |
| **wait** | 等待 | 等待其他依赖(如atomics) | 减少atomic操作、使用warp-level primitives |

### 3. Instruction/Execution Stalls (指令/执行相关)

| Stall Type | 中文名 | 原因 | 优化建议 |
|------------|--------|------|----------|
| **math_pipe_throttle** | 数学管道throttle | 数学单元(FP/INT)满载 | 增加内存操作交替、检查算法 |
| **dispatch_stall** | 调度stall | 指令调度器忙 | 通常表示高效利用 |
| **no_instruction** | 无指令 | 没有可发射的指令 | 增加ILP |
| **not_selected** | 未被选中 | warp就绪但调度器选了其他warp | 正常调度行为 |
| **selected** | 被选中 | warp被选中发射指令 | 这是"活跃"状态 |
| **imc_miss** | 指令缓存miss | 指令缓存未命中 | 减少代码大小、改善局部性 |
| **branch_resolving** | 分支解析 | 等待分支结果 | 减少分支发散 |

### 4. Other Stalls (其他)

| Stall Type | 中文名 | 原因 | 优化建议 |
|------------|--------|------|----------|
| **sleeping** | 休眠 | warp显式休眠 | __nanosleep等 |
| **misc** | 其他 | 未分类的stall | 需要具体分析 |

## 典型Kernel的Stall模式

### Memory-Bound Kernels (如Vector Add, Reduction)
```
主要stall: long_scoreboard (50-80%)
次要stall: not_selected, lg_throttle
```

### Compute-Bound Kernels (如矩阵乘法核心)
```
主要stall: not_selected (warp充足时)
或: math_pipe_throttle (计算密集时)
```

### Shared Memory Intensive Kernels
```
主要stall: short_scoreboard (有bank冲突时)
次要stall: barrier (频繁同步时)
```

### Random Access Patterns
```
主要stall: long_scoreboard (80%+)
伴随: lg_throttle
原因: 缓存miss率高
```

### Atomic-Heavy Kernels
```
主要stall: wait (原子操作等待)
或: lg_throttle
```

## 优化决策树

```
如果 long_scoreboard > 50%:
    → 内存带宽受限
    → 优化: 合并访问、减少访问量、使用shared memory缓存
    
如果 short_scoreboard > 30%:
    → 共享内存bank冲突
    → 优化: 重新设计shared memory布局、添加padding
    
如果 barrier > 30%:
    → 同步开销大
    → 优化: 减少__syncthreads调用、warp-level primitives
    
如果 math_pipe_throttle > 30%:
    → 计算密集
    → 可能是好事(高效利用)，或需要内存操作隐藏延迟
    
如果 not_selected > 50%:
    → 有足够的warp隐藏延迟
    → 可能需要更多workload或减少warp数
    
如果 lg_throttle > 20%:
    → 内存请求队列满
    → 减少outstanding请求、增加计算
    
如果 wait > 20%:
    → 原子操作或其他依赖
    → 减少atomics、使用reduction等替代
```

## NCU 命令参考

### 快速收集warp stall统计
```bash
ncu --metrics smsp__warp_issue_stalled_long_scoreboard_per_issue_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_issue_active.pct,\
smsp__warp_issue_stalled_barrier_per_issue_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_issue_active.pct,\
smsp__warp_issue_stalled_not_selected_per_issue_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_issue_active.pct,\
smsp__warp_issue_stalled_wait_per_issue_active.pct \
./your_program
```

### 完整profiling (包含所有信息)
```bash
ncu --set full -o report ./your_program
ncu-ui report.ncu-rep  # 图形界面查看
```

### 导出为CSV分析
```bash
ncu --import report.ncu-rep --csv > report.csv
```

### 特定kernel profiling
```bash
ncu --kernel-name "your_kernel" --launch-skip 3 --launch-count 1 ./your_program
```

## Warp State 计算

所有warp state的百分比之和 = 100%:
- **Stalled states**: 各种stall原因
- **Active state**: `selected` (正在发射指令)

理想情况:
- `selected` 尽可能高 → 高效执行
- 某一个stall dominate → 明确的优化目标
- 多个stall分散 → 需要综合优化

## 参考资料

1. NVIDIA Nsight Compute Documentation
2. CUDA C++ Programming Guide - Warp Level Primitives
3. CUDA Optimization Best Practices

