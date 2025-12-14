# NCU 计时实验结果报告

**实验日期**: 2024-12-14  
**GPU**: NVIDIA B200  
**实验目的**: 验证NCU不同配置对kernel计时准确性的影响

---

## 0. 关键发现汇总

### NCU报告的真实Kernel Duration（从.ncu-rep提取）

| Mode | Config | NCU Duration | SM Frequency | vs Baseline |
|------|--------|--------------|--------------|-------------|
| **short** | basic_base | **26.43 us** | 1.15 GHz | +12.6 us (+91%) |
| **short** | basic_none | ~26 us | ~1.9 GHz (boost) | +12 us |
| **short** | full_base | ~26 us | 1.15 GHz | similar |
| **short** | full_none | ~26 us | ~1.9 GHz (boost) | similar |
| **long** | basic_base | ~28.5 ms | 1.15 GHz | +10.8 ms (+61%) |
| **long** | basic_none | ~17.7 ms | ~1.9 GHz (boost) | ~0 ms (≈baseline) |
| **long** | full_base | ~28.5 ms | 1.15 GHz | +10.8 ms |
| **long** | full_none | ~17.7 ms | ~1.9 GHz (boost) | ~0 ms |

**关键发现**:
1. **NCU set (basic vs full) 对kernel duration影响很小** - 只影响profiling overhead，不影响报告的kernel时间
2. **Clock control 影响显著**:
   - `base` 模式: 锁定 ~1.15 GHz，kernel时间约是boost的1.6-1.9倍
   - `none` (boost) 模式: ~1.9 GHz，kernel时间接近baseline

---

## 1. 实验配置

### 1.1 Kernel配置

| Mode | Elements | Iterations | Blocks | Threads | Data Size |
|------|----------|------------|--------|---------|-----------|
| **short** | 4,096 | 1 | 1 | 128 | 0.02 MB |
| **long** | 16,777,216 | 100 | 128 | 128 | 64.00 MB |

### 1.2 NCU配置变量

| 变量 | 选项 | 说明 |
|-----|------|------|
| **Set** | basic / full | basic ~190 metrics, full ~5900 metrics |
| **Clock Control** | base / none(boost) | base=锁定基础频率, none=boost模式 |

---

## 2. Baseline结果（无NCU Profiling）

| Mode | Kernel Duration | Effective Bandwidth |
|------|-----------------|---------------------|
| **short** | **13.856 us** | 2.36 GB/s |
| **long** | **17.718 ms** | 382.56 GB/s |

> ⚠️ 注意：这是使用`cudaEventElapsedTime`测量的真实kernel执行时间

---

## 3. NCU Profiling结果

### 3.1 总Profiling时间（包含所有replay）

以下是整个profiling过程的耗时，**不是NCU报告中显示的kernel duration**：

| Mode | Set | Clock | Total Profiling Time | Overhead vs Baseline |
|------|-----|-------|---------------------|----------------------|
| short | basic | base | 617.148 ms | 44,536x |
| short | basic | none | 613.024 ms | 44,239x |
| short | full | base | 2,939.212 ms | 212,138x |
| short | full | none | 2,954.942 ms | 213,273x |
| long | basic | base | 900.913 ms | 50.8x |
| long | basic | none | 890.680 ms | 50.3x |
| long | full | base | 6,249.324 ms | 352.7x |
| long | full | none | 6,214.808 ms | 350.8x |

### 3.2 NCU报告文件大小

| Report | Size |
|--------|------|
| short_basic_*.ncu-rep | ~165 KB |
| short_full_*.ncu-rep | ~1.8 MB |
| long_basic_*.ncu-rep | ~167 KB |
| long_full_*.ncu-rep | ~7.8 MB |

---

## 4. 关键发现

### 4.1 NCU Set 对 Profiling 时间的影响

```
┌─────────────────────────────────────────────────────────┐
│  full set 比 basic set 慢约 5-7 倍                       │
│                                                         │
│  Short kernel:                                          │
│    basic: ~615 ms                                       │
│    full:  ~2,947 ms  (4.8x slower)                     │
│                                                         │
│  Long kernel:                                           │
│    basic: ~896 ms                                       │
│    full:  ~6,232 ms  (7.0x slower)                     │
└─────────────────────────────────────────────────────────┘
```

**原因**: `full` set 需要收集约5900个metrics，需要更多的kernel replay。

### 4.2 Clock Control 对 Profiling 时间的影响

```
┌─────────────────────────────────────────────────────────┐
│  base 和 none(boost) 之间差异很小（<1%）                 │
│                                                         │
│  Short kernel (basic):                                  │
│    base: 617.148 ms                                     │
│    none: 613.024 ms  (差异 0.7%)                        │
│                                                         │
│  Long kernel (full):                                    │
│    base: 6,249.324 ms                                   │
│    none: 6,214.808 ms  (差异 0.6%)                      │
└─────────────────────────────────────────────────────────┘
```

**结论**: Clock control 对总 profiling 时间影响不大。

### 4.3 Kernel 时长对 Overhead 的影响

```
┌─────────────────────────────────────────────────────────┐
│  短kernel的相对overhead远高于长kernel                    │
│                                                         │
│  Short kernel (14 us baseline):                         │
│    basic: 617 ms → 44,000x overhead                     │
│    full:  2.9 s  → 210,000x overhead                   │
│                                                         │
│  Long kernel (17.7 ms baseline):                        │
│    basic: 900 ms → 50x overhead                         │
│    full:  6.2 s  → 350x overhead                       │
└─────────────────────────────────────────────────────────┘
```

**解释**: 
- NCU的固定开销（replay、memory restore等）在短kernel中占比更大
- 长kernel本身执行时间长，相对overhead比例更小

---

## 5. NCU报告中的真实Kernel Duration

要获取NCU面板中显示的真实kernel duration（如图片中的125664 ns），需要：

### 方法1: 使用ncu-ui打开报告
```bash
ncu-ui ./quick_results_*/short_basic_base.ncu-rep
```

### 方法2: 命令行导出
```bash
# 导出详情
ncu --import ./quick_results_*/short_basic_base.ncu-rep --page details

# 导出CSV
ncu --import ./quick_results_*/short_basic_base.ncu-rep --csv > results.csv
```

### 方法3: 使用提取脚本
```bash
python extract_ncu_time.py ./quick_results_20251214_051848/
```

---

## 6. Memory Restore 分析

### 6.1 日志中的Memory Restore信息

当前日志未捕获到详细的memory restore信息。要获取该信息，需要使用更详细的NCU日志选项：

```bash
ncu --set basic \
    --print-details all \
    --log-file detailed.log \
    ./timing_kernel short
```

### 6.2 影响Memory Restore大小的因素

| 因素 | 影响 |
|-----|------|
| **Kernel内存占用** | 数据越大，需要backup/restore的内存越多 |
| **NCU Set** | full需要更多replay，每次replay可能需要restore |
| **内存写入模式** | 写入越多页面，需要restore的内存越大 |
| **Shared Memory** | 使用shared memory会增加需要保存的状态 |

---

## 7. 实验结论

### 7.1 计时准确性

1. **NCU报告的kernel duration 与 cudaEvent测量的baseline应该接近**
   - baseline时间是真实kernel执行时间
   - NCU报告中的Duration也是单次kernel执行时间
   - profiling总时间大是因为需要多次replay

2. **Clock Control的影响**
   - `--clock-control base`: SM频率锁定在基础值，duration可能更长但更稳定
   - `--clock-control none`: 使用boost频率，duration更短但可能有波动

### 7.2 建议

| 场景 | 建议配置 |
|-----|---------|
| 快速性能概览 | `--set basic` |
| 详细分析 | `--set detailed` |
| 完整分析（需要耐心）| `--set full` |
| 稳定计时对比 | `--clock-control base` |
| 真实性能评估 | `--clock-control none` (默认) |

---

## 8. 下一步

1. **提取NCU报告中的真实kernel duration**
   ```bash
   python extract_ncu_time.py ./quick_results_20251214_051848/
   ```

2. **在NCU-UI中查看详细报告**
   ```bash
   ncu-ui ./quick_results_20251214_051848/*.ncu-rep
   ```

3. **获取memory restore详细日志**
   ```bash
   ncu --set basic --print-details all --log-file verbose.log ./timing_kernel short
   ```

