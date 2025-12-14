# NCU Timing Accuracy Experiments

验证NCU (NVIDIA Nsight Compute) 计时准确性的实验工具集。

## 实验目标

1. **验证kernel时长对计时准确性的影响**
   - 短时间kernel (~10us): kernel launch开销与kernel时间同量级
   - 长时间kernel (~10ms): kernel时间远大于launch开销

2. **验证NCU set对计时的影响**
   - `basic`: 最小set，约190个metrics
   - `detailed`: 中等set，约560个metrics
   - `full`: 完整set，约5900个metrics
   - `roofline`: roofline分析，约5260个metrics

3. **验证clock-control对计时的影响**
   - `base`: 使用base频率
   - `none`/`boost`: 使用boost频率（默认）

4. **分析memory restore行为**
   - 每次NCU replay restore的memory大小
   - 影响restore大小的因素

## 文件说明

- `timing_kernel.cu` - CUDA kernel源代码，通过参数控制读写量来控制运行时间
- `run_ncu_experiments.sh` - 实验运行脚本，自动运行所有配置组合
- `parse_ncu_results.py` - 结果解析脚本，提取kernel duration等信息
- `analyze_memory_restore.py` - Memory restore分析脚本

## 使用方法

### 1. 在SLURM容器中运行

```bash
# 进入容器后
cd /path/to/ut/ncu_timing

# 编译kernel
nvcc -O3 -arch=sm_90 timing_kernel.cu -o timing_kernel

# 测试kernel
./timing_kernel short   # 短时间kernel
./timing_kernel long    # 长时间kernel

# 运行完整实验
chmod +x run_ncu_experiments.sh
./run_ncu_experiments.sh ./results_$(date +%Y%m%d)

# 或者手动运行单个NCU实验
ncu --set basic --clock-control base -o short_basic_base ./timing_kernel short
ncu --set full --clock-control none -o long_full_boost ./timing_kernel long
```

### 2. 解析结果

```bash
# 解析所有结果
python parse_ncu_results.py ./results_dir

# 导出CSV
python parse_ncu_results.py ./results_dir --export-csv

# 分析memory restore
python analyze_memory_restore.py ./results_dir
python analyze_memory_restore.py ./results_dir --verbose

# 只查看因素分析
python analyze_memory_restore.py --factors
```

### 3. 手动NCU命令示例

```bash
# 基础profiling（最快）
ncu --set basic ./timing_kernel short

# 详细profiling
ncu --set detailed ./timing_kernel short

# 完整profiling（最慢）
ncu --set full ./timing_kernel short

# 带频率控制
ncu --set basic --clock-control base ./timing_kernel short
ncu --set basic --clock-control none ./timing_kernel short  # boost

# 保存报告
ncu --set basic -o report_name ./timing_kernel short

# 带详细日志
ncu --set basic --log-file ncu.log --print-details all ./timing_kernel short

# 查看特定section
ncu --section SpeedOfLight ./timing_kernel short
ncu --section MemoryWorkloadAnalysis ./timing_kernel short
```

## 实验配置说明

### NCU Sets

| Set | Metrics数量 | 用途 |
|-----|------------|------|
| basic | ~190 | 快速概览，基本性能指标 |
| detailed | ~560 | 详细分析，包含内存分析 |
| full | ~5900 | 完整分析，包含所有metrics |
| nvlink | ~52 | NVLink分析 |
| pmsampling | ~186 | PM Sampling |
| roofline | ~5260 | Roofline分析 |

### Clock Control

| 选项 | 说明 |
|-----|------|
| base | 锁定到基础频率，时间更稳定但不代表实际性能 |
| none | 默认boost模式，代表实际运行频率 |

### Kernel参数

```bash
# 短时间kernel (~10us)
./timing_kernel short
# 配置: 4096 elements, 1 iteration, 1 block

# 长时间kernel (~10ms)
./timing_kernel long
# 配置: 16M elements, 100 iterations, 128 blocks

# 自定义
./timing_kernel custom --size 1000000 --iterations 10 --blocks 64 --threads 256
```

## 预期结果

### 时间测量差异

1. **短kernel (us级别)**
   - NCU时间可能比实际运行偏大（launch开销影响）
   - 不同set之间差异较大

2. **长kernel (ms级别)**
   - NCU时间与实际运行更接近
   - 不同set之间差异较小

### Clock Control影响

- `base`模式: 时间更长，但replay之间更一致
- `boost`模式: 时间更短，但可能有波动

### Memory Restore

影响因素:
1. Kernel的内存footprint大小
2. NCU set（更多metrics = 更多replay = 更多restore）
3. 内存修改模式（写入越多，restore越大）

## 故障排除

### NCU命令找不到
```bash
# 检查CUDA toolkit路径
which ncu
# 或者使用完整路径
/usr/local/cuda/bin/ncu ...
```

### 权限问题
```bash
# 可能需要CAP_SYS_ADMIN
sudo ncu ...
# 或者在容器中以root运行
```

### 报告太大
```bash
# 使用更小的set
ncu --set basic ...
# 或只采集特定section
ncu --section SpeedOfLight ...
```

