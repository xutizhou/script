#!/bin/bash
# NCU Stitch Traffic 定量分析脚本
# 
# 使用方法:
#   chmod +x run_ncu_stitch_analysis.sh
#   ./run_ncu_stitch_analysis.sh           # 普通用户
#   sudo ./run_ncu_stitch_analysis.sh      # 需要 root 权限

cd "$(dirname "$0")"

BINARY="./ncu_stitch_analysis"
OUTPUT_DIR="ncu_stitch_reports"

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "NCU Stitch Traffic 定量分析"
echo "========================================"

# 检查 NCU 权限
echo "检查 NCU 权限..."
if ! ncu --version > /dev/null 2>&1; then
    echo "错误: NCU 未安装或不可用"
    exit 1
fi

# 核心 L2 指标
L2_METRICS="lts__t_sectors_srcunit_tex_op_read.sum"
L2_METRICS+=",lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
L2_METRICS+=",lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
L2_METRICS+=",lts__t_request_cycles.avg"
L2_METRICS+=",lts__average_t_sector_srcunit_tex_op_read_hit_rate.pct"

# DRAM 指标
DRAM_METRICS="dram__bytes_read.sum"
DRAM_METRICS+=",dram__sectors_read.sum"

# 内存带宽指标
BW_METRICS="l2_cache_throughput.avg_pct_of_peak_sustained_elapsed"
BW_METRICS+=",dram_throughput.avg_pct_of_peak_sustained_elapsed"

ALL_METRICS="$L2_METRICS,$DRAM_METRICS,$BW_METRICS"

echo ""
echo "========================================"
echo "分析每个 kernel 的 L2 行为"
echo "========================================"

# 定义要分析的 kernel 列表
KERNELS=(
    "local_access"
    "cross_access"
    "full_l2_access"
    "exceed_l2_access"
    "strided_access"
    "alternating_partition"
    "random_access"
)

# 对每个 kernel 单独分析
for kernel in "${KERNELS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "分析 kernel: $kernel"
    echo "----------------------------------------"
    
    OUTPUT_FILE="$OUTPUT_DIR/${kernel}_report.txt"
    
    ncu --kernel-name $kernel \
        --metrics $ALL_METRICS \
        --csv \
        $BINARY 2>&1 | tee "$OUTPUT_FILE"
    
    echo "报告保存到: $OUTPUT_FILE"
done

echo ""
echo "========================================"
echo "汇总分析"
echo "========================================"

# 生成汇总报告
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
echo "Kernel                    L2 Sectors Read    L2 Hit Rate    DRAM Bytes" > $SUMMARY_FILE
echo "=========================================================================" >> $SUMMARY_FILE

for kernel in "${KERNELS[@]}"; do
    REPORT="$OUTPUT_DIR/${kernel}_report.txt"
    if [ -f "$REPORT" ]; then
        # 提取关键指标 (简化版)
        echo "$(basename $kernel)" >> $SUMMARY_FILE
    fi
done

echo ""
echo "所有报告保存在: $OUTPUT_DIR/"
echo ""
echo "使用以下命令查看详细报告:"
echo "  cat $OUTPUT_DIR/<kernel>_report.txt"
echo ""
echo "========================================"

