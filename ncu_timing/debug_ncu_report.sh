#!/bin/bash
# Debug NCU Report - 查看NCU报告的原始格式
#
# 用法: ./debug_ncu_report.sh <ncu-rep-file>

if [ -z "$1" ]; then
    echo "Usage: $0 <ncu-rep-file>"
    echo ""
    echo "Example:"
    echo "  $0 ./quick_results_*/short_basic_base.ncu-rep"
    exit 1
fi

REP_FILE="$1"

if [ ! -f "$REP_FILE" ]; then
    echo "Error: File not found: $REP_FILE"
    exit 1
fi

echo "========================================"
echo "Debug NCU Report: $(basename $REP_FILE)"
echo "========================================"
echo ""

echo "--- Method 1: --page raw ---"
ncu --import "$REP_FILE" --page raw 2>&1 | head -100
echo ""

echo "--- Method 2: --page details ---"
ncu --import "$REP_FILE" --page details 2>&1 | head -100
echo ""

echo "--- Method 3: --csv (first 20 lines) ---"
ncu --import "$REP_FILE" --csv 2>&1 | head -20
echo ""

echo "--- Method 4: --print-summary per-kernel ---"
ncu --import "$REP_FILE" --print-summary per-kernel 2>&1 | head -50
echo ""

echo "--- Method 5: Query specific metrics ---"
ncu --import "$REP_FILE" --query-metrics 'regex:.*duration.*|regex:.*time.*|regex:.*cycle.*' 2>&1 | head -50
echo ""

echo "--- Method 6: List all available metrics ---"
echo "(showing first 30 lines only)"
ncu --import "$REP_FILE" --query-metrics all 2>&1 | head -30
echo ""

echo "========================================"
echo "Debug complete"
echo "========================================"

