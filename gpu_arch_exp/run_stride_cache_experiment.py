#!/usr/bin/env python3
"""
Run L1/L2 cache stride experiments with Nsight Compute.

This script compiles the CUDA benchmark (stride_cache_benchmark.cu) and then
executes it under Nsight Compute for a series of stride sizes. Each stride
value corresponds to one experiment where every thread loads 4 bytes from
global memory with the specified inter-thread stride. The script collects
L1/L2 related metrics and stores the Nsight Compute reports on disk so they
can be inspected later with `ncu-ui` or `ncu --import`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CUDA_FILE = SCRIPT_DIR / "stride_cache_benchmark.cu"
BINARY_PATH = SCRIPT_DIR / "stride_cache_benchmark"
REPORT_DIR = SCRIPT_DIR / "ncu_stride_reports"

DEFAULT_STRIDES = list(range(4, 129, 4))


def check_binary_paths(nvcc: str, ncu: str) -> None:
    for tool_name, tool_path in {"nvcc": nvcc, "ncu": ncu}.items():
        if shutil.which(tool_path) is None:
            raise FileNotFoundError(
                f"{tool_name} executable '{tool_path}' not found in PATH. "
                "Please ensure CUDA Toolkit/Nsight Compute is installed."
            )


def compile_cuda_benchmark(nvcc: str, extra_flags: list[str]) -> None:
    cmd = [nvcc, "-O3", "--std=c++17", str(CUDA_FILE), "-o", str(BINARY_PATH)]
    cmd.extend(extra_flags)
    print(f"[compile] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_ncu_for_stride(
    ncu: str,
    stride_bytes: int,
    set_name: str,
) -> Path:
    report_base = REPORT_DIR / f"stride_{stride_bytes:03d}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        ncu,
        "--set",
        set_name,
        "--target-processes",
        "all",
        "--launch-skip",
        "0",
        "--launch-count",
        "1",
        "--export",
        str(report_base),
        str(BINARY_PATH),
        str(stride_bytes),
    ]
    print(f"[ncu] {' '.join(cmd)}")
    completed = subprocess.run(cmd, capture_output=True, text=True)

    log_path = report_base.with_suffix(".log")
    log_path.write_text(
        "STDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr
    )
    if completed.returncode != 0:
        combined_error = (completed.stderr or "") + (completed.stdout or "")
        if "ERR_NVGPUCTRPERM" in combined_error:
            raise RuntimeError(
                "Nsight Compute reported ERR_NVGPUCTRPERM. "
                "GPU performance counters are restricted on this system. "
                "Run the profiler with administrator permissions or follow "
                "the instructions at https://developer.nvidia.com/ERR_NVGPUCTRPERM "
                f"to enable access. See log: {log_path}"
            )
        raise RuntimeError(
            f"Nsight Compute failed for stride {stride_bytes} (exit "
            f"{completed.returncode}). Check log: {log_path}"
        )
    return report_base.with_suffix(".ncu-rep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark stride-based loads and capture Nsight reports."
    )
    parser.add_argument(
        "--nvcc",
        default="nvcc",
        help="Path to nvcc. Defaults to looking it up on PATH.",
    )
    parser.add_argument(
        "--ncu",
        default="ncu",
        help="Path to Nsight Compute CLI executable.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=DEFAULT_STRIDES,
        help="Stride sizes (in bytes) to test. Defaults to 4..128 step 4.",
    )
    parser.add_argument(
        "--ncu-set",
        type=str,
        default="full",
        help="Nsight Compute set to use (default: 'full').",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip nvcc compilation step (assumes binary already exists).",
    )
    parser.add_argument(
        "--extra-nvcc-flags",
        type=str,
        default="",
        help="Additional nvcc flags (space separated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    check_binary_paths(args.nvcc, args.ncu)

    if not args.skip_compile:
        extra_flags = args.extra_nvcc_flags.split()
        compile_cuda_benchmark(args.nvcc, extra_flags)
    elif not BINARY_PATH.exists():
        print(
            "[error] --skip-compile set but binary is missing. "
            "Run without --skip-compile first.",
            file=sys.stderr,
        )
        return 1

    generated_reports = []
    for stride in args.strides:
        if stride <= 0 or stride % 4 != 0:
            print(f"[warn] Skipping invalid stride {stride} (must be positive multiple of 4).")
            continue
        try:
            report_path = run_ncu_for_stride(args.ncu, stride, args.ncu_set)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            return 1
        else:
            generated_reports.append(report_path)

    if not generated_reports:
        print("[warn] No reports were generated.")
        return 1

    print("\nExperiments complete. Reports saved:")
    for report in generated_reports:
        print(f"  - {report}")
    print("\nOpen a report with: ncu-ui --import <report_path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

