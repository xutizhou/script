from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

import torch

x = torch.randn(0, 7168, dtype=torch.bfloat16,device="cuda")

x_q, x_s = sglang_per_token_group_quant_fp8(x, 128)

print(x_q.shape, x_s.shape)
print(x_q.dtype, x_s.dtype)


x_new = (
    torch.empty_like(
        x,
        device=x.device,
        dtype=torch.float8_e4m3fn,
    ),
    torch.empty(
        x.shape[:-1] + (x.shape[-1] // 128,),
        device=x.device,
        dtype=torch.float32,
    ),
)

print(x_q.shape, x_s.shape)
print(x_q.dtype, x_s.dtype)