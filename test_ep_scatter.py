import torch
from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter, ep_gather
import time

N, D, D_ = 235234*3, 7168, 56
E = 9
N_ = 304000*3
recv_x = torch.randn((N, D), device="cuda").to(torch.float8_e4m3fn)
recv_x_scale = torch.randn((N, D_), device="cuda").to(torch.float)
num_recv_tokens_per_expert = 3*torch.tensor([0, 107904, 178304, 240000, 273536, 284288, 292736, 296832, 301440, 304000], device="cuda").to(torch.int32)
num_recv_tokens_per_expert[1:] = num_recv_tokens_per_expert[1:] - num_recv_tokens_per_expert[:-1]
recv_topk = torch.empty((N, E), device="cuda").to(torch.int64)
recv_topk.fill_(-1)
recv_topk_weight = torch.empty((N, E), device="cuda").to(torch.float)
for i in range(E):
    num = num_recv_tokens_per_expert[i + 1].item()
    recv_topk[-num:, i] = i
expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)
output_tensor = torch.zeros((N_, D), device="cuda").to(torch.float8_e4m3fn)
output_tensor_scale = torch.empty((N_, D_), device="cuda").to(torch.float)
m_indices = torch.empty(N_, device="cuda").to(torch.int32)
output_index = torch.randn((N, E), device="cuda").to(torch.int64)

ep_scatter(
    recv_x,
    recv_x_scale,
    recv_topk,
    num_recv_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_scale,
    m_indices,
    output_index,
)

down_output = torch.randn((N_, D), device="cuda").to(torch.bfloat16)
gather_out = torch.empty((N, D), device="cuda").to(torch.bfloat16)

ep_gather(down_output, recv_topk, recv_topk_weight, output_index, gather_out)

print(output_tensor[0])
print(gather_out[0])
print("passed examine")