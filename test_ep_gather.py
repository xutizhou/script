down_output = torch.randn((N_, D), device="cuda").to(torch.bfloat16)
gather_out = torch.empty((N, D), device="cuda").to(torch.bfloat16)

ep_gather(down_output, recv_topk, recv_topk_weight, output_index, gather_out)

print(output_tensor[0])
print(gather_out[0])
print("passed examine")