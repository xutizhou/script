import numpy as np
import torch
import os
import time
import torch.distributed as dist
from utils import init_dist,bench

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
torch.set_printoptions(threshold=np.inf)

def build_dispatch_map(rank_matrix):
    """ 生成每个rank对应的token索引 """
    dispatch_map = defaultdict(list)
    
    for token_idx in range(rank_matrix.size(0)):
        # 去重并排序目标rank
        target_ranks = torch.unique(rank_matrix[token_idx]).tolist()

        for rank in target_ranks:
            if rank == -1:
                continue
            dispatch_map[rank].append(token_idx)

    return dispatch_map

def prepare_multi_dispatch(input_tensor, dispatch_map, num_ranks):
    """ 生成各rank接收的数据分片 """
    partitioned_inputs = []
    split_counts = []

    for rank in range(num_ranks):
        token_indices = dispatch_map.get(rank, [])
        
        # 记录分片大小
        split_counts.append(len(token_indices))
        
        # 提取对应token数据
        if len(token_indices) > 0:
            partitioned = input_tensor[token_indices]  # shape [k, hidden_dim]
        else:
            partitioned = torch.empty((0, input_tensor.size(1)), 
                                    dtype=input_tensor.dtype)
        partitioned_inputs.append(partitioned)
        
    split_counts = torch.tensor(split_counts,device='cuda')
    return split_counts, partitioned_inputs


def test_nccl_dispatch(input_tensor_list,output_tensor):
   
    dist.all_to_all(output_tensor, input_tensor_list)


def nccl_dispatch(input_tensor_list,input_split,output_split,num_ranks,rank,hidden_size):
   

    # 执行 all_to_all 分发数据

    # 准备接收其他进程的块大小
    dist.all_to_all_single(output_split, input_split) # latency? bw240->200
    
    # #通过各进程块大小计算预留buffer
    output_tensor_list = [torch.zeros((size,hidden_size), device='cuda') for size in output_split]# 300->240  how to optimize?nsys
    dist.all_to_all(output_tensor_list, input_tensor_list)  #300+
    # print(f'Rank {rank} after dispatch ,output_split_pt = {output_tensor_list}') 
    # print()

    return output_tensor_list,0

def nccl_combine(split_tensors, input_split , num_ranks,rank, hidden_size,dim=0):

    
    # 准备接收返回的块
    output_tensor = [torch.zeros(size,hidden_size).cuda() for size in input_split]
    
    # 执行 all_to_all 收集数据
    dist.all_to_all(output_tensor, split_tensors)
    
    dist.barrier()
    print(f'Rank {rank} after combine ,output_split_pt = {output_tensor}') 
    print()
    # # 拼接成原始形状
    combined = torch.cat(output_tensor, dim=dim)
    return combined


def test_loop(local_rank: int, num_local_ranks: int):
    # 初始化分布式环境
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)


    # -----------------pure nccl alltoall bench

    # num_tokens, hidden = 4096, 7168
    # x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    # input = [x for _ in range(num_ranks)]
    # output = [torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 10 for _ in range(num_ranks)]
    # dist.barrier()
    # t = bench(lambda: test_nccl_dispatch(input,output))[0]
    # recv = torch.cat(output)
    # print(recv.numel())
    # print(f'NCCL rank:{local_rank} , {recv.numel()*2 / 1e9 / t:.2f} GB/s (NCCL), avg_t={t * 1e6:.2f} us')
    # if rank==0:
    #     print("$"*50)
    #     print(recv)
    # assert 2==1







    # ------------ep test

    input_tensor = torch.tensor([
        [[0, 1, 2]],  # token0
        [[3, 4, 5]],   # token1
        [[6, 7, 8]],   # token2
    ],dtype=torch.bfloat16).squeeze(1)  
    rank_matrix = torch.tensor([   #internode
        [1, 3, 1, 2],  # token0发送到rank1、2、3（去重后）
        [4, 5, 0, 7],   # token1发送到rank4,5,1,7
        [13, 14, 1, 15]   # token2发送到rank4,5,1,7
    ])
    rank_matrix = torch.tensor([   #intranode
        [1, 3, 1, -1],  # token0发送到rank1、2、3（去重后）
        [4, 5, 0, 7],   # token1发送到rank4,5,1,7
        [1, 3, 1, -1],  # token2发送到rank1、2、3（去重后）
    ])
    dispatch_map = build_dispatch_map(rank_matrix)

    input_split, input_tensor_list = prepare_multi_dispatch(
        input_tensor, dispatch_map, num_ranks
    )
    output_split = torch.empty(num_ranks,dtype=torch.int64, device='cuda')
    if rank==0:
        print(dispatch_map) #[1, 2, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        print(input_split)
        print(input_tensor_list)
    dist.barrier()

    #  -------------dispatch verify
    #准备数据
    # input_split = [1 for i in range(num_ranks)]
    # hidden_size = 2 
    # input_tensor = torch.tensor(torch.ones((sum(input_split),hidden_size), dtype=torch.bfloat16, device='cuda') * rank)  # 不同进程的输入不同
    # #假设每个进程有一个输入张量（示例数据）
    # input_tensor_list = list(input_tensor.split(input_split)) #根据预发送的数据量切分input成list

    print(f'Rank {rank} before dispatch ,input = {input_tensor_list}') 
    print()
    dist.barrier()
    # Dispatch阶段：分发数据
    recv,output_split = nccl_dispatch(input_tensor_list,input_split,output_split,num_ranks,rank,3)
    # # Combine阶段：收集结果
    # result = nccl_combine(recv, input_split,num_ranks,rank,3)

    # print(f"Rank {rank}: Combined result:\n{result}")

if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)