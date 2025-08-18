```mermaid
sequenceDiagram
    participant ST as Scheduler Thread<br/>(scheduler_stream)
    participant IQ as input_queue
    participant OQ as output_queue
    participant RQ as result_queue
    participant FT as Forward Thread<br/>(forward_stream)
    participant GPU as GPU
    
    Note over ST,GPU: 初始化阶段
    ST->>ST: 创建 scheduler_stream (默认流)
    FT->>FT: 创建 forward_stream (专用流)
    ST->>RQ: 初始化 result_queue = deque()
    
    Note over ST,GPU: 第一次迭代 (Dummy Batch)
    rect rgb(255, 240, 240)
        Note over ST: 处理第一个真实批次
        ST->>ST: batch = get_next_batch_to_run()
        ST->>ST: batch.launch_done = threading.Event()
        ST->>ST: 调用 run_batch(batch)
        
        Note over ST: forward_batch_generation
        ST->>ST: 创建 sampling_info 副本
        ST->>ST: sync_event = torch.cuda.Event()
        ST->>ST: sync_event.record(scheduler_stream)
        ST->>IQ: put((batch, future_token_ids_ct, sync_event))
        ST->>ST: 分配 future_next_token_ids (负数)
        ST->>RQ: append((batch.copy(), result))
        
        Note over ST: 创建 Dummy Batch
        ST->>ST: tmp_batch = ScheduleBatch(forward_mode=DUMMY_FIRST)
        ST->>ST: process_batch_result(tmp_batch, None, batch.launch_done)
        ST->>ST: batch.launch_done 传递给 dummy batch
    end
    
    Note over FT,GPU: Forward Thread 处理第一个批次
    rect rgb(240, 255, 240)
        FT->>IQ: get() 获取 (batch, future_token_ids_ct, sync_event)
        FT->>FT: sync_event.wait()
        Note right of FT: 等待 scheduler_stream 完成
        
        FT->>FT: 创建 copy_done = torch.cuda.Event()
        FT->>FT: resolve_future_token_ids(input_ids, future_token_ids_map)
        Note right of FT: 将负数索引替换为真实 token
        
        FT->>GPU: forward_batch_generation() [在 forward_stream 上]
        GPU-->>FT: 返回 logits_output, next_token_ids
        
        FT->>FT: launch_done.set()
        Note right of FT: GPU forward 完成后立即设置
        
        FT->>FT: 更新 future_token_ids_map
        Note right of FT: map[future_token_ids_ct+1:+bs+1] = next_token_ids
        
        FT->>GPU: 异步拷贝到 CPU (non_blocking=True)
        Note right of FT: logprobs.to("cpu"), next_token_ids.to("cpu")
        FT->>FT: copy_done.record() [在 forward_stream 上]
        FT->>OQ: put((copy_done, logits_output, next_token_ids, can_run_cuda_graph))
    end
    
    Note over ST,GPU: 后续迭代 (正常批次)
    loop 每个批次
        rect rgb(255, 240, 240)
            Note over ST: Scheduler Thread 操作
            ST->>ST: recv_reqs = recv_requests()
            ST->>ST: process_input_requests(recv_reqs)
            ST->>ST: batch = get_next_batch_to_run()
            
            alt batch 不为空
                ST->>ST: batch.launch_done = threading.Event()
                ST->>ST: result = run_batch(batch)
                Note right of ST: 返回 future_next_token_ids (负数)
                ST->>RQ: append((batch.copy(), result))
            end
            
            alt last_batch 存在
                ST->>RQ: popleft() 获取 (tmp_batch, tmp_result)
                ST->>ST: tmp_batch.next_batch_sampling_info = 当前 sampling_info
                
                Note over ST: 处理上一批次结果
                ST->>ST: process_batch_result(tmp_batch, tmp_result, 当前batch.launch_done)
                
                Note over ST: resolve_last_batch_result 内部
                ST->>OQ: get() 获取 (copy_done, logits_output, next_token_ids, can_run_cuda_graph)
                
                alt launch_done 不为 None
                    ST->>ST: launch_done.wait()
                    Note right of ST: 等待当前批次发射完成
                end
                
                ST->>ST: copy_done.synchronize()
                Note right of ST: 等待 GPU->CPU 拷贝完成
                
                ST->>ST: 处理结果，更新请求状态
            end
        end
        
        rect rgb(240, 255, 240)
            Note over FT: Forward Thread 并行操作
            FT->>IQ: get() 阻塞等待新批次
            FT->>FT: sync_event.wait()
            FT->>FT: resolve_future_token_ids()
            Note right of FT: 解析上一轮的 future tokens
            FT->>GPU: forward_batch_generation()
            FT->>FT: launch_done.set()
            Note right of FT: 在 forward 完成后立即通知
            FT->>FT: 更新 future_token_ids_map
            FT->>GPU: 异步拷贝结果到 CPU
            FT->>FT: copy_done.record()
            FT->>OQ: put(结果)
        end
    end
    
    Note over ST,GPU: 关键同步点说明
    Note over ST: 1. sync_event: 确保 scheduler_stream 操作完成<br/>2. launch_done: CPU 线程间同步，确保批次已发射<br/>3. copy_done: GPU->CPU 拷贝完成同步
    Note over FT: 1. 在 forward_stream 上执行所有 GPU 操作<br/>2. launch_done.set() 在 forward 后立即调用<br/>3. 异步拷贝实现流水线