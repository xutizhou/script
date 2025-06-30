sequenceDiagram
    participant Caller
    participant DLayer as DeepseekV2DecoderLayer.forward

    Caller->>+DLayer: (positions, hidden_states, ...)

    DLayer->>DLayer: 1. Comm.prepare_attn(hs, res)
    
    box "2. self_attn.forward (DeepseekV2AttentionMLA)"
        participant Attn as self_attn.forward
    end
    DLayer->>+Attn: forward(positions, hs, ...)
    
    Attn->>Attn: dispatch_attn_forward_method(forward_batch)
    Note right of Attn: Determines path (MHA, MLA, etc.)
    
    alt AttnForwardMethod is MLA
        Attn->>Attn: forward_absorb_prepare(...)
        Attn->>Attn: forward_absorb_core(...)
        Note right of Attn: Uses weight absorption
    else AttnForwardMethod is MHA
        Attn->>Attn: forward_normal_prepare(...)
        Attn->>Attn: forward_normal_core(...)
        Note right of Attn: Standard Multi-Head Attention
    end
    
    Attn-->>-DLayer: hidden_states

    DLayer->>DLayer: 3. Comm.prepare_mlp(hs, res)
    
    box "4. mlp.forward"
        participant MLP as mlp.forward
    end
    DLayer->>+MLP: forward(hidden_states, ...)
    
    alt is_dense (uses DeepseekV2MLP)
        MLP->>MLP: gate_up_proj(hs) -> act_fn -> down_proj
    
    else is_layer_sparse (uses DeepseekV2MoE)
        box "Within DeepseekV2MoE.forward"
            participant MoE as MoE.forward
        end
        MLP->>+MoE: (hidden_states, ...)
        
        alt _enable_deepep_moe
            box "path: forward_deepep"
                participant DeepEP as forward_deepep
            end
            MoE->>+DeepEP: (hs, ...)
            DeepEP->>DeepEP: select_experts -> dispatch -> experts -> combine
            Note right of DeepEP: Expert Parallelism via All-to-All
            DeepEP-->>-MoE: result
        else not _enable_deepep_moe
            box "path: forward_normal"
                participant Normal as forward_normal
            end
            MoE->>+Normal: (hidden_states)
            Normal->>Normal: gate -> experts -> all_reduce
            Note right of Normal: Standard MoE execution
            Normal-->>-MoE: result
        end
        MoE-->>-MLP: hidden_states
    end
    MLP-->>-DLayer: hidden_states

    DLayer->>DLayer: 5. Comm.postprocess_layer(hs, res)

    DLayer-->>-Caller: final hidden_states