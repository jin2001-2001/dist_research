from torch.distributed.pipelining.schedules import _Action, _ComputationType


def generate_1f1b_pipeline_actions(num_stages: int, num_microbatches: int):
    
    actions_per_rank = {}
    
    for stage_idx in range(num_stages):
        rank = stage_idx  # stage == rank
        actions = []
        
        
        warmup_chunks = min(num_microbatches, num_stages - stage_idx)
        
       
        fwd_mb = 0  
        bwd_mb = 0  
        
        # Warmup
        for _ in range(warmup_chunks):
            # Recv F
            if stage_idx > 0:
                actions.append(_Action(stage_idx, rank, _ComputationType.RECV_F, fwd_mb, rank - 1))
            
            # Forward
            actions.append(_Action(stage_idx, rank, _ComputationType.FORWARD, fwd_mb, None))
            
            # Send F
            if stage_idx < num_stages - 1:
                actions.append(_Action(stage_idx, rank, _ComputationType.SEND_F, fwd_mb, rank + 1))
            
            fwd_mb += 1
        
        # 1B1F阶段
        while fwd_mb < num_microbatches:
            # 1B
            # Recv B
            if stage_idx < num_stages - 1:
                actions.append(_Action(stage_idx, rank, _ComputationType.RECV_B, bwd_mb, rank + 1))
            
            # Backward
            actions.append(_Action(stage_idx, rank, _ComputationType.FULL_BACKWARD, bwd_mb, None))
            
            # Send B
            if stage_idx > 0:
                actions.append(_Action(stage_idx, rank, _ComputationType.SEND_B, bwd_mb, rank - 1))
            
            bwd_mb += 1
            
            # 1F
            # Recv F
            if stage_idx > 0:
                actions.append(_Action(stage_idx, rank, _ComputationType.RECV_F, fwd_mb, rank - 1))
            
            # Forward
            actions.append(_Action(stage_idx, rank, _ComputationType.FORWARD, fwd_mb, None))
            
            # Send F
            if stage_idx < num_stages - 1:
                actions.append(_Action(stage_idx, rank, _ComputationType.SEND_F, fwd_mb, rank + 1))
            
            fwd_mb += 1
        
        #Cooldown
        while bwd_mb < num_microbatches:
            # Recv B
            if stage_idx < num_stages - 1:
                actions.append(_Action(stage_idx, rank, _ComputationType.RECV_B, bwd_mb, rank + 1))
            
            # Backward
            actions.append(_Action(stage_idx, rank, _ComputationType.FULL_BACKWARD, bwd_mb, None))
            
            # Send B
            if stage_idx > 0:
                actions.append(_Action(stage_idx, rank, _ComputationType.SEND_B, bwd_mb, rank - 1))
            
            bwd_mb += 1
        
        actions_per_rank[rank] = actions
    
    return actions_per_rank


def print_pipeline_actions(num_stages, num_microbatches):

    actions = generate_1f1b_pipeline_actions(num_stages, num_microbatches)
    
    print("def create_pipeline_actions():")
    
    for rank in range(num_stages):
        print(f"\n    # Rank {rank} (Stage {rank}) 的操作序列")
        print(f"    rank{rank}_actions = [")
        for action in actions[rank]:
            print(f"        {action},")
        print("    ]")
    
    print("\n    return {" + ", ".join([f"{rank}: rank{rank}_actions" for rank in range(num_stages)]) + "}")
    
    return actions


if __name__ == "__main__":
    print_pipeline_actions(3, 8)