import math
import torch

def flash_attention(q, k, v, n_heads, block_size=128, is_causal=False):
    """
    Implement flash attention
    
    Args:
        q (torch.Tensor): Query Tensor of shape (batch, sequence_length, embedding_dim).
        k (torch.Tensor): Key Tensor of shape (batch, sequence_length, embedding_dim).
        v (torch.Tensor): Value Tensor of shape (batch, sequence_length, embedding_dim).
        n_heads (int): number of attention heads.
        blocksize (int): Size of blocks to process at a time.
        causal (bool): Whether to apply causal masking.

    Returns: 
        torch.Tensor: Attention output Tensor of shape (batch, sequence_length, n_heads, head_dim).
    """

    # Get dimensions and validate the validity of the number of heads
    batch_size, sequence_length, embedding_dim = q.shape
    assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
    head_dim = embedding_dim // n_heads

    # Get device and dtype
    device = q.device
    dtype = q.dtype

    # Reshape q, k, v tensors into separate heads
    # e.g., from (batch_size, sequence_length, embedding_dim) to (batch_size, sequence_length, n_heads, head_dim)
    q = torch.reshape(q, [batch_size, sequence_length, n_heads, head_dim])
    k = torch.reshape(k, [batch_size, sequence_length, n_heads, head_dim])
    v = torch.reshape(v, [batch_size, sequence_length, n_heads, head_dim])
    
    # Initialize output tensor to hold attention outputs
    output = torch.zeros_like(q)

    # Define scaling factor
    scale = 1.0 / math.sqrt(head_dim)

    # Process each batch/head block-wise
    for batch_index in range(batch_size):
        for head_index in range(n_heads):
            # Slice q, k, v tensors for the current batch and head
            q_sliced = q[batch_index, :, head_index, :]
            k_sliced = k[batch_index, :, head_index, :]
            v_sliced = v[batch_index, :, head_index, :]

            # Initialize online softmax variables
            running_max = torch.full((sequence_length, ), float("-inf"), device=q.device)
            running_denom = torch.zeros((sequence_length, ), device=q.device)
            running_accumulator = torch.zeros((sequence_length, head_dim), device=q.device)

            # Iterate over key blocks (simulate streaming)
            for start in range(0, sequence_length, block_size):
                end = min(start + block_size, sequence_length)
                k_block = k_sliced[start:end]
                v_block = v_sliced[start:end]

                # First step of FlashAttention recurrence: 
                # Compute local attention scores within the block
                # The tensor shape of the sores is (sequence_length, block_len), where block_len = end - start
                scores = (q_sliced @ k_block.T) * scale

                if is_causal:
                    # Build a causal mask of shape (sequence_length, block_len):
                    # Scores are only allowed if key_index <= query_index
                    # Shape of key_indices: (1, block_len)
                    key_indices = torch.arange(start, end, device=device).unsqueeze(0)
                    # Shape of query_indices: (sequence_length, 1)
                    query_indices = torch.arange(sequence_length, device=device).unsqueeze(1)
                    # Shape of causal_mask: (sequence_length, block_len)
                    causal_mask = (key_indices <= query_indices)
                    scores = scores.masked_fill_(causal_mask.logical_not(), float("-inf"))

                # Second step of FlashAttention recurrence:
                # Get the block max from the sores, e.g. max(scores_i)
                # Update running max, e.g. max_{1:i} = max(max_{1:i-1}, max(scores_i))
                block_max = torch.max(scores, dim=-1).values
                new_max = torch.maximum(running_max, block_max)

                # Third step of FlashAttention recurrence:
                # Update running denom with the running max, new max, and block scores
                # e.g. denom_{1:i} = denom_{1:i-1} * exp(old_max - new_max) + sum(exp(scores_i - new_max))
                exp_old = torch.exp(running_max - new_max)
                exp_block = torch.exp(scores - new_max.unsqueeze(-1))
                new_running_denom = running_denom * exp_old + exp_block.sum(dim=-1)

                # Fourth step of FlashAttention recurrence:
                # Update running accumulator with results from step 3
                # e.g. accumulator_{1:i} = accumulator_{1:i-1} * exp(old_max - new_max) + exp(scores_i - new_max) @ v_i
                new_running_accumulator = running_accumulator * exp_old.unsqueeze(-1) + exp_block @ v_block
                
                # Finalize running variables for the next block
                running_max = new_max
                running_denom = new_running_denom
                running_accumulator = new_running_accumulator

            # After processing all blocks, compute the final output for one head in one batch
            output[batch_index, :, head_index, :] = running_accumulator / running_denom.unsqueeze(-1)
    
    # Reshape output back to (batch_size, sequence_length, embedding_dim)
    output = torch.reshape(output, [batch_size, sequence_length, embedding_dim])

    return output

# Example usage:
if __name__ == "__main__":
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    n_heads = 4

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = flash_attention(q, k, v, n_heads, is_causal=True)
    print(output.shape)  # Expected: (4, 1024, 256)