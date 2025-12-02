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

    batch_size, sequence_length, embedding_dim = q.shape
    assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
    head_dim = embedding_dim // n_heads

    device = q.device
    dtype = q.dtype

    # reshape: (B, T, C) -> (B, T, H, D)
    q = torch.reshape(q, [batch_size, sequence_length, n_heads, head_dim])
    k = torch.reshape(k, [batch_size, sequence_length, n_heads, head_dim])
    v = torch.reshape(v, [batch_size, sequence_length, n_heads, head_dim])
    
    output = torch.zeros_like(q)

    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)

    # choose a large negative for masked positions (dtype-safe)
    neg_inf = torch.tensor(-1e9, device=device, dtype=dtype)

    # Process each batch/head block-wise
    for batch_index in range(batch_size):
        for head_index in range(n_heads):
            q_sliced = q[batch_index, :, head_index, :]  # (T, D)
            k_sliced = k[batch_index, :, head_index, :]
            v_sliced = v[batch_index, :, head_index, :]

            # Initialize online softmax variables
            running_max = torch.full((sequence_length, ), float("-inf"), device=q.device)
            running_denom = torch.zeros((sequence_length, ), device=q.device)
            running_acc = torch.zeros((sequence_length, head_dim), device=q.device)

            # Iterate over key blocks (simulate streaming)
            for start in range(0, sequence_length, block_size):
                end = min(start + block_size, sequence_length)
                k_block = k_sliced[start:end]
                v_block = v_sliced[start:end]

                # Compute local attention scores
                scores = (q_sliced @ k_block.T) * scale  # (T, block_size), block_len = end - start

                if is_causal:
                    # Build a causal mask of shape (sequence_length, block_len):
                    # allowed if key_index <= query_index
                    # key indices span [start, end)
                    key_indices = torch.arange(start, end, device=device).unsqueeze(0)    # (1, block_len)
                    query_indices = torch.arange(sequence_length, device=device).unsqueeze(1)  # (sequence_length, 1)
                    causal_mask = (key_indices <= query_indices)  # (sequence_length, block_len)
                    scores = scores.masked_fill(~causal_mask, neg_inf)

                block_max = torch.max(scores, dim=-1).values
                new_max = torch.maximum(running_max, block_max)

                # Compute exponentials in a numerically stable way
                exp_old = torch.exp(running_max - new_max)
                exp_block = torch.exp(scores - new_max.unsqueeze(-1))

                running_denom = running_denom * exp_old + exp_block.sum(dim=-1)
                running_acc = running_acc * exp_old.unsqueeze(-1) + exp_block @ v_block

                running_max = new_max

            output[batch_index, :, head_index, :] = running_acc / running_denom.unsqueeze(-1)
    
    # Merge heads back: out_heads shape (B, T, H, D) -> (B, T, H*D)
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