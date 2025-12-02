import math
import torch

def get_attn_mask(sequence_length, stride, device=None) -> torch.Tensor:
    """
    Build a strided causal mask shaped (1, 1, sequence_length, sequence_length) on the given device.
    Keep position (query index, key index) if:
        - query index >= key index  (causal)
        - (query index - key index) % stride == 0  (strided)
    
    Args:
        sequence_length (int): length of the sequence
        stride (int): stride for strided sparse attention
        device (torch.device, optional): device to create the mask on; defaults to CPU if None
        
    Returns: 
        torch.Tensor: boolean mask by default.
    """
    # Validate inputs to ensure it's not None
    if stride is None:
        raise ValueError("stride (local_attn_ctx) must be provided and not None")
    # Set default device to CPU if none provided
    if device is None:
        device = torch.device("cpu")

    # Initialize two tensors respectively with the shape of (sequence_length, 1), and (1, sequence_length)
    # These tensors will be broadcasted to (sequence_length, sequence_length) for comparison
    q_indices = torch.arange(sequence_length, device=device, dtype=torch.bool).view(sequence_length, 1)
    k_indices = torch.arange(sequence_length, device=device, dtype=torch.bool).view(1, sequence_length)

    # First condition: query index >= key index  (causal)
    # Generates a (sequence_length, sequence_length) boolean tensor for causal masking
    cond_causal = q_indices >= k_indices
    
    # Second condition: (query index - key index) % stride == 0  (strided)
    # Generates a (sequence_length, sequence_length) boolean tensor for strided masking
    diff = q_indices - k_indices
    cond_stride = (diff % stride == 0)

    # Combine both conditions for the final masking boolean tensor
    mask = cond_causal & cond_stride
    
    # Reshape to (1, 1, sequence_length, sequence_length) for compatibility in attention mechanisms
    mask = torch.reshape(mask, (1, 1, sequence_length, sequence_length)).to(device=device, dtype=torch.bool)

    return mask

def split_heads(x, n_heads) -> torch.Tensor:
    """
    Reshape (batch, sequence_length, embedding_dim) -> (batch, n_heads, sequence_length, head_dim)

    Args:
        x (torch.Tensor): Input Tensor of shape (batch, sequence_length, embedding_dim)
        n_heads (int): number of attention heads
    
    Returns:
        torch.Tensor: Reshaped Tensor of shape (batch, n_heads, sequence_length, head_dim)
    """
    # Get dimensions
    batch_size, sequence_length, embedding_dim = x.size()
    
    # Validate the validity of n_heads
    assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
    
    # Calculate the number of embedding dimensions per head
    # Reshape the input tensor to separate heads
    head_dim = embedding_dim // n_heads
    x_split = torch.reshape(x, [batch_size, sequence_length, n_heads, head_dim])

    # Adjust dimensions to move heads to the second dimension
    x_split = x_split.permute(0, 2, 1, 3).contiguous()

    return x_split

def merge_heads(x) -> torch.Tensor:
    """
    Reshape (batch, n_heads, sequence_length, head_dim) -> (batch, sequence_length, embedding_dim)

    Args:
        x (torch.Tensor): Input Tensor of shape (batch, sequence_length, n_heads, head_dim)
    
    Returns:
        torch.Tensor: Reshaped Tensor of shape (batch, sequence_length, embedding_dim)
    """
    # Adjust dimensions to move heads back to last-two dims
    # Get dimensions
    x = x.permute(0, 2, 1, 3).contiguous()
    batch_size, sequence_length, n_heads, head_dim = x.size()

    # Calculate the original embedding dimension
    # Reshape the input tensor to merge heads back into the original embedding dimension
    embedding_dim = n_heads * head_dim
    x_merged = torch.reshape(x, [batch_size, sequence_length, embedding_dim])

    return x_merged

def strided_sparse_attention(queries, keys, values, n_heads, stride) -> torch.Tensor:
    """
    Arg: 
        queries (torch.Tensor): Query Tensor of shape (batch, sequence_length, embedding_dim)
        keys (torch.Tensor): Key Tensor of shape (batch, sequence_length, embedding_dim)
        values (torch.Tensor): Value Tensor of shape (batch, sequence_length, embedding_dim)
        n_heads(int): number of attention heads
        local_attn_ctx (int): stride for strided sparse attention
    
    Returns: 
        torch.Tensor: Attention output Tensor of shape (batch, sequence_length, embedding_dim)
    
    NOTES:
      - This implementation masks logits for a strided causal pattern.
      - It still computes full (sequence_length x sequence_length) logits (O(N^2)).
    """
    # Get dimensions
    batch_size, sequence_length, embedding_dim = queries.shape

    # Split query, key, and value tensors into multiple heads
    q = split_heads(queries, n_heads)
    k = split_heads(keys, n_heads)
    v = split_heads(values, n_heads)

    # Scaled dot-product of query and key tensors to get raw attention logits
    head_dim = q.size(-1)
    scale = 1.0 / math.sqrt(head_dim)
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Build and apply boolean masking tensor to mask sequences from the calculation of attention tensor
    mask_bool = get_attn_mask(sequence_length, stride, device=queries.divece)
    logits = logits.masked_fill_(mask_bool.logical_not(), float("-inf"))

    # Softmax of the masked logits to get attention weights
    # Compute attention output with the dot-product of attention weights and value tensor
    attn_weights = torch.softmax(logits, dim=-1)
    attn = torch.matmul(attn_weights, v)  # (batch, heads, sequence_length, head_dim)

    # Merge heads of the attention tensor to restore the original embedding dimension
    out = merge_heads(attn)
    
    return out

# Example usage:
if __name__ == "__main__":
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    n_heads = 4

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = strided_sparse_attention(q, k, v, n_heads, stride=64)
    print(output.shape)  # Expected: (4, 1024, 256)