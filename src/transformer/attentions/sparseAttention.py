import math
import torch

def get_attn_mask(sequence_length, stride, device=None, dtype=None) -> torch.Tensor:
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
        dtype = torch.long if dtype is None else dtype

    # Initialize two tensors respectively with the shape of (sequence_length, 1), and (1, sequence_length)
    # These tensors will be broadcasted to (sequence_length, sequence_length) for comparison
    q_indices = torch.arange(sequence_length, device=device, dtype=dtype).view(sequence_length, 1)
    k_indices = torch.arange(sequence_length, device=device, dtype=dtype).view(1, sequence_length)

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
    Implement strided sparse attention using dense operations with masking.
    
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
      - It still computes full (sequence_length x sequence_length) logits with O(N^2) complexity.
      - Used for the verification of more efficient sparse implementations.
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
    mask_bool = get_attn_mask(sequence_length, stride, device=queries.device)
    logits = logits.masked_fill_(mask_bool.logical_not(), float("-inf"))

    # Softmax of the masked logits to get attention weights
    # Compute attention output with the dot-product of attention weights and value tensor
    attn_weights = torch.softmax(logits, dim=-1)
    attn = torch.matmul(attn_weights, v)  # (batch, heads, sequence_length, head_dim)

    # Merge heads of the attention tensor to restore the original embedding dimension
    out = merge_heads(attn)
    
    return out

def build_sparse_indices(sequence_length, stride, device=None):
    """
    Construct, for every query position, the list of indexes allowed by the strided-causal rule. 
    Frankly, generate a rectangular tensor indexing the allowed tokens contributing to the attention calculation in each position. 

    Args:
        sequence_length (int): length of the sequence
        stride (int): stride for strided sparse attention
        device (torch.device, optional): device to create the tensors on; defaults to CPU if None

    Output:
        key_indices_per_query (torch.Tensor): LongTensor of shape (sequence_length, max_keys)
        mask_per_query (torch.Tensor): BoolTensor of same shape (sequence_length, max_keys)
        max_keys (int): maximum number of keys selected for any query position
    
    NOTES: 
        - key_indices_per_query[q] gives the indices of keys allowed for the query position q.
        - mask_per_query[q] indicates which entries in key_indices_per_query[q] are valid (not padding).
        - Terminologies: "keys" refer to the sequence positions being attended to, "queries" refer to the positions attending. 
          Both in this function differ from the input query/key tensors into the attention mechanism. 
    """
    # Set default device to CPU if none provided
    if device is None:
        device = torch.device("cpu")
    
    # Initialize list to hold allowed key indices for each query position
    all_indices = []
    max_len = 0

    for q in range(sequence_length):
        # Construct a 1D tensor holding keys satisfying causal constraint, i.e., k <= q
        k = torch.arange(0, q+1)

        # Apply stride rule to 1D tensor into a , i.e., (q-k) % stride == 0
        cond = ((q - k) % stride) == 0

        # Filter keys based on the causal and strided conditions
        # Store the allowed key indices for the current query position
        # Update the maximum length of allowed keys across all query positions
        allowed = k[cond]
        all_indices.append(allowed)
        max_len = max(max_len, allowed.numel())

    # Initialize tensors to hold key indices and masks, with the shape (sequence_length, max_len)
    key_indices = torch.full((sequence_length, max_len), fill_value=0, dtype=torch.long)
    mask = torch.full((sequence_length, max_len), fill_value=0, dtype=torch.bool)
    
    # For each query position, fill in the allowed key indices and set the corresponding mask entries to True
    for q, k_allowed in enumerate(all_indices):
        L = len(k_allowed)
        key_indices[q, :L] = k_allowed
        mask[q, :L] = True

    return key_indices.to(device), mask.to(device), max_len


def sliced_strided_sparse_attention(queries, keys, values, n_heads, stride):
    """
    Implement strided sparse attention using sliced tensor for efficent oeerations. 
    
    Arg: 
        queries (torch.Tensor): Query Tensor of shape (batch, sequence_length, embedding_dim)
        keys (torch.Tensor): Key Tensor of shape (batch, sequence_length, embedding_dim)
        values (torch.Tensor): Value Tensor of shape (batch, sequence_length, embedding_dim)
        n_heads (int): number of attention heads
        stride (int): stride for strided sparse attention

    Returns:
        torch.Tensor: Attention output Tensor of shape (batch, sequence_length, embedding_dim)
    """
    # Assign device and get dimensions
    device = queries.device
    batch_size, sequence_length, embedding_dim = queries.shape
    head_dim = embedding_dim // n_heads

    # Split query, key, and value tensors into multiple heads
    q = split_heads(queries, n_heads)
    k = split_heads(keys, n_heads)
    v = split_heads(values, n_heads)

    # Construct tensors holding sparse indices and masks with the shape (sequence_length, max_keys)
    key_index_map, key_mask, max_keys = build_sparse_indices(sequence_length, stride, device)

    # Expand tensors of sparse indices and masks from 2D to 4D, with the shape (batch, heads, sequence_length, max_keys)
    # This operation is the preparation for broadcasting both tensors in all batches and heads
    key_indices = key_index_map.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, sequence_length, max_keys)
    key_mask = key_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, sequence_length, max_keys)

    # Gather selected keys and values according to sparse indices, so that tensors of allowed keys/values 
    # have the shape (batch, heads, sequence_length, max_keys, head_dim), e.g., for each query position q in 
    # each batch and head, selected_keys[batch, head, q, :] contains the keys/values allowed for atention calculation
    selected_keys = torch.gather(
        k.unsqueeze(3).expand(-1, -1, -1, max_keys, -1),
        dim=2,
        index=key_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        )
    selected_values = torch.gather(
        v.unsqueeze(3).expand(-1, -1, -1, max_keys, -1),
        dim=2,
        index=key_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        )
    
    # Extend query tensor for broadcasting in dot-product attention calculation
    # The shape is expanded from (batch, heads, sequence_length, head_dim) to (batch, heads, sequence_length, 1, head_dim)
    q_expanded = q.unsqueeze(3)

    # Compute attention weights with only selected keys 
    # Dot-product the query tensors with only selected key tensors
    # Sum over the last dimension (head_dim) to get raw attention logits for each selected key
    # Shape of logits: (batch, heads, sequence_length, max_keys)
    logits = (q_expanded * selected_keys).sum(dim=-1)

    # Apply scaling to the logits
    logits = logits / math.sqrt(head_dim)

    # Not all queries have the same number of allowed keys, so we mask out 
    # the padding positions in the logits using the precomputed key mask
    logits = logits.masked_fill_(key_mask.logical_not(), float("-inf"))

    # Softmax of the masked logits to get attention weights
    attn_weights = torch.softmax(logits, dim=-1)

    # Compute attention output with the expanded attention weights and only selected values
    # Sum over the max_keys dimension to ensure the attention output of each token in the sequence can be with the same embedding length with the input
    # attn_weights: (batch, heads, sequence_length, max_keys, 1)
    # selected_values: (batch, heads, sequence_length, max_keys, head_dim)
    attn_output = (attn_weights.unsqueeze(-1) * selected_values).sum(dim=-2)

    # Merge heads of the attention tensor to restore the original embedding dimension
    attn_output = merge_heads(attn_output)

    return attn_output

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

    out_sparse = sliced_strided_sparse_attention(q, k, v, n_heads, stride=64)
    out_dense = strided_sparse_attention(q, k, v, n_heads, stride=64)
    print(torch.allclose(out_sparse, out_dense, atol=1e-5))  # Expected: True