import torch

def validate_qkv(q, k, v, *, self_attention=False):
    """
    Helper function to validate the query, key, and value tensors for attention computation.

    Args:
        q (torch.Tensor): The query input into the attention computation.
                          The assumend tensor has a shape (batch, heads, sequence, dim)
        k (torch.Tensor): The key tensor into the attention computation. 
                          The assumend tensor has a shape (batch, heads, sequence, dim)
        v (torch.Tensor): The value tensor into the attention computation. 
                          The assumend tensor has a shape (batch, heads, sequence, dim)
        self_attention (bool): Default is False to make it unavailable for the full attention and flash attention. 
                               When computing sparse attention, this should be set to True to ensure self attention is used. 
                               When computing self attention, query and key should have the same sequence length.
    """
    # Validate that q, k, v are all 4D floating point tensors with the same device and dtype
    if any(x.ndim != 4 or not x.is_floating_point() for x in (q, k, v)):
        raise ValueError("q, k, v must be floating tensors shaped (batch, heads, sequence, dim)")
    if not (q.device == k.device == v.device and q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must have the same device and dtype")
    
    # Validate that the batch and head dimensions match, and that the key/value sequence lengths match
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("Batch/head dimensions and key/value sequence lengths must match")
    
    # Validate that the query and key head dimensions match, and that all dimensions are positive
    if q.shape[-1] != k.shape[-1] or any(min(x.shape) < 1 for x in (q, k, v)):
        raise ValueError("All dimensions must be positive and query/key head dimensions must match")
    
    # Only allow self-attention for sparse attention, as it is not implemented for cross-attention
    if self_attention and q.shape[-2] != k.shape[-2]:
        raise ValueError("Sparse attention currently supports self-attention only")