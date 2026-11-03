"""
This module is based on the code from pytorch documentation: 
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import math
import torch
from numbers import Real
from transformer.attentions.helperFunctions import validate_qkv

def scaled_dot_product_attention(query, key, value, attn_mask=None, 
                                 dropout_p=0.0, is_causal=False, 
                                 scale=None) -> torch.Tensor:
    """
    Compute scale dot product attention

    Args:
        query (torch.Tensor): Query Tensor of shape (N ... Hq, L, E)
        key (torch.Tensor): Key Tensor of shape (N ... H, S, E)
        value (torch.Tensor): Value Tensor of shape (N ... H, S, Ev)
        attn_mask (optional, Tensor): Whether to apply attention mask; 
            if default with None, no custom mask is applied;
            if provided, the Tensor shape must be (N ... L, S)
        dropout_p (float): Dropout probability; 
            if default with 0.0, no dropout is applied
        is_causal (bool): Whether to use causal attention; 
            if default with False, it is not causal
        scale (optional, float): Customized scale factor for attention weights; 
            if default with None, it is set to 1/sqrt(d_k)

    Returns:
        torch.Tensor: Attention output Tensor of shape (N ... Hq, L, Ev)

    Notes: 
        N: Batch size
        Hq: Number of query heads
        H: Number of key heads
        L: Length of target sequence
        S: Length of source sequence
        E: Embedding dimension of query and key
        Ev: Embedding dimension of value
    """
    # Initialize attention bias and scale factor
    # The scale factor is used to scale the attention weights
    validate_qkv(query, key, value)
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # Implement a default attention mask if is_causal is True and attn_mask is None
    # Or, implemant a specific attention mask if is_causal is False and attn_mask is not None
    if is_causal and attn_mask is not None:
        raise ValueError("Use either is_causal=True or an explicit attention mask")
    # Check the validity of the attention mask if it is provided
    if attn_mask is not None:
        if not isinstance(attn_mask, torch.Tensor):
            raise TypeError("attn_mask must be a tensor or None")
        if attn_mask.device != query.device:
            raise ValueError("attn_mask must be on the query device")
        if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
            raise TypeError("attn_mask must be Boolean or floating point")
    # Check the validity of the input arguments
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p must be in [0, 1]")
    if scale is not None and not isinstance(scale, Real):
        raise TypeError("scale must be a number or None")
    
    # Compute the attention scores as the dot product of query and key, scaled by the scale factor
    scores = query @ key.transpose(-2, -1) * scale_factor
    
    # Attention mask should be a lower triangular mask
    if is_causal:
        # Create a lower triangular mask with the type of torch.bool
        # Use -inf to mask out the future tokens 
        temp_mask = torch.ones(L, S, device=query.device, dtype=torch.bool).tril(diagonal=0)
        scores.masked_fill_(temp_mask.logical_not(), float("-inf"))

    # Attention mask is applied to hide attention weights to certain positions
    # A lower triangular mask is applied to the attention bias
    if attn_mask is not None:
        # A boolean mask where a value of True indicates that the element should take part in attention.
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(attn_mask.logical_not(), float("-inf"))
        # A float mask of the same type as query, key, value that is added to the attention score
        else:
            scores = scores + attn_mask.to(query.dtype)
    
    # Get the attention weights by applying softmax to the attention scores along the last dimension
    attn_weight = torch.softmax(scores, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    # Compute the attention output as the weighted sum of value
    attn = attn_weight @ value
    
    return attn

# Example usage:
if __name__ == "__main__":
    n_batch = 4
    n_heads = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.randn(n_batch, n_heads, n_ctx, n_embd)
    k = torch.randn(n_batch, n_heads, n_ctx, n_embd)
    v = torch.randn(n_batch, n_heads, n_ctx, n_embd)

    output = scaled_dot_product_attention(q, k, v)
    print('output.shape:', output.shape) # Expected: (4, 4, 1024, 256)