"""
This module is based on the code from pytorch documentation: 
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import math
import torch

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
            if default with None, no mask is applied;
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
    # The bias matrix is for the attention mask
    # The scale factor is used to scale the attention weights
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    # Causal attention mask is applied to hide attention weights of future tokens
    # If the attention is causal, the atten_mask can't be None
    # Attention mask should be a lower triangular mask
    if is_causal and attn_mask is None:
        raise ValueError("Causal attention must have an attention mask")
    elif is_causal and attn_mask is not None:
        # Create a lower triangular mask with the type of torch.bool
        # Use -inf to mask out the future tokens
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    # Attention mask is applied to hide attention weights to certain positions
    # A lower triangular mask is applied to the attention bias
    if attn_mask is not None:
        # A boolean mask where a value of True indicates that the element should take part in attention.
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        # A float mask of the same type as query, key, value that is added to the attention score
        else:
            attn_bias = attn_mask + attn_bias

    # Compute the attention weights
    # The attention weights are computed as the dot product of query and key, 
    # scaled by the scale factor, and softmaxed to get the attention distribution
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn = attn_weight @ value
    
    return attn

# Example usage:
if __name__ == "__main__":
    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = scaled_dot_product_attention(q, k, v)
    print(output.shape) # Expected: (4, 1024, 256)