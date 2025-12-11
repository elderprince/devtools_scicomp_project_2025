import torch
import torch.nn as nn
from transformer.attentions import scaleDotProductAttention, sparseAttention, flashAttention

class BaseAttention(nn.Module):
    """
    Abstract attention class.
    Child classes must implement forward(q, k, v).
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        raise NotImplementedError

class FullAttention(BaseAttention):
    """
    Standard full attention class using scaled dot-product attention.
    """
    def __init__(self, attn_mask=None, 
                 dropout_p=0.0, is_causal=False, 
                 scale=None):
        super().__init__()
        self.attn_mask = attn_mask
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale

    def forward(self, q, k, v):
        return scaleDotProductAttention.scaled_dot_product_attention(q, k, v, 
                                                                     self.attn_mask,
                                                                     self.dropout_p,
                                                                     self.is_causal,
                                                                     self.scale)

class StridedSparseAttention(BaseAttention):
    """
    Strided sparse attention class.
    """
    def __init__(self, stride=64):
        super().__init__()
        self.stride = stride

    def forward(self, q, k, v):
        return sparseAttention.sliced_strided_sparse_attention(q, k, v,
                                                               self.stride)

class FlashAttention(BaseAttention):
    """
    Flash attention class.
    """
    def __init__(self, block_size=64, is_causal=False):
        super().__init__()
        self.block_size = block_size
        self.is_causal = is_causal

    def forward(self, q, k, v):
        return flashAttention.flash_attention(q, k, v, 
                                              self.block_size, 
                                              self.is_causal)

class ModularMultiHeadAttention(nn.Module):
    """
    Multi-head attention module that can utilize different attention mechanisms.

    Args:
        embed_dim (int): The total embedding length.
        num_heads (int): Number of attention heads.
        attention_module (BaseAttention): An instance of a class derived from BaseAttention.
    """
    def __init__(self, embed_dim, num_heads, attention_module):
        super().__init__()
        
        # Validate the validity of n_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Get dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define the assigned attention module
        self.attn = attention_module

        # Define the projection layers for query, key, value, and output
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x) -> torch.Tensor:
        """
        Reshape (batch, sequence_length, embedding_dim) -> (batch, n_heads, sequence_length, head_dim)

        Args:
            x (torch.Tensor): Input Tensor of shape (batch, sequence_length, embedding_dim)
        
        Returns:
            torch.Tensor: Reshaped Tensor of shape (batch, n_heads, sequence_length, head_dim)
        """
        # Get dimensions
        batch_size, sequence_length, embedding_dim = x.size()

        # Validate the embedding dimension
        assert embedding_dim == self.embed_dim, "Embedding dimension does not match"
        
        # Reshape the input tensor to separate heads
        x_split = torch.reshape(x, 
                                [batch_size, sequence_length, 
                                 self.num_heads, self.head_dim])

        # Adjust dimensions to move heads to the second dimension
        x_split = x_split.permute(0, 2, 1, 3).contiguous()

        return x_split

    def merge_heads(self, x) -> torch.Tensor:
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

        # Validate the number of heads and head dimension
        assert n_heads == self.num_heads, "Number of heads does not match"
        assert head_dim == self.head_dim, "Head dimension does not match"

        # Reshape the input tensor to merge heads back into the original embedding dimension
        x_merged = torch.reshape(x, [batch_size, 
                                     sequence_length, self.embed_dim])

        return x_merged

    def forward(self, x):
        # Split input tensors into multiple heads
        q = self.split_heads(self.Wq(x))
        k = self.split_heads(self.Wk(x))
        v = self.split_heads(self.Wv(x))

        # Apply the attention mechanism
        out = self.attn(q, k, v)
        
        # Merge the multiple heads back into the original embedding dimension
        out = self.merge_heads(out)

        return self.out_proj(out)