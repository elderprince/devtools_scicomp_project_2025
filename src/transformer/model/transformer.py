import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformer.attentions.attentionRegistry import register_attention, build_attention
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

@register_attention("full")
class FullAttention(BaseAttention):
    """
    Standard full attention class using scaled dot-product attention.
    Implement the scaled dot-product attention function in transformer/attentions/scaleDotProductAttention.py
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

@register_attention("strided")
class StridedSparseAttention(BaseAttention):
    """
    Strided sparse attention class.
    Implement the sliced strided sparse attention function in transformer/attentions/sparseAttention.py
    """
    def __init__(self, stride=64):
        super().__init__()
        self.stride = stride

    def forward(self, q, k, v):
        return sparseAttention.sliced_strided_sparse_attention(q, k, v,
                                                               self.stride)

@register_attention("flash")
class FlashAttention(BaseAttention):
    """
    Flash attention class.
    Implement the flash attention function in transformer/attentions/flashAttention.py
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
        embed_dim (int): Dimensionality of each token representation.
        num_heads (int): Number of attention heads.
        attention_config (dic): Configuration dictionary for the attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, attention_config):
        super().__init__()
        
        # Validate the validity of n_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Get dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define the assigned attention module from the registry
        self.attn = build_attention(attention_config)

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

class SinusoidalPositionalEncoding(nn.Module):
    """
    Implement the same fixed sinusoidal positional encoding in the orifinal 
    Transformer paper (Attention Is All You Need (Vaswani et al., 2017)). 

    Based on the implementation from PyTorchLightning tutorial: 
    https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    
    Args:
        embed_dim (int): Dimensionality of each token representation.
        max_seq_len (int): Maximum sequence length for positional encodings.
    """

    def __init__(self, embed_dim, max_seq_len=2048):
        super().__init__()

        # Create positional encoding matrix of shape (max_seq_len, embed_dim)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len, device):
        """
        Args:
            seq_len (int): Length of the input sequence.
            device (torch.device): Device to place the positional encodings on.

        Returns:
            Positional encodings of shape (1, seq_len, embed_dim)
        """
        return self.pe[ : seq_len].unsqueeze(0).to(device)

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of pluggable multi-head attention, MLP with residual connections, and layer normalization.

    Args:
        embed_dim (int): Dimensionality of each token representation.
        num_heads (int): Number of attention heads.
        attention_config (dic): Configuration dictionary for the attention mechanism.
        mlp_ratio (int): Ratio to determine the hidden dimension of the MLP relative to embed_dim.
    """
    def __init__(self, embed_dim, num_heads, attention_config, mlp_ratio=4):
        super().__init__()

        # Define normalization layer before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Define the pluggable multi-head attention module
        self.attn = ModularMultiHeadAttention(embed_dim, num_heads, attention_config)

        # Define normalization layer before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # Define the MLP module
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x
    
class SmallTransformerLM(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating a small Transformer-based language model.
    
    Args:
        attention_config (dic): Configuration dictionary for the attention mechanism.
        max_seq_len (int): Maximum sequence length for positional encodings.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the token embeddings.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer blocks.
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, attention_config, max_seq_len=2048, vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, attention_config)
            for i in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, idx):
        # Get dimensions of the token index tensor of the input batch
        batch_size, seq_len = idx.shape
        # Embedding tokens with the shape of (batch_size, seq_len, embed_dim)
        tok = self.token_emb(idx)
        # Adding positional encodings to the token embeddings
        pos = self.pos_enc(seq_len, idx.device)
        x = tok + pos

        # Passing through the Transformer blocks sequentially for the final logits 
        # The shape of final logits is (batch_size, seq_len, vocab_size)
        for b in self.blocks:
            x = b(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))
        self.log("train/loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)