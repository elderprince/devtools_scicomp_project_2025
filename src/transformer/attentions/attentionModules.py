import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel
from transformer.attentions.baseAttention import BaseAttention
from transformer.attentions.attentionRegistry import register_attention
from transformer.attentions import scaleDotProductAttention, sparseAttention, flashAttention


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

@register_attention("pytorch_full")
class PyTorchFullAttention(BaseAttention):
    """
    PyTorch full attention class. 
    Implement the same functionality as FullAttention with PyTorch's built-in function.
    """
    def __init__(self, is_causal=True):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v):
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, 
                                                  is_causal=self.is_causal)

@register_attention("pytorch_flash")
class PyTorchFlashAttention(BaseAttention):
    """
    PyTorch flash attention class.
    Implement the same functionality as FlashAttention with PyTorch's built-in function.
    """
    def __init__(self, is_causal=True):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v):
        if q.device.type != "cuda":
            raise RuntimeError("pytorch_flash requires CUDA device.")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, 
                                                  is_causal=self.is_causal)