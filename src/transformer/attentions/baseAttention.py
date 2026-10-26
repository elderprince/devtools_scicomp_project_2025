import torch.nn as nn

class BaseAttention(nn.Module):
    """
    Abstract attention class.
    Child classes must implement forward(q, k, v).
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        raise NotImplementedError
