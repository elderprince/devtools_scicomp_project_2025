"""
Attention Registry

With different constructor signatures (e.g. full, strided, flash), this centralized registry for 
all attention mechanisms can simply instantiate attention modules from the configuration.
"""

from typing import Dict, Type
from transformer.model import BaseAttention
import inspect

_ATTENTION_REGISTRY: Dict[str, Type[BaseAttention]] = {}


def register_attention(name: str):
    """
    Decorator to register an attention module.

    Example:
        @register_attention("flash")
        class FlashAttention(nn.Module):
            ...
    """
    def decorator(cls: Type[BaseAttention]) -> Type[BaseAttention]:
        if name in _ATTENTION_REGISTRY.keys():
            raise ValueError(f"Attention '{name}' already registered")
        _ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator

def build_attention(cfg: Dict) -> Type[BaseAttention]:
    """
    Build an attention module from config.

    Args:
        cfg (dict): Must contain at least:
            {
                "type": "flash" | "full" | "strided" | ...
                ... attention-specific parameters ...
            }

    Returns:
        nn.Module: instantiated attention module
    """
    if "type" not in cfg:
        raise KeyError("Attention config must contain key 'type'")

    attn_type = cfg["type"]

    if attn_type not in _ATTENTION_REGISTRY:
        raise KeyError(
            f"Unknown attention type '{attn_type}'. "
            f"Available: {list(_ATTENTION_REGISTRY.keys())}"
        )

    attn_cls = _ATTENTION_REGISTRY[attn_type]

    # Filter cfg to only accepted constructor arguments
    sig = inspect.signature(attn_cls.__init__)
    valid_args = {
        k: v for k, v in cfg.items()
        if k in sig.parameters and k != "self"
    }

    return attn_cls(**valid_args)

def list_available_attentions():
    """Return list of registered attention types."""
    return list(_ATTENTION_REGISTRY.keys())