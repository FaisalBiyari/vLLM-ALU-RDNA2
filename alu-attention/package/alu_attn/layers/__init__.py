"""flash_attn.layers compatibility modules.

Third-party code imports these directly (for example, `from flash_attn.layers.rotary import
apply_rotary_emb`). The implementations here use pure torch so drop-in compatibility does not
fail at import time.
"""
from . import rotary

__all__ = ["rotary"]
