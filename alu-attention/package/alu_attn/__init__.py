"""alu_attn — drop-in flash-attn implementation for AMD RDNA2/gfx1030 on ROCm.

Exact FlashAttention (forward, backward, and decode) on HIP `v_dot2_f32_f16` kernels with a
flash-attn-compatible entry points. Unsupported combinations are explicitly rejected.
See docs/CONCURRENCY.md for the alpha candidate scope. For supported configurations:

    import alu_attn; alu_attn.install_as_flash_attn()   # `import flash_attn` now resolves here

or import the API directly:

    from alu_attn import flash_attn_func
"""
import sys

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_combine,
    flash_attn_with_kvcache,
)
from .paged import paged_attention, create_decode_workspace, DecodeWorkspace
from . import bert_padding
from . import torch_op

# Register the optional torch operator used by framework attention integrations.
torch_op.register()

__all__ = [
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_combine",
    "flash_attn_with_kvcache",
    "bert_padding",
    "paged_attention",
    "create_decode_workspace",
    "DecodeWorkspace",
    "install_as_flash_attn",
]

# Native ALU Attention package version.
__version__ = "0.6.3"

# Version advertised when this package stands in for flash_attn. This is part of the drop-in API
# contract, not a claim of authorship: downstream libraries feature-gate behavior by version
# (`parse(flash_attn.__version__) >= parse("2.4.2")` is a common check before passing window_size
# or softcap). Advertising only "0.2.0" would make consumers silently disable features that are
# implemented here. The value advertises FA-2-style feature gates, not complete behavioral coverage; the
# local +alu suffix keeps the alternative implementation visible.
FLASH_ATTN_API_VERSION = "2.8.3+alu." + __version__


def install_as_flash_attn():
    """Register the package under the import names expected by third-party code.

      flash_attn                        — FA-2 argument ordering
      flash_attn.flash_attn_interface   — the same API as a submodule
      flash_attn.bert_padding           — unpad_input/pad_input compatibility
      flash_attn_interface              — FA-3 argument ordering (see fa3_interface)

    FA-2 and FA-3 intentionally remain separate because their positional argument order differs:
    the fourth FA-2 argument is `dropout_p`, while the fourth FA-3 argument is `softmax_scale`.
    Combining them into one function could silently compute the wrong operation.
    """
    from . import flash_attn_interface as _fi
    from . import fa3_interface as _fi3
    from . import bert_padding as _bp
    from . import layers as _layers
    from .layers import rotary as _rotary
    from . import modules as _modules
    from .modules import mha as _mha
    from . import ops as _ops
    from .ops import triton as _triton
    from .ops.triton import layer_norm as _tln
    from .ops import rms_norm as _rms, layer_norm as _ln

    me = sys.modules[__name__]
    me.__dict__.setdefault("_alu_own_version", __version__)
    me.__version__ = FLASH_ATTN_API_VERSION      # See the version-contract comment above.
    _fi.__version__ = FLASH_ATTN_API_VERSION

    sys.modules["flash_attn"] = me
    sys.modules["flash_attn.flash_attn_interface"] = _fi
    sys.modules["flash_attn.bert_padding"] = _bp
    # Third-party code imports these submodules directly. They are functional torch
    # implementations, not placeholders, and are required for import-level drop-in compatibility.
    sys.modules["flash_attn.layers"] = _layers
    sys.modules["flash_attn.layers.rotary"] = _rotary
    sys.modules["flash_attn.modules"] = _modules
    sys.modules["flash_attn.modules.mha"] = _mha
    sys.modules["flash_attn.ops"] = _ops
    sys.modules["flash_attn.ops.triton"] = _triton
    sys.modules["flash_attn.ops.triton.layer_norm"] = _tln
    sys.modules["flash_attn.ops.rms_norm"] = _rms
    sys.modules["flash_attn.ops.layer_norm"] = _ln
    sys.modules["flash_attn_interface"] = _fi3
    return me
