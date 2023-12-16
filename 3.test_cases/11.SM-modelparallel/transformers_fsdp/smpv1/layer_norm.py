# Third Party
import torch

from .core import local_rank

# First Party
apex_is_available = True
try:
    from smdistributed.modelparallel.torch.apex.normalization.fused_layer_norm import (
        FusedLayerNorm as ApexFusedLayerNorm,
    )
    from smdistributed.modelparallel.torch.apex.normalization.fused_layer_norm import (
        MixedFusedLayerNorm as ApexMixedFusedLayerNorm,
    )
except ImportError:
    apex_is_available = False
    print("Apex not available")

if apex_is_available:

    class FusedLayerNorm(ApexFusedLayerNorm):
        def forward(self, x):
            torch.cuda.set_device(local_rank())
            return super(FusedLayerNorm, self).forward(x)

    # From apex: Why "mixed"?
    # MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
    # as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
    # See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
    class MixedFusedLayerNorm(ApexMixedFusedLayerNorm):
        def forward(self, x):
            torch.cuda.set_device(local_rank())
            return super(MixedFusedLayerNorm, self).forward(x)
