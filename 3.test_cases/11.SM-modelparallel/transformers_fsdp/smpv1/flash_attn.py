from typing import Optional, Tuple

import torch
import torch.nn as nn
from flash_attn.flash_attn_triton import flash_attn_qkvpacked_func
from torch import Tensor


class TritonFlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale

    def forward(
        self, qkv, attn_mask: Optional[Tensor] = None, causal: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:

        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if attn_mask is not None and attn_mask.requires_grad:
            attn_mask = attn_mask.detach()

        attn_output = flash_attn_qkvpacked_func(qkv, attn_mask, causal, self.softmax_scale)
        return attn_output, None
