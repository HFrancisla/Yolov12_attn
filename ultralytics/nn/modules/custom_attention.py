# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Custom Attention Modules for YOLOv12.

This module provides various attention mechanisms that can replace the default AreaAttention:
- HTA: Height-wise Transposed Attention (W×W)
- WTA: Width-wise Transposed Attention (H×H)
- IRS: Intra-channel Row Self-attention (C×H×H)
- ICS: Intra-channel Column Self-attention (C×W×W)
- MDTA: Multi-DConv Head Transposed Self-Attention (standard multi-head)
"""

import torch
import torch.nn as nn
from einops import rearrange

from .conv import Conv

__all__ = ("HTA", "WTA", "IRS", "ICS", "MDTA", "CustomABlock")


class HTA(nn.Module):
    """
    Height-wise Transposed Attention (W×W attention matrix).
    
    Computes attention along the width dimension, allowing each position to attend
    to all positions in the same row across the width.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in convolutions. Default: False
        
    Examples:
        >>> import torch
        >>> model = HTA(dim=64, num_heads=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # (2, 64, 32, 32)
    """

    def __init__(self, dim, num_heads, bias=False):
        super(HTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        """Forward pass through HTA attention."""
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (c h) w", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # w×w
        attn = attn.softmax(dim=-2)

        out = v @ attn

        out = rearrange(
            out, "b head (c h) w -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class WTA(nn.Module):
    """
    Width-wise Transposed Attention (H×H attention matrix).
    
    Computes attention along the height dimension, allowing each position to attend
    to all positions in the same column across the height.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in convolutions. Default: False
        
    Examples:
        >>> import torch
        >>> model = WTA(dim=64, num_heads=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # (2, 64, 32, 32)
    """

    def __init__(self, dim, num_heads, bias=False):
        super(WTA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        """Forward pass through WTA attention."""
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        v1 = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        q1 = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k1 = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)

        out = attn1 @ v1
        out = rearrange(
            out, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


class IRS(nn.Module):
    """
    Intra-channel Row Self-attention (C×H×H attention per channel).
    
    Computes row-wise attention independently for each channel, allowing
    fine-grained spatial interactions within channels.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in convolutions. Default: True
        
    Examples:
        >>> import torch
        >>> model = IRS(dim=64, num_heads=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # (2, 64, 32, 32)
    """

    def __init__(self, dim, num_heads, bias=True):
        super(IRS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        """Forward pass through IRS attention."""
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Split into heads: [b, (head c), h, w] -> [b, head, c, h, w]
        q = rearrange(q, "b (head c) h w -> b head c h w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c h w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c h w", head=self.num_heads)

        # Normalize along width dimension for row-wise attention
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Row attention: H×H attention matrix per head per channel
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        # Merge heads back
        out = rearrange(out, "b head c h w -> b (head c) h w", head=self.num_heads)
        out = self.project_out(out)
        return out


class ICS(nn.Module):
    """
    Intra-channel Column Self-attention (C×W×W attention per channel).
    
    Computes column-wise attention independently for each channel, allowing
    fine-grained spatial interactions within channels.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in convolutions. Default: True
        
    Examples:
        >>> import torch
        >>> model = ICS(dim=64, num_heads=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # (2, 64, 32, 32)
    """

    def __init__(self, dim, num_heads, bias=True):
        super(ICS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        """Forward pass through ICS attention."""
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Split into heads: [b, (head c), h, w] -> [b, head, c, h, w]
        q = rearrange(q, "b (head c) h w -> b head c h w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c h w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c h w", head=self.num_heads)

        # Normalize along height dimension for column-wise attention
        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)

        # Column attention: W×W attention matrix per head per channel
        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-2)

        out = v @ attn

        # Merge heads back
        out = rearrange(out, "b head c h w -> b (head c) h w", head=self.num_heads)
        out = self.project_out(out)
        return out


class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention.
    
    Standard multi-head self-attention with depthwise convolution for Q, K, V projection.
    Computes global attention across all spatial positions.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in convolutions. Default: False
        
    Examples:
        >>> import torch
        >>> model = MDTA(dim=64, num_heads=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # (2, 64, 32, 32)
    """

    def __init__(self, dim, num_heads, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward pass through MDTA attention."""
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class CustomABlock(nn.Module):
    """
    Custom Attention Block with configurable attention mechanism.
    
    This block can use any of the custom attention mechanisms (HTA, WTA, IRS, ICS, MDTA)
    and includes a feed-forward MLP layer.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio. Default: 1.2
        attn_type (str): Type of attention mechanism. Options: 'HTA', 'WTA', 'IRS', 'ICS', 'MDTA'. Default: 'MDTA'
        bias (bool): Whether to use bias in attention layers. Default: False
        
    Examples:
        >>> import torch
        >>> model = CustomABlock(dim=64, num_heads=2, attn_type='HTA')
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # (2, 64, 32, 32)
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, attn_type="MDTA", bias=False):
        super().__init__()
        
        # Select attention mechanism
        attn_dict = {
            "HTA": HTA,
            "WTA": WTA,
            "IRS": IRS,
            "ICS": ICS,
            "MDTA": MDTA,
        }
        
        if attn_type not in attn_dict:
            raise ValueError(f"Unknown attention type: {attn_type}. Choose from {list(attn_dict.keys())}")
        
        self.attn = attn_dict[attn_type](dim, num_heads=num_heads, bias=bias)
        
        # MLP layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),
            Conv(mlp_hidden_dim, dim, 1, act=False)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through CustomABlock."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
