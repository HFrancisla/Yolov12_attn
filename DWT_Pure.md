# DWT_Pure 完整迁移参考

> **用途**: 将 DWT_PureAttn 注意力机制迁移到其他 U-Net / Encoder-Decoder 网络时的完整代码参考。
> 本文档包含所有必需的组件代码，可直接复制使用。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [依赖项](#2-依赖项)
3. [文件 1: torch_wavelets.py — DWT/IDWT 基础算子](#3-文件-1-torch_waveletspy--dwtidwt-基础算子)
4. [文件 2: DWT_PureAttn — 核心注意力模块](#4-文件-2-dwt_pureattn--核心注意力模块)
5. [文件 3: 辅助模块 — LayerNorm / FeedForward / TransformerBlock](#5-文件-3-辅助模块--layernorm--feedforward--transformerblock)
6. [文件 4: 上下采样模块 — Downsample / Upsample](#6-文件-4-上下采样模块--downsample--upsample)
7. [文件 5: 完整 U-Net 骨架 — Restormer](#7-文件-5-完整-u-net-骨架--restormer)
8. [YAML 配置示例](#8-yaml-配置示例)
9. [迁移检查清单](#9-迁移检查清单)
10. [张量形状速查](#10-张量形状速查)

---

## 1. 整体架构概览

```
输入图像 (B, 3, H, W)
    │
    ▼
┌─────────────────────┐
│  OverlapPatchEmbed   │  3×3 Conv → (B, dim, H, W)        dim=48
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Encoder Level 1    │  N × TransformerBlock(DWT_PureAttn) dim=48
│                     │
│  out_enc_level1 ────────────────────────────────────┐ skip connection
└─────────┬───────────┘                               │
          ▼ Downsample (PixelUnshuffle)               │
┌─────────────────────┐                               │
│   Encoder Level 2    │  N × TransformerBlock          dim=96
│                     │
│  out_enc_level2 ───────────────────────────┐ skip    │
└─────────┬───────────┘                      │         │
          ▼ Downsample                       │         │
┌─────────────────────┐                      │         │
│   Encoder Level 3    │  N × TransformerBlock  dim=192 │
│                     │                                │
│  out_enc_level3 ──────────────────┐ skip   │         │
└─────────┬───────────┘             │        │         │
          ▼ Downsample              │        │         │
┌─────────────────────┐             │        │         │
│   Latent (Level 4)   │  N × TB     dim=384 │         │
└─────────┬───────────┘             │        │         │
          ▼ Upsample (PixelShuffle) │        │         │
┌─────────────────────┐             │        │         │
│   Decoder Level 3    │  cat + 1×1 + N×TB   │         │
│                     │◄────────────┘        │         │
└─────────┬───────────┘                      │         │
          ▼ Upsample                         │         │
┌─────────────────────┐                      │         │
│   Decoder Level 2    │  cat + 1×1 + N×TB   │         │
│                     │◄─────────────────────┘         │
└─────────┬───────────┘                                │
          ▼ Upsample                                   │
┌─────────────────────┐                                │
│   Decoder Level 1    │  cat (无 1×1) + N×TB  dim=96  │
│                     │◄───────────────────────────────┘
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Refinement         │  N × TransformerBlock  dim=96
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Output Conv 3×3    │  96 → 3
└─────────┬───────────┘
          ▼
     output + inp_img   (全局残差连接)
```

**关键点**:
- 每个 TransformerBlock 内部: `x = x + Attn(LN(x))` → `x = x + FFN(LN(x))`
- DWT_PureAttn 替换的是上面的 `Attn` 部分
- 下采样: `Conv2d(C→C//2) + PixelUnshuffle(2)` → 通道 ×2, 空间 ÷2
- 上采样: `Conv2d(C→C*2) + PixelShuffle(2)` → 通道 ÷2, 空间 ×2

---

## 2. 依赖项

```
pip install pywt einops
```

| 包名 | 用途 |
|------|------|
| `pywt` | 生成 Haar 小波的分解/重构滤波器系数 |
| `einops` | `rearrange` 用于多头注意力的维度变换 |
| `torch` | PyTorch 核心 |

---

## 3. 文件 1: torch_wavelets.py — DWT/IDWT 基础算子

> **此文件可直接复制到目标项目中**, 无需修改。
> 它提供 `DWT_2D` 和 `IDWT_2D` 两个 `nn.Module`。

```python
# torch_wavelets.py
# 基于 PyWavelets 的 2D 离散小波变换 (Haar)，支持 autograd

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DWT_Function(Function):
    """2D DWT 前向: (B, C, H, W) → (B, 4C, H/2, W/2)
       输出通道排列: [LL, LH, HL, HH] 各 C 个通道
    """
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        w_ll = w_ll.to(x.dtype)
        w_lh = w_lh.to(x.dtype)
        w_hl = w_hl.to(x.dtype)
        w_hh = w_hh.to(x.dtype)

        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).to(dx.dtype).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class IDWT_Function(Function):
    """2D IDWT 前向: (B, 4C, H/2, W/2) → (B, C, H, W)"""
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.to(x.dtype).repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters.to(dx.dtype), dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class DWT_2D(nn.Module):
    """2D 离散小波变换模块
    
    Args:
        wave: 小波类型, 如 "haar"
    
    输入:  (B, C, H, W)     — H, W 必须为偶数
    输出:  (B, 4C, H/2, W/2) — 通道排列: [LL, LH, HL, HH]
    """
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_2D(nn.Module):
    """2D 逆离散小波变换模块
    
    Args:
        wave: 小波类型, 如 "haar"
    
    输入:  (B, 4C, H/2, W/2) — 通道排列必须为 [LL, LH, HL, HH]
    输出:  (B, C, H, W)
    """
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)
```

---

## 4. 文件 2: DWT_PureAttn — 核心注意力模块

> **这是你需要迁移的核心模块。**
> 它是一个即插即用的注意力模块，接口: `(B, C, H, W) → (B, C, H, W)`，可替换任何同接口的注意力层。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_wavelets import DWT_2D, IDWT_2D  # ← 调整为你的实际导入路径


class DWT_PureAttn(nn.Module):
    """DWT-Frequency Self-Attention (DWT-FSA)
    
    在 Haar 小波域的 4 个子带 (LL, LH, HL, HH) 上分别执行
    MDTA 风格的转置通道注意力 (C×C attention)，然后 IDWT 重建回空间域。
    
    Args:
        dim:       输入通道数 C
        num_heads: 多头注意力的头数 (必须能整除 dim)
        bias:      卷积层是否使用 bias
    
    输入:  (B, C, H, W)  — H, W 可以是任意尺寸 (奇数会自动 pad)
    输出:  (B, C, H, W)
    """
    def __init__(self, dim, num_heads, bias):
        super(DWT_PureAttn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # ──── Q 投影: 1×1 conv + 3×3 depthwise conv ────
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )

        # ──── K 投影: 1×1 conv + 3×3 depthwise conv ────
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )

        # ──── V 投影: 1×1 conv + 3×3 depthwise conv ────
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )

        # ──── DWT / IDWT (Haar 小波) ────
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        # ──── 输出 1×1 conv ────
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # ━━━━ Step 1: Q, K, V 投影 ━━━━
        # 1×1 pointwise → 3×3 depthwise (局部空间信息注入)
        q = self.q_dwconv(self.q_conv(x))   # (B, C, H, W)
        k = self.k_dwconv(self.k_conv(x))   # (B, C, H, W)
        v = self.v_dwconv(self.v_conv(x))   # (B, C, H, W)

        # ━━━━ Step 2: Pad 到偶数尺寸 (DWT 要求) ━━━━
        pad_h = h % 2   # 0 或 1
        pad_w = w % 2   # 0 或 1
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode="reflect")
            k = F.pad(k, (0, pad_w, 0, pad_h), mode="reflect")
            v = F.pad(v, (0, pad_w, 0, pad_h), mode="reflect")

        # ━━━━ Step 3: DWT 分解 ━━━━
        # (B, C, H', W') → (B, 4C, H'/2, W'/2)
        # 其中 H' = H + pad_h, W' = W + pad_w
        q_dwt = self.dwt(q)
        k_dwt = self.dwt(k)
        v_dwt = self.dwt(v)

        # ━━━━ Step 4: 拆分为 4 个子带 ━━━━
        # 每个子带: (B, C, H'/2, W'/2)
        # 顺序: LL (低频), LH (水平细节), HL (垂直细节), HH (对角细节)
        q_subs = q_dwt.chunk(4, dim=1)
        k_subs = k_dwt.chunk(4, dim=1)
        v_subs = v_dwt.chunk(4, dim=1)

        h2, w2 = (h + pad_h) // 2, (w + pad_w) // 2
        attended_subs = []

        # ━━━━ Step 5: 对每个子带独立做 MDTA 风格通道注意力 ━━━━
        for q_s, k_s, v_s in zip(q_subs, k_subs, v_subs):
            # 重排为多头格式: (B, C, H/2, W/2) → (B, head, C//head, H*W/4)
            q_s = rearrange(
                q_s, "b (head c) h w -> b head c (h w)", head=self.num_heads
            )
            k_s = rearrange(
                k_s, "b (head c) h w -> b head c (h w)", head=self.num_heads
            )
            v_s = rearrange(
                v_s, "b (head c) h w -> b head c (h w)", head=self.num_heads
            )

            # L2 归一化 Q, K (沿空间维度)
            q_s = torch.nn.functional.normalize(q_s, dim=-1)
            k_s = torch.nn.functional.normalize(k_s, dim=-1)

            # 转置注意力: (C//head × C//head) 通道亲和矩阵
            attn = (q_s @ k_s.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)

            # 注意力加权 V
            out_s = attn @ v_s

            # 恢复空间维度: (B, head, C//head, H*W/4) → (B, C, H/2, W/2)
            out_s = rearrange(
                out_s,
                "b head c (h w) -> b (head c) h w",
                head=self.num_heads,
                h=h2,
                w=w2,
            )
            attended_subs.append(out_s)

        # ━━━━ Step 6: 合并 4 个子带 ━━━━
        # (B, C, H/2, W/2) × 4 → (B, 4C, H/2, W/2)
        out_dwt = torch.cat(attended_subs, dim=1)

        # ━━━━ Step 7: IDWT 重建 ━━━━
        # (B, 4C, H/2, W/2) → (B, C, H', W')
        out = self.idwt(out_dwt)

        # ━━━━ Step 8: 裁剪回原始尺寸 ━━━━
        if pad_h or pad_w:
            out = out[:, :, :h, :w]

        # ━━━━ Step 9: 输出投影 ━━━━
        out = self.project_out(out)   # (B, C, H, W)
        return out
```

### 数据流图

```
输入 x: (B, C, H, W)
    │
    ├─→ q = dwconv(conv1x1(x))  ──┐
    ├─→ k = dwconv(conv1x1(x))  ──┤
    └─→ v = dwconv(conv1x1(x))  ──┤
                                   │
                        [Pad to even H, W]
                                   │
                        ┌──────────┴──────────┐
                        │     DWT (Haar)       │
                        │  (B,C,H,W)→(B,4C,H/2,W/2) │
                        └──────────┬──────────┘
                                   │
                    chunk(4, dim=1) — 拆分 4 子带
                    ┌──────┬───────┬───────┐
                   LL     LH     HL     HH
                    │      │      │       │
              ┌─────┴──────┴──────┴───────┴─────┐
              │  对每个子带独立执行:                │
              │  1. rearrange → 多头格式           │
              │  2. normalize(Q, K)               │
              │  3. attn = softmax(Q·Kᵀ / τ)     │
              │  4. out = attn · V                │
              │  5. rearrange → (B, C, H/2, W/2) │
              └─────┬──────┬──────┬───────┬─────┘
                   LL'    LH'    HL'    HH'
                    │      │      │       │
                    └──────┴──────┴───────┘
                              │
                        cat(dim=1) → (B, 4C, H/2, W/2)
                              │
                        ┌─────┴─────┐
                        │ IDWT (Haar)│
                        │ → (B,C,H,W)│
                        └─────┬─────┘
                              │
                        [Crop if padded]
                              │
                        conv1x1 (project_out)
                              │
                        输出: (B, C, H, W)
```

---

## 5. 文件 3: 辅助模块 — LayerNorm / FeedForward / TransformerBlock

> TransformerBlock 是将 DWT_PureAttn 包装成可堆叠单元的容器。

```python
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


# ==================== LayerNorm ====================

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensor.
    
    内部先 reshape 为 (B, HW, C) 做 LN, 再 reshape 回来。
    """
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ==================== GDFN (Gated-Dconv Feed-Forward Network) ====================

class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network
    
    结构: Conv1×1(C→2H) → DWConv3×3 → GELU门控 → Conv1×1(H→C)
    其中 H = C × ffn_expansion_factor
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2       # 门控机制
        x = self.project_out(x)
        return x


# ==================== TransformerBlock ====================

class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block
    
    结构: x → LN → Attention → 残差 → LN → FFN → 残差
    
    Args:
        dim:                  通道数
        num_heads:            注意力头数
        ffn_expansion_factor: FFN 隐藏层扩展倍数 (默认 2.66)
        bias:                 是否使用 bias
        LayerNorm_type:       "WithBias" 或 "BiasFree"
        use_checkpoint:       是否使用梯度检查点节省显存
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type, use_checkpoint=False):
        super(TransformerBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DWT_PureAttn(dim, num_heads, bias)   # ← 核心: 使用 DWT_Pure
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))   # 残差 + 注意力
        x = x + self.ffn(self.norm2(x))    # 残差 + FFN
        return x
```

---

## 6. 文件 4: 上下采样模块 — Downsample / Upsample

```python
import torch.nn as nn


class Downsample(nn.Module):
    """下采样: 空间 ÷2, 通道 ×2
    
    结构: Conv2d(C → C//2, 3×3) → PixelUnshuffle(2)
    
    PixelUnshuffle(2) 将 (B, C//2, H, W) → (B, C//2 × 4, H/2, W/2) = (B, 2C, H/2, W/2)
    
    输入: (B, C, H, W)
    输出: (B, 2C, H/2, W/2)
    """
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """上采样: 空间 ×2, 通道 ÷2
    
    结构: Conv2d(C → C×2, 3×3) → PixelShuffle(2)
    
    PixelShuffle(2) 将 (B, C×2, H, W) → (B, C×2 / 4, H×2, W×2) = (B, C/2, H×2, W×2)
    
    输入: (B, C, H, W)
    输出: (B, C/2, H×2, W×2)
    """
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
```

---

## 7. 文件 5: 完整 U-Net 骨架 — Restormer

> 以下为 Restormer 完整 `__init__` + `forward`，展示 DWT_Pure 如何嵌入 4 级 U-Net。
> 迁移时可参考此结构中的 **skip connection** 和 **通道变化** 逻辑。

```python
import torch
import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    """输入嵌入: 3×3 Conv 将 RGB 映射到 dim 维特征"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,                                    # 基础通道数
        num_blocks=[4, 6, 6, 8],                   # 每层 TransformerBlock 数量
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],                        # 每层注意力头数
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        use_checkpoint=False,
    ):
        super(Restormer, self).__init__()

        # ──── 输入嵌入 ────
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ════════════════ Encoder ════════════════
        # Level 1: dim=48
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[0])
        ])

        # Level 1 → 2: 48 → 96
        self.down1_2 = Downsample(dim)

        # Level 2: dim=96
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[1])
        ])

        # Level 2 → 3: 96 → 192
        self.down2_3 = Downsample(dim * 2)

        # Level 3: dim=192
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[2])
        ])

        # Level 3 → 4: 192 → 384
        self.down3_4 = Downsample(dim * 4)

        # ════════════════ Bottleneck (Latent) ════════════════
        # Level 4: dim=384
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[3])
        ])

        # ════════════════ Decoder ════════════════
        # Level 4 → 3: 384 → 192
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        #                         ↑ cat后通道=384, 用 1×1 conv 降回 192

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[2])
        ])

        # Level 3 → 2: 192 → 96
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[1])
        ])

        # Level 2 → 1: 96 → 48, cat 后变 96 (不用 1×1 降通道)
        self.up2_1 = Upsample(dim * 2)

        # Level 1 decoder: dim=96 (因为 cat 了 encoder_level1 的 48)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_blocks[0])
        ])

        # ──── Refinement ────
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias,
                             LayerNorm_type, use_checkpoint)
            for _ in range(num_refinement_blocks)
        ])

        # ──── 输出层 ────
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # ──── Encoder ────
        inp_enc_level1 = self.patch_embed(inp_img)           # (B, 48, H, W)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # (B, 48, H, W)

        inp_enc_level2 = self.down1_2(out_enc_level1)         # (B, 96, H/2, W/2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # (B, 96, H/2, W/2)

        inp_enc_level3 = self.down2_3(out_enc_level2)         # (B, 192, H/4, W/4)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # (B, 192, H/4, W/4)

        inp_enc_level4 = self.down3_4(out_enc_level3)         # (B, 384, H/8, W/8)
        latent = self.latent(inp_enc_level4)                  # (B, 384, H/8, W/8)

        # ──── Decoder ────
        inp_dec_level3 = self.up4_3(latent)                              # (B, 192, H/4, W/4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # (B, 384, H/4, W/4)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)          # (B, 192, H/4, W/4)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)             # (B, 192, H/4, W/4)

        inp_dec_level2 = self.up3_2(out_dec_level3)                      # (B, 96, H/2, W/2)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # (B, 192, H/2, W/2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)          # (B, 96, H/2, W/2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)             # (B, 96, H/2, W/2)

        inp_dec_level1 = self.up2_1(out_dec_level2)                      # (B, 48, H, W)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # (B, 96, H, W)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)             # (B, 96, H, W)

        out_dec_level1 = self.refinement(out_dec_level1)                 # (B, 96, H, W)

        # ──── 输出 + 全局残差 ────
        out_dec_level1 = self.output(out_dec_level1) + inp_img           # (B, 3, H, W)
        return out_dec_level1
```

---

## 8. YAML 配置示例

```yaml
network_g:
  type: Restormer
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: WithBias
  attn_types: ["DWT_Pure", "DWT_Pure", "DWT_Pure", "DWT_Pure"]
  use_checkpoint: True
```

---

## 9. 迁移检查清单

### 必须完成

- [ ] **复制 `torch_wavelets.py`** 到目标项目, 确保 `DWT_2D` 和 `IDWT_2D` 可正常导入
- [ ] **安装依赖**: `pip install pywt einops`
- [ ] **复制 `DWT_PureAttn` 类** 到目标网络的 attention 模块文件
- [ ] **调整导入路径**: `from xxx import DWT_2D, IDWT_2D`
- [ ] **确保 `dim` 能被 `num_heads` 整除**: dim % num_heads == 0
- [ ] **在 TransformerBlock 中替换注意力层**: `self.attn = DWT_PureAttn(dim, num_heads, bias)`

### 注意事项

- [ ] DWT_PureAttn 的 **输入输出形状不变**: (B, C, H, W) → (B, C, H, W)，因此可直接替换任何同形状的注意力模块
- [ ] 奇数空间尺寸会自动 **reflect-pad** 到偶数，无需手动处理
- [ ] 小波滤波器通过 `register_buffer` 注册，**不参与梯度更新**，但会随模型 `.cuda()` / `.to()` 自动迁移设备
- [ ] 如果目标网络不使用 Restormer 的 LayerNorm，可用 PyTorch 原生 `nn.LayerNorm` 替代
- [ ] `temperature` 是**可学习参数**，所有 4 个子带共享同一个 temperature

### 可选调整

- [ ] 更换小波基: 将 `wave="haar"` 改为 `"db2"`, `"sym4"` 等 (滤波器尺寸会变大，需相应 pad)
- [ ] 子带独立 temperature: 可改为 4 个独立的 `nn.Parameter`
- [ ] 移除某些子带: 例如只保留 LL 子带做注意力，其他子带直接 pass-through

---

## 10. 张量形状速查

以 `dim=48, num_heads=1, H=128, W=128` 为例:

| 阶段 | 张量 | 形状 |
|------|------|------|
| 输入 | `x` | `(B, 48, 128, 128)` |
| Q/K/V 投影后 | `q, k, v` | `(B, 48, 128, 128)` |
| DWT 后 | `q_dwt` | `(B, 192, 64, 64)` |
| chunk 后每个子带 | `q_s` | `(B, 48, 64, 64)` |
| rearrange 多头 | `q_s` | `(B, 1, 48, 4096)` |
| 注意力矩阵 | `attn` | `(B, 1, 48, 48)` |
| 注意力输出 | `out_s` | `(B, 1, 48, 4096)` |
| rearrange 回 | `out_s` | `(B, 48, 64, 64)` |
| cat 4 子带 | `out_dwt` | `(B, 192, 64, 64)` |
| IDWT 后 | `out` | `(B, 48, 128, 128)` |
| 最终输出 | `out` | `(B, 48, 128, 128)` |

以 `dim=384, num_heads=8, H=16, W=16` (Latent 层) 为例:

| 阶段 | 张量 | 形状 |
|------|------|------|
| 输入 | `x` | `(B, 384, 16, 16)` |
| DWT 后 | `q_dwt` | `(B, 1536, 8, 8)` |
| chunk 后每个子带 | `q_s` | `(B, 384, 8, 8)` |
| rearrange 多头 | `q_s` | `(B, 8, 48, 64)` |
| 注意力矩阵 | `attn` | `(B, 8, 48, 48)` |
