# 基于 Haar DWT 的上下采样模块

> 从 Restormer_LLIE 项目中剥离，方便移植到其他 U-Net / Encoder-Decoder 网络。

---

## 1. 核心思想

本项目将其替换为 **Haar 离散小波变换 (DWT)** 来下采样、**Haar 逆离散小波变换 (IDWT)** 来上采样，主要优势：

| 对比项   | PixelUnshuffle       | Haar DWT                                     |
| -------- | -------------------- | -------------------------------------------- |
| 频率分解 | 无，仅做像素重排     | 分解出 LL / LH / HL / HH 四个子带            |
| 信息保留 | 简单 rearrange       | 显式保留低频近似 + 三方向高频细节            |
| 可逆性   | pixel_shuffle 即可逆 | IDWT 是精确逆变换                            |
| 计算量   | reshape              | reshape + 四次加减法（无卷积），速度更快     |
| 适用场景 | 通用                 | 对频域细节敏感的任务（去噪、低光增强等）更优 |

### 1.1 数据流

**下采样 (`DWT_Downsample`)**：
```
(B, C, H, W) ──DWT──▶ (B, 4C, H/2, W/2) ──1×1 Conv──▶ (B, 2C, H/2, W/2)
```
- DWT 将空间缩小 2× 的同时通道扩展 4×（LL+LH+HL+HH 拼接）
- 1×1 卷积将 4C 投影到 2C，与原始 Restormer 的通道翻倍策略保持一致

**上采样 (`IDWT_Upsample`)**：
```
(B, C, H/2, W/2) ──1×1 Conv──▶ (B, 2C, H/2, W/2) ──IDWT──▶ (B, C/2, H, W)
```
- 1×1 卷积将 C 扩展到 2C（使 IDWT 除以 4 后恰好输出 C/2 通道）
- IDWT 将空间放大 2×，通道缩小 4×

### 1.2 Haar 小波公式

**正变换（2D DWT）**：将 2×2 像素块 `(ee, eo, oe, oo)` 分解为四个子带：

```
LL = 0.5 × (ee + eo + oe + oo)    # 低频近似（均值）
LH = 0.5 × (ee - eo + oe - oo)    # 水平高频
HL = 0.5 × (ee + eo - oe - oo)    # 垂直高频
HH = 0.5 × (ee - eo - oe + oo)    # 对角高频
```

**逆变换（2D IDWT）**：从四个子带恢复 2×2 像素块：

```
ee = 0.5 × (LL + LH + HL + HH)
eo = 0.5 × (LL - LH + HL - HH)
oe = 0.5 × (LL + LH - HL - HH)
oo = 0.5 × (LL - LH - HL + HH)
```

---

## 2. 完整可移植代码

### 2.1 `haar_wavelets.py` — DWT / IDWT 基础算子

```python
"""Haar DWT / IDWT implemented via pixel_unshuffle / pixel_shuffle.

Drop-in replacement for conv2d-based DWT.
Advantages:
  - No convolution: uses only reshape + elementwise add/sub → 3-5× faster.
  - No custom autograd Function: standard ops, autograd handles backward.
  - Lower memory: no conv intermediate buffers.
  - Numerically identical to the conv2d Haar implementation.

Limitation: only supports the Haar wavelet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT_2D(nn.Module):
    """Haar 2-D Discrete Wavelet Transform via ``pixel_unshuffle``.

    Input : (B, C, H, W)   — H, W must be even.
    Output: (B, 4C, H/2, W/2) — channel order [LL, LH, HL, HH] per input channel.
    """

    def __init__(self, wave="haar"):
        super(DWT_2D, self).__init__()
        if wave != "haar":
            raise ValueError(
                f"Only 'haar' wavelet is supported by this module, got '{wave}'"
            )

    def forward(self, x):
        # pixel_unshuffle: (B, C, H, W) → (B, 4C, H/2, W/2)
        # Per-channel sub-pixel order: [ee, eo, oe, oo]
        #   ee = x[..., 0::2, 0::2]   eo = x[..., 0::2, 1::2]
        #   oe = x[..., 1::2, 0::2]   oo = x[..., 1::2, 1::2]
        xu = F.pixel_unshuffle(x, 2)
        B, C4, H2, W2 = xu.shape
        C = C4 // 4
        xu = xu.view(B, C, 4, H2, W2)
        x_ee = xu[:, :, 0]
        x_eo = xu[:, :, 1]
        x_oe = xu[:, :, 2]
        x_oo = xu[:, :, 3]

        # Haar DWT
        ll = (x_ee + x_eo + x_oe + x_oo) * 0.5
        lh = (x_ee - x_eo + x_oe - x_oo) * 0.5
        hl = (x_ee + x_eo - x_oe - x_oo) * 0.5
        hh = (x_ee - x_eo - x_oe + x_oo) * 0.5

        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT_2D(nn.Module):
    """Haar 2-D Inverse Discrete Wavelet Transform via ``pixel_shuffle``.

    Input : (B, 4C, H/2, W/2) — channel order [LL, LH, HL, HH] per output channel.
    Output: (B, C, H, W)
    """

    def __init__(self, wave="haar"):
        super(IDWT_2D, self).__init__()
        if wave != "haar":
            raise ValueError(
                f"Only 'haar' wavelet is supported by this module, got '{wave}'"
            )

    def forward(self, x):
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        # Input layout: [all_LL, all_LH, all_HL, all_HH] (C channels each)
        x = x.view(B, 4, C, H2, W2)
        ll = x[:, 0]  # (B, C, H2, W2)
        lh = x[:, 1]
        hl = x[:, 2]
        hh = x[:, 3]

        # Inverse Haar transform — recover sub-pixel values.
        x_ee = (ll + lh + hl + hh) * 0.5
        x_eo = (ll - lh + hl - hh) * 0.5
        x_oe = (ll + lh - hl - hh) * 0.5
        x_oo = (ll - lh - hl + hh) * 0.5

        # Per-channel interleaved order for pixel_shuffle
        x = torch.stack([x_ee, x_eo, x_oe, x_oo], dim=2)  # (B, C, 4, H2, W2)
        x = x.view(B, C * 4, H2, W2)
        return F.pixel_shuffle(x, 2)
```

### 2.2 `dwt_sampling.py` — 网络中直接使用的上下采样模块

```python
import torch.nn as nn
from haar_wavelets import DWT_2D, IDWT_2D  # 调整 import 路径以适配你的项目


class DWT_Downsample(nn.Module):
    """DWT-based downsampling with channel projection: C → 2C, spatial ÷2.

    用法: 替换原有的 stride-2 卷积 / PixelUnshuffle 下采样。

    Args:
        n_feat: 输入通道数 C。输出通道数为 2C。
        wave:   小波类型，目前仅支持 'haar'。
    """

    def __init__(self, n_feat, wave="haar"):
        super(DWT_Downsample, self).__init__()
        self.dwt = DWT_2D(wave)
        # DWT produces 4C channels; reduce to 2C to keep memory manageable
        self.proj = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W) → DWT → (B, 4C, H/2, W/2) → proj → (B, 2C, H/2, W/2)
        return self.proj(self.dwt(x))


class IDWT_Upsample(nn.Module):
    """IDWT-based upsampling with channel projection: C → C/2, spatial ×2.

    用法: 替换原有的 PixelShuffle / 反卷积上采样。

    Args:
        n_feat: 输入通道数 C。输出通道数为 C/2。
        wave:   小波类型，目前仅支持 'haar'。
    """

    def __init__(self, n_feat, wave="haar"):
        super(IDWT_Upsample, self).__init__()
        # Expand n_feat → 2*n_feat so IDWT (which divides channels by 4) outputs n_feat/2
        self.proj = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=False)
        self.idwt = IDWT_2D(wave)

    def forward(self, x):
        # x: (B, C, H/2, W/2) → proj → (B, 2C, H/2, W/2) → IDWT → (B, C/2, H, W)
        return self.idwt(self.proj(x))
```

---

## 3. 移植指南

### 3.1 快速替换步骤

1. **复制两个文件**到目标项目中（`haar_wavelets.py` + `dwt_sampling.py`）
2. **修改 import 路径**：让 `dwt_sampling.py` 能正确 import `DWT_2D` / `IDWT_2D`
3. **在网络中替换下采样层**：
   ```python
   # Before (PixelUnshuffle)
   self.down = nn.Sequential(
       nn.Conv2d(C, C // 2, 3, 1, 1, bias=False),
       nn.PixelUnshuffle(2),
   )
   # After (DWT)
   self.down = DWT_Downsample(C)
   ```
4. **在网络中替换上采样层**：
   ```python
   # Before (PixelShuffle)
   self.up = nn.Sequential(
       nn.Conv2d(C, C * 2, 3, 1, 1, bias=False),
       nn.PixelShuffle(2),
   )
   # After (IDWT)
   self.up = IDWT_Upsample(C)
   ```

### 3.2 通道数变换规则

| 模块                | 输入 shape         | 输出 shape          |
| ------------------- | ------------------ | ------------------- |
| `DWT_Downsample(C)` | `(B, C, H, W)`     | `(B, 2C, H/2, W/2)` |
| `IDWT_Upsample(C)`  | `(B, C, H/2, W/2)` | `(B, C/2, H, W)`    |

如果你的网络通道倍增策略不同（例如希望 `C → C` 而不是 `C → 2C`），修改 `proj` 的输出通道数即可。

### 3.3 注意事项

- **空间尺寸**必须为偶数（H, W 均为 2 的倍数），否则 `pixel_unshuffle` 会报错
- **仅支持 Haar 小波**。如需 db2、sym4 等其他小波，需要替换为基于卷积滤波器组的实现
- `DWT_2D` / `IDWT_2D` **无可学习参数**（纯数学变换），所有可学习性集中在 `proj`（1×1 卷积）
- 该实现利用 `pixel_unshuffle` / `pixel_shuffle` 做像素重排，避免了自定义 CUDA kernel，**兼容所有 PyTorch ≥ 1.7 版本**

### 3.4 依赖

```
torch >= 1.7.0
```

无其他第三方依赖。

---

## 4. 在原项目中的使用示例

在 Restormer 4 层 Encoder-Decoder 中，DWT/IDWT 用于所有层间连接：

```python
# Encoder: 每层之间用 DWT 下采样
self.down1_2 = DWT_Downsample(dim)           # dim   → 2*dim
self.down2_3 = DWT_Downsample(dim * 2)       # 2*dim → 4*dim
self.down3_4 = DWT_Downsample(dim * 4)       # 4*dim → 8*dim

# Decoder: 每层之间用 IDWT 上采样
self.up4_3 = IDWT_Upsample(dim * 8)          # 8*dim → 4*dim
self.up3_2 = IDWT_Upsample(dim * 4)          # 4*dim → 2*dim
self.up2_1 = IDWT_Upsample(dim * 2)          # 2*dim → dim
```

注意：该网络**没有使用 skip connection**（encoder → decoder 的跳跃连接），仅做纯粹的 down → latent → up 结构。如果你需要 skip connection，需在 decoder 侧拼接/相加 encoder 特征后，添加 1×1 卷积调整通道数。
