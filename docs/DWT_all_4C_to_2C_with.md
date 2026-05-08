# DWT4C 上下采样方案 — 完整设计文档（可移植）

> 从 `restormer_arch.py` 和 `haar_wavelets.py` 中提取完整的上下采样模块、skip connection 机制和 forward 数据流，方便对照迁移到其他网络中。

---

## 1. 核心思想

本方案用 **Haar 离散小波变换 (DWT/IDWT)** 替代传统 PixelShuffle/PixelUnshuffle，关键特点：

- **下采样 (DWT_Downsample)**：DWT 将特征分解为 4C 通道（LL + LH/HL/HH），通过 1×1 Conv 融合 4C → 2C 作为编码器主路径，同时将高频 [LH, HL, HH]（3C）作为 **detail skip** 传给解码器。
- **上采样 (IDWT_Upsample)**：解码器特征经 1×1 Conv 降通道后作为 LL 分量，与编码器保存的高频 detail 拼接后通过 IDWT 逆变换恢复空间分辨率。
- **双 skip connection**：detail skip（高频，频域传递）+ encoder skip（特征，空间域 cat），比传统方式多了一条高频细节跳跃连接。

---

## 2. 整体数据流

```
Encoder                                         Decoder

Level 1: (B, C, H, W)                          Level 1: (B, 2C, H, W)
    |                                               ^
    | DWT_Downsample                                | IDWT_Upsample
    |  +-- 4C fuse -> (B,2C,H/2,W/2)               |  +-- x(decoder) -> channel_reduce -> LL
    |  +-- detail(LH,HL,HH) ----[detail skip]------>|  +-- + detail(LH,HL,HH) -> IDWT -> (B,C,H,W)
    v                                               |
Level 2: (B, 2C, H/2, W/2)                     Level 2: (B, 2C, H/2, W/2)
    |                                               ^
    | DWT_Downsample                                | IDWT_Upsample
    v                                               |
Level 3: (B, 4C, H/4, W/4)                     Level 3: (B, 4C, H/4, W/4)
    |                                               ^
    | DWT_Downsample                                | IDWT_Upsample
    v                                               |
Level 4 (Latent): (B, 8C, H/8, W/8) ------------>--+
```

**关键**：每次 DWT 下采样产生的 `detail` (3×输入通道) 需缓存，在对应层级的 IDWT 上采样时使用。解码器中 IDWT 上采样结果还会与编码器同层的输出进行常规 cat + 1×1 Conv 降通道。

---

## 3. Haar 小波数学原理

对于 2x2 像素块 `[ee, eo, oe, oo]`（行偶/奇 x 列偶/奇），Haar DWT 定义为：

```
正变换 (DWT):
  LL = (ee + eo + oe + oo) * 0.5    # 低频近似（均值）
  LH = (ee - eo + oe - oo) * 0.5    # 水平高频（列方向差异）
  HL = (ee + eo - oe - oo) * 0.5    # 垂直高频（行方向差异）
  HH = (ee - eo - oe + oo) * 0.5    # 对角高频

逆变换 (IDWT):
  ee = (LL + LH + HL + HH) * 0.5
  eo = (LL - LH + HL - HH) * 0.5
  oe = (LL + LH - HL - HH) * 0.5
  oo = (LL - LH - HL + HH) * 0.5
```

正逆变换互为精确逆运算，无信息损失。

---

## 4. 底层实现：DWT_2D / IDWT_2D

> 文件: `haar_wavelets.py`。无可学习参数，纯数学变换。

### 4.1 DWT_2D（正变换）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT_2D(nn.Module):
    """Haar 2-D Discrete Wavelet Transform via pixel_unshuffle.

    Input : (B, C, H, W)   — H, W 必须是偶数
    Output: (B, 4C, H/2, W/2) — 通道顺序 [LL, LH, HL, HH] (每组 C 个通道)
    """

    def __init__(self, wave="haar"):
        super(DWT_2D, self).__init__()
        if wave != "haar":
            raise ValueError(f"Only 'haar' wavelet is supported, got '{wave}'")

    def forward(self, x):
        # pixel_unshuffle: (B, C, H, W) → (B, 4C, H/2, W/2)
        xu = F.pixel_unshuffle(x, 2)
        B, C4, H2, W2 = xu.shape
        C = C4 // 4
        xu = xu.view(B, C, 4, H2, W2)
        x_ee = xu[:, :, 0]
        x_eo = xu[:, :, 1]
        x_oe = xu[:, :, 2]
        x_oo = xu[:, :, 3]

        # Haar 系数
        ll = (x_ee + x_eo + x_oe + x_oo) * 0.5  # 低频近似
        lh = (x_ee - x_eo + x_oe - x_oo) * 0.5  # 水平高频
        hl = (x_ee + x_eo - x_oe - x_oo) * 0.5  # 垂直高频
        hh = (x_ee - x_eo - x_oe + x_oo) * 0.5  # 对角高频

        return torch.cat([ll, lh, hl, hh], dim=1)
```

### 4.2 IDWT_2D（逆变换）

```python
class IDWT_2D(nn.Module):
    """Haar 2-D Inverse Discrete Wavelet Transform via pixel_shuffle.

    Input : (B, 4C, H/2, W/2) — 通道顺序 [LL, LH, HL, HH]
    Output: (B, C, H, W)
    """

    def __init__(self, wave="haar"):
        super(IDWT_2D, self).__init__()
        if wave != "haar":
            raise ValueError(f"Only 'haar' wavelet is supported, got '{wave}'")

    def forward(self, x):
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        x = x.view(B, 4, C, H2, W2)
        ll = x[:, 0]
        lh = x[:, 1]
        hl = x[:, 2]
        hh = x[:, 3]

        # 逆 Haar 变换 — 恢复子像素值
        x_ee = (ll + lh + hl + hh) * 0.5
        x_eo = (ll - lh + hl - hh) * 0.5
        x_oe = (ll + lh - hl - hh) * 0.5
        x_oo = (ll - lh - hl + hh) * 0.5

        x = torch.stack([x_ee, x_eo, x_oe, x_oo], dim=2)  # (B, C, 4, H2, W2)
        x = x.view(B, C * 4, H2, W2)
        return F.pixel_shuffle(x, 2)
```

---

## 5. 上下采样封装模块（网络直接调用）

封装了通道数变换逻辑，使 DWT/IDWT 与 U-Net 通道倍增规则兼容。

### 5.1 DWT_Downsample（当前使用）

> 关键点：DWT 输出 4C 通道，1×1 Conv 融合为 2C；高频 [LH, HL, HH]（3C）作为 detail skip 传递给对应的 IDWT_Upsample。

```python
class DWT_Downsample(nn.Module):
    """DWT 下采样：融合全部子带给编码器，同时保留高频细节作为 skip。

    输入:  (B, C, H, W)
    输出:
      - x_fused: (B, 2C, H/2, W/2)  — 编码器主路径
      - x_detail: (B, 3C, H/2, W/2) — [LH, HL, HH] 高频细节 skip
    """

    def __init__(self, n_feat, wave="haar"):
        super(DWT_Downsample, self).__init__()
        self.dwt = DWT_2D(wave)
        # 1×1 Conv 将 4C 融合为 2C
        self.channel_fuse = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        dwt_out = self.dwt(x)           # (B, 4C, H/2, W/2)
        C = dwt_out.shape[1] // 4
        x_detail = dwt_out[:, C:, :, :] # (B, 3C, H/2, W/2) = [LH, HL, HH]
        x_fused = self.channel_fuse(dwt_out)  # (B, 2C, H/2, W/2)
        return x_fused, x_detail
```

### 5.2 IDWT_Upsample（当前使用）

> 关键点：接收解码器特征 + 编码器保存的高频 detail skip，通过 IDWT 逆变换恢复高分辨率特征。

```python
class IDWT_Upsample(nn.Module):
    """IDWT 上采样：将解码器特征作为 LL，与编码器的高频细节合并后逆变换。

    输入:
      - x: (B, 2C, H/2, W/2)     — 解码器特征
      - detail: (B, 3C, H/2, W/2) — 编码器 DWT 保存的 [LH, HL, HH]
    输出: (B, C, H, W)
    """

    def __init__(self, n_feat, wave="haar"):
        super(IDWT_Upsample, self).__init__()
        self.idwt = IDWT_2D(wave)
        # 1×1 Conv 将 2C 降回 C，作为 LL 子带
        self.channel_reduce = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, bias=False)

    def forward(self, x, detail):
        x_ll = self.channel_reduce(x)         # (B, C, H/2, W/2)
        idwt_in = torch.cat([x_ll, detail], dim=1)  # (B, 4C, H/2, W/2)
        out = self.idwt(idwt_in)              # (B, C, H, W)
        return out
```

### 5.3 备用方案：PixelUnshuffle / PixelShuffle

```python
class Downsample(nn.Module):
    """PixelUnshuffle 下采样。

    流程: C --[3×3 Conv]--> C/2 --[PixelUnshuffle(2)]--> 2C
    输入: (B, C, H, W)
    输出: (B, 2C, H/2, W/2)
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """PixelShuffle 上采样。

    流程: C --[3×3 Conv]--> 2C --[PixelShuffle(2)]--> C/2
    输入: (B, C, H, W)
    输出: (B, C/2, 2H, 2W)
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
```

---

## 6. 辅助模块：PatchEmbed & ChannelReduce

### 6.1 OverlapPatchEmbed（输入嵌入）

```python
class OverlapPatchEmbed(nn.Module):
    """3×3 Conv 将输入图像映射到特征空间。

    输入: (B, in_c, H, W)   — in_c=3
    输出: (B, embed_dim, H, W) — embed_dim=48
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)
```

### 6.2 reduce_chan（解码器 skip 融合后降通道）

```python
# 不是独立类，而是在 Restormer.__init__ 中定义的 1×1 Conv：
# cat(上采样结果, encoder_skip) 之后通道数翻倍，用 1×1 Conv 压回来

# Level 3: cat → 8C → 4C
self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2),
                                    kernel_size=1, bias=bias)
# Level 2: cat → 4C → 2C
self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1),
                                    kernel_size=1, bias=bias)
# Level 1: 不需要 reduce，因为 cat(C, C) = 2C 直接输入 decoder_level1（本身就是 2C）
```

---

## 7. 在 U-Net 中的使用方式

### 7.1 模块实例化

```python
dim = 48  # 基础通道数 C

# ---- 下采样 ----
self.down1_2 = DWT_Downsample(dim)           # C → 2C,  detail = 3C
self.down2_3 = DWT_Downsample(dim * 2)       # 2C → 4C, detail = 6C
self.down3_4 = DWT_Downsample(dim * 4)       # 4C → 8C, detail = 12C

# ---- 上采样 ----
self.up4_3 = IDWT_Upsample(dim * 8)         # 8C + detail(12C) → 4C
self.up3_2 = IDWT_Upsample(dim * 4)         # 4C + detail(6C)  → 2C
self.up2_1 = IDWT_Upsample(dim * 2)         # 2C + detail(3C)  → C

# ---- skip 融合后的通道压缩 ----
self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
# Level 1 不需要 reduce（cat 后 2C 直接作为 decoder 输入）
```

### 7.2 前向传播

```python
def forward(self, inp_img):
    # ===== 输入嵌入 =====
    inp_enc_level1 = self.patch_embed(inp_img)
    # inp_img:        (B, 3, H, W)
    # inp_enc_level1: (B, C, H, W)        C = dim = 48

    # ===== Encoder Level 1 =====
    out_enc_level1 = self.encoder_level1(inp_enc_level1)
    # out_enc_level1: (B, C, H, W)        ← 保存为 skip_1

    # ===== 下采样 1→2 (DWT) =====
    inp_enc_level2, detail1 = self.down1_2(out_enc_level1)
    # inp_enc_level2: (B, 2C, H/2, W/2)   ← 主路径继续
    # detail1:        (B, 3C, H/2, W/2)   ← 高频 skip（给 IDWT_Up 2→1 用）

    # ===== Encoder Level 2 =====
    out_enc_level2 = self.encoder_level2(inp_enc_level2)
    # out_enc_level2: (B, 2C, H/2, W/2)   ← 保存为 skip_2

    # ===== 下采样 2→3 (DWT) =====
    inp_enc_level3, detail2 = self.down2_3(out_enc_level2)
    # inp_enc_level3: (B, 4C, H/4, W/4)
    # detail2:        (B, 6C, H/4, W/4)   ← 高频 skip（给 IDWT_Up 3→2 用）

    # ===== Encoder Level 3 =====
    out_enc_level3 = self.encoder_level3(inp_enc_level3)
    # out_enc_level3: (B, 4C, H/4, W/4)   ← 保存为 skip_3

    # ===== 下采样 3→4 (DWT) =====
    inp_enc_level4, detail3 = self.down3_4(out_enc_level3)
    # inp_enc_level4: (B, 8C, H/8, W/8)
    # detail3:        (B, 12C, H/8, W/8)  ← 高频 skip（给 IDWT_Up 4→3 用）

    # ===== Bottleneck (Latent) =====
    latent = self.latent(inp_enc_level4)
    # latent: (B, 8C, H/8, W/8)

    # ===== 上采样 4→3 (IDWT) + Skip Connection =====
    inp_dec_level3 = self.up4_3(latent, detail3)
    # IDWT_Upsample: latent(8C) + detail3(12C) → (B, 4C, H/4, W/4)
    inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
    # cat: (B, 4C + 4C, H/4, W/4) = (B, 8C, H/4, W/4)
    inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
    # 1×1 Conv: 8C → 4C

    # ===== Decoder Level 3 =====
    out_dec_level3 = self.decoder_level3(inp_dec_level3)
    # out_dec_level3: (B, 4C, H/4, W/4)

    # ===== 上采样 3→2 (IDWT) + Skip Connection =====
    inp_dec_level2 = self.up3_2(out_dec_level3, detail2)
    # IDWT_Upsample: (4C) + detail2(6C) → (B, 2C, H/2, W/2)
    inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
    # cat: (B, 2C + 2C, H/2, W/2) = (B, 4C, H/2, W/2)
    inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
    # 1×1 Conv: 4C → 2C

    # ===== Decoder Level 2 =====
    out_dec_level2 = self.decoder_level2(inp_dec_level2)
    # out_dec_level2: (B, 2C, H/2, W/2)

    # ===== 上采样 2→1 (IDWT) + Skip Connection =====
    inp_dec_level1 = self.up2_1(out_dec_level2, detail1)
    # IDWT_Upsample: (2C) + detail1(3C) → (B, C, H, W)
    inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
    # cat: (B, C + C, H, W) = (B, 2C, H, W)
    # 注意: Level 1 没有 reduce_chan，decoder_level1 直接处理 2C

    # ===== Decoder Level 1 =====
    out_dec_level1 = self.decoder_level1(inp_dec_level1)
    # out_dec_level1: (B, 2C, H, W)

    # ===== Refinement =====
    out_dec_level1 = self.refinement(out_dec_level1)
    # out_dec_level1: (B, 2C, H, W)

    # ===== 输出 + 全局残差 =====
    out_dec_level1 = self.output(out_dec_level1) + inp_img
    # self.output: 3×3 Conv, 2C → 3
    # 全局残差: 输出 + 原始输入图像

    return out_dec_level1
    # 最终输出: (B, 3, H, W)
```

---

## 8. 通道数变化总结表

以 `dim=48 (C=48)` 为例：

| 阶段            | 操作            | 输入形状                  | 输出形状                                          | 说明              |
| --------------- | --------------- | ------------------------- | ------------------------------------------------- | ----------------- |
| PatchEmbed      | 3×3 Conv        | (B, 3, H, W)              | (B, 48, H, W)                                     | 输入嵌入          |
| **Encoder L1**  | Transformer×N   | (B, 48, H, W)             | (B, 48, H, W)                                     | —                 |
| Down 1→2        | DWT + 1×1 Conv  | (B, 48, H, W)             | **(B, 96, H/2, W/2)** + detail(B, 144, H/2, W/2)  | 主路径 + 高频skip |
| **Encoder L2**  | Transformer×N   | (B, 96, H/2, W/2)         | (B, 96, H/2, W/2)                                 | —                 |
| Down 2→3        | DWT + 1×1 Conv  | (B, 96, H/2, W/2)         | **(B, 192, H/4, W/4)** + detail(B, 288, H/4, W/4) | —                 |
| **Encoder L3**  | Transformer×N   | (B, 192, H/4, W/4)        | (B, 192, H/4, W/4)                                | —                 |
| Down 3→4        | DWT + 1×1 Conv  | (B, 192, H/4, W/4)        | **(B, 384, H/8, W/8)** + detail(B, 576, H/8, W/8) | —                 |
| **Latent**      | Transformer×N   | (B, 384, H/8, W/8)        | (B, 384, H/8, W/8)                                | —                 |
| Up 4→3          | IDWT            | (B, 384) + detail(B, 576) | (B, 192, H/4, W/4)                                | —                 |
| cat + reduce    | cat + 1×1       | (B, 192+192)              | (B, 192, H/4, W/4)                                | skip_3            |
| **Decoder L3**  | Transformer×N   | (B, 192, H/4, W/4)        | (B, 192, H/4, W/4)                                | —                 |
| Up 3→2          | IDWT            | (B, 192) + detail(B, 288) | (B, 96, H/2, W/2)                                 | —                 |
| cat + reduce    | cat + 1×1       | (B, 96+96)                | (B, 96, H/2, W/2)                                 | skip_2            |
| **Decoder L2**  | Transformer×N   | (B, 96, H/2, W/2)         | (B, 96, H/2, W/2)                                 | —                 |
| Up 2→1          | IDWT            | (B, 96) + detail(B, 144)  | (B, 48, H, W)                                     | —                 |
| cat (no reduce) | cat             | (B, 48+48)                | (B, 96, H, W)                                     | skip_1            |
| **Decoder L1**  | Transformer×N   | (B, 96, H, W)             | (B, 96, H, W)                                     | —                 |
| Refinement      | Transformer×N   | (B, 96, H, W)             | (B, 96, H, W)                                     | —                 |
| Output          | 3×3 Conv + 残差 | (B, 96, H, W)             | (B, 3, H, W)                                      | + inp_img         |

---

## 9. 移植要点

1. **依赖**：仅需 PyTorch (>=1.8, 需 `pixel_unshuffle`)，无额外依赖
2. **需要复制的类**：`DWT_2D` / `IDWT_2D`（无参数）、`DWT_Downsample`（含 1×1 Conv channel_fuse）、`IDWT_Upsample`（含 1×1 Conv channel_reduce）
3. **接口变更**：下采样返回 **两个值** `(features, detail)`，上采样接收 **两个参数** `(features, detail)`，需在网络 forward 中缓存 detail
4. **空间约束**：输入 H、W 必须为偶数（三层下采样需 H、W 为 8 的倍数）
5. **两类 skip connection**：
   - **detail skip**（高频）：`DWT_Downsample` → `IDWT_Upsample`，在频域传递
   - **encoder skip**（特征）：encoder 输出 → `cat` 到 decoder，在空间域传递
6. **通道数公式**：每层下采样通道 ×2，上采样 ×(1/2)；detail 通道 = 输入通道 ×3
7. **如果不想用 DWT**：可用备用 `Downsample`/`Upsample`（PixelUnshuffle/Shuffle），但此时没有 detail skip

---

## 10. 与原始 PixelShuffle 方案的对比

| 维度       | PixelShuffle 方案                | DWT 方案                    |
| ---------- | -------------------------------- | --------------------------- |
| 下采样     | Conv(C->C/2) + PixelUnshuffle(2) | DWT分解 + 1×1Conv(4C->2C)   |
| 上采样     | Conv(C->2C) + PixelShuffle(2)    | 1×1Conv(2C->C) + IDWT合成   |
| 高频信息   | 混入所有通道, 可能丢失           | 显式保留为 detail skip      |
| 额外连接   | 仅常规 skip connection           | 常规skip + detail skip      |
| 可学习参数 | Conv 中的权重                    | 仅 1×1 Conv, DWT 本身无参数 |
| 信息保持   | 有损（依赖学习）                 | 数学上无损（DWT/IDWT 互逆） |
