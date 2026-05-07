# DWT 上下采样方案 — 完整设计文档（可移植）

> ，用于替代传统 PixelShuffle/PixelUnshuffle 的上下采样模块。

---

## 1. 核心思想

本方案用 **Haar 离散小波变换 (DWT/IDWT)** 替代，关键优势：

- **下采样 (DWT)**：将特征分解为低频 LL（主体结构）和高频 LH/HL/HH（细节纹理），高频分量作为 **detail skip** 传给解码器，避免高频信息在下采样中丢失。
- **上采样 (IDWT)**：解码器特征作为 LL 分量，与编码器保存的高频 detail 重新合成，实现无损恢复空间分辨率。

这比传统方式多了一条 **高频细节跳跃连接**，可保留更多纹理和边缘信息。

---

## 2. 整体数据流

```
Encoder                                         Decoder

Level 1: (B, C, H, W)                          Level 1: (B, 2C, H, W)
    |                                               ^
    | DWT_Downsample                                | IDWT_Upsample
    |  +-- LL -> channel_expand -> (B,2C,H/2,W/2)  |  +-- x(decoder) -> channel_reduce -> LL
    |  +-- detail(LH,HL,HH) ----[detail skip]------>|  +-- + detail(LH,HL,HH) -> IDWT -> (B,C,H,W)
    v                                               |
Level 2: (B, 2C, H/2, W/2)                     Level 2: ...
    |                                               ^
    | DWT_Downsample                                | IDWT_Upsample
    v                                               |
Level 3: (B, 4C, H/4, W/4)                     Level 3: ...
    |                                               ^
    | DWT_Downsample                                | IDWT_Upsample
    v                                               |
Level 4 (Latent): (B, 8C, H/8, W/8) ------------>--+
```

**关键**：每次 DWT 下采样产生的 `detail` (3C 通道) 需缓存，在对应层级的 IDWT 上采样时使用。

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

### 4.1 方案 A：pixel_unshuffle 实现（推荐，零额外依赖）

仅支持 Haar 小波，但速度快 3-5x，无 conv 开销，无自定义 autograd Function。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT_2D(nn.Module):
    """Haar 2-D DWT via pixel_unshuffle.
    Input:  (B, C, H, W)       H,W must be even
    Output: (B, 4C, H/2, W/2)  channel order [LL, LH, HL, HH] per input channel
    """
    def __init__(self, wave="haar"):
        super().__init__()
        if wave != "haar":
            raise ValueError(f"Only 'haar' supported, got '{wave}'")

    def forward(self, x):
        xu = F.pixel_unshuffle(x, 2)          # (B, 4C, H/2, W/2)
        B, C4, H2, W2 = xu.shape
        C = C4 // 4
        xu = xu.view(B, C, 4, H2, W2)
        x_ee, x_eo, x_oe, x_oo = xu[:,:,0], xu[:,:,1], xu[:,:,2], xu[:,:,3]

        ll = (x_ee + x_eo + x_oe + x_oo) * 0.5
        lh = (x_ee - x_eo + x_oe - x_oo) * 0.5
        hl = (x_ee + x_eo - x_oe - x_oo) * 0.5
        hh = (x_ee - x_eo - x_oe + x_oo) * 0.5

        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT_2D(nn.Module):
    """Haar 2-D IDWT via pixel_shuffle.
    Input:  (B, 4C, H/2, W/2)  channel order [LL, LH, HL, HH]
    Output: (B, C, H, W)
    """
    def __init__(self, wave="haar"):
        super().__init__()
        if wave != "haar":
            raise ValueError(f"Only 'haar' supported, got '{wave}'")

    def forward(self, x):
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        x = x.view(B, 4, C, H2, W2)
        ll, lh, hl, hh = x[:,0], x[:,1], x[:,2], x[:,3]

        x_ee = (ll + lh + hl + hh) * 0.5
        x_eo = (ll - lh + hl - hh) * 0.5
        x_oe = (ll + lh - hl - hh) * 0.5
        x_oo = (ll - lh - hl + hh) * 0.5

        x = torch.stack([x_ee, x_eo, x_oe, x_oo], dim=2)  # (B, C, 4, H2, W2)
        x = x.view(B, C * 4, H2, W2)
        return F.pixel_shuffle(x, 2)  # (B, C, H, W)
```

### 4.2 方案 B：conv2d 实现（支持任意小波基，需 `pywt`）

```python
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        w_ll, w_lh, w_hl, w_hh = (w.to(x.dtype) for w in (w_ll, w_lh, w_hl, w_hh))
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

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
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.to(x.dtype).repeat(C, 1, 1, 1)
        return F.conv_transpose2d(x, filters, stride=2, groups=C)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters.to(dx.dtype), dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C,-1,-1,-1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C,-1,-1,-1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C,-1,-1,-1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C,-1,-1,-1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        self.register_buffer("w_ll", (dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1))[None,None])
        self.register_buffer("w_lh", (dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1))[None,None])
        self.register_buffer("w_hl", (dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1))[None,None])
        self.register_buffer("w_hh", (dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1))[None,None])

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi, rec_lo = torch.Tensor(w.rec_hi), torch.Tensor(w.rec_lo)
        w_ll = (rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1))[None,None]
        w_lh = (rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1))[None,None]
        w_hl = (rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1))[None,None]
        w_hh = (rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1))[None,None]
        self.register_buffer("filters", torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0))

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)
```

---

## 5. 上下采样封装模块（网络直接调用）

封装了通道数变换逻辑，使 DWT/IDWT 与 U-Net 通道倍增规则兼容。

```python
class DWT_Downsample(nn.Module):
    """DWT 下采样: LL 作为主特征并通道扩展, detail(LH/HL/HH) 作为跳跃连接传递给解码器。

    输入:  (B, C, H, W)
    输出:  x_ll:    (B, 2C, H/2, W/2)   -- 送入下一层编码器
           detail:  (B, 3C, H/2, W/2)   -- 缓存, 供对应层IDWT上采样使用
    """
    def __init__(self, n_feat, wave="haar"):
        super().__init__()
        self.dwt = DWT_2D(wave)
        # LL 分量通道为 C, 需扩展到 2C 以匹配下一层编码器通道数
        self.channel_expand = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        dwt_out = self.dwt(x)                      # (B, 4C, H/2, W/2)
        C = dwt_out.shape[1] // 4
        x_ll = dwt_out[:, :C, :, :]                # (B, C, H/2, W/2)  -- LL
        x_detail = dwt_out[:, C:, :, :]             # (B, 3C, H/2, W/2) -- [LH, HL, HH]
        x_ll = self.channel_expand(x_ll)            # (B, 2C, H/2, W/2)
        return x_ll, x_detail


class IDWT_Upsample(nn.Module):
    """IDWT 上采样: 将解码器特征(作为LL)与编码器传来的高频detail重新合成。

    输入:  x:       (B, 2C, H/2, W/2)   -- 解码器特征
           detail:  (B, 3C, H/2, W/2)   -- 编码器DWT产生的 [LH, HL, HH]
    输出:  (B, C, H, W)
    """
    def __init__(self, n_feat, wave="haar"):
        super().__init__()
        self.idwt = IDWT_2D(wave)
        # 解码器特征通道为 2C, 需缩减到 C 以作为 LL 分量
        self.channel_reduce = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, bias=False)

    def forward(self, x, detail):
        # x: (B, 2C, H/2, W/2), detail: (B, 3C, H/2, W/2)
        x_ll = self.channel_reduce(x)               # (B, C, H/2, W/2)
        idwt_in = torch.cat([x_ll, detail], dim=1)  # (B, 4C, H/2, W/2) = [LL, LH, HL, HH]
        out = self.idwt(idwt_in)                     # (B, C, H, W)
        return out
```

---

## 6. 在 U-Net 中的使用方式

### 6.1 模块实例化

```python
dim = 48  # 基础通道数

# 编码器下采样
self.down1_2 = DWT_Downsample(dim)              # C=48  -> 2C=96,  detail: 3*48=144
self.down2_3 = DWT_Downsample(dim * 2)          # C=96  -> 2C=192, detail: 3*96=288
self.down3_4 = DWT_Downsample(dim * 4)          # C=192 -> 2C=384, detail: 3*192=576

# 解码器上采样
self.up4_3 = IDWT_Upsample(dim * 8)             # 2C=384 -> C=192
self.up3_2 = IDWT_Upsample(dim * 4)             # 2C=192 -> C=96
self.up2_1 = IDWT_Upsample(dim * 2)             # 2C=96  -> C=48
```

### 6.2 前向传播

```python
def forward(self, inp_img):
    # === 编码器 ===
    inp_enc_level1 = self.patch_embed(inp_img)       # (B, C, H, W)
    out_enc_level1 = self.encoder_level1(inp_enc_level1)

    # 下采样: DWT分解, 缓存detail用于解码器
    inp_enc_level2, detail1 = self.down1_2(out_enc_level1)
    out_enc_level2 = self.encoder_level2(inp_enc_level2)

    inp_enc_level3, detail2 = self.down2_3(out_enc_level2)
    out_enc_level3 = self.encoder_level3(inp_enc_level3)

    inp_enc_level4, detail3 = self.down3_4(out_enc_level3)
    latent = self.latent(inp_enc_level4)

    # === 解码器 ===
    # 上采样: IDWT合成, 使用编码器缓存的detail
    inp_dec_level3 = self.up4_3(latent, detail3)
    inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # 常规skip concat
    inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)          # 1x1 conv降通道
    out_dec_level3 = self.decoder_level3(inp_dec_level3)

    inp_dec_level2 = self.up3_2(out_dec_level3, detail2)
    inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
    inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
    out_dec_level2 = self.decoder_level2(inp_dec_level2)

    inp_dec_level1 = self.up2_1(out_dec_level2, detail1)
    inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
    out_dec_level1 = self.decoder_level1(inp_dec_level1)

    return self.output(out_dec_level1) + inp_img
```

---

## 7. 通道数变化总结表

| 操作                | 输入                     | DWT/IDWT 输出            | Conv 调整后         | 说明              |
| ------------------- | ------------------------ | ------------------------ | ------------------- | ----------------- |
| `DWT_Downsample(C)` | `(B,C,H,W)`              | LL: `(B,C,H/2,W/2)`      | `(B,2C,H/2,W/2)`    | 1x1 Conv 扩展通道 |
|                     |                          | detail: `(B,3C,H/2,W/2)` | 不变，直接缓存      |                   |
| `IDWT_Upsample(2C)` | x: `(B,2C,H/2,W/2)`      | —                        | LL: `(B,C,H/2,W/2)` | 1x1 Conv 缩减通道 |
|                     | detail: `(B,3C,H/2,W/2)` | `(B,C,H,W)`              | —                   | cat后IDWT重建     |

---

## 8. 移植要点

1. **依赖**：方案A仅需 PyTorch (>=1.8, 需 `pixel_unshuffle`)；方案B额外需 `pywt`
2. **接口变更**：下采样返回 **两个值** `(features, detail)`，上采样接收 **两个参数** `(features, detail)`，需在网络 forward 中缓存 detail
3. **空间约束**：输入 H、W 必须为偶数（每层下采样一次，三层下采样需 H、W 为 8 的倍数）
4. **与原始 skip connection 共存**：IDWT 上采样后仍可 concat 编码器的常规 skip，再用 1x1 Conv 降通道
5. **无可学习参数**：DWT_2D / IDWT_2D 本身无参数，通道变换的 1x1 Conv 是唯一可学习部分

---

## 9. 与原始 PixelShuffle 方案的对比

| 维度       | PixelShuffle 方案                | DWT 方案                    |
| ---------- | -------------------------------- | --------------------------- |
| 下采样     | Conv(C->C/2) + PixelUnshuffle(2) | DWT分解 + 1x1Conv(C->2C)    |
| 上采样     | Conv(C->2C) + PixelShuffle(2)    | 1x1Conv(2C->C) + IDWT合成   |
| 高频信息   | 混入所有通道, 可能丢失           | 显式保留为 detail skip      |
| 额外连接   | 仅常规 skip connection           | 常规skip + detail skip      |
| 可学习参数 | Conv 中的权重                    | 仅 1x1 Conv, DWT 本身无参数 |
| 信息保持   | 有损（依赖学习）                 | 数学上无损（DWT/IDWT 互逆） |
