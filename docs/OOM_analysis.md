# YOLOv12m 训练验证阶段 CUDA OOM 排查报告

> 日期：2025-05-01  
> 环境：NVIDIA GeForce RTX 4090 (24564 MiB) / PyTorch 2.2.2+cu121 / Python 3.11  
> 模型：yolov12m (533 layers, 19,670,784 parameters, 60.4 GFLOPs)  
> 训练脚本：`train_voc.py`（batch=16, imgsz=640, workers=8）

---

## 一、问题描述

训练 YOLOv12m 模型时，第 1 个 epoch 的训练阶段正常完成（GPU 显存占用约 11GB），但在进入**验证阶段**后立即报错：

```
RuntimeError: Caught RuntimeError in pin memory thread for device 0.
  ...
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
```

关键异常现象：**RTX 4090 拥有 24GB 显存，训练阶段仅使用约 11GB，理论上剩余 13GB 足够验证使用，却仍然 OOM。**

---

## 二、错误堆栈分析

```
trainer.py:433  →  self.metrics, self.fitness = self.validate()
trainer.py:606  →  metrics = self.validator(self)
validator.py:171  →  for batch_i, batch in enumerate(bar):   # 遍历验证数据
build.py:48  →  yield next(self.iterator)                    # InfiniteDataLoader
dataloader.py:1346  →  self._next_data()
pin_memory.py:57  →  data.pin_memory(device)                 # ← OOM 发生点
```

错误表面上发生在 `pin_memory` 线程中，但 PyTorch 明确提示：
> *"CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect."*

这意味着 **`pin_memory` 线程只是错误的报告者，不一定是真正的触发者**。

---

## 三、排查过程

### 3.1 初步假设：训练→验证切换时显存未释放

**依据**：训练结束后 CUDA 缓存分配器（caching allocator）可能持有大量已释放但未归还系统的显存块。

**尝试**：在 `ultralytics/engine/trainer.py` 验证前添加显存清理：
```python
# ultralytics/engine/trainer.py, line 432-434
if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
    gc.collect()
    torch.cuda.empty_cache()
    self.metrics, self.fitness = self.validate()
```

**结果**：❌ 失败。`nvidia-smi` 监控显示显存确实从 ~11GB 降至 ~1.3GB，但 OOM 依然发生。  
**结论**：排除"训练显存残留"假设。OOM 发生在验证过程自身的内存分配中。

---

### 3.2 第二假设：验证 batch size 过大（batch × 2）

**发现**：Ultralytics 框架默认将验证 batch size 翻倍（因为验证无需反向传播，通常可用更大 batch）：

```python
# ultralytics/engine/trainer.py, line 290-292
self.test_loader = self.get_dataloader(
    self.testset,
    batch_size=batch_size if self.args.task == "obb" else batch_size * 2,  # ← 验证用 32
    rank=-1, mode="val"
)
```

训练 batch=16，验证 batch=**32**。对于传统 YOLO 模型（纯卷积），这通常没问题。但 YOLOv12m 引入了 `A2C2f` 注意力模块，前向传播的显存开销显著高于传统卷积。

**尝试**：将验证 batch 改回与训练相同的 16。

**结果**：❌ 仍然失败。  
**结论**：batch size 翻倍是加剧因素，但并非根本原因。

---

### 3.3 第三假设：`pin_memory` 机制引发 CUDA 显存竞争

**深入分析 `pin_memory` 机制**：

当 `pin_memory=True`（默认）时，DataLoader 会启动一个独立的 **pin_memory 线程**，该线程调用 `tensor.pin_memory(device)` 将 CPU 内存注册为 CUDA 页锁定内存（page-locked memory）。这个操作：

1. 分配页锁定主机内存
2. 通过 `cudaHostAlloc` 将其映射到 CUDA 设备的统一虚拟地址空间（UVA）
3. 这个映射过程会**占用 CUDA 设备端的虚拟地址资源和少量显存**

**关键问题**：

验证阶段同时存在以下显存压力：

| 来源                                                              | 估算占用     |
| ----------------------------------------------------------------- | ------------ |
| 训练模型权重（FP32，仍在 GPU 上）                                 | ~80 MB       |
| 优化器状态（SGD momentum）                                        | ~80 MB       |
| EMA 模型（FP16，用于验证推理）                                    | ~40 MB       |
| CUDA 上下文 + cuDNN workspace                                     | ~1-2 GB      |
| 训练 DataLoader 的 pin_memory 线程（InfiniteDataLoader 不会停止） | 持续占用     |
| 验证前向传播激活值（batch=32, A2C2f 注意力块）                    | **数 GB**    |
| 验证 DataLoader 的 pin_memory 线程                                | **新增占用** |

`InfiniteDataLoader`（训练用）在验证期间**并不会停止其 worker 和 pin_memory 线程**，它们会持续预取并固定下一个 epoch 的训练数据。这意味着训练和验证的 pin_memory 线程**同时运行**，与验证前向传播的 CUDA 内存分配产生竞争。

**验证方式**：使用环境变量 `PIN_MEMORY=false` 全局禁用 pin_memory，并配合 `CUDA_LAUNCH_BLOCKING=1` 使错误同步报告：

```bash
PIN_MEMORY=false CUDA_LAUNCH_BLOCKING=1 python train_voc.py
```

**结果**：✅ **验证成功通过**，训练连续完成多个 epoch 无报错。

---

### 3.4 精确定位与最小化修复

进一步测试确认：
- 仅禁用**验证** DataLoader 的 `pin_memory`（训练保持 `True`）即可解决问题
- 不需要 `CUDA_LAUNCH_BLOCKING=1`
- 不需要修改验证 batch size（可保持默认的 `batch × 2`）

---

## 四、根本原因

**CUDA 显存 OOM 的根本原因是验证 DataLoader 的 `pin_memory(device)` 操作与验证前向传播竞争 CUDA 显存资源。**

具体机制：
1. 训练结束后，训练模型、优化器状态、EMA 模型仍在 GPU 上（尽管 `empty_cache()` 释放了缓存块）
2. 训练 `InfiniteDataLoader` 的 pin_memory 线程仍在后台运行
3. 验证开始时，YOLOv12m 的 `A2C2f` 注意力模块前向传播（batch=32）需要分配大量 CUDA 显存
4. 验证 DataLoader 的 pin_memory 线程**同时**尝试通过 `cudaHostAlloc` 注册页锁定内存
5. 这些并发的 CUDA 内存分配请求累积超过可用资源，触发 OOM
6. 由于 CUDA 操作的异步性，OOM 被报告在 pin_memory 线程中，而非实际的触发点

**为什么只在 YOLOv12 上出现**：传统 YOLO 模型（v5/v8/v11）使用纯卷积，前向传播显存占用较低且可预测。YOLOv12 引入了 `A2C2f` → `ABlock` → `AAttn`（区域注意力）模块，其前向传播产生的中间张量（Q/K/V 投影、注意力计算）显著增加了显存峰值。

---

## 五、修复方案

### 修改文件清单

| 文件                                      | 修改内容                                               | 作用                                     |
| ----------------------------------------- | ------------------------------------------------------ | ---------------------------------------- |
| `ultralytics/data/build.py`               | `build_dataloader()` 增加 `pin_memory` 参数            | 允许调用方控制 pin_memory                |
| `ultralytics/models/yolo/detect/train.py` | 验证模式下传入 `pin_memory=False`                      | **核心修复**：消除验证时 pin_memory 竞争 |
| `ultralytics/models/yolo/detect/val.py`   | `get_dataloader()` 传入 `pin_memory=False`             | 独立验证（非训练中调用）时同样生效       |
| `ultralytics/engine/trainer.py`           | 验证前添加 `gc.collect()` + `torch.cuda.empty_cache()` | 辅助措施：释放训练阶段的 CUDA 缓存       |

### 具体代码变更

**1. `ultralytics/data/build.py`**（第 135 行）
```python
# 修改前
def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    ...
    pin_memory=PIN_MEMORY,

# 修改后
def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, pin_memory=PIN_MEMORY):
    ...
    pin_memory=pin_memory,
```

**2. `ultralytics/models/yolo/detect/train.py`**（第 54-56 行）
```python
# 修改前
workers = self.args.workers if mode == "train" else self.args.workers * 2
return build_dataloader(dataset, batch_size, workers, shuffle, rank)

# 修改后
workers = self.args.workers if mode == "train" else self.args.workers * 2
pin_memory = mode == "train"
return build_dataloader(dataset, batch_size, workers, shuffle, rank, pin_memory=pin_memory)
```

**3. `ultralytics/models/yolo/detect/val.py`**（第 244 行）
```python
# 修改前
return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

# 修改后
return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1, pin_memory=False)
```

**4. `ultralytics/engine/trainer.py`**（第 432-434 行）
```python
if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
    gc.collect()
    torch.cuda.empty_cache()
    self.metrics, self.fitness = self.validate()
```

---

## 六、修复验证

修复后连续训练 2 个完整 epoch，训练和验证均正常完成：

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/200      9.65G      3.338      5.064      4.246         54        640: 100% 1035/1035 [02:52, 5.99it/s]
           Class     Images  Instances      Box(P     R      mAP50  mAP50-95): 100% 155/155 [00:20, 7.41it/s]
             all       4952      12032    0.00513  0.0704   0.00318    0.00108

2/200      9.76G      2.797      4.445      3.484         36        640: 100% 1035/1035 [02:46, 6.22it/s]
           Class     Images  Instances      Box(P     R      mAP50  mAP50-95): 100% 155/155 [00:22, 6.86it/s]
             all       4952      12032      0.399  0.0568    0.0302    0.0111
```

- **训练速度**：~6.0 it/s（与修改前一致，无性能损失）
- **验证速度**：~7.1 it/s（pin_memory=False 对验证速度影响极小）
- **验证 batch size**：保持默认的 32（batch × 2），无需缩减
- **显存峰值**：~9.8 GB，远低于 24GB 上限

### 扩展验证：batch=32 训练 + batch=64 验证

进一步测试将训练 batch size 从 16 提升至 32（验证自动翻倍为 64），验证修复方案在高显存压力下的稳定性。

**测试结果**：✅ 训练和验证均正常通过。

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/200      20.4G       3.34      5.077      4.236         41        640: 100% 518/518 [02:49, 3.06it/s]
           Class     Images  Instances      Box(P     R      mAP50  mAP50-95): 100% 155/155 [00:27, 5.67it/s]
             all       4952      12032    0.00428  0.051   0.00238   0.000806

2/200      20.5G      3.342      5.039      4.217        159        640: ...
```

| 配置            | batch=16 (原始) | batch=32 (扩展) |
| --------------- | --------------- | --------------- |
| 训练 batch      | 16              | 32              |
| 训练显存        | 9.65 GB         | 20.4 GB         |
| 训练速度        | ~6.0 it/s       | ~3.0 it/s       |
| 验证 batch (×2) | 32              | 64              |
| 验证状态        | ✅               | ✅               |

**为什么验证 batch=64 不会 OOM**：

验证阶段的显存占用远低于训练阶段，核心原因在于**不需要为反向传播保存中间激活值**：

| 项目                                    | 训练时 (batch=32)        | 验证时 (batch=64)      |
| --------------------------------------- | ------------------------ | ---------------------- |
| 模型权重                                | FP32, ~80 MB             | FP16 (half), ~40 MB    |
| 优化器状态 (SGD momentum)               | ~80 MB                   | 仍在 GPU 上, ~80 MB    |
| 梯度缓冲区                              | ~80 MB                   | 仍在 GPU 上, 被清零    |
| **前向传播中间激活值（用于 backward）** | **~16-18 GB** ← 主要开销 | **不保存，用完即释放** |
| 前向推理激活值（当前层）                | 包含在上面               | ~几百 MB (FP16)        |
| CUDA 上下文 + workspace                 | ~1-2 GB                  | ~1-2 GB                |
| **总计**                                | **~20.4 GB**             | **~4-5 GB**            |

- **训练时**：PyTorch 的 `autograd` 需要保留每一层的输出张量来计算梯度（`loss.backward()`），A2C2f 注意力块的 Q/K/V 投影、注意力矩阵等全部缓存在显存中，这占了 **~16-18 GB**
- **验证时**：`torch.inference_mode()` 下前向传播不记录计算图，每层的中间结果用完立即释放，同时模型以 FP16 运行，所以即使 batch=64（训练的 2 倍），实际显存也仅需 ~4-5 GB

### 附：GPU 显存占用与 batch size 的关系

GPU 显存中各项数据与 batch size 的关系如下：

| 项目                   | 大小          | 是否随 batch 变化 | 说明                                            |
| ---------------------- | ------------- | ----------------- | ----------------------------------------------- |
| 模型权重               | ~75 MB        | ❌ 固定            | 由模型参数量决定（19,670,784 × 4 bytes）        |
| 梯度缓冲区             | ~75 MB        | ❌ 固定            | 每个参数对应一个梯度，形状与权重完全相同        |
| 优化器状态 (SGD)       | ~75 MB        | ❌ 固定            | 每个参数存一份动量，形状与权重完全相同          |
| **前向传播中间激活值** | **~16-18 GB** | **✅ 线性增长**    | 每个样本产生独立的特征图，全部缓存用于 backward |
| CUDA 上下文            | ~1-2 GB       | ❌ 固定            | 驱动、cuDNN workspace 等                        |

**关键结论**：模型权重、梯度、优化器状态的大小**只与模型参数量有关，与 batch size 完全无关**。真正随 batch size 线性增长的是前向传播的中间激活值，这才是显存的主要消耗。

以某一层输出 `[B, 256, 80, 80]` 为例：
- batch=16 → `16 × 256 × 80 × 80 × 2 bytes (FP16) ≈ 50 MB`
- batch=32 → `32 × 256 × 80 × 80 × 2 bytes (FP16) ≈ 100 MB`

模型有数十层，每层都需要缓存激活值用于反向传播，因此累计占用 **16-18 GB**。

> 注：若使用 Adam 优化器（需额外存储一阶矩和二阶矩），优化器状态会从 ~75 MB 增至 ~150 MB，但仍与 batch size 无关，相对激活值而言很小。

---

## 七、关键概念详解

### 7.1 `CUDA_LAUNCH_BLOCKING=1` 的作用

默认情况下，CUDA 操作是**异步**的——CPU 发出 GPU 指令后立即返回，GPU 在后台执行。如果 GPU 上某个操作 OOM，错误不会立即报告，而是在下一个需要同步的 CUDA 调用时才抛出，导致**报错位置与实际出错位置不一致**。

```
默认（异步）：
  GPU 操作 A  → 实际 OOM（但 CPU 不知道）
  GPU 操作 B  → 正常
  pin_memory() → 需要同步 → 这时才报告 A 的 OOM  ← 报错位置不准

CUDA_LAUNCH_BLOCKING=1（同步）：
  GPU 操作 A  → 立即报告 OOM  ← 报错位置准确
```

**本次用途**：纯调试工具，帮助定位真正的 OOM 发生点。会显著降低训练速度（~6 it/s → ~2.4 it/s），正式训练不应使用。

### 7.2 `pin_memory` 的作用

`pin_memory` 控制 DataLoader 是否将加载的数据放入**页锁定内存（pinned/page-locked memory）**，以加速 CPU→GPU 的数据传输。

```
pin_memory=False（普通流程）：
  磁盘 → CPU 普通内存(可分页) → [需要先复制到 pinned 内存] → GPU
                                    ↑ PyTorch 内部自动做，但多一次拷贝

pin_memory=True（优化流程）：
  磁盘 → CPU 页锁定内存(不可分页) → GPU（DMA 直传，更快）
                                      ↑ 少一次拷贝，速度更快
```

**但 `pin_memory` 有代价**：调用 `tensor.pin_memory(device)` 时，PyTorch 通过 `cudaHostAlloc` 向 CUDA 驱动注册这块内存，这个过程会：
1. 在 CUDA 设备的虚拟地址空间中建立映射
2. 占用少量 GPU 端资源
3. 如果此时 GPU 显存紧张，这个注册操作本身就可能触发 `CUDA error: out of memory`

### 7.3 为什么训练时开启 `pin_memory`，验证时关闭

**不是因为 `pin_memory` 本身占用大量显存**（它占用很少），而是因为**时序冲突**：

```
时间线：
━━━━━━━━━ 训练阶段 ━━━━━━━━━━━━━━━━━━━━ 验证阶段 ━━━━━━━━━━━

训练 DataLoader:
  [pin_memory线程] █████████████████████████████████████████  ← 不会停！InfiniteDataLoader 持续预取

训练 forward/backward:
  [GPU 显存]       █████████████████████                      ← 训练结束，empty_cache 释放

验证 DataLoader (pin_memory=True 时):
                                      [pin_memory线程] █████  ← 新的 pin_memory 注册请求

验证 forward:
                                      [GPU 显存]       █████  ← 同时进行前向传播

                                      ↑ 三者同时竞争 CUDA 资源 → OOM!
```

关键在于**训练的 `InfiniteDataLoader` 在验证期间仍然活跃**。它的 pin_memory 线程持续运行，不断向 CUDA 注册新的页锁定内存。如果验证 DataLoader 也开启 pin_memory，就会出现：

- **训练 pin_memory 线程**：持续注册
- **验证 pin_memory 线程**：也在注册
- **验证前向传播**：需要分配 GPU 显存

三方同时竞争 CUDA 资源，在显存边界情况下触发 OOM。

**训练阶段不会有这个问题**，因为训练时只有一个 DataLoader 的 pin_memory 线程在运行，没有竞争。因此训练保持 `pin_memory=True` 以获得最佳数据加载速度，验证设为 `pin_memory=False` 消除竞争。

---

## 八、`gc.collect()` + `torch.cuda.empty_cache()` 的必要性分析

### 8.1 工作原理

```python
gc.collect()               # 回收 Python 层面的循环引用对象（可能持有 CUDA 张量）
torch.cuda.empty_cache()   # 将 CUDA caching allocator 缓存的空闲块归还给驱动
```

PyTorch 的 CUDA 内存分配器会**缓存**已释放的显存块以便复用（避免频繁调用 `cudaMalloc`/`cudaFree`）。`empty_cache()` 强制将这些缓存块归还给 CUDA 驱动，使其可被其他分配请求使用。

### 8.2 batch=16 时：非必要

| 阶段                   | 显存占用            | 剩余空间 |
| ---------------------- | ------------------- | -------- |
| 训练                   | ~9.65 GB            | ~14.4 GB |
| 验证（训练缓存未释放） | 缓存 + 验证 ≈ 数 GB | 充裕     |

batch=16 时训练仅占 ~9.65GB，即使不释放缓存，剩余 ~14GB 也足够验证使用。实测加与不加都能正常训练。

### 8.3 batch=32 时：必要（刚需）

| 阶段     | 无 empty_cache                                  | 有 empty_cache     |
| -------- | ----------------------------------------------- | ------------------ |
| 训练结束 | 缓存占 ~20.4 GB                                 | 释放缓存后 ~2-3 GB |
| 验证分配 | 需额外数 GB，总计逼近 24 GB → **系统卡顿/卡死** | 充裕空间，流畅运行 |

**实测对比**：

```
# 无 empty_cache，batch=32：
训练 20.4GB → 验证开始 → 系统极度卡顿 → 进程无响应 → 手动终止

# 有 empty_cache，batch=32：
训练 20.4GB → empty_cache → 验证 batch=64 → 流畅完成 ✅
1/200      20.4G  ...  100% 518/518 [02:51, 3.02it/s]
           Class  ...  100% 78/78 [00:21, 3.58it/s]      ← 验证流畅
2/200      20.5G  ...                                     ← 正常进入下一 epoch
```

**原因**：训练结束后，CUDA caching allocator 仍持有 ~18GB 的缓存块（虽然逻辑上已释放，但未归还驱动）。验证的前向传播需要分配新的显存块，allocator 的缓存块与新分配请求的大小不匹配时，会向驱动申请新块，导致总占用逼近 24GB 物理极限，系统卡顿甚至卡死。`empty_cache()` 将缓存归还驱动后，验证可以自由分配，不再有压力。

### 8.4 结论

| batch size                | `gc.collect()` + `empty_cache()` | 必要性                         |
| ------------------------- | -------------------------------- | ------------------------------ |
| batch=16（训练 ~9.65 GB） | 可选                             | 显存充裕，加不加均可           |
| batch=32（训练 ~20.4 GB） | **必要**                         | 不加会导致系统卡顿/卡死        |
| 通用建议                  | **保留**                         | 开销极小（<1ms），提供安全保障 |

---

## 九、最终结论

| 项目         | 内容                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------- |
| **根因**     | 验证 DataLoader 的 `pin_memory(device)` 与 A2C2f 注意力模块前向传播竞争 CUDA 显存        |
| **核心修复** | 仅对验证 DataLoader 禁用 `pin_memory`，训练 DataLoader 保持不变                          |
| **辅助修复** | 验证前执行 `gc.collect()` + `torch.cuda.empty_cache()` 释放训练缓存（大 batch 时为刚需） |
| **影响**     | 无训练速度损失，无精度影响，验证速度影响可忽略                                           |
| **适用范围** | 所有使用注意力机制的 YOLO 变体（YOLOv12 系列），尤其在 GPU 显存利用率较高时              |
| **batch=16** | 训练 ~9.65 GB，验证 batch=32，流畅稳定                                                   |
| **batch=32** | 训练 ~20.4 GB，验证 batch=64，需配合 `empty_cache()` 使用                                |
