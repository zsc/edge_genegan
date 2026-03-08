# SPEC: EdgeMap-Guided Video Continuation via GeneGAN-Style Cross-Modal Disentanglement

## 0. Project Context (This Repo)

本实现运行于本机以下约束/资源之上：

- 参考代码：`~/d/genegan_torch`（CelebA GeneGAN PyTorch）。本仓库会复用其工程习惯（`pyproject.toml` 依赖、checkpoint/logging 形式、`python -m <pkg>.cli.*` 的 CLI 组织方式），但不会复用其包名以避免冲突。
- 数据：`/mnt/sdc2/Vimeo-90k/vimeo_septuplet.zip`（Vimeo-90K septuplet）。默认工作流以“先解压到目录，再做 edge cache”为准；直接从 zip 读属于可选优化而非必需。
- 包名：为了避免与参考仓库的 `genegan` 包冲突，本项目 Python package 命名为 `edge_genegan`。

---

## 1. Summary

实现一个基于 **GeneGAN 风格 disentanglement + cross-modal swap** 的视频续写系统：

- 输入过去若干 RGB 帧 `F[1:T]`
- 输入未来若干 edge map `E[T+1:T+K]`
- 输出未来 RGB 帧 `F_hat[T+1:T+K]`

核心思想：

- **edge 不是 epsilon**。
- `epsilon` 是 **edge encoder 的 modality-private residual**，训练中被 nulling loss 压到 0。
- edge 是“只有结构、没有外观纹理”的完整观测，可视为 `x_{z0}`。
- RGB 是“结构 + 外观”的观测，可视为 `x_{za}`。
- 训练时通过同一 shot 内两帧的交换学习：
  - `(E_t, F_s) -> (F_t, E_s)`
  - 对称地 `(E_s, F_t) -> (F_s, E_t)`
- double-swap / cycle 仅做 **弱一致性正则**，不是主监督。

本 SPEC 面向 Codex / 代码生成代理。要求它按本文件实现一个 **可训练、可评估、可推理** 的 PyTorch 工程，而不是只写 demo。

---

## 2. Goals

### 2.1 Primary Goal

构建一个训练与推理完整可运行的系统，满足：

1. 在 Vimeo 类短视频数据集上，按 **同 shot、短间隔** 采样两帧 RGB `F_t, F_s`
2. 为它们生成 edge map `E_t, E_s`
3. 学习共享结构 latent `z` 与 RGB 外观 latent `a`
4. 学习 edge-only 表达 `(z, epsilon)`，其中 `epsilon -> 0`
5. 通过 swap 训练，让模型可以使用未来 edge 和过去 appearance 生成未来 RGB
6. 支持 rollout：从历史 RGB 提取 appearance memory，对未来 edge 序列逐帧渲染未来 RGB

### 2.2 Deliverables

必须产出：

- 数据预处理脚本
- 训练脚本
- 验证 / 测试脚本
- 单条视频或 batch 推理脚本
- 默认配置文件
- README（简明运行说明）
- 基础单元测试 / smoke tests

### 2.3 Non-Goals

本阶段不做：

- 端到端预测未来 edge（假设未来 edge 已给定）
- 文本条件控制
- 音频条件控制
- 多相机 / 3D 建模
- 复杂 object tracking supervision
- 超大规模分布式优化

---

## 3. Method Definition

### 3.1 Notation

- `F_t`: time `t` 的 RGB frame，shape `[3, H, W]`
- `E_t`: time `t` 的 edge map，shape `[1, H, W]` 或 `[C_e, H, W]`
- `z_t`: 结构 latent（shared structural state）
- `a_t`: RGB 外观 latent（appearance code）
- `epsilon_t`: edge branch private residual，应被压到 0

### 3.2 Encoders / Decoders

定义：

```text
(z_t^E, epsilon_t) = EncE(E_t)
(z_t^F, a_t)       = EncF(F_t)

E_hat_t = DecE(z_t)
F_hat_t = DecF(z_t, a)
```

其中：

- `EncE`：输入 edge，仅保留结构信息；输出 `z_t^E` 与极小私有残差 `epsilon_t`
- `EncF`：输入 RGB，输出结构 `z_t^F` 与 appearance `a_t`
- `DecE`：仅用 `z` 重建 edge
- `DecF`：用 `z + a` 重建 RGB

### 3.3 Training Swap

对于来自同一 shot 的两帧 `t, s`：

```text
F_rec_t = DecF(z_t^F, a_t)
F_rec_s = DecF(z_s^F, a_s)
E_rec_t = DecE(z_t^E)
E_rec_s = DecE(z_s^E)

F_swap_t = DecF(z_t^E, a_s)   # (E_t, F_s) -> F_t
F_swap_s = DecF(z_s^E, a_t)   # (E_s, F_t) -> F_s

E_swap_t = DecE(z_t^F)        # RGB structural code should recover edge
E_swap_s = DecE(z_s^F)
```

### 3.4 Inference for Video Continuation

给定历史 RGB `F[1:T]` 与未来 edge `E[T+1:T+K]`：

```text
a_star = AggAppearance(a_1, ..., a_T)
for k in 1..K:
    z_future = EncE(E_{T+k}).z
    F_hat_{T+k} = DecF(z_future, a_star)
```

`AggAppearance` 默认对历史 appearance 做 mean / attention pooling，并可选 EMA。

---

## 4. Critical Design Decisions

### 4.1 Edge Is Not Epsilon

必须严格遵守：

- **不要**把 `E_t` 直接实现为 `epsilon`
- `epsilon` 只表示 edge encoder 中应该被 null 掉的私有残差
- `E_t` 是“结构可观测但外观缺失”的输入模态

这是本实现最重要的理论约束之一。

### 4.2 Same-Shot, Small Temporal Gap

训练数据必须满足：

- `F_t, F_s` 来自同一 shot / 同一 clip
- 初始阶段 temporal gap `|s - t|` 较小
- 推荐 curriculum：从 `1~3` 帧开始，再扩展到 `1~5` / `1~7`

不允许随机跨视频或跨 shot 配对。

### 4.3 Swap Is Primary, Cycle Is Weak Regularization

必须：

- 以 `swap reconstruction` 为主训练目标
- `double-swap` / `cycle` 仅作小权重 regularizer
- 不要把 cycle 设计成主要监督，否则训练不稳定风险较大

### 4.4 Use Shared Structural Latent

必须显式加入 `z_t^E` 与 `z_t^F` 的一致性约束。

### 4.5 RGB GAN Only

对抗损失仅施加在 RGB 输出上：

- `F_rec`
- `F_swap`
- rollout 生成的 `F_hat`

edge 分支优先使用 `L1/BCE/Dice`，不强制 edge GAN。

---

## 5. Dataset and Preprocessing

### 5.1 Dataset

优先支持：

- Vimeo-90K triplet / septuplet 或等价 Vimeo 风格短视频数据

要求数据加载器支持：

- 以 clip 为单位读取连续帧
- 在一个 clip 内采样两个索引 `t, s`
- 可控制 temporal gap 上限

#### 5.1.1 Vimeo-90K Septuplet (local default)

本机数据源：

- zip：`/mnt/sdc2/Vimeo-90k/vimeo_septuplet.zip`

解压后的期望目录结构（zip 内带 `vimeo_septuplet/` 前缀）：

```text
vimeo_septuplet/
  sep_trainlist.txt
  sep_testlist.txt
  readme.txt
  sequences/
    00001/0001/im1.png ... im7.png
    00001/0002/im1.png ... im7.png
    ...
```

其中：

- `sep_{train,test}list.txt` 每行是一个 clip id（例如 `00001/0001`）
- clip 对应帧路径为 `sequences/<clip_id>/im{1..7}.png`

默认约定：

- `data.root` 指向解压后的 `vimeo_septuplet/` 目录（例如 `/mnt/sdc2/Vimeo-90k/vimeo_septuplet`）
- `edge_root` 指向 edge cache 根目录（例如 `/mnt/sdc2/Vimeo-90k/vimeo_septuplet_edges`）
- rollout 默认使用 septuplet 的固定切分：`history_len=3`（im1..im3），`future_len=4`（im4..im7）

### 5.2 Preprocessing

统一预处理：

- resize 到 `256x256`（默认）
- RGB 归一化到 `[-1, 1]`
- edge 归一化到 `[0, 1]`

### 5.3 Edge Extraction

需要支持两种 edge 管线：

1. `offline_canny`：离线 Canny，简单稳定
2. `offline_soft_edge`：离线 soft edge（默认用 **Sobel 梯度幅值** 作为连续 edge 强度图；后续可替换为 HED / DexiNed 等预训练网络输出）

实现要求：

- edge extraction 作为独立脚本
- 将 edge 缓存到磁盘，避免训练时重复计算
- 数据集类优先读取缓存 edge

缓存目录约定（必须与原图一一对应，保持相同层级与文件名）：

```text
<edge_root>/<mode>/sequences/<clip_id>/im1.png ... im7.png
```

edge 文件要求：

- 单通道 PNG（uint8，取值 `0..255`）
- 训练读取时再归一化到 `[0,1]`

### 5.4 Dataset Output Schema

每个 sample 至少返回：

```python
{
    "frame_t": Tensor[3, H, W],
    "frame_s": Tensor[3, H, W],
    "edge_t": Tensor[1, H, W],
    "edge_s": Tensor[1, H, W],
    "t_index": int,
    "s_index": int,
    "gap": int,
    "clip_id": str,
}
```

如果做 rollout 验证，再额外返回：

```python
{
    "history_frames": Tensor[T, 3, H, W],
    "future_edges": Tensor[K, 1, H, W],
    "future_frames": Tensor[K, 3, H, W],
}
```

---

## 6. Architecture Requirements

### 6.1 High-Level Architecture

实现以下模块：

- `EdgeEncoder`
- `RgbEncoder`
- `EdgeDecoder`
- `RgbDecoder`
- `AppearanceAggregator`
- `EdgeAdherenceExtractor`（固定，不训练）
- `PatchDiscriminator`（RGB only）

### 6.2 Structural Latent `z`

`z` 不应仅为全局向量，默认应为 **spatial latent feature map**，以保留几何结构。

推荐：

- `z`: `[Cz, H/16, W/16]`
- 默认 `Cz = 256`

### 6.3 Appearance Latent `a`

`a` 为全局 appearance code：

- shape 推荐 `[Ca]`
- 默认 `Ca = 256`

注入方式：

- 优先实现 `AdaIN` 或 `FiLM`
- 备选：broadcast + concat

### 6.4 Encoders

#### EdgeEncoder

输入：`[B, 1, H, W]`
输出：

- `z_e`: spatial latent `[B, Cz, H/16, W/16]`
- `eps`: global or low-rank residual `[B, Ceps]`

要求：

- `eps` 分支轻量化
- 便于施加 `nulling loss`

#### RgbEncoder

输入：`[B, 3, H, W]`
输出：

- `z_f`: spatial latent `[B, Cz, H/16, W/16]`
- `a`: appearance code `[B, Ca]`

要求：

- 结构 / appearance 在架构上部分解耦
- 可通过 global average pooling + MLP 获得 `a`

### 6.5 Decoders

#### EdgeDecoder

输入：`z`
输出：`edge_hat`

要求：

- U-Net-like upsampling
- 输出 shape 与 `E_t` 一致
- 支持 `sigmoid`

#### RgbDecoder

输入：`z, a`
输出：`frame_hat`

要求：

- appearance 通过 AdaIN / FiLM 注入 decoder blocks
- 输出 `[-1, 1]`
- 支持 skip / residual blocks

### 6.6 Optional Temporal State

本阶段默认不强制 recurrent state。

但代码结构必须允许后续扩展：

```python
frame_hat, state = rgb_decoder(z, a, prev_state=None)
```

当前实现里 `prev_state` 可默认为 `None`。

---

## 7. Losses

必须实现下列损失。

### 7.1 Reconstruction Loss

```text
L_rec = L1(F_rec_t, F_t) + L1(F_rec_s, F_s)
      + lambda_e_rec * [L1(E_rec_t, E_t) + L1(E_rec_s, E_s)]
```

RGB 可附加：

- `LPIPS(F_rec, F)`

### 7.2 Swap Reconstruction Loss

```text
L_swap = L1(F_swap_t, F_t) + L1(F_swap_s, F_s)
       + lambda_e_swap * [L1(E_swap_t, E_t) + L1(E_swap_s, E_s)]
```

这是主损失之一。

### 7.3 Shared Structural Consistency

```text
L_shared = L1(z_t^E, z_t^F) + L1(z_s^E, z_s^F)
```

如果 `z` 是 feature map，直接按 feature map 做 `L1`。

### 7.4 Nulling Loss

```text
L_null = ||epsilon_t||_1 + ||epsilon_s||_1
```

必须确保 edge branch 的 private residual 被压缩。

### 7.5 Latent Drift Constraint

对 swap 后的 RGB 再编码，约束其保持正确结构与外观：

```text
(z_hat_t, a_hat_t) = EncF(F_swap_t)
(z_hat_s, a_hat_s) = EncF(F_swap_s)

L_drift = L1(z_hat_t, z_t^E) + L1(a_hat_t, a_s)
        + L1(z_hat_s, z_s^E) + L1(a_hat_s, a_t)
```

### 7.6 Edge Adherence Loss

使用固定 edge extractor `H`：

```text
L_edge = L1(H(F_swap_t), E_t) + L1(H(F_swap_s), E_s)
```

如果 rollout 训练时使用未来监督，也对 rollout 输出施加同类约束。

### 7.7 Adversarial Loss (RGB Only)

采用 hinge GAN 或 vanilla PatchGAN 均可，默认用 hinge。

判别器目标：

- 真样本：`F_t, F_s`
- 假样本：`F_rec_t, F_rec_s, F_swap_t, F_swap_s`

### 7.8 Weak Cycle Loss

仅作为可选小权重 regularizer：

```text
(E_cyc_t, F_cyc_s) = T(E_swap_t, F_swap_s)
(E_cyc_s, F_cyc_t) = T(E_swap_s, F_swap_t)

L_cyc = L1(E_cyc_t, E_t) + L1(F_cyc_s, F_s)
      + L1(E_cyc_s, E_s) + L1(F_cyc_t, F_t)
```

这里 `T` 表示再次做一次 cross-modal swap。

默认：

- 第 1 阶段关闭 `L_cyc`
- 第 2 阶段小权重启用

### 7.9 Optional Parallelogram-Like Constraint

不在 raw pixel 上硬套原版 parallelogram。

仅允许在共享 embedding 上实现可选版本，默认关闭。

### 7.10 Total Loss

```text
L_total = lambda_rec   * L_rec
        + lambda_swap  * L_swap
        + lambda_shared* L_shared
        + lambda_null  * L_null
        + lambda_drift * L_drift
        + lambda_edge  * L_edge
        + lambda_adv   * L_adv
        + lambda_lpips * L_lpips
        + lambda_cyc   * L_cyc
```

### 7.11 Default Weights

初始默认值：

```yaml
lambda_rec: 10.0
lambda_swap: 10.0
lambda_shared: 2.0
lambda_null: 0.1
lambda_drift: 2.0
lambda_edge: 5.0
lambda_adv: 1.0
lambda_lpips: 1.0
lambda_cyc: 0.5   # stage 2 only
lambda_e_rec: 5.0
lambda_e_swap: 5.0
```

这些值写入默认 config，可在训练时覆盖。

---

## 8. Training Stages

必须实现 staged training。

### Stage 1: Reconstruction + Swap Warmup

目标：先学会稳定编码、解码和 swap。

启用：

- `L_rec`
- `L_swap`
- `L_shared`
- `L_null`
- `L_drift`
- `L_edge`
- `L_adv`
- `L_lpips`

关闭：

- `L_cyc`

数据策略：

- `gap <= 3`

### Stage 2: Larger Gap + Weak Cycle

目标：提升更大 motion 下的泛化。

启用：

- Stage 1 全部损失
- `L_cyc`（小权重）

数据策略：

- `gap <= 5` 或 `gap <= 7`

### Stage 3: Rollout Fine-Tuning (Optional but Supported)

目标：更贴近测试时的 video continuation。

训练 sample 结构：

- 历史 RGB `F[1:T]`
- 未来 edge `E[T+1:T+K]`
- GT future RGB `F[T+1:T+K]`

过程：

```text
a_star = AggAppearance(history RGB)
for each future step:
    z_k = EncE(E_{T+k}).z
    F_hat_{T+k} = DecF(z_k, a_star)
```

loss：

- future RGB reconstruction
- LPIPS
- edge adherence
- optional adversarial on future frames

---

## 9. Evaluation

### 9.1 Frame-Level Metrics

实现：

- `PSNR`
- `SSIM`
- `LPIPS`

### 9.2 Edge Consistency Metrics

实现至少一种：

- `L1(H(F_hat), E_gt)`
- edge precision / recall / F1（在阈值化后计算）

### 9.3 Video-Level Metric

可选实现：

- `FVD`

如果实现成本过高，可先预留接口并在 README 说明。

### 9.4 Qualitative Outputs

验证脚本必须导出：

- `F_t`, `F_s`, `E_t`, `E_s`
- `F_rec_*`, `E_rec_*`
- `F_swap_*`, `E_swap_*`
- rollout 结果 GIF / MP4
- 拼图可视化 PNG

---

## 10. Inference Requirements

### 10.1 Input Format

推理脚本支持：

1. 一段历史 RGB 帧目录 + 一段未来 edge 帧目录
2. 一个 `.npz` / `.pt` 文件，内含历史 RGB 和未来 edge

### 10.2 Output Format

输出：

- 未来 RGB PNG 序列
- 合成 MP4
- 可选中间结果：提取的 `a_star`、decoded edge preview

### 10.3 Appearance Aggregation

默认实现：

```text
a_star = mean(a_1, ..., a_T)
```

同时预留：

- `ema`
- `attention_pool`

通过 config 切换。

---

## 11. Repository Layout

Codex 必须按如下结构组织代码：

```text
project/
  SPEC.md
  README.md
  pyproject.toml
  configs/
    default.yaml
    stage1.yaml
    stage2.yaml
    rollout.yaml
  edge_genegan/
    __init__.py
    cli/
      __init__.py
      train.py
      validate.py
      infer_rollout.py
      export_samples.py
    preprocess/
      __init__.py
      extract_edges.py
      build_splits.py
    data/
      __init__.py
      vimeo_dataset.py
      edge_cache_dataset.py
      samplers.py
      transforms.py
    models/
      __init__.py
      blocks.py
      edge_encoder.py
      rgb_encoder.py
      edge_decoder.py
      rgb_decoder.py
      appearance_aggregator.py
      discriminator.py
      system.py
    losses/
      __init__.py
      reconstruction.py
      adversarial.py
      perceptual.py
      edge_adherence.py
      cycle.py
    trainers/
      __init__.py
      trainer.py
      hooks.py
    evaluators/
      __init__.py
      metrics.py
      rollout_eval.py
      visualizer.py
    utils/
      __init__.py
      ckpt.py
      config.py
      image_io.py
      logging.py
      seed.py
  tests/
    test_dataset.py
    test_model_shapes.py
    test_forward_smoke.py
    test_train_step.py
```

### 11.1 Repo Baseline

本仓库默认按以下本地路径与参考实现执行（默认值以 `configs/*.yaml` 为主）：

- 参考实现约束：`~/d/genegan_torch`
- Vimeo-90k 解压根目录：`/mnt/sdc2/Vimeo-90k/vimeo_septuplet`
- edge cache 根目录：`/mnt/sdc2/Vimeo-90k/vimeo_septuplet_edges`
- 边缘 zip：`/mnt/sdc2/Vimeo-90k/vimeo_septuplet.zip`

边缘缓存路径约定：

- `<edge_root>/<mode>/sequences/<clip_id>/im*.png`

---

## 12. Implementation Requirements

### 12.1 Framework

- Python 3.10+
- PyTorch
- torchvision
- PyYAML
- tqdm
- Pillow / OpenCV

可选：

- LPIPS library
- einops
- imageio / moviepy

### 12.2 Code Quality

必须：

- 类型注解尽量完善
- 函数与类写清楚 docstring
- 模块职责明确
- 不要把所有逻辑塞进单个脚本
- 训练、验证、推理解耦

### 12.3 Reproducibility

必须支持：

- random seed
- deterministic flags（尽可能）
- checkpoint save / resume
- config dump 到输出目录

### 12.4 Logging

至少支持：

- stdout logging
- 每 N step 保存图像拼图
- 每 N epoch 保存 checkpoint
- 记录 loss 曲线

如果实现 tensorboard / wandb，需为可选依赖，默认关闭。

### 12.5 Config + Execution Contract

配置生效优先级（高到低）：

1. 命令行参数
2. 任务配置（如 `configs/stage1.yaml`）
3. `configs/default.yaml`

`train.amp` 与 `experiment.amp` 应支持回退逻辑（若 `train.amp` 未设置则使用 `experiment.amp`）。

验收要求（建议执行）：

- 训练脚本支持 `--steps`，并在到达上限后停止
- 训练与验证写入 `train.log`，包含 `L_total`、`L_rec`、`L_swap`、`L_shared`、`L_drift`、`L_edge`
- 每个 epoch 生成 `checkpoints/latest.pt` 与可选 `checkpoints/epoch_XXXX.pt`
- 验证脚本产出 `samples/` 可视化图
- 推理脚本产出 `frames/`、`grid.png`、`rollout.mp4`（或同等可播放视频）

---

## 13. Minimal Public APIs

Codex 必须提供以下 Python 接口。

### 13.1 Model System

```python
class EdgeRgbSwapSystem(nn.Module):
    def encode_edge(self, edge: torch.Tensor) -> dict: ...
    def encode_rgb(self, frame: torch.Tensor) -> dict: ...
    def decode_edge(self, z: torch.Tensor) -> torch.Tensor: ...
    def decode_rgb(
        self,
        z: torch.Tensor,
        appearance: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def forward_pair(
        self,
        frame_t: torch.Tensor,
        frame_s: torch.Tensor,
        edge_t: torch.Tensor,
        edge_s: torch.Tensor,
    ) -> dict: ...

    def rollout(
        self,
        history_frames: torch.Tensor,
        future_edges: torch.Tensor,
    ) -> dict: ...
```

### 13.2 Trainer

```python
class Trainer:
    def train_step(self, batch: dict) -> dict: ...
    def validate_step(self, batch: dict) -> dict: ...
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...
```

---

## 14. Acceptance Criteria

只有在满足以下条件时，任务才算完成。

### 14.1 Data Pipeline Acceptance

- 能从 Vimeo 风格 clip 中读取连续帧
- 能缓存并读取 edge
- 能返回成对 `t, s` 样本
- 能控制 temporal gap
- smoke test 通过

### 14.2 Model Acceptance

- `forward_pair()` 可运行并返回完整中间结果
- shape 全部正确
- `rollout()` 可运行并输出 `K` 帧未来 RGB
- 单卡上至少可完成一个 train step

### 14.3 Loss Acceptance

- 所有必需损失都已实现
- 可通过 config 开关控制 `L_cyc`、LPIPS、GAN
- loss 字典中能看到各分量

### 14.4 Training Acceptance

- 训练脚本支持 stage1 / stage2 config
- 能保存 checkpoint
- 能 resume
- 能导出可视化样本

### 14.5 Inference Acceptance

- 可对给定历史 RGB + 未来 edge 进行 rollout
- 能输出 PNG 序列与 MP4

### 14.6 Test Acceptance

至少包含：

- dataset shape test
- model shape test
- forward smoke test
- one-step train smoke test

---

## 15. Config Specification

默认配置文件示例：

```yaml
data:
  dataset: vimeo_septuplet
  root: /mnt/sdc2/Vimeo-90k/vimeo_septuplet
  edge_root: /mnt/sdc2/Vimeo-90k/vimeo_septuplet_edges
  image_size: 256
  edge_mode: offline_soft_edge
  history_len: 3
  future_len: 4
  max_gap_stage1: 3
  max_gap_stage2: 5
  num_workers: 8

model:
  z_channels: 256
  a_channels: 256
  base_channels: 64
  norm: instance
  rgb_decoder_inject: adain
  use_temporal_state: false

train:
  stage: stage1
  batch_size: 16
  epochs: 100
  lr_g: 0.0002
  lr_d: 0.0002
  betas: [0.5, 0.999]
  amp: true
  seed: 42
  log_every: 100
  vis_every: 500
  save_every_epoch: 1

loss:
  lambda_rec: 10.0
  lambda_swap: 10.0
  lambda_shared: 2.0
  lambda_null: 0.1
  lambda_drift: 2.0
  lambda_edge: 5.0
  lambda_adv: 1.0
  lambda_lpips: 1.0
  lambda_cyc: 0.5
  lambda_e_rec: 5.0
  lambda_e_swap: 5.0
  use_gan: true
  use_lpips: true
  use_cycle: false

infer:
  appearance_pool: mean
  save_png: true
  save_mp4: true
```

---

## 16. Suggested Implementation Order

Codex 应严格按以下顺序实现，避免一次性写太多不可调试代码。

1. 数据集与 edge 缓存脚本
2. 基础 encoder / decoder
3. `forward_pair()` 跑通
4. reconstruction + swap losses
5. shared/null/drift/edge losses
6. trainer + logger + checkpoint
7. GAN
8. rollout 推理
9. stage2 weak cycle
10. tests + README

不要先实现复杂 rollout，再补基础模块。

---

## 17. Common Failure Modes and Required Safeguards

### 17.1 Appearance Leakage into Edge Branch

现象：`EncE` 偷偷编码纹理信息。

要求：

- 保留 `epsilon` 分支并施加 `L_null`
- 控制 `EncE` 容量不要过大
- 监控 swap 时是否无 edge 也能“背答案”

### 17.2 Structure Ignored by RGB Decoder

现象：`DecF` 只靠 appearance 生成模糊平均图。

要求：

- 强制 `L_edge`
- 强制 `L_shared`
- 监控 `H(F_swap)` 与目标 edge 的误差

### 17.3 Cycle Destabilizes Training

要求：

- stage1 默认关闭 `L_cyc`
- stage2 才启用，且权重低
- 允许通过 config 完全关闭

### 17.4 Cross-Shot Misalignment

要求：

- 数据层面禁止跨 shot 配对
- 若数据源无法提供显式 shot 标注，则默认一个 clip 内视为同 shot

---

## 18. README Requirements

README 至少包括：

- 项目简介
- 环境安装
- 数据准备
- edge 提取
- stage1 训练命令
- stage2 训练命令
- rollout 推理命令
- 输出说明

README 不需要长篇论文解释，但要足够让工程可运行。

---

## 19. Example Commands

```bash
# 1) 解压数据（推荐解压到同目录，得到 /mnt/sdc2/Vimeo-90k/vimeo_septuplet/）
unzip /mnt/sdc2/Vimeo-90k/vimeo_septuplet.zip -d /mnt/sdc2/Vimeo-90k/

# 2) 离线提取 edge 并缓存
python -m edge_genegan.preprocess.extract_edges \
  --data-root /mnt/sdc2/Vimeo-90k/vimeo_septuplet \
  --output-root /mnt/sdc2/Vimeo-90k/vimeo_septuplet_edges \
  --mode soft

# 3) 训练
python -m edge_genegan.cli.train --config configs/stage1.yaml
python -m edge_genegan.cli.train --config configs/stage2.yaml

# 4) rollout 推理（历史 RGB + 未来 edge）
python -m edge_genegan.cli.infer_rollout \
  --config configs/rollout.yaml \
  --checkpoint ./outputs/stage2/latest.pt \
  --history-dir ./example/history_rgb \
  --future-edge-dir ./example/future_edges \
  --output-dir ./outputs/demo_rollout
```

---

## 20. Final Instruction to Codex

按本 SPEC 直接实现代码。

优先保证：

- 结构正确
- 模块清晰
- forward / train / infer 全链路可运行
- test 和 README 完整

不要擅自改动核心方法定义：

- 不要把 edge 当成 `epsilon`
- 不要把 cycle 变成主损失
- 不要用跨 shot 随机配对
- 不要省略 `shared structural consistency`
- 不要省略 `nulling loss`

如遇实现细节未定义，遵循以下优先级：

1. 保持与本 SPEC 的理论约束一致
2. 选择最简单、最稳妥、最容易调试的实现
3. 不引入与本方法无关的复杂组件
