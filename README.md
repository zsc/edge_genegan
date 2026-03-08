# Edge-Guided Video Continuation via GeneGAN-style Cross-Modal Disentanglement

本项目实现基于 `Vimeo-90K` 短视频的 `Edge -> RGB` 跨模态续写流程，核心是 GeneGAN 风格的共享结构编码与外观编码解耦：

- 输入：一段历史 RGB (`F[1:T]`)
- 输入：未来 edge map (`E[T+1:T+K]`)
- 输出：未来 RGB (`F_hat[T+1:T+K]`)

与默认设置无关的信息会在命令行和配置里覆盖，核心代码按 `Spec` 执行，支持 `stage1 -> stage2 -> rollout`。

## 环境

```bash
python -m pip install -e .
```

## 目录与数据准备

当前仓库默认数据源为本机路径：

- `/mnt/sdc2/Vimeo-90k/vimeo_septuplet.zip`
- 解压后目录建议为 `/mnt/sdc2/Vimeo-90k/vimeo_septuplet`

先解压 septuplet：

```bash
mkdir -p /mnt/sdc2/Vimeo-90k
unzip /mnt/sdc2/Vimeo-90k/vimeo_septuplet.zip -d /mnt/sdc2/Vimeo-90k
```

## Edge 缓存提取

离线提取 Canny / Soft（默认 Sobel）：

```bash
python -m edge_genegan.preprocess.extract_edges \
  --data-root /mnt/sdc2/Vimeo-90k/vimeo_septuplet \
  --output-root /mnt/sdc2/Vimeo-90k/vimeo_septuplet_edges \
  --mode soft \
  --processes 8
```

参数说明：

- `--mode canny`: OpenCV `Canny` 阈值边缘
- `--mode soft`: Sobel 梯度连续边缘（默认，替代 HED/DexiNed 的轻量版）
- `--image-size`: 统一重采样分辨率

## 训练

```bash
python -m edge_genegan.cli.train --config configs/stage1.yaml
python -m edge_genegan.cli.train --config configs/stage2.yaml
```

如果希望直接从 stage1 续训：

```bash
python -m edge_genegan.cli.train --config configs/stage2.yaml --resume /path/to/stage1/checkpoints/latest.pt
```

## 验证与可视化

```bash
python -m edge_genegan.cli.validate \
  --config configs/stage2.yaml \
  --checkpoint /path/to/stage2/checkpoints/latest.pt \
  --output-dir ./outputs/val
```

验证默认会输出：

- `F_t, F_s, E_t, E_s`
- `F_rec_t/F_rec_s`, `F_swap_t/F_swap_s`
- `E_rec_t/E_rec_s`, `E_swap_t/E_swap_s`
- 以及拼图图像与若干指标汇总

## Rollout 推理

给定历史 RGB 目录和未来 edge 目录：

```bash
python -m edge_genegan.cli.infer_rollout \
  --config configs/rollout.yaml \
  --checkpoint outputs/stage2/checkpoints/latest.pt \
  --history-dir examples/history_frames \
  --future-edge-dir examples/future_edges \
  --output-dir outputs/rollout_demo
```

也可以从 `.npz` / `.pt` 加载：

```bash
python -m edge_genegan.cli.infer_rollout \
  --config configs/rollout.yaml \
  --checkpoint outputs/stage2/checkpoints/latest.pt \
  --sample-path examples/sample.npz \
  --output-dir outputs/rollout_demo
```

`.npz/.pt` 需要包含：

- `history_frames`: `[T, 3, H, W]`, float in `[-1, 1]`
- `future_edges`: `[K, 1, H, W]`, float in `[0, 1]`

## 输出

- `outputs/<exp>/checkpoints/latest.pt`：最新权重
- `outputs/<exp>/checkpoints/epoch_xxx.pt`：定期快照
- `outputs/<exp>/samples`：训练可视化拼图
- `outputs/<exp>/train.log`: 日志文本
- `outputs/<exp>/config_dump.yaml`: 当前运行配置

## 测试

```bash
pytest -q
```
