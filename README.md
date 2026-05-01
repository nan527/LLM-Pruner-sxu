# LLM 结构化剪枝与量化 — 课程设计实验

基于 [LLM-Pruner](https://github.com/horseee/LLM-Pruner) 框架，在消费级 GPU（RTX 4060 Laptop, 8GB VRAM）上完成 **TinyLlama-1.1B** 的全流程压缩实验：结构化剪枝 → LoRA 微调恢复 → 权重量化对比。

## 项目简介

本项目探索大语言模型在资源受限环境下的压缩方案，核心工作包括：

1. **结构化剪枝**：使用 Taylor 一阶重要性估计，对 Attention 与 MLP 模块进行 block-wise 剪枝（25% / 50% 两组对比）。
2. **LoRA 微调恢复**：在剪枝后模型上应用 LoRA（Low-Rank Adaptation）进行低成本后训练，恢复语言能力。
3. **量化评估**：对比 GPTQ（W8 / W4）与 PyTorch Dynamic INT8 两种量化策略对模型体积与精度的影响。

评估指标：Wikitext-2 Perplexity（PPL）、模型体积、压缩比。

## 文件结构

```
.
├── llama3.py                  # 剪枝主脚本（支持 GQA）
├── post_training.py           # LoRA 微调脚本
├── gptq_quantize.py           # 25% 组量化对比（W8 / W4 + Dynamic INT8）
├── gptq_quantize_50.py        # 50% 组量化对比
├── LLMPruner/                 # LLM-Pruner 核心库（含兼容性修复）
│   ├── torch_pruning/         # 依赖图分析与剪枝引擎
│   ├── datasets/              # 数据集与评估工具
│   └── evaluator/             # PPL 评估器
├── scripts/                   # 辅助脚本
├── docs/                      # 原始项目文档与参考
│   └── README_ORIGINAL.md     # LLM-Pruner 原始 README
├── PROJECT_EXPERIMENT_SUMMARY.md   # 实验摘要（含结论建议）
└── EXPERIMENT_PARAMS_LOG.md        # 详细参数与训练日志
```

## 环境准备

```bash
pip install -r requirement.txt
```

主要依赖：`torch`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `auto-gptq`（量化）。

## 实验步骤

### Step 1：结构化剪枝（Pruning）

使用 `llama3.py` 对 TinyLlama 执行 block-wise Taylor 剪枝：

**25% 剪枝组**

```bash
python llama3.py \
    --base_model <PATH_TO_TINYLLAMA> \
    --pruning_ratio 0.25 \
    --device cuda --eval_device cuda \
    --block_wise \
    --block_mlp_layer_start 4 --block_mlp_layer_end 20 \
    --block_attention_layer_start 4 --block_attention_layer_end 20 \
    --pruner_type taylor --taylor param_first \
    --save_ckpt_log_name tinyllama_prune \
    --test_before_train --test_after_train --save_model
```

**50% 剪枝组**

```bash
python llama3.py \
    --base_model <PATH_TO_TINYLLAMA> \
    --pruning_ratio 0.5 \
    --device cuda --eval_device cuda \
    --block_wise \
    --block_mlp_layer_start 4 --block_mlp_layer_end 20 \
    --block_attention_layer_start 4 --block_attention_layer_end 20 \
    --pruner_type taylor --taylor param_first \
    --save_ckpt_log_name tinyllama_prune_50 \
    --test_before_train --test_after_train --save_model
```

**关键参数说明：**
- `--pruning_ratio`：剪枝比例（0.25 表示移除 25% 的组）
- `--block_wise`：按 block（Attention + MLP）进行结构化剪枝
- `--block_*_layer_start/end`：指定参与剪枝的层范围（首尾层保留）
- `--taylor param_first`：基于一阶梯度估计重要性
- 输出：`prune_log/tinyllama_prune/pytorch_model.bin`

### Step 2：LoRA 微调恢复（Post-Training）

使用 Alpaca 数据集（52K 条指令数据）对剪枝后模型进行 LoRA 微调：

**25% 组微调**

```bash
python post_training.py \
    --prune_model prune_log/tinyllama_prune/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir tune_log/tinyllama_tune_25 \
    --lora_r 8 --num_epochs 2 \
    --learning_rate 1e-4 --batch_size 4
```

**50% 组微调**

```bash
python post_training.py \
    --prune_model prune_log/tinyllama_prune_50/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir tune_log/tinyllama_tune_50 \
    --lora_r 8 --num_epochs 2 \
    --learning_rate 1e-4 --batch_size 4
```

**训练配置：**
- LoRA rank `r=8`, alpha=16, dropout=0.05
- 优化器：AdamW，FP16 训练
- 最大步数 500，warmup 50 步，每 200 步评估/保存
- 约 18 分钟完成（500 steps，RTX 4060）

### Step 3：量化评估（Quantization）

在剪枝+微调后的模型上执行权重量化对比：

```bash
# 25% 组量化
python gptq_quantize.py

# 50% 组量化
python gptq_quantize_50.py
```

量化方法：
- **GPTQ W8 / W4**：权重量化，group_size=128，per-group min-max 对称量化
- **Dynamic INT8**：PyTorch 官方动态量化（仅作对比，精度损失较大）

## 实验结果

| 模型版本 | PPL | 体积 (MB) | 压缩比 |
|---|---:|---:|---:|
| 原始 TinyLlama (FP16) | 17.42 | 2200 | 1.0x |
| 剪枝 25% | 35.95 | 1762 | 1.2x |
| 剪枝 25% + LoRA (FP16) | **26.71** | 1762 | 1.2x |
| 剪枝 25% + LoRA + W8 | **26.71** | 969 | 2.3x |
| 剪枝 25% + LoRA + W4 | 30.75 | 560 | **3.9x** |
| 剪枝 50% | 90.72 | 1426 | 1.5x |
| 剪枝 50% + LoRA (FP16) | 45.18 | 1426 | 1.5x |
| 剪枝 50% + LoRA + W8 | 45.18 | 796 | 2.8x |
| 剪枝 50% + LoRA + W4 | 50.50 | 471 | **4.7x** |
| 剪枝 + LoRA + Dynamic INT8 | 81.82 | 250 | 8.8x |

**结论建议：**
- 综合精度与压缩率，推荐 **剪枝 25% + LoRA + W4**（PPL 30.75，3.9x 压缩）。
- 若追求极限压缩可选 **剪枝 50% + LoRA + W4**（4.7x），但 PPL 明显升高。
- Dynamic INT8 虽压缩最大，但对 LLM 精度破坏显著，不建议作为主部署方案。

## 参考文档

- [PROJECT_EXPERIMENT_SUMMARY.md](PROJECT_EXPERIMENT_SUMMARY.md) — 实验摘要、详细步骤、最终结论
- [EXPERIMENT_PARAMS_LOG.md](EXPERIMENT_PARAMS_LOG.md) — 完整参数表、训练日志、量化细节
- [docs/README_ORIGINAL.md](docs/README_ORIGINAL.md) — LLM-Pruner 原始项目文档（支持模型、更多示例）
- [examples/README.md](examples/README.md) — 其他模型（BLOOM、Baichuan 等）剪枝指南

## 引用

本项目基于 LLM-Pruner：

```bibtex
@inproceedings{ma2023llmpruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Xinyin Ma and Gongfan Fang and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
