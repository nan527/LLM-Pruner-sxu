# 实验详细参数与日志记录

## Step 1：结构化剪枝（Taylor Pruning）

### 脚本
`llama3.py`

### 公共参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--base_model` | `f:\AA课程设计\llm_prune_quant_project\tinyllama_model` | 基础模型路径 |
| `--device` | `cuda` | 剪枝计算设备 |
| `--eval_device` | `cuda` | 评估设备 |
| `--block_wise` | ✅ | 按 block 进行结构化剪枝 |
| `--block_mlp_layer_start` | `4` | MLP 剪枝起始层 |
| `--block_mlp_layer_end` | `20` | MLP 剪枝终止层 |
| `--block_attention_layer_start` | `4` | Attention 剪枝起始层 |
| `--block_attention_layer_end` | `20` | Attention 剪枝终止层 |
| `--pruner_type` | `taylor` | 使用 Taylor 一阶重要性估计 |
| `--taylor` | `param_first` | 基于参数梯度估计重要性 |
| `--test_before_train` | ✅ | 剪枝前测试 PPL |
| `--test_after_train` | ✅ | 剪枝后测试 PPL |
| `--save_model` | ✅ | 保存剪枝后模型 |

### 实验 A：25% 剪枝

| 参数 | 值 |
|---|---|
| `--pruning_ratio` | `0.25` |
| `--save_ckpt_log_name` | `tinyllama_prune` |
| 执行时间 | 2026-04-23 |
| 输出路径 | `prune_log/tinyllama_prune/pytorch_model.bin` |

**结果：**
- 剪枝前 PPL：17.42
- 剪枝后 PPL：35.95
- 模型大小：1762 MB

### 实验 B：50% 剪枝

| 参数 | 值 |
|---|---|
| `--pruning_ratio` | `0.5` |
| `--save_ckpt_log_name` | `tinyllama_prune_50` |
| 执行时间 | 2026-04-30 |
| 输出路径 | `prune_log/tinyllama_prune_50/pytorch_model.bin` |

**结果：**
- 剪枝前 PPL：17.42
- 剪枝后 PPL：90.72
- 模型大小：1426 MB

---

## Step 2：LoRA 微调（Post-Training）

### 脚本
`post_training.py`

### 公共参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--data_path` | `yahma/alpaca-cleaned` | 训练数据集（52K 条指令数据） |
| `--lora_r` | `8` | LoRA 秩 |
| `--lora_alpha` | `16` | LoRA 缩放系数 |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj` | LoRA 作用的模块 |
| `--num_epochs` | `2` | 训练轮数 |
| `--learning_rate` | `1e-4` | 学习率 |
| `--batch_size` | `4` | 批大小 |
| `--micro_batch_size` | `4` | micro batch 大小 |
| `--max_steps` | `500` | 最大训练步数 |
| `--warmup_steps` | `50` | 学习率预热步数 |
| `--eval_steps` | `200` | 评估间隔步数 |
| `--save_steps` | `200` | 保存间隔步数 |
| `--fp16` | `True` | 使用半精度训练 |
| `--optim` | `adamw_torch` | 优化器 |

### 实验 A：25% 剪枝模型微调

| 参数 | 值 |
|---|---|
| `--prune_model` | `prune_log/tinyllama_prune/pytorch_model.bin` |
| `--output_dir` | `tune_log/tinyllama_tune_25` |
| 执行时间 | 2026-04-30 |
| 训练时长 | ~18 分钟（500 steps） |

**训练日志（关键指标）：**

| step | loss | learning_rate |
|---:|---:|---:|
| 100 | 1.403 | 2.222e-05 |
| 200 | — | — |
| 500 | — | — |
| final | 1.557 (train_loss) | — |

- eval loss：1.521
- train_samples_per_second：1.8
- 训练完成后 PPL（wikitext2）：26.71

**Checkpoint 列表：**
- `checkpoint-200/`
- `checkpoint-400/`
- `checkpoint-500/`（最终使用）

### 实验 B：50% 剪枝模型微调

| 参数 | 值 |
|---|---|
| `--prune_model` | `prune_log/tinyllama_prune_50/pytorch_model.bin` |
| `--output_dir` | `tune_log/tinyllama_tune_50` |

**结果：**
- 训练后 PPL：45.18

---

## Step 3：量化评估

### 脚本
- `gptq_quantize.py`（25% 组）
- `gptq_quantize_50.py`（50% 组）

### 量化方法参数

#### W8 / W4 权重量化（weight-only）

| 参数 | 值 | 说明 |
|---|---|---|
| `bits` | `8` / `4` | 量化位宽 |
| `group_size` | `128` | 每组独立计算 scale 和 zero point |
| 量化目标 | `torch.nn.Linear` 层 | 只量化权重，激活保持 FP16 |
| 量化方式 | 对称 min-max per-group | 每 128 个权重一组 |

#### Dynamic INT8（对比组）

| 参数 | 值 | 说明 |
|---|---|---|
| API | `torch.quantization.quantize_dynamic` | PyTorch 官方动态量化 |
| 量化目标 | `torch.nn.Linear` 层 | |
| `dtype` | `torch.qint8` | |
| 激活处理 | 运行时动态校准 | 不预先校准激活分布 |

### 量化结果汇总

| 模型版本 | PPL | 大小(MB) | 压缩比 |
|---|---:|---:|---:|
| 原始 TinyLlama (FP16) | 17.42 | 2200 | 1.0x |
| 剪枝 25% | 35.95 | 1762 | 1.2x |
| 剪枝 25% + LoRA (FP16) | 26.71 | 1762 | 1.2x |
| 剪枝 25% + LoRA + W8 | 26.71 | 969 | 2.3x |
| 剪枝 25% + LoRA + W4 | 30.75 | 560 | 3.9x |
| 剪枝 50% | 90.72 | 1426 | 1.5x |
| 剪枝 50% + LoRA (FP16) | 45.18 | 1426 | 1.5x |
| 剪枝 50% + LoRA + W8 | 45.18 | 796 | 2.8x |
| 剪枝 50% + LoRA + W4 | 50.50 | 471 | 4.7x |
| 剪枝 25% + LoRA + Dynamic INT8 | 81.82 | 250 | 8.8x |

---

## 评估方法

| 项目 | 说明 |
|---|---|
| 评估数据集 | Wikitext-2 (raw, test split) |
| 评估指标 | Perplexity (PPL) |
| 序列长度 | 128 tokens |
| Batch size | 1 |
| 计算设备 | CUDA (量化模型) / CPU (Dynamic INT8) |
| 评估脚本 | `LLMPruner/evaluator/ppl.py` |
