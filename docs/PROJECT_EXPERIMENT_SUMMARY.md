# LLM-Pruner 课程设计实验整理（TinyLlama-1.1B）

## 1. 实验目标
在 RTX 4060 Laptop（8GB VRAM）上完成压缩全流程：
- 结构化剪枝（Taylor）
- LoRA 微调恢复
- 量化对比（W8/W4 与 Dynamic INT8）

---

## 2. 实验步骤（可复现）

### Step A：剪枝（两组比例）
- 25% 剪枝：
```bash
f:\anaconda\envs\llm_prune\python.exe llama3.py --base_model f:\AA课程设计\llm_prune_quant_project\tinyllama_model --pruning_ratio 0.25 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 20 --block_attention_layer_start 4 --block_attention_layer_end 20 --save_ckpt_log_name tinyllama_prune --pruner_type taylor --taylor param_first --test_before_train --test_after_train --save_model
```
- 50% 剪枝：
```bash
f:\anaconda\envs\llm_prune\python.exe llama3.py --base_model f:\AA课程设计\llm_prune_quant_project\tinyllama_model --pruning_ratio 0.5 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 20 --block_attention_layer_start 4 --block_attention_layer_end 20 --save_ckpt_log_name tinyllama_prune_50 --pruner_type taylor --taylor param_first --test_before_train --test_after_train --save_model
```

### Step B：LoRA 微调（恢复精度）
- 25% 剪枝模型微调：
```bash
f:\anaconda\envs\llm_prune\python.exe post_training.py --prune_model prune_log/tinyllama_prune/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/tinyllama_tune_25 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 4
```
- 50% 剪枝模型微调：
```bash
f:\anaconda\envs\llm_prune\python.exe post_training.py --prune_model prune_log/tinyllama_prune_50/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/tinyllama_tune_50 --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 4
```

### Step C：量化对比
- 25% 剪枝组量化评估：
```bash
f:\anaconda\envs\llm_prune\python.exe gptq_quantize.py
```
- 50% 剪枝组量化评估：
```bash
f:\anaconda\envs\llm_prune\python.exe gptq_quantize_50.py
```

---

## 3. 主要文件说明

### 核心执行脚本
- `llama3.py`：剪枝主脚本（支持 GQA）
- `post_training.py`：LoRA 微调脚本
- `gptq_quantize.py`：25% 组量化对比（W8/W4 + Dynamic INT8）
- `gptq_quantize_50.py`：50% 组量化对比

### 核心模块（已做兼容修复）
- `LLMPruner/torch_pruning/dependency.py`
- `LLMPruner/datasets/example_samples.py`
- `LLMPruner/datasets/ppl_dataset.py`

### 结果产物目录
- `prune_log/tinyllama_prune/`
- `prune_log/tinyllama_prune_50/`
- `tune_log/tinyllama_tune_25/`
- `tune_log/tinyllama_tune_50/`

---

## 4. 最终结果表

| Model Version | PPL | Size(MB) | Ratio |
|---|---:|---:|---:|
| Original TinyLlama (FP16) | 17.42 | 2200 | 1.0x |
| Pruned 25% | 35.95 | 1762 | 1.2x |
| Pruned 25% + LoRA (FP16) | 26.71 | 1762 | 1.2x |
| Pruned 25% + LoRA + W8 | 26.71 | 969 | 2.3x |
| Pruned 25% + LoRA + W4 | 30.75 | 560 | 3.9x |
| Pruned 50% | 90.72 | 1426 | 1.5x |
| Pruned 50% + LoRA (FP16) | 45.18 | 1426 | 1.5x |
| Pruned 50% + LoRA + W8 | 45.18 | 796 | 2.8x |
| Pruned 50% + LoRA + W4 | 50.50 | 471 | 4.7x |
| Pruned + LoRA + Dynamic INT8 | 81.82 | 250 | 8.8x |

---

## 5. 结论建议（报告可直接引用）
1. 综合精度与压缩率，推荐主方案：**Pruned 25% + LoRA + W4**（PPL 30.75，3.9x 压缩）。
2. 若追求极限压缩：**Pruned 50% + LoRA + W4**（4.7x），但精度更差。
3. Dynamic INT8 虽压缩大，但对 LLM 精度破坏明显，不建议作为主部署方案。

---

## 6. 已清理文件
已删除以下无用临时文件：
- `debug_run.py`
- `debug_output.txt`
- `output.log`
- `quantize_eval.py`


| 模型版本                    | PPL  | 大小     | 压缩比 |
| 原始 TinyLlama             | 17.42 | 2200 MB | 1x    |
| 剪枝 25%                   | 35.95 | 1762 MB | 1.25x |
| 剪枝 + LoRA (FP16)         | 26.71 | 1762 MB | 1.25x |
| 剪枝 + LoRA + W8           | 26.71 | 969 MB  | 2.3x  |
| 剪枝 + LoRA + W4           | 30.75 | 560 MB  | 3.9x  |
| 剪枝 + LoRA + Dynamic INT8 | 81.82 | 250 MB  | 8.8x  |