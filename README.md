# LLM 压缩实验平台

基于 [LLM-Pruner](https://github.com/horseee/LLM-Pruner)（NeurIPS 2023）框架，在消费级 GPU 上完成 **TinyLlama-1.1B** 的全流程压缩实验。

## 实验概览

| 阶段 | 方法 | 脚本 |
|:---|:---|:---|
| 结构化剪枝 | Taylor 一阶重要性估计，block-wise 剪枝 | `scripts/llama3.py` |
| LoRA 微调恢复 | Low-Rank Adaptation 后训练恢复 | `scripts/post_training.py` |
| 权重量化 | GPTQ W8/W4 + Dynamic INT8 对比 | `scripts/gptq_quantize.py` |

## 实验结果

| 模型版本 | PPL | 体积 (MB) | 压缩比 |
|:---|---:|---:|---:|
| 原始 TinyLlama (FP16) | 17.42 | 2200 | 1.0x |
| 剪枝 25% + LoRA + W8 | 26.71 | 969 | 2.3x |
| **剪枝 25% + LoRA + W4** | **30.75** | **560** | **3.9x** |
| 剪枝 50% + LoRA + W4 | 50.50 | 471 | 4.7x |

> 推荐方案：**剪枝 25% + LoRA + W4**（PPL 30.75，3.9x 压缩，560MB）

## 环境要求

- Python >= 3.9
- CUDA >= 12.0
- GPU 显存 >= 8GB（RTX 4060 Laptop 测试通过）

## 快速开始

```bash
# 1. 创建 conda 环境
conda create -n llm_prune python=3.10
conda activate llm_prune

# 2. 安装 PyTorch（CUDA 12.4）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 启动 Web UI（推荐）
python app.py
```

## 实验步骤

### Step 1：结构化剪枝

```bash
python scripts/llama3.py \
    --base_model <模型路径> \
    --pruning_ratio 0.25 \
    --device cuda --eval_device cuda \
    --block_wise \
    --block_mlp_layer_start 4 --block_mlp_layer_end 20 \
    --block_attention_layer_start 4 --block_attention_layer_end 20 \
    --pruner_type taylor --taylor param_first \
    --save_ckpt_log_name tinyllama_prune \
    --test_before_train --test_after_train --save_model
```

### Step 2：LoRA 微调恢复

```bash
python scripts/post_training.py \
    --prune_model prune_log/tinyllama_prune/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir tune_log/tinyllama_tune_25 \
    --lora_r 8 --num_epochs 2 \
    --learning_rate 1e-4 --batch_size 4
```

### Step 3：量化评估

```bash
python scripts/gptq_quantize.py     # 25% 组
python scripts/gptq_quantize_50.py  # 50% 组
```

## Web UI 平台

项目提供基于 Gradio 的 Web 界面，支持可视化操作：

```bash
conda activate llm_prune
python app.py
```

功能包括：
- **结果看板**：实验数据表格 + Plotly 交互图表
- **实验操作**：剪枝 / 微调 / 量化三步流水线，实时日志
- **推理演示**：单模型推理 + A/B 对比（原始 vs 剪枝）

## 项目结构

```
LLM-Pruner-main/
├── app.py                      # Web UI 主入口
├── requirements.txt            # 依赖清单
├── README.md                   # 项目文档
│
├── scripts/                    # 实验脚本
│   ├── llama3.py               # 剪枝主脚本
│   ├── post_training.py        # LoRA 微调脚本
│   ├── gptq_quantize.py        # 25% 量化评估
│   ├── gptq_quantize_50.py     # 50% 量化评估
│   └── test_speedup.py         # 模型性能分析
│
├── modules/                    # Web UI 模块
│   ├── dashboard.py            # 结果看板
│   ├── experiment.py           # 实验操作
│   └── inference.py            # 推理演示
│
├── LLMPruner/                  # 核心剪枝库
├── docs/                       # 文档
│   ├── USAGE_GUIDE.md          # 使用手册
│   ├── PROJECT_STRUCTURE.md    # 结构说明
│   ├── PROJECT_EXPERIMENT_SUMMARY.md  # 实验摘要
│   └── EXPERIMENT_PARAMS_LOG.md       # 参数日志
│
├── prune_log/                  # 剪枝模型输出
├── tune_log/                   # LoRA 微调输出
└── quant_log/                  # 量化评估输出
```

## 文档导航

- [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) — 完整使用手册
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) — 项目结构详解
- [docs/PROJECT_EXPERIMENT_SUMMARY.md](docs/PROJECT_EXPERIMENT_SUMMARY.md) — 实验摘要与结论
- [docs/EXPERIMENT_PARAMS_LOG.md](docs/EXPERIMENT_PARAMS_LOG.md) — 详细参数与训练日志
- [docs/README_ORIGINAL.md](docs/README_ORIGINAL.md) — LLM-Pruner 原始文档

## 引用

```bibtex
@inproceedings{ma2023llmpruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Xinyin Ma and Gongfan Fang and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## 许可证

Apache 2.0 License
