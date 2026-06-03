# 使用手册

本手册介绍如何搭建环境、运行实验、使用 Web UI 平台。

---

## 1. 环境搭建

### 1.1 创建 Conda 环境

```bash
conda create -n llm_prune python=3.10
conda activate llm_prune
```

### 1.2 安装 PyTorch

根据你的 CUDA 版本选择对应的 PyTorch：

```bash
# CUDA 12.4（推荐，RTX 4060 测试通过）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证安装：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 期望输出: 2.x.x+cu124  True
```

### 1.3 安装项目依赖

```bash
cd LLM-Pruner-main
pip install -r requirements.txt
```

### 1.4 验证环境

```bash
python -c "
import torch
import gradio as gr
import transformers
from LLMPruner.peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric
print('All imports OK')
"
```

> **注意**：`torch` 必须在 `gradio` 之前导入，否则可能产生 segfault。

---

## 2. Web UI 平台

### 2.1 启动

```bash
conda activate llm_prune
python app.py
```

启动后访问 `http://127.0.0.1:7860`（端口自动递增，避免冲突）。

### 2.2 功能说明

#### Tab 1：结果看板

- 展示所有实验结果的汇总数据
- 三张摘要卡片：推荐方案 / 极限压缩 / 不推荐
- 彩色 HTML 表格：困惑度用绿/黄/红编码，压缩比用绿/黄编码
- Plotly 交互图表：困惑度对比柱状图、体积压缩比双轴图

#### Tab 2：实验操作

按顺序执行三步流水线，每步有实时日志输出：

| 步骤 | 功能 | 可调参数 |
|:---|:---|:---|
| Step 1 — 结构化剪枝 | 调用 `llama3.py` 做 Taylor 剪枝 | 剪枝比例、MLP/Attn 层范围、输出名称 |
| Step 2 — LoRA 微调 | 调用 `post_training.py` 恢复能力 | LoRA rank、epochs、学习率、batch size |
| Step 3 — 量化评估 | 对剪枝+LoRA 模型做 W8/W4 量化 | 选择 GPTQ-W8 / GPTQ-W4 |

#### Tab 3：推理演示

- **单模型推理**：选择模型版本（原始/剪枝25%/剪枝50%），输入 prompt，调节 temperature/top_p
- **A/B 对比**：同一 prompt 分别喂给两个模型，逐个加载省显存，并排对比

### 2.3 模型路径配置

Web UI 中的模型路径需要根据你的实际情况修改：

| 预设 | 默认路径 |
|:---|:---|
| 原始 TinyLlama | `F:\Download\tinyllama_model` |
| 剪枝 25% | `prune_log/tinyllama_prune/pytorch_model.bin` |
| LoRA 25% | `tune_log/tinyllama_tune_25` |
| 剪枝 50% | `prune_log/tinyllama_prune_50/pytorch_model.bin` |
| LoRA 50% | `tune_log/tinyllama_tune_50` |

> 如果模型在 HuggingFace 上，可直接使用模型名（如 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`），会自动下载。

---

## 3. 命令行实验

如果不使用 Web UI，可以按步骤手动执行。

### 3.1 结构化剪枝

**25% 剪枝**

```bash
python llama3.py \
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

**50% 剪枝**

```bash
python llama3.py \
    --base_model <模型路径> \
    --pruning_ratio 0.5 \
    --device cuda --eval_device cuda \
    --block_wise \
    --block_mlp_layer_start 4 --block_mlp_layer_end 20 \
    --block_attention_layer_start 4 --block_attention_layer_end 20 \
    --pruner_type taylor --taylor param_first \
    --save_ckpt_log_name tinyllama_prune_50 \
    --test_before_train --test_after_train --save_model
```

**参数说明**

| 参数 | 说明 |
|:---|:---|
| `--pruning_ratio` | 剪枝比例（0.25 = 移除 25% 的组） |
| `--block_wise` | 按 block（Attention + MLP）结构化剪枝 |
| `--block_*_layer_start/end` | 参与剪枝的层范围（首尾层保留） |
| `--taylor param_first` | 一阶梯度估计重要性 |
| `--save_ckpt_log_name` | 输出目录名（在 `prune_log/` 下） |

**输出**：`prune_log/tinyllama_prune/pytorch_model.bin`

### 3.2 LoRA 微调

```bash
# 25% 组
python post_training.py \
    --prune_model prune_log/tinyllama_prune/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir tune_log/tinyllama_tune_25 \
    --lora_r 8 --num_epochs 2 \
    --learning_rate 1e-4 --batch_size 4

# 50% 组
python post_training.py \
    --prune_model prune_log/tinyllama_prune_50/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir tune_log/tinyllama_tune_50 \
    --lora_r 8 --num_epochs 2 \
    --learning_rate 1e-4 --batch_size 4
```

**训练配置**

| 参数 | 值 |
|:---|:---|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Dropout | 0.05 |
| 优化器 | AdamW |
| 精度 | FP16 |
| 最大步数 | 500 |
| Warmup | 50 步 |
| 评估间隔 | 200 步 |
| 训练时间 | ~18 分钟（RTX 4060） |

### 3.3 量化评估

```bash
# 25% 组量化
python gptq_quantize.py

# 50% 组量化
python gptq_quantize_50.py
```

量化方法：
- **GPTQ W8**：8-bit 权重量化，group_size=128
- **GPTQ W4**：4-bit 权重量化，group_size=128
- **Dynamic INT8**：PyTorch 动态量化（仅作对比）

---

## 4. 模型分析

### 4.1 参数量/FLOPs 分析

```bash
python test_speedup.py
```

输出模型的参数量、FLOPs、GPU 显存占用。

### 4.2 生成测试

```bash
python generate.py
```

启动 Gradio 文本生成界面，用于测试模型生成质量。

---

## 5. 常见问题

### Q: `ImportError: cannot import name 'HfFolder'`

`huggingface_hub` 版本过新。降级：

```bash
pip install "huggingface_hub>=0.16,<1.0"
```

### Q: `torch` 和 `gradio` 一起导入时 segfault

必须先导入 `torch`：

```python
import torch      # 先
import gradio     # 后
```

### Q: `ModuleNotFoundError: No module named 'gradio'`

确认使用了正确的 conda 环境：

```bash
conda activate llm_prune
pip install gradio
```

### Q: CUDA out of memory

- 减小 batch size（`--batch_size 1`）
- 使用更小的剪枝比例
- 关闭其他占用 GPU 的程序

### Q: HuggingFace 下载慢

设置国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell:
$env:HF_ENDPOINT="https://hf-mirror.com"
```

---

## 6. 硬件要求

| 项目 | 最低要求 | 推荐配置 |
|:---|:---|:---|
| GPU | 8GB VRAM | RTX 4060 Laptop |
| 内存 | 16GB | 32GB |
| 磁盘 | 10GB 可用空间 | SSD |
| CUDA | 12.0+ | 12.4 |

## 7. 依赖版本参考

以下版本在 RTX 4060 Laptop + CUDA 12.4 环境下测试通过：

| 包 | 版本 |
|:---|:---|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 5.6.0 |
| Gradio | 5.50.0 |
| Accelerate | 1.13.0 |
| Datasets | 4.8.4 |
| PEFT | 0.19.1 |
| SentencePiece | 0.2.1 |
