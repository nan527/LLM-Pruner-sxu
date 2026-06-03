# 项目结构说明

## 顶层目录

```
LLM-Pruner-main/
├── app.py                      # Web UI 主入口（Gradio）
├── llama3.py                   # 剪枝主脚本（支持 GQA/TinyLlama）
├── post_training.py            # LoRA 微调脚本
├── gptq_quantize.py            # 25% 组量化对比
├── gptq_quantize_50.py         # 50% 组量化对比
├── generate.py                 # Gradio 文本生成 UI（旧版）
├── test_speedup.py             # FLOPs/参数量/显存分析
├── hf_prune.py                 # HuggingFace 模型剪枝辅助
├── plot_final_results.py       # 结果图表生成
├── plot_ppl_comparison.py      # PPL 对比图表
├── plot_pruning_results.py     # 剪枝结果图表
│
├── requirements.txt            # Python 依赖清单
├── README.md                   # 项目主文档
├── PROJECT_STRUCTURE.md        # 项目结构说明（本文件）
├── USAGE_GUIDE.md              # 使用手册
├── PROJECT_EXPERIMENT_SUMMARY.md  # 实验摘要与结论
├── EXPERIMENT_PARAMS_LOG.md    # 详细参数与训练日志
├── LICENSE                     # Apache 2.0 许可证
│
├── modules/                    # Web UI 模块
├── LLMPruner/                  # 核心剪枝库
├── examples/                   # 其他模型剪枝示例
├── docs/                       # 文档与参考
├── lm-evaluation-harness/      # EleutherAI 评估工具
├── figures/                    # 生成的图表
├── scripts/                    # 辅助脚本
├── more_results/               # 扩展实验结果
│
├── prune_log/                  # 剪枝模型输出
├── tune_log/                   # LoRA 微调输出
└── quant_log/                  # 量化评估输出
```

## 核心脚本

| 文件 | 功能 | 输入 | 输出 |
|:---|:---|:---|:---|
| `llama3.py` | Taylor 一阶重要性剪枝 | 原始模型 | `prune_log/*/pytorch_model.bin` |
| `post_training.py` | LoRA 微调恢复 | 剪枝模型 | `tune_log/*/` (adapter) |
| `gptq_quantize.py` | W8/W4 量化评估 | 剪枝+LoRA 模型 | PPL 报告 |
| `gptq_quantize_50.py` | 50% 组量化评估 | 剪枝+LoRA 模型 | PPL 报告 |
| `app.py` | Web 实验平台 | — | http://127.0.0.1:7860 |

## Web UI 模块 (`modules/`)

```
modules/
├── __init__.py
├── dashboard.py        # 结果看板：数据表格 + Plotly 图表 + 摘要卡片
├── experiment.py       # 实验操作：剪枝/微调/量化三步流水线
└── inference.py        # 推理演示：单模型推理 + A/B 对比
```

- `dashboard.py`：展示实验结果，包含 PPL 对比柱状图、体积压缩比双轴图、HTML 色彩编码表格
- `experiment.py`：封装 `llama3.py` / `post_training.py` / 量化逻辑，通过 Gradio 界面调用
- `inference.py`：加载模型进行文本生成，支持原始/剪枝/LoRA 多版本切换

## 核心库 (`LLMPruner/`)

```
LLMPruner/
├── __init__.py
├── torch_pruning/              # 剪枝引擎（依赖图分析）
│   ├── importance.py           # 重要性估计器（Magnitude, BNScale, LAMP, Random）
│   ├── ops.py                  # 操作类型定义
│   ├── _helpers.py             # 索引映射工具
│   └── pruner/
│       ├── function.py         # 基础剪枝函数（Conv, Linear, BN, LN, Embed, MHA）
│       └── algorithms/
│           ├── metapruner.py   # MetaPruner 核心剪枝器
│           └── scheduler.py    # 线性稀疏度调度
│
├── pruner/                     # LLM 专用剪枝器
│   ├── hf_llama_pruner.py     # LLaMA/TinyLlama 剪枝器（含权重合并）
│   ├── hf_baichuan_pruner.py  # Baichuan 剪枝器
│   ├── hf_chatglm_pruner.py   # ChatGLM 剪枝器
│   └── llama_pruner.py        # 原始 Meta LLaMA 剪枝器
│
├── models/                     # 自定义模型实现（依赖图兼容）
│   ├── hf_llama/              # HuggingFace LLaMA 模型
│   ├── hf_baichuan/           # Baichuan 模型
│   ├── hf_bloom/              # BLOOM 模型
│   ├── hf_chatglm/            # ChatGLM 模型
│   └── llama/                 # 原始 Meta LLaMA
│
├── peft/                       # 内置 PEFT 库（v0.3.0.dev0）
│   ├── tuners/
│   │   ├── lora.py            # LoRA 实现
│   │   ├── adalora.py         # AdaLoRA
│   │   ├── prefix_tuning.py   # Prefix Tuning
│   │   └── prompt_tuning.py   # Prompt Tuning
│   └── utils/                 # PEFT 工具函数
│
├── evaluator/
│   └── ppl.py                 # Wikitext-2 困惑度评估器
│
├── datasets/
│   ├── ppl_dataset.py         # PPL 评估数据集加载
│   ├── example_samples.py     # BookCorpus 样本（Taylor 重要性）
│   └── dialogue.py            # 对话数据集
│
├── templates/
│   └── prompts.py             # 生成测试 Prompt 模板
│
└── utils/
    ├── logger.py              # 结构化日志
    └── prompter.py            # Alpaca 指令模板
```

## 输出目录

### `prune_log/` — 剪枝模型

```
prune_log/
├── tinyllama_prune/            # 25% 剪枝
│   ├── pytorch_model.bin       # 剪枝后模型权重
│   ├── tokenizer_config.json
│   └── 2026-04-23-*/           # 时间戳日志
└── tinyllama_prune_50/         # 50% 剪枝
    ├── pytorch_model.bin
    └── 2026-04-30-*/
```

### `tune_log/` — LoRA 适配器

```
tune_log/
├── tinyllama_tune_25/          # 25% 组微调
│   ├── adapter_model.bin       # LoRA 权重
│   ├── adapter_config.json
│   ├── checkpoint-200/         # 训练检查点
│   ├── checkpoint-400/
│   └── checkpoint-500/
└── tinyllama_tune_50/          # 50% 组微调
    ├── adapter_model.bin
    └── checkpoint-*/
```

### `quant_log/` — 量化日志

```
quant_log/
├── tinyllama_quant_25/         # 25% 组量化评估日志
└── tinyllama_quant_50/         # 50% 组量化评估日志
```

## 数据流

```
原始模型 (TinyLlama-1.1B FP16)
    │
    ▼  llama3.py (Taylor 剪枝)
剪枝模型 (prune_log/)
    │
    ▼  post_training.py (LoRA 微调)
剪枝 + LoRA 模型 (tune_log/)
    │
    ├──▶ gptq_quantize.py (W8/W4 量化)
    │       │
    │       ▼  量化后 PPL 评估
    │
    └──▶ app.py (Web UI 全流程)
```

## 外部依赖目录

| 目录 | 说明 |
|:---|:---|
| `lm-evaluation-harness/` | EleutherAI 评估工具（bundled） |
| `examples/` | BLOOM、Baichuan 剪枝示例 |
| `docs/` | 原始 LLM-Pruner 文档 |
| `figures/` | 生成的结果图表 |
| `more_results/` | Llama-3、Llama-3.1 等扩展结果 |
