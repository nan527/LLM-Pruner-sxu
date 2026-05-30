# LLM 压缩实验平台 — 前端设计文档

## 概述

基于 Gradio 构建全功能 Web 平台，整合三个模块：实验结果看板、实验操作平台、模型推理演示。与现有 Python 实验脚本（llama3.py, post_training.py, gptq_quantize*.py）复用。

## 技术选型

- **框架**：Gradio（Python 原生，与现有代码直接复用）
- **图表**：Plotly（交互式图表，Gradio 原生支持）
- **部署**：本地运行，`python app.py` 单入口

## 文件结构

```
app.py                     # Gradio 入口，Tab 组装
modules/
    dashboard.py           # 模块 A：结果看板
    experiment.py          # 模块 B：实验操作平台
    inference.py           # 模块 C：推理演示
```

## 模块 A — 结果看板

**功能**：静态展示已完成的实验结果。

**内容**：
- 实验结果表格（与 README 一致，Markdown 渲染）
- 三张 Plotly 柱状图：
  1. PPL 对比（各模型版本 PPL，越低越好）
  2. 模型体积对比（MB，叠加显示压缩比标注）
  3. 压缩比对比
- 结论建议文字（推荐方案 + 不推荐方案）

**数据来源**：硬编码实验结果数据（dict 形式），无外部依赖。

## 模块 B — 实验操作平台

**功能**：在浏览器中启动剪枝/微调/量化实验，后台执行现有脚本，实时输出日志。

**三个子区（Gradio Accordion）**：

### B1. 结构化剪枝
| 参数 | 类型 | 默认值 |
|------|------|--------|
| 基础模型路径 | Textbox | (需填写) |
| 剪枝比例 | Slider | 0.25 |
| MLP 剪枝起始层 | Number | 4 |
| MLP 剪枝终止层 | Number | 20 |
| Attention 剪枝起始层 | Number | 4 |
| Attention 剪枝终止层 | Number | 20 |
| 输出名称 | Textbox | tinyllama_prune |

点击"开始剪枝"→ 后台 `subprocess` 调用 `llama3.py`，TextArea 实时显示 stdout。

### B2. LoRA 微调
| 参数 | 类型 | 默认值 |
|------|------|--------|
| 剪枝后模型路径 | Textbox | (需填写) |
| 数据集 | Dropdown | yahma/alpaca-cleaned |
| LoRA rank | Number | 8 |
| Epochs | Number | 2 |
| 学习率 | Number | 1e-4 |
| Batch Size | Number | 4 |
| 输出目录 | Textbox | tune_log/xxx |

点击"开始微调"→ 后台调用 `post_training.py`。

### B3. 量化评估
| 参数 | 类型 | 默认值 |
|------|------|--------|
| 模型路径 | Textbox | (需填写) |
| 量化方法 | CheckboxGroup | GPTQ-W8, GPTQ-W4, Dynamic INT8 |

点击"开始量化"→ 后台调用 `gptq_quantize.py`。

**执行方式**：`subprocess.Popen` 异步执行，逐行读取 stdout，yield 到 Gradio TextArea 实现实时日志。

## 模块 C — 推理演示

**功能**：加载选定压缩模型，输入 prompt 实时生成文本。

**输入**：
| 参数 | 类型 | 默认值 |
|------|------|--------|
| 模型选择 | Dropdown | 列出可用模型版本 |
| Prompt | Textbox (多行) | (用户输入) |
| Max Tokens | Slider | 256 |
| Temperature | Slider | 0.7 |
| Top-p | Slider | 0.9 |

**输出**：
- 生成文本
- 生成耗时（秒）

**可选模型列表**：
- 原始 TinyLlama (FP16)
- 剪枝 25% + LoRA (FP16)
- 剪枝 25% + LoRA + W4
- 剪枝 25% + LoRA + W8
- 剪枝 50% + LoRA (FP16)
- 剪枝 50% + LoRA + W4
- 剪枝 50% + LoRA + W8

**加载策略**：用户选择模型后点击"加载模型"，加载完成后可输入 prompt 生成。模型保持在内存中直到切换。

## 依赖

在现有 `requirement.txt` 基础上新增：
```
gradio
plotly
```

其余依赖（torch, transformers, peft, accelerate, bitsandbytes, auto-gptq）已存在。

## 边界与约束

- 仅本地运行，不考虑远程部署、认证、多用户
- 模块 B 实验执行期间，前端会阻塞在日志输出流
- 模块 C 同一时间只加载一个推理模型（8GB VRAM 限制）
- 不持久化用户数据，实验结果手动保存路径已在 README 中指定
