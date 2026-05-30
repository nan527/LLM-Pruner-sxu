# LLM 压缩实验平台 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Gradio 构建三Tab全功能Web平台：实验结果看板 + 实验操作（剪枝/微调/量化）+ 模型推理演示

**Architecture:** 单入口 `app.py` 组装三个 Tab，每个 Tab 由 `modules/` 下独立模块构建 UI。实验模块通过 subprocess 调用现有 Python 脚本，推理模块直接加载模型使用 transformers 生成。

**Tech Stack:** Gradio, Plotly, PyTorch, Transformers, PEFT

---

### Task 1: 创建项目骨架

**Files:**
- Create: `modules/__init__.py`
- Create: `modules/dashboard.py` (skeleton)
- Create: `modules/experiment.py` (skeleton)
- Create: `modules/inference.py` (skeleton)

- [ ] **Step 1: 创建 modules 目录和 init 文件**

```bash
mkdir -p modules
```

- [ ] **Step 2: 写入 modules/__init__.py**

```python
# LLM Compression Platform - Modules
```

- [ ] **Step 3: 写入 modules/dashboard.py 骨架**

```python
"""Module A: Results Dashboard"""

def build_dashboard():
    """Build the dashboard UI. Called within a gr.Blocks context."""
    pass
```

- [ ] **Step 4: 写入 modules/experiment.py 骨架**

```python
"""Module B: Experiment Platform"""

def build_experiment():
    """Build the experiment UI. Called within a gr.Blocks context."""
    pass
```

- [ ] **Step 5: 写入 modules/inference.py 骨架**

```python
"""Module C: Inference Demo"""

def build_inference():
    """Build the inference UI. Called within a gr.Blocks context."""
    pass
```

- [ ] **Step 6: 验证文件结构**

Run: `ls modules/`
Expected: `__init__.py  dashboard.py  experiment.py  inference.py`

---

### Task 2: 实现模块 A — 结果看板

**Files:**
- Modify: `modules/dashboard.py`

- [ ] **Step 1: 替换 dashboard.py 完整内容**

```python
"""Module A: Results Dashboard — 静态实验数据可视化"""
import gradio as gr
import plotly.graph_objects as go

RESULTS = [
    {"版本": "原始 TinyLlama (FP16)", "PPL": 17.42, "体积(MB)": 2200, "压缩比": 1.0},
    {"版本": "剪枝 25%", "PPL": 35.95, "体积(MB)": 1762, "压缩比": 1.2},
    {"版本": "剪枝 25% + LoRA (FP16)", "PPL": 26.71, "体积(MB)": 1762, "压缩比": 1.2},
    {"版本": "剪枝 25% + LoRA + W8", "PPL": 26.71, "体积(MB)": 969, "压缩比": 2.3},
    {"版本": "剪枝 25% + LoRA + W4", "PPL": 30.75, "体积(MB)": 560, "压缩比": 3.9},
    {"版本": "剪枝 50%", "PPL": 90.72, "体积(MB)": 1426, "压缩比": 1.5},
    {"版本": "剪枝 50% + LoRA (FP16)", "PPL": 45.18, "体积(MB)": 1426, "压缩比": 1.5},
    {"版本": "剪枝 50% + LoRA + W8", "PPL": 45.18, "体积(MB)": 796, "压缩比": 2.8},
    {"版本": "剪枝 50% + LoRA + W4", "PPL": 50.50, "体积(MB)": 471, "压缩比": 4.7},
    {"版本": "剪枝 + LoRA + Dynamic INT8", "PPL": 81.82, "体积(MB)": 250, "压缩比": 8.8},
]

CONCLUSION_MD = """## 结论建议

| 策略 | PPL | 压缩比 | 评价 |
|------|-----|--------|------|
| 剪枝 25% + LoRA + W4 | 30.75 | 3.9x | **推荐** — 精度与体积最佳平衡 |
| 剪枝 50% + LoRA + W4 | 50.50 | 4.7x | 极限压缩，精度损失较大 |
| Dynamic INT8 | 81.82 | 8.8x | **不推荐** — 精度破坏严重 |

- 综合推荐主部署方案为 **剪枝 25% + LoRA + W4**
- Dynamic INT8 虽压缩率最高，但对 LLM 生成质量影响显著，不建议作为主力方案
"""


def _make_bar_chart(x, y, title, color):
    fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color=color)])
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        height=450,
        margin=dict(b=200, l=80, r=20, t=50),
    )
    return fig


def build_dashboard():
    gr.Markdown("# 实验结果看板")

    with gr.Tab("数据表格"):
        gr.Dataframe(RESULTS, label="压缩实验结果汇总", interactive=False)
        gr.Markdown(CONCLUSION_MD)

    versions = [d["版本"] for d in RESULTS]

    with gr.Tab("PPL 对比"):
        ppl_chart = _make_bar_chart(
            versions, [d["PPL"] for d in RESULTS],
            "Perplexity 对比（越低越好）", "steelblue"
        )
        gr.Plot(ppl_chart)

    with gr.Tab("模型体积对比"):
        size_chart = _make_bar_chart(
            versions, [d["体积(MB)"] for d in RESULTS],
            "模型体积对比 (MB)", "coral"
        )
        gr.Plot(size_chart)

    with gr.Tab("压缩比对比"):
        ratio_chart = _make_bar_chart(
            versions, [d["压缩比"] for d in RESULTS],
            "压缩比对比（越高越好）", "seagreen"
        )
        gr.Plot(ratio_chart)
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from modules.dashboard import build_dashboard; print('OK')"`
Expected: `OK`

---

### Task 3: 实现模块 B — 实验操作平台

**Files:**
- Modify: `modules/experiment.py`

- [ ] **Step 1: 替换 experiment.py 完整内容**

```python
"""Module B: Experiment Platform — 剪枝 / 微调 / 量化"""
import os
import sys
import subprocess
import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(cmd_args):
    """Generator: run a script with subprocess, yield accumulated output."""
    cmd = [sys.executable] + cmd_args
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
        bufsize=1,
    )
    log_lines = []
    for line in process.stdout:
        log_lines.append(line)
        yield "".join(log_lines)
    process.wait()
    log_lines.append(f"\n=== 进程结束，退出码: {process.returncode} ===")
    yield "".join(log_lines)


# ---------- B1: Pruning ----------

def run_pruning(base_model, pruning_ratio, mlp_start, mlp_end,
                attn_start, attn_end, save_name):
    args = [
        "llama3.py",
        "--base_model", base_model,
        "--pruning_ratio", str(pruning_ratio),
        "--device", "cuda", "--eval_device", "cuda",
        "--block_wise",
        "--block_mlp_layer_start", str(int(mlp_start)),
        "--block_mlp_layer_end", str(int(mlp_end)),
        "--block_attention_layer_start", str(int(attn_start)),
        "--block_attention_layer_end", str(int(attn_end)),
        "--pruner_type", "taylor", "--taylor", "param_first",
        "--save_ckpt_log_name", save_name,
        "--test_before_train", "--test_after_train", "--save_model",
    ]
    yield from _run_script(args)


# ---------- B2: Fine-tuning ----------

def run_finetuning(prune_model, data_path, lora_r, num_epochs,
                   learning_rate, batch_size, output_dir, base_model):
    args = [
        "post_training.py",
        "--prune_model", prune_model,
        "--data_path", data_path,
        "--lora_r", str(int(lora_r)),
        "--num_epochs", str(int(num_epochs)),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(int(batch_size)),
        "--output_dir", output_dir,
        "--base_model", base_model,
    ]
    yield from _run_script(args)


# ---------- B3: Quantization ----------

def run_quantization(prune_path, lora_path, quant_methods):
    """Run quantization using the core functions from gptq_quantize."""
    import torch
    import time
    from LLMPruner.peft import PeftModel
    from LLMPruner.evaluator.ppl import PPLMetric
    from gptq_quantize import quantize_model_weights, calc_model_size_bits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_lines = []

    def log(msg):
        log_lines.append(msg + "\n")
        return "".join(log_lines)

    yield log(f"Loading pruned model from {prune_path}...")
    pruned_dict = torch.load(prune_path, map_location="cpu")
    tokenizer = pruned_dict["tokenizer"]
    model = pruned_dict["model"]
    model.half().to(device)

    if lora_path and os.path.exists(lora_path):
        yield log(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
        model = model.merge_and_unload()
        model.to(device)
        yield log("LoRA merged.")

    results = []

    if "GPTQ-W8" in quant_methods:
        yield log("=== GPTQ INT8 Weight-Only Quantization ===")
        est = calc_model_size_bits(model, quant_bits=8, group_size=128)
        yield log(f"Estimated INT8 size: {est:.1f} MB")
        model_int8 = quantize_model_weights(model, bits=8, group_size=128)
        start = time.time()
        ppl = PPLMetric(model_int8, tokenizer, ["wikitext2"], seq_len=128, batch_size=1, device=device)
        elapsed = time.time() - start
        results.append(f"GPTQ-W8 | PPL: {ppl['wikitext2']:.2f} | Size: {est:.0f} MB | Time: {elapsed:.0f}s")
        yield log(results[-1])
        del model_int8
        torch.cuda.empty_cache()

    if "GPTQ-W4" in quant_methods:
        yield log("=== GPTQ INT4 Weight-Only Quantization ===")
        # Reload model
        pruned_dict2 = torch.load(prune_path, map_location="cpu")
        model2 = pruned_dict2["model"]
        model2.half().to(device)
        if lora_path and os.path.exists(lora_path):
            model2 = PeftModel.from_pretrained(model2, lora_path, torch_dtype=torch.float16)
            model2 = model2.merge_and_unload()
            model2.to(device)

        est = calc_model_size_bits(model2, quant_bits=4, group_size=128)
        yield log(f"Estimated INT4 size: {est:.1f} MB")
        model_int4 = quantize_model_weights(model2, bits=4, group_size=128)
        start = time.time()
        ppl = PPLMetric(model_int4, tokenizer, ["wikitext2"], seq_len=128, batch_size=1, device=device)
        elapsed = time.time() - start
        results.append(f"GPTQ-W4 | PPL: {ppl['wikitext2']:.2f} | Size: {est:.0f} MB | Time: {elapsed:.0f}s")
        yield log(results[-1])
        del model_int4
        torch.cuda.empty_cache()

    yield log("\n=== 量化评估完成 ===")
    yield log("\n".join(results))


def build_experiment():
    gr.Markdown("# 实验操作平台")

    # ---- Pruning ----
    with gr.Accordion("Step 1: 结构化剪枝", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                prune_base = gr.Textbox(label="基础模型路径", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                prune_ratio = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="剪枝比例")
                prune_save = gr.Textbox(label="输出名称", value="tinyllama_prune")
            with gr.Column(scale=1):
                prune_mlp_start = gr.Number(label="MLP 起始层", value=4, precision=0)
                prune_mlp_end = gr.Number(label="MLP 终止层", value=20, precision=0)
                prune_attn_start = gr.Number(label="Attention 起始层", value=4, precision=0)
                prune_attn_end = gr.Number(label="Attention 终止层", value=20, precision=0)

        prune_btn = gr.Button("开始剪枝", variant="primary")
        prune_log = gr.Textbox(label="运行日志", lines=15, max_lines=30, autoscroll=True)

        prune_btn.click(
            fn=run_pruning,
            inputs=[prune_base, prune_ratio, prune_mlp_start, prune_mlp_end,
                    prune_attn_start, prune_attn_end, prune_save],
            outputs=prune_log,
        )

    # ---- Fine-tuning ----
    with gr.Accordion("Step 2: LoRA 微调", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                ft_prune = gr.Textbox(label="剪枝模型路径 (.bin)", value="prune_log/tinyllama_prune/pytorch_model.bin")
                ft_base = gr.Textbox(label="基础模型名", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                ft_data = gr.Dropdown(["yahma/alpaca-cleaned"], value="yahma/alpaca-cleaned", label="数据集")
                ft_out = gr.Textbox(label="输出目录", value="tune_log/tinyllama_tune_25")
            with gr.Column(scale=1):
                ft_lora_r = gr.Number(label="LoRA Rank", value=8, precision=0)
                ft_epochs = gr.Number(label="Epochs", value=2, precision=0)
                ft_lr = gr.Number(label="Learning Rate", value=1e-4)
                ft_bs = gr.Number(label="Batch Size", value=4, precision=0)

        ft_btn = gr.Button("开始微调", variant="primary")
        ft_log = gr.Textbox(label="运行日志", lines=15, max_lines=30, autoscroll=True)

        ft_btn.click(
            fn=run_finetuning,
            inputs=[ft_prune, ft_data, ft_lora_r, ft_epochs, ft_lr, ft_bs, ft_out, ft_base],
            outputs=ft_log,
        )

    # ---- Quantization ----
    with gr.Accordion("Step 3: 量化评估", open=False):
        with gr.Row():
            quant_prune = gr.Textbox(label="剪枝模型路径 (.bin)", value="prune_log/tinyllama_prune/pytorch_model.bin")
            quant_lora = gr.Textbox(label="LoRA 模型路径", value="tune_log/tinyllama_tune_25")
            quant_methods = gr.CheckboxGroup(
                ["GPTQ-W8", "GPTQ-W4"],
                value=["GPTQ-W8", "GPTQ-W4"],
                label="量化方法"
            )

        quant_btn = gr.Button("开始量化评估", variant="primary")
        quant_log = gr.Textbox(label="运行日志", lines=15, max_lines=30, autoscroll=True)

        quant_btn.click(
            fn=run_quantization,
            inputs=[quant_prune, quant_lora, quant_methods],
            outputs=quant_log,
        )
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from modules.experiment import build_experiment; print('OK')"`
Expected: `OK`

---

### Task 4: 实现模块 C — 推理演示

**Files:**
- Modify: `modules/inference.py`

- [ ] **Step 1: 替换 inference.py 完整内容**

```python
"""Module C: Inference Demo — 加载压缩模型进行文本生成"""
import os
import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PRESETS = {
    "原始 TinyLlama (FP16)": {
        "type": "hf",
        "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
    "剪枝 25% + LoRA (FP16)": {
        "type": "pruned_lora",
        "prune_path": "prune_log/tinyllama_prune/pytorch_model.bin",
        "lora_path": "tune_log/tinyllama_tune_25",
    },
    "剪枝 50% + LoRA (FP16)": {
        "type": "pruned_lora",
        "prune_path": "prune_log/tinyllama_prune_50/pytorch_model.bin",
        "lora_path": "tune_log/tinyllama_tune_50",
    },
}

_current_model = None
_current_tokenizer = None


def load_model(model_name, hf_path, prune_path, lora_path):
    """Load or reload the selected model into global state."""
    global _current_model, _current_tokenizer

    # Free previous model
    _current_model = None
    _current_tokenizer = None
    torch.cuda.empty_cache()

    preset = MODEL_PRESETS.get(model_name, {})
    model_type = preset.get("type", "hf")

    try:
        if model_type == "hf":
            path = hf_path or preset.get("model_path", "")
            _current_tokenizer = AutoTokenizer.from_pretrained(path)
            _current_model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16, device_map="auto"
            )

        elif model_type == "pruned_lora":
            prune = prune_path or preset.get("prune_path", "")
            lora = lora_path or preset.get("lora_path", "")

            from LLMPruner.peft import PeftModel

            pruned_dict = torch.load(prune, map_location="cpu")
            _current_tokenizer = pruned_dict["tokenizer"]
            _current_model = pruned_dict["model"]
            _current_model.half().to(DEVICE)

            if lora and os.path.exists(lora):
                _current_model = PeftModel.from_pretrained(
                    _current_model, lora, torch_dtype=torch.float16
                )
                _current_model = _current_model.merge_and_unload()
                _current_model.to(DEVICE)

        _current_model.eval()
        return f"模型加载成功: {model_name}\n设备: {DEVICE}"

    except Exception as e:
        return f"加载失败: {str(e)}"


def generate_text(prompt, max_tokens, temperature, top_p):
    """Generate text from the currently loaded model."""
    global _current_model, _current_tokenizer

    if _current_model is None or _current_tokenizer is None:
        return "请先加载模型"

    try:
        inputs = _current_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        start = time.time()

        with torch.no_grad():
            outputs = _current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=_current_tokenizer.pad_token_id or _current_tokenizer.eos_token_id,
            )

        elapsed = time.time() - start
        response = _current_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return f"{response}\n\n--- 生成耗时: {elapsed:.1f}s, {len(response)} 字符 ---"
    except Exception as e:
        return f"生成失败: {str(e)}"


def unload_model():
    """Release GPU memory."""
    global _current_model, _current_tokenizer
    if _current_model is not None:
        del _current_model
        _current_model = None
    if _current_tokenizer is not None:
        _current_tokenizer = None
    torch.cuda.empty_cache()
    return "模型已卸载"


def _on_preset_change(preset_name):
    """Auto-fill paths when preset changes."""
    preset = MODEL_PRESETS.get(preset_name, {})
    return (
        preset.get("model_path", ""),
        preset.get("prune_path", ""),
        preset.get("lora_path", ""),
    )


def build_inference():
    gr.Markdown("# 模型推理演示")

    with gr.Row():
        with gr.Column(scale=1):
            model_preset = gr.Dropdown(
                list(MODEL_PRESETS.keys()),
                value=list(MODEL_PRESETS.keys())[0],
                label="选择模型版本",
            )
            hf_path = gr.Textbox(label="HF 模型路径", visible=False)
            prune_path = gr.Textbox(label="剪枝 Checkpoint 路径")
            lora_path = gr.Textbox(label="LoRA 路径")

            with gr.Row():
                load_btn = gr.Button("加载模型", variant="primary")
                unload_btn = gr.Button("卸载模型")

            load_status = gr.Textbox(label="加载状态", lines=3)

        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="输入 Prompt",
                lines=4,
                placeholder="请输入文本...",
            )
            with gr.Row():
                max_tokens = gr.Slider(16, 512, value=128, step=16, label="Max Tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

            gen_btn = gr.Button("生成文本", variant="secondary")
            output_text = gr.Textbox(label="生成结果", lines=12, autoscroll=True)

    # Events
    model_preset.change(
        fn=_on_preset_change,
        inputs=model_preset,
        outputs=[hf_path, prune_path, lora_path],
    )

    load_btn.click(
        fn=load_model,
        inputs=[model_preset, hf_path, prune_path, lora_path],
        outputs=load_status,
    )

    unload_btn.click(fn=unload_model, inputs=[], outputs=load_status)

    gen_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_p],
        outputs=output_text,
    )
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from modules.inference import build_inference; print('OK')"`
Expected: `OK`

---

### Task 5: 创建主入口 app.py

**Files:**
- Create: `app.py`

- [ ] **Step 1: 写入 app.py**

```python
"""LLM Compression Experiment Platform — Gradio Web UI"""
import gradio as gr
from modules.dashboard import build_dashboard
from modules.experiment import build_experiment
from modules.inference import build_inference

TITLE = "LLM 压缩实验平台"
DESCRIPTION = """
TinyLlama-1.1B 结构化剪枝与量化实验平台。
支持：实验结果可视化 | 剪枝/微调/量化实验操作 | 压缩模型推理演示
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Tab("结果看板"):
        build_dashboard()

    with gr.Tab("实验操作"):
        build_experiment()

    with gr.Tab("推理演示"):
        build_inference()

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
```

- [ ] **Step 2: 验证 app.py 语法正确**

Run: `python -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

---

### Task 6: 启动验证

- [ ] **Step 1: 确认依赖已安装**

Run: `python -c "import gradio, plotly; print(f'Gradio {gradio.__version__}, Plotly installed')"`
Expected: 显示版本号（如缺失，运行 `pip install gradio plotly`）

- [ ] **Step 2: 启动应用（前台测试）**

Run: `python app.py`
Expected: 控制台输出 `Running on local URL: http://127.0.0.1:7860`

- [ ] **Step 3: 按 Ctrl+C 停止，确认无导入错误**
