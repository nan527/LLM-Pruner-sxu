"""Module B: Experiment Platform - pruning / fine-tuning / quantization"""
import os
import sys
import subprocess
import time
import ssl
import torch
import gradio as gr
from LLMPruner.peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
ssl._create_default_https_context = ssl._create_unverified_context


def _run_script(cmd_args):
    """Generator: run a script with subprocess, yield accumulated output."""
    cmd = [sys.executable] + cmd_args
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
        bufsize=1,
        env=env,
    )
    log_lines = []
    for line in process.stdout:
        log_lines.append(line)
        yield "".join(log_lines)
    process.wait()
    rc = process.returncode
    tag = "完成" if rc == 0 else "失败"
    log_lines.append(f"\n{'='*50}\n  进程结束 [{tag}]  退出码: {rc}\n{'='*50}")
    yield "".join(log_lines)


# ============================================================
# Quantization functions (unchanged)
# ============================================================

def _quantize_weight_per_group(weight, bits=4, group_size=128):
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    w = weight.float()
    if w.shape[-1] % group_size != 0:
        pad_size = group_size - (w.shape[-1] % group_size)
        w = torch.nn.functional.pad(w, (0, pad_size))
    w_grouped = w.reshape(-1, group_size)
    w_min = w_grouped.min(dim=1, keepdim=True).values
    w_max = w_grouped.max(dim=1, keepdim=True).values
    qmax = 2 ** bits - 1
    scales = (w_max - w_min) / qmax
    scales = scales.clamp(min=1e-8)
    zeros = w_min
    w_quant = torch.round((w_grouped - zeros) / scales).clamp(0, qmax)
    w_deq = w_quant * scales + zeros
    w_deq = w_deq.reshape(w.shape)
    if w_deq.shape != orig_shape:
        w_deq = w_deq[:orig_shape[0], :orig_shape[1]]
    return w_deq.to(orig_dtype)


def _quantize_model_weights(model, bits=4, group_size=128):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = _quantize_weight_per_group(
                module.weight.data, bits=bits, group_size=group_size
            )
            count += 1
    return model


def _calc_model_size_bits(model, quant_bits=4, group_size=128):
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            numel = module.weight.numel()
            n_groups = (numel + group_size - 1) // group_size
            total_bits += numel * quant_bits + n_groups * 16 * 2
        elif hasattr(module, 'weight') and module.weight is not None:
            if not any(isinstance(m, torch.nn.Linear) for m in module.children()):
                total_bits += module.weight.numel() * module.weight.element_size() * 8
    return total_bits / 8 / 1024 / 1024


# ============================================================
# Step 1: Pruning
# ============================================================

def run_pruning(base_model, pruning_ratio, mlp_start, mlp_end,
                attn_start, attn_end, save_name):
    base_model = base_model.strip().strip('"').strip("'")
    save_name = save_name.strip().strip('"').strip("'")
    args = [
        "scripts/llama3.py",
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


# ============================================================
# Step 2: LoRA Fine-tuning
# ============================================================

def run_finetuning(prune_model, data_path, lora_r, num_epochs,
                   learning_rate, batch_size, output_dir, base_model):
    prune_model = prune_model.strip().strip('"').strip("'")
    data_path = data_path.strip().strip('"').strip("'")
    output_dir = output_dir.strip().strip('"').strip("'")
    base_model = base_model.strip().strip('"').strip("'")
    args = [
        "scripts/post_training.py",
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


# ============================================================
# Step 3: Quantization
# ============================================================

def run_quantization(prune_path, lora_path, quant_methods):
    prune_path = prune_path.strip().strip('"').strip("'")
    lora_path = lora_path.strip().strip('"').strip("'") if lora_path else lora_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_lines = []

    def log(msg):
        log_lines.append(msg + "\n")
        return "".join(log_lines)

    if not quant_methods:
        yield "未选择量化方法"
        return

    yield log(f"加载剪枝模型: {prune_path}...")
    pruned_dict = torch.load(prune_path, map_location="cpu", weights_only=False)
    tokenizer = pruned_dict["tokenizer"]
    model = pruned_dict["model"]
    model.half().to(device)

    if lora_path and os.path.exists(lora_path):
        yield log(f"加载 LoRA 适配器: {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
        model = model.merge_and_unload()
        model.to(device)
        yield log("LoRA 已合并.")

    results = []

    if "GPTQ-W8" in quant_methods:
        yield log("\n[1/2] GPTQ INT8 Weight-Only Quantization")
        est = _calc_model_size_bits(model, quant_bits=8, group_size=128)
        yield log(f"  预估大小: {est:.1f} MB")
        model_int8 = _quantize_model_weights(model, bits=8, group_size=128)
        start = time.time()
        ppl = PPLMetric(model_int8, tokenizer, ["wikitext2"], seq_len=128, batch_size=1, device=device)
        elapsed = time.time() - start
        results.append(f"GPTQ-W8  PPL={ppl['wikitext2']:.2f}  Size={est:.0f}MB  Time={elapsed:.0f}s")
        yield log(f"  {results[-1]}")
        del model_int8
        torch.cuda.empty_cache()

    if "GPTQ-W4" in quant_methods:
        yield log("\n[2/2] GPTQ INT4 Weight-Only Quantization")
        pruned_dict2 = torch.load(prune_path, map_location="cpu", weights_only=False)
        model2 = pruned_dict2["model"]
        model2.half().to(device)
        if lora_path and os.path.exists(lora_path):
            model2 = PeftModel.from_pretrained(model2, lora_path, torch_dtype=torch.float16)
            model2 = model2.merge_and_unload()
            model2.to(device)
        est = _calc_model_size_bits(model2, quant_bits=4, group_size=128)
        yield log(f"  预估大小: {est:.1f} MB")
        model_int4 = _quantize_model_weights(model2, bits=4, group_size=128)
        start = time.time()
        ppl = PPLMetric(model_int4, tokenizer, ["wikitext2"], seq_len=128, batch_size=1, device=device)
        elapsed = time.time() - start
        results.append(f"GPTQ-W4  PPL={ppl['wikitext2']:.2f}  Size={est:.0f}MB  Time={elapsed:.0f}s")
        yield log(f"  {results[-1]}")
        del model_int4
        torch.cuda.empty_cache()

    yield log(f"\n{'='*50}")
    yield log("量化评估完成")
    for r in results:
        yield log(f"  {r}")
    yield log(f"{'='*50}")


# ============================================================
# UI
# ============================================================

def build_experiment():
    gr.Markdown(
        "## 实验操作面板\n"
        "按顺序执行：结构化剪枝 → LoRA 微调恢复 → 量化评估"
    )

    # ---- Step 1: Pruning ----
    with gr.Accordion("Step 1 — 结构化剪枝 (Taylor)", open=True):
        gr.Markdown("使用 Taylor 一阶重要性估计，对 Attention 与 MLP 模块进行 block-wise 剪枝")
        with gr.Row():
            with gr.Column(scale=2):
                prune_base = gr.Textbox(
                    label="基础模型路径",
                    value="F:\\Download\\tinyllama_model",
                    placeholder="HuggingFace 模型名或本地路径",
                )
                prune_save = gr.Textbox(label="输出名称", value="tinyllama_prune")
            with gr.Column(scale=1):
                prune_ratio = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="剪枝比例")
            with gr.Column(scale=1):
                prune_mlp_start = gr.Number(label="MLP 起始层", value=4, precision=0)
                prune_mlp_end = gr.Number(label="MLP 终止层", value=20, precision=0)
            with gr.Column(scale=1):
                prune_attn_start = gr.Number(label="Attn 起始层", value=4, precision=0)
                prune_attn_end = gr.Number(label="Attn 终止层", value=20, precision=0)

        prune_btn = gr.Button("开始剪枝", variant="primary", size="lg")
        prune_log = gr.Textbox(
            label="运行日志", lines=12, max_lines=30,
            autoscroll=True, interactive=False,
            placeholder="点击按钮开始剪枝，日志将实时显示在此处...",
        )
        prune_btn.click(
            fn=run_pruning,
            inputs=[prune_base, prune_ratio, prune_mlp_start, prune_mlp_end,
                    prune_attn_start, prune_attn_end, prune_save],
            outputs=prune_log,
        )

    # ---- Step 2: Fine-tuning ----
    with gr.Accordion("Step 2 — LoRA 微调恢复", open=False):
        gr.Markdown("在剪枝后模型上应用 LoRA 进行低成本后训练，恢复语言能力")
        with gr.Row():
            with gr.Column(scale=2):
                ft_prune = gr.Textbox(
                    label="剪枝模型路径 (.bin)",
                    value="prune_log/tinyllama_prune/pytorch_model.bin",
                )
                ft_base = gr.Textbox(label="基础模型名", value="F:\\Download\\tinyllama_model")
                ft_out = gr.Textbox(label="输出目录", value="tune_log/tinyllama_tune_25")
            with gr.Column(scale=1):
                ft_data = gr.Dropdown(
                    ["yahma/alpaca-cleaned"],
                    value="yahma/alpaca-cleaned",
                    label="训练数据集",
                )
                ft_lora_r = gr.Number(label="LoRA Rank", value=8, precision=0)
            with gr.Column(scale=1):
                ft_epochs = gr.Number(label="训练轮数", value=2, precision=0)
                ft_lr = gr.Number(label="学习率", value=1e-4)
            with gr.Column(scale=1):
                ft_bs = gr.Number(label="批量大小", value=4, precision=0)

        ft_btn = gr.Button("开始微调", variant="primary", size="lg")
        ft_log = gr.Textbox(
            label="运行日志", lines=12, max_lines=30,
            autoscroll=True, interactive=False,
            placeholder="点击按钮开始微调，日志将实时显示在此处...",
        )
        ft_btn.click(
            fn=run_finetuning,
            inputs=[ft_prune, ft_data, ft_lora_r, ft_epochs, ft_lr, ft_bs, ft_out, ft_base],
            outputs=ft_log,
        )

    # ---- Step 3: Quantization ----
    with gr.Accordion("Step 3 — 量化评估", open=False):
        gr.Markdown("对比 GPTQ W8 / W4 权重量化对模型体积与精度的影响")
        with gr.Row():
            with gr.Column(scale=1):
                quant_prune = gr.Textbox(
                    label="剪枝模型路径 (.bin)",
                    value="prune_log/tinyllama_prune/pytorch_model.bin",
                )
                quant_lora = gr.Textbox(
                    label="LoRA 适配器路径",
                    value="tune_log/tinyllama_tune_25",
                )
            with gr.Column(scale=1):
                quant_methods = gr.CheckboxGroup(
                    ["GPTQ-W8", "GPTQ-W4"],
                    value=["GPTQ-W8", "GPTQ-W4"],
                    label="量化方法",
                )

        quant_btn = gr.Button("开始量化评估", variant="primary", size="lg")
        quant_log = gr.Textbox(
            label="运行日志", lines=12, max_lines=30,
            autoscroll=True, interactive=False,
            placeholder="点击按钮开始量化评估，日志将实时显示在此处...",
        )
        quant_btn.click(
            fn=run_quantization,
            inputs=[quant_prune, quant_lora, quant_methods],
            outputs=quant_log,
        )
