"""Module C: Inference Demo — 模型推理 & 剪枝前后对比"""
import os
import time
import ssl
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PRESETS = {
    "原始 TinyLlama (FP16)": {
        "type": "hf",
        "model_path": "F:\\Download\\tinyllama_model",
    },
    "剪枝 25% + LoRA (FP16)": {
        "type": "pruned_lora",
        "prune_path": "prune_log/tinyllama_prune/pytorch_model.bin",
        "lora_path": "tune_log/tinyllama_tune_25",
    },
    "剪枝 25% (无微调)": {
        "type": "pruned_lora",
        "prune_path": "prune_log/tinyllama_prune/pytorch_model.bin",
        "lora_path": "",
    },
    "剪枝 50% + LoRA (FP16)": {
        "type": "pruned_lora",
        "prune_path": "prune_log/tinyllama_prune_50/pytorch_model.bin",
        "lora_path": "tune_log/tinyllama_tune_50",
    },
}

_current_model = None
_current_tokenizer = None


def _load_one(model_name, hf_path, prune_path, lora_path):
    """Load a single model and tokenizer, return (model, tokenizer) or raise."""
    preset = MODEL_PRESETS.get(model_name, {})
    model_type = preset.get("type", "hf")

    if model_type == "hf":
        path = hf_path or preset.get("model_path", "")
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto"
        )
    elif model_type == "pruned_lora":
        prune = prune_path or preset.get("prune_path", "")
        lora = lora_path or preset.get("lora_path", "")
        from LLMPruner.peft import PeftModel
        pruned_dict = torch.load(prune, map_location="cpu", weights_only=False)
        tok = pruned_dict["tokenizer"]
        mdl = pruned_dict["model"]
        mdl.half().to(DEVICE)
        if lora and os.path.exists(lora):
            mdl = PeftModel.from_pretrained(mdl, lora, torch_dtype=torch.float16)
            mdl = mdl.merge_and_unload()
            mdl.to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    mdl.eval()
    return mdl, tok


def _generate(mdl, tok, prompt, max_tokens, temperature, top_p):
    """Run generation, return (text, elapsed_seconds)."""
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    start = time.time()
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    elapsed = time.time() - start
    response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response, elapsed


# ============================================================
# Single-model mode
# ============================================================

def load_model(model_name, hf_path, prune_path, lora_path):
    global _current_model, _current_tokenizer
    _current_model = None
    _current_tokenizer = None
    torch.cuda.empty_cache()

    try:
        _current_model, _current_tokenizer = _load_one(model_name, hf_path, prune_path, lora_path)
        return f" 模型加载成功: {model_name}\n 设备: {DEVICE}"
    except Exception as e:
        return f" 加载失败: {str(e)}"


def generate_text(prompt, max_tokens, temperature, top_p):
    global _current_model, _current_tokenizer
    if _current_model is None or _current_tokenizer is None:
        return " 请先加载模型"
    try:
        response, elapsed = _generate(_current_model, _current_tokenizer, prompt,
                                      max_tokens, temperature, top_p)
        return f"{response}\n\n━━━━━━━━━━━━━━━━━━━━\n {elapsed:.1f}s | {len(response)} 字符 | {len(response)/max(elapsed,0.01):.0f} chars/s"
    except Exception as e:
        return f" 生成失败: {str(e)}"


def unload_model():
    global _current_model, _current_tokenizer
    _current_model = None
    _current_tokenizer = None
    torch.cuda.empty_cache()
    return " 模型已卸载，GPU 内存已释放"


def _on_preset_change(preset_name):
    preset = MODEL_PRESETS.get(preset_name, {})
    return (
        preset.get("model_path", ""),
        preset.get("prune_path", ""),
        preset.get("lora_path", ""),
    )


# ============================================================
# Comparison mode: A/B side-by-side
# ============================================================

def compare_models(model_a, hf_a, prune_a, lora_a,
                   model_b, hf_b, prune_b, lora_b,
                   prompt, max_tokens, temperature, top_p):
    """Load model A → generate → unload → load model B → generate → show both."""
    results = []
    log_lines = []

    # --- Model A ---
    log_lines.append(f"[1/2] 加载 {model_a} ...")
    try:
        mdl_a, tok_a = _load_one(model_a, hf_a, prune_a, lora_a)
        log_lines.append(f"      生成中...")
        text_a, t_a = _generate(mdl_a, tok_a, prompt, max_tokens, temperature, top_p)
        results.append(f"### {model_a}\n{text_a}\n\n*{t_a:.1f}s | {len(text_a)} chars*")
        log_lines.append(f"      完成 ({t_a:.1f}s)")
        del mdl_a, tok_a
        torch.cuda.empty_cache()
    except Exception as e:
        results.append(f"### {model_a}\n 加载/生成失败: {e}")
        log_lines.append(f"      失败: {e}")

    # --- Model B ---
    log_lines.append(f"[2/2] 加载 {model_b} ...")
    try:
        mdl_b, tok_b = _load_one(model_b, hf_b, prune_b, lora_b)
        log_lines.append(f"      生成中...")
        text_b, t_b = _generate(mdl_b, tok_b, prompt, max_tokens, temperature, top_p)
        results.append(f"### {model_b}\n{text_b}\n\n*{t_b:.1f}s | {len(text_b)} chars*")
        log_lines.append(f"      完成 ({t_b:.1f}s)")
        del mdl_b, tok_b
        torch.cuda.empty_cache()
    except Exception as e:
        results.append(f"### {model_b}\n 加载/生成失败: {e}")
        log_lines.append(f"      失败: {e}")

    log_lines.append(" 对比完成")
    return results[0], results[1], "\n".join(log_lines)


# ============================================================
# UI
# ============================================================

def build_inference():
    gr.HTML("""
    <div style="margin-bottom:16px;">
        <h2 style="color:#c9d1d9;font-size:20px;font-weight:600;margin:0;">模型推理演示</h2>
        <p style="color:#556070;font-size:13px;margin:4px 0 12px 0;">单模型推理 & 剪枝前后 A/B 对比</p>
    </div>
    """)

    with gr.Tabs():

        # ========== Tab 1: Single Model ==========
        with gr.TabItem(" 单模型推理"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML("""<p style="color:#8896aa;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin:0 0 8px 0;">模型加载</p>""")
                    model_preset = gr.Dropdown(
                        list(MODEL_PRESETS.keys()),
                        value=list(MODEL_PRESETS.keys())[0],
                        label="选择模型版本",
                    )
                    hf_path = gr.Textbox(label="HF 模型路径", placeholder="HuggingFace 模型名")
                    prune_path = gr.Textbox(label="剪枝 Checkpoint 路径", placeholder="pytorch_model.bin 路径")
                    lora_path = gr.Textbox(label="LoRA 适配器路径", placeholder="tune_log/...")

                    with gr.Row():
                        load_btn = gr.Button(" 加载模型", variant="primary")
                        unload_btn = gr.Button(" 卸载")
                    load_status = gr.Textbox(label="状态", lines=4, interactive=False)

                with gr.Column(scale=2, min_width=400):
                    gr.HTML("""<p style="color:#8896aa;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin:0 0 8px 0;">文本生成</p>""")
                    prompt_input = gr.Textbox(label="输入 Prompt", lines=4,
                                              placeholder="在此输入文本...")
                    with gr.Row():
                        max_tokens = gr.Slider(16, 512, value=128, step=16, label="Max Tokens")
                        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

                    gen_btn = gr.Button(" 生成文本", variant="secondary")
                    output_text = gr.Textbox(label="生成结果", lines=14, autoscroll=True)

            model_preset.change(fn=_on_preset_change, inputs=model_preset,
                                outputs=[hf_path, prune_path, lora_path])
            load_btn.click(fn=load_model,
                           inputs=[model_preset, hf_path, prune_path, lora_path],
                           outputs=load_status)
            unload_btn.click(fn=unload_model, inputs=[], outputs=load_status)
            gen_btn.click(fn=generate_text,
                          inputs=[prompt_input, max_tokens, temperature, top_p],
                          outputs=output_text)

        # ========== Tab 2: Comparison ==========
        with gr.TabItem(" A/B 对比"):
            gr.HTML("""<p style="color:#66758a;font-size:13px;margin:0 0 12px 0;">输入同一个 Prompt，对比两个模型的生成效果（逐个加载，不占双倍显存）</p>""")

            with gr.Row():
                with gr.Column():
                    gr.HTML("""<p style="color:#4da8da;font-size:12px;font-weight:600;margin:0 0 4px 0;">模型 A</p>""")
                    cmp_a_preset = gr.Dropdown(list(MODEL_PRESETS.keys()), value="原始 TinyLlama (FP16)", label="版本")
                    cmp_a_hf = gr.Textbox(label="HF 路径")
                    cmp_a_prune = gr.Textbox(label="剪枝路径")
                    cmp_a_lora = gr.Textbox(label="LoRA 路径")
                    cmp_a_output = gr.Textbox(label="输出", lines=12, autoscroll=True)

                with gr.Column():
                    gr.HTML("""<p style="color:#f0a050;font-size:12px;font-weight:600;margin:0 0 4px 0;">模型 B</p>""")
                    cmp_b_preset = gr.Dropdown(list(MODEL_PRESETS.keys()), value="剪枝 25% (无微调)", label="版本")
                    cmp_b_hf = gr.Textbox(label="HF 路径")
                    cmp_b_prune = gr.Textbox(label="剪枝路径")
                    cmp_b_lora = gr.Textbox(label="LoRA 路径")
                    cmp_b_output = gr.Textbox(label="输出", lines=12, autoscroll=True)

            cmp_prompt = gr.Textbox(label="输入 Prompt", lines=3, placeholder="同一个 prompt 会同时喂给两个模型...")
            with gr.Row():
                cmp_max_tokens = gr.Slider(16, 512, value=128, step=16, label="Max Tokens")
                cmp_temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                cmp_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

            cmp_btn = gr.Button(" 对比生成", variant="primary")
            cmp_log = gr.Textbox(label="执行日志", lines=4, interactive=False)

            # Fill paths on preset change
            cmp_a_preset.change(fn=_on_preset_change, inputs=cmp_a_preset,
                                outputs=[cmp_a_hf, cmp_a_prune, cmp_a_lora])
            cmp_b_preset.change(fn=_on_preset_change, inputs=cmp_b_preset,
                                outputs=[cmp_b_hf, cmp_b_prune, cmp_b_lora])
            cmp_btn.click(fn=compare_models,
                          inputs=[cmp_a_preset, cmp_a_hf, cmp_a_prune, cmp_a_lora,
                                  cmp_b_preset, cmp_b_hf, cmp_b_prune, cmp_b_lora,
                                  cmp_prompt, cmp_max_tokens, cmp_temperature, cmp_top_p],
                          outputs=[cmp_a_output, cmp_b_output, cmp_log])
