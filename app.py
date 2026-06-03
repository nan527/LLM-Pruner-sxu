"""LLM Compression Experiment Platform - Gradio Web UI"""
import torch
import gradio as gr
from modules.dashboard import build_dashboard
from modules.experiment import build_experiment
from modules.inference import build_inference

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep: #06080f;
    --bg-mid: #0c1019;
    --glass-bg: rgba(12, 18, 32, 0.65);
    --glass-bg-hover: rgba(18, 26, 48, 0.72);
    --glass-border: rgba(100, 160, 255, 0.08);
    --glass-border-hover: rgba(100, 160, 255, 0.18);
    --glass-glow: rgba(60, 140, 255, 0.06);
    --accent: #3c8cff;
    --accent-soft: rgba(60, 140, 255, 0.12);
    --cyan: #00d4ff;
    --emerald: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --violet: #a78bfa;
    --text-1: #e8edf5;
    --text-2: #a4b0c8;
    --text-3: #6b7a94;
    --text-4: #3d4a60;
    --radius: 14px;
    --radius-sm: 10px;
    --font-sans: 'Noto Sans SC', 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
}

/* ===== Root ===== */
.gradio-container {
    background: var(--bg-deep) !important;
    color: var(--text-1) !important;
    font-family: var(--font-sans) !important;
    min-height: 100vh !important;
    position: relative !important;
    overflow-x: hidden !important;
}
.gradio-container::before {
    content: '' !important;
    position: fixed !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background:
        radial-gradient(ellipse 600px 400px at 20% 20%, rgba(60,140,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 80% 30%, rgba(139,92,246,0.05) 0%, transparent 70%),
        radial-gradient(ellipse 400px 300px at 50% 80%, rgba(0,212,255,0.04) 0%, transparent 70%) !important;
    pointer-events: none !important;
    z-index: 0 !important;
    animation: aurora 20s ease-in-out infinite alternate !important;
}
@keyframes aurora {
    0%   { transform: translate(0, 0) rotate(0deg); }
    50%  { transform: translate(-2%, 1%) rotate(1deg); }
    100% { transform: translate(1%, -1%) rotate(-0.5deg); }
}
.gradio-container > * { position: relative !important; z-index: 1 !important; }
.gradio-container .contain {
    max-width: 1320px !important;
    padding: 20px 24px 40px !important;
}

/* ===== Glass mixin (applied via class) ===== */
/* We apply glass styling to .block, .accordion, .panel etc */

/* ===== Tab nav ===== */
.tab-nav {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(24px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.5) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 3px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.03) !important;
}
.tab-nav > button {
    border-radius: var(--radius-sm) !important;
    padding: 11px 28px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    color: var(--text-3) !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.tab-nav > button:hover {
    color: var(--text-2) !important;
    background: rgba(60, 140, 255, 0.06) !important;
}
.tab-nav > button.selected {
    color: #fff !important;
    background: linear-gradient(135deg, rgba(60,140,255,0.25), rgba(139,92,246,0.18)) !important;
    box-shadow: 0 0 20px rgba(60,140,255,0.12), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(60,140,255,0.2) !important;
}

/* ===== Accordion ===== */
.accordion {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.4) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18) !important;
    overflow: hidden !important;
    transition: border-color 0.3s ease !important;
}
.accordion:hover {
    border-color: var(--glass-border-hover) !important;
}
.accordion > .label-wrap {
    background: rgba(10, 16, 28, 0.5) !important;
    padding: 14px 20px !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    color: var(--text-1) !important;
    border-bottom: 1px solid var(--glass-border) !important;
    letter-spacing: 0.01em !important;
}
.accordion > .label-wrap:hover {
    background: rgba(14, 22, 40, 0.6) !important;
}
.accordion > .label-wrap .icon-wrap svg {
    color: var(--text-3) !important;
}

/* ===== Buttons ===== */
button.primary {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    border: 1px solid rgba(96, 165, 250, 0.25) !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.02em !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 28px !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(37, 99, 235, 0.25), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}
button.primary::after {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, transparent, rgba(255,255,255,0.08)) !important;
    opacity: 0 !important;
    transition: opacity 0.25s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.35), inset 0 1px 0 rgba(255,255,255,0.12) !important;
}
button.primary:hover::after { opacity: 1 !important; }

button.secondary {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-2) !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 28px !important;
    transition: all 0.25s ease !important;
}
button.secondary:hover {
    border-color: rgba(60, 140, 255, 0.25) !important;
    color: var(--cyan) !important;
    background: var(--glass-bg-hover) !important;
}

/* ===== Inputs ===== */
input, textarea, select {
    background: rgba(8, 14, 26, 0.6) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
    font-family: var(--font-sans) !important;
    font-size: 13.5px !important;
    transition: all 0.25s ease !important;
}
input:focus, textarea:focus, select:focus {
    border-color: rgba(60, 140, 255, 0.35) !important;
    box-shadow: 0 0 0 3px rgba(60, 140, 255, 0.08), 0 0 24px rgba(60, 140, 255, 0.06) !important;
}
input::placeholder, textarea::placeholder {
    color: var(--text-4) !important;
}
label, .label-text {
    color: var(--text-3) !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ===== Table / Dataframe ===== */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid rgba(100, 160, 255, 0.12) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25) !important;
}
thead th {
    background: rgba(16, 24, 44, 0.85) !important;
    color: #c0cee0 !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 14px 18px !important;
    text-align: center !important;
    border-bottom: 2px solid rgba(60, 140, 255, 0.2) !important;
}
tbody td {
    padding: 12px 18px !important;
    border-top: 1px solid rgba(100, 160, 255, 0.06) !important;
    background: rgba(10, 16, 30, 0.6) !important;
    color: #dce4f0 !important;
    text-align: center !important;
    font-size: 14px !important;
    font-family: var(--font-mono) !important;
    transition: background 0.2s ease !important;
}
tbody tr:hover td {
    background: rgba(60, 140, 255, 0.1) !important;
}
/* Dataframe wrapper */
.gradio-dataframe {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
.gradio-dataframe table {
    width: 100% !important;
}

/* ===== Markdown / Prose ===== */
.prose, .prose p, .prose li, .prose span, .prose td {
    color: var(--text-2) !important;
    font-family: var(--font-sans) !important;
    line-height: 1.7 !important;
}
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: var(--text-1) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
.prose h2 {
    font-size: 20px !important;
    margin-bottom: 4px !important;
}
.prose h3 {
    font-size: 16px !important;
}
.prose code {
    background: rgba(60, 140, 255, 0.1) !important;
    color: var(--cyan) !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    font-family: var(--font-mono) !important;
    font-size: 12.5px !important;
}
.prose strong { color: var(--text-1) !important; }
.prose em { color: var(--text-3) !important; }
.prose a { color: var(--accent) !important; }

/* ===== Block (glass card for each component) ===== */
.block {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.12) !important;
    transition: border-color 0.3s ease !important;
}
.block:hover {
    border-color: var(--glass-border-hover) !important;
}

/* ===== Plotly ===== */
.js-plotly-plot {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18) !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(100, 160, 255, 0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(100, 160, 255, 0.3);
}

/* ===== Slider ===== */
input[type=range] { accent-color: var(--accent) !important; }
input[type=checkbox] { accent-color: var(--accent) !important; }

/* ===== Dropdown options ===== */
ul.options {
    background: rgba(12, 18, 32, 0.9) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
}
ul.options > li {
    color: var(--text-2) !important;
}
ul.options > li:hover {
    background: rgba(60, 140, 255, 0.1) !important;
    color: var(--text-1) !important;
}

/* ===== Footer ===== */
footer { display: none !important; }

/* ===== Status textbox ===== */
textarea[aria-label="状态"],
textarea[aria-label="Status"],
textarea[aria-label="运行日志"],
textarea[aria-label="Log Output"],
textarea[aria-label="执行日志"],
textarea[aria-label="Execution Log"] {
    font-family: var(--font-mono) !important;
    font-size: 12.5px !important;
    line-height: 1.6 !important;
    color: var(--emerald) !important;
}

/* ===== Number inputs ===== */
input[type=number] {
    font-family: var(--font-mono) !important;
}

/* ===== Panel cards (glass) ===== */
.gradio-group {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
}
"""

HEADER_HTML = """
<div style="text-align:center;padding:32px 0 12px 0;position:relative;">
    <div style="display:inline-flex;align-items:center;gap:8px;
        background:rgba(12,18,32,0.5);
        backdrop-filter:blur(16px);
        border:1px solid rgba(100,160,255,0.1);
        border-radius:24px;
        padding:6px 20px;
        margin-bottom:16px;
        box-shadow:0 2px 16px rgba(0,0,0,0.15);">
        <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#34d399;box-shadow:0 0 8px #34d399;"></span>
        <span style="color:#6b7a94;font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">
            TinyLlama-1.1B Compression Pipeline
        </span>
    </div>
    <h1 style="color:#e8edf5;font-size:34px;font-weight:700;margin:0 0 8px 0;
        letter-spacing:-0.03em;
        font-family:'Noto Sans SC','Segoe UI',system-ui,sans-serif;">
        LLM <span style="background:linear-gradient(135deg,#3c8cff,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">压缩实验</span>平台
    </h1>
    <p style="color:#4d5e78;font-size:14px;margin:0;max-width:480px;margin:0 auto;line-height:1.7;letter-spacing:0.01em;">
        结构化剪枝 · LoRA 微调恢复 · 权重量化<br/>
        <span style="color:#3d4a60;font-size:12px;">RTX 4060 Laptop · 消费级 GPU 全流程压缩</span>
    </p>
</div>
"""

with gr.Blocks(title="LLM 压缩实验平台", css=CUSTOM_CSS,
               theme=gr.themes.Base()) as app:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.TabItem("  结果看板"):
            build_dashboard()
        with gr.TabItem("  实验操作"):
            build_experiment()
        with gr.TabItem("  推理演示"):
            build_inference()

if __name__ == "__main__":
    port = 7860
    for p in range(7860, 7880):
        try:
            import socket
            s = socket.socket()
            s.bind(("127.0.0.1", p))
            s.close()
            port = p
            break
        except OSError:
            continue

    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
    )
