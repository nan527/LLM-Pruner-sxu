"""LLM Compression Experiment Platform — Gradio Web UI"""
import gradio as gr
from modules.dashboard import build_dashboard
from modules.experiment import build_experiment
from modules.inference import build_inference

CUSTOM_CSS = """
/* ===== Dark Dashboard Theme ===== */
:root {
    --bg-deep: #090d13;
    --bg-surface: #131822;
    --bg-card: #1a2030;
    --border: #252d3a;
    --accent: #4da8da;
    --accent-glow: rgba(77, 168, 218, 0.18);
    --green: #3fb986;
    --orange: #f0a050;
    --red: #e0556a;
    --text-primary: #e2e6ed;
    --text-secondary: #8896aa;
    --text-muted: #556070;
    --radius: 10px;
    --shadow: 0 2px 12px rgba(0,0,0,0.3);
}

/* Page background */
body, .gradio-container {
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* Main container */
.gradio-container .contain {
    max-width: 1280px !important;
    padding-top: 24px !important;
}

/* Tabs */
.tabs {
    border: none !important;
    background: transparent !important;
}
.tabs > .tab-nav {
    background: var(--bg-surface) !important;
    border-radius: var(--radius) !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.tabs > .tab-nav > button {
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.tabs > .tab-nav > button:hover {
    color: var(--text-primary) !important;
    background: rgba(255,255,255,0.04) !important;
}
.tabs > .tab-nav > button.selected {
    color: #fff !important;
    background: var(--accent) !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}

/* Accordions */
.accordion {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-card) !important;
    margin-bottom: 16px !important;
    overflow: hidden !important;
}
.accordion > .label-wrap {
    background: var(--bg-surface) !important;
    padding: 14px 20px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
}
.accordion > .label-wrap:hover {
    background: #1a2436 !important;
}

/* Buttons */
button.primary {
    background: var(--accent) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px var(--accent-glow) !important;
    color: #fff !important;
}
button.primary:hover {
    filter: brightness(1.15) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
button.secondary {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(77,168,218,0.06) !important;
}

/* Inputs */
input, textarea, select {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 8px 14px !important;
    font-size: 14px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
label, .label-text {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    margin-bottom: 4px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Slider */
.slider input[type=range] {
    accent-color: var(--accent) !important;
}

/* Dataframe / Table */
table {
    border-collapse: collapse !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}
thead th {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    text-align: center !important;
    font-size: 13px !important;
}
tbody td {
    padding: 10px 16px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    text-align: center !important;
    font-size: 14px !important;
}
tbody tr:hover td {
    background: rgba(77,168,218,0.05) !important;
}

/* Markdown */
.prose {
    color: var(--text-primary) !important;
}
.prose h1, .prose h2, .prose h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
.prose table {
    width: 100% !important;
}

/* Footer */
footer {
    display: none !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-deep);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Plotly charts */
.js-plotly-plot {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* Checkbox group */
.checkbox-group {
    gap: 8px !important;
}
.checkbox-group label {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 14px !important;
}
.checkbox-group label.selected {
    border-color: var(--accent) !important;
    background: rgba(77,168,218,0.1) !important;
    color: var(--accent) !important;
}
"""

with gr.Blocks(title="LLM 压缩实验平台") as app:
    # Hero header
    gr.HTML("""
    <div style="text-align:center;padding:32px 0 8px 0;">
        <div style="display:inline-block;background:linear-gradient(135deg,#1a2540,#1a2030);border:1px solid #252d3a;border-radius:12px;padding:3px 14px;margin-bottom:16px;">
            <span style="color:#4da8da;font-size:12px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">
                TinyLlama-1.1B Compression Pipeline
            </span>
        </div>
        <h1 style="color:#e2e6ed;font-size:32px;font-weight:700;margin:0 0 8px 0;letter-spacing:-0.02em;">
            LLM 压缩实验平台
        </h1>
        <p style="color:#66758a;font-size:15px;margin:0;max-width:560px;margin:0 auto;line-height:1.6;">
            结构化剪枝 &middot; LoRA 微调恢复 &middot; 权重量化<br/>RTX 4060 Laptop 消费级 GPU 全流程压缩
        </p>
    </div>
    """)

    with gr.Tabs() as tabs:
        with gr.TabItem(" 结果看板", id="dashboard"):
            build_dashboard()

        with gr.TabItem(" 实验操作", id="experiment"):
            build_experiment()

        with gr.TabItem(" 推理演示", id="inference"):
            build_inference()

if __name__ == "__main__":
    # Auto-find available port to avoid OSError on restart
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
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )
