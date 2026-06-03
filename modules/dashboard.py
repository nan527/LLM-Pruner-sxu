"""Module A: Results Dashboard - experiment data visualization"""
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS = [
    {"Version": "原始 TinyLlama (FP16)", "PPL": 17.42, "Size(MB)": 2200, "Ratio": 1.0},
    {"Version": "剪枝 25%", "PPL": 35.95, "Size(MB)": 1762, "Ratio": 1.2},
    {"Version": "剪枝 25% + LoRA (FP16)", "PPL": 26.71, "Size(MB)": 1762, "Ratio": 1.2},
    {"Version": "剪枝 25% + LoRA + W8", "PPL": 26.71, "Size(MB)": 969, "Ratio": 2.3},
    {"Version": "剪枝 25% + LoRA + W4", "PPL": 30.75, "Size(MB)": 560, "Ratio": 3.9},
    {"Version": "剪枝 50%", "PPL": 90.72, "Size(MB)": 1426, "Ratio": 1.5},
    {"Version": "剪枝 50% + LoRA (FP16)", "PPL": 45.18, "Size(MB)": 1426, "Ratio": 1.5},
    {"Version": "剪枝 50% + LoRA + W8", "PPL": 45.18, "Size(MB)": 796, "Ratio": 2.8},
    {"Version": "剪枝 50% + LoRA + W4", "PPL": 50.50, "Size(MB)": 471, "Ratio": 4.7},
    {"Version": "剪枝 + LoRA + Dynamic INT8", "PPL": 81.82, "Size(MB)": 250, "Ratio": 8.8},
]

SHORT = [
    "原始FP16", "剪枝25%", "+LoRA", "+W8", "+W4",
    "剪枝50%", "+LoRA", "+W8", "+W4", "+DynINT8"
]

# Design tokens — dark blue glass palette
C = {
    "bg": "rgba(8, 14, 26, 0)",
    "paper": "rgba(10, 18, 36, 0)",
    "plot": "rgba(10, 18, 36, 0.4)",
    "grid": "rgba(59, 130, 246, 0.06)",
    "text": "#94a3b8",
    "text2": "#cbd5e1",
    "title": "#e8edf5",
    "accent": "#3b82f6",
    "cyan": "#06b6d4",
    "emerald": "#10b981",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "violet": "#8b5cf6",
}


def _layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=C["title"], family="Noto Sans SC"), x=0.5, y=0.97),
        paper_bgcolor=C["paper"],
        plot_bgcolor=C["plot"],
        font=dict(color=C["text"], size=12, family="JetBrains Mono, Consolas, monospace"),
        margin=dict(b=90, l=60, r=40, t=55),
        height=400,
        xaxis=dict(
            tickangle=-30,
            tickfont=dict(size=11, color=C["text"]),
            gridcolor=C["grid"],
            linecolor=C["grid"],
        ),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["grid"]),
        legend=dict(font=dict(color=C["text2"], size=12), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _ppl_chart():
    ppls = [d["PPL"] for d in RESULTS]
    colors = [C["emerald"] if p < 30 else C["amber"] if p < 50 else C["rose"] for p in ppls]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=SHORT, y=ppls,
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{p:.1f}" for p in ppls],
        textposition="outside",
        textfont=dict(color=C["text2"], size=12, family="JetBrains Mono"),
        hovertemplate="<b>%{x}</b><br>PPL: %{y:.2f}<extra></extra>",
        opacity=0.95,
    ))
    fig.add_hline(
        y=17.42, line_dash="dot", line_color=C["accent"], line_width=1.5,
        annotation_text="原始 PPL 17.42",
        annotation_position="top right",
        annotation_font=dict(color=C["accent"], size=11, family="Noto Sans SC"),
    )
    return _layout(fig, "困惑度对比 — 越低越好")


def _size_chart():
    sizes = [d["Size(MB)"] for d in RESULTS]
    ratios = [d["Ratio"] for d in RESULTS]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=SHORT, y=sizes, name="模型体积 (MB)",
        marker_color=C["accent"],
        marker_line=dict(width=0),
        opacity=0.85,
        text=[f"{s}" for s in sizes],
        textposition="outside",
        textfont=dict(color=C["text"], size=10.5, family="JetBrains Mono"),
        hovertemplate="<b>%{x}</b><br>体积: %{y} MB<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=SHORT, y=ratios, name="压缩比",
        mode="lines+markers",
        line=dict(color=C["cyan"], width=3),
        marker=dict(size=7, color=C["cyan"], line=dict(width=1, color="rgba(6,182,212,0.3)")),
        hovertemplate="<b>%{x}</b><br>压缩比: %{y:.1f}x<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="体积 (MB)", secondary_y=False, gridcolor=C["grid"], title_font=dict(color=C["text"], size=11))
    fig.update_yaxes(title_text="压缩比", secondary_y=True, gridcolor="rgba(0,0,0,0)", title_font=dict(color=C["cyan"], size=11))
    return _layout(fig, "模型体积与压缩比")


def _cards_html():
    best = RESULTS[4]
    extreme = RESULTS[8]

    card_style = (
        "flex:1;min-width:220px;"
        "background:rgba(12,20,44,0.5);"
        "backdrop-filter:blur(24px) saturate(1.6);"
        "-webkit-backdrop-filter:blur(24px) saturate(1.6);"
        "border:1px solid rgba(59,130,246,0.08);"
        "border-top:1px solid rgba(59,130,246,0.15);"
        "border-radius:16px;"
        "padding:26px 28px;"
        "box-shadow:0 1px 2px rgba(0,0,0,0.3),0 4px 12px rgba(0,0,0,0.25),0 12px 40px rgba(0,0,0,0.2),inset 0 1px 0 rgba(59,130,246,0.06);"
        "transition:all 0.35s cubic-bezier(0.4,0,0.2,1);"
        "animation:cardFadeIn 0.6s cubic-bezier(0.4,0,0.2,1) both;"
    )
    label_style = "font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;font-family:'Outfit',sans-serif;"
    val_style = "font-family:'JetBrains Mono',monospace;font-weight:700;"
    sub_style = "color:#64748b;font-size:11px;font-family:'Noto Sans SC',sans-serif;letter-spacing:0.02em;"
    gap_style = "display:flex;gap:32px;margin-top:16px;"

    return f"""
    <style>
    @keyframes cardFadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .glass-card-metric:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35),0 8px 32px rgba(0,0,0,0.3),0 24px 64px rgba(0,0,0,0.25),inset 0 1px 0 rgba(59,130,246,0.08) !important;
        border-color: rgba(59,130,246,0.25) !important;
    }}
    </style>
    <div style="display:flex;gap:16px;margin:20px 0;flex-wrap:wrap;">
        <div class="glass-card-metric" style="{card_style}border-top:3px solid rgba(59,130,246,0.4);animation-delay:0.05s;">
            <div style="{label_style}color:#60a5fa;">✦ 推荐方案</div>
            <div style="color:#e8edf5;font-size:16px;font-weight:700;margin-bottom:4px;font-family:'Noto Sans SC',sans-serif;">剪枝 25% + LoRA + W4</div>
            <div style="{gap_style}">
                <div><span style="{sub_style}">困惑度</span><br/><span style="{val_style}color:#60a5fa;font-size:26px;">{best['PPL']}</span></div>
                <div><span style="{sub_style}">体积</span><br/><span style="{val_style}color:#e8edf5;font-size:26px;">{best['Size(MB)']}<span style="font-size:12px;color:#64748b;"> MB</span></span></div>
                <div><span style="{sub_style}">压缩比</span><br/><span style="{val_style}color:#f59e0b;font-size:26px;">{best['Ratio']}x</span></div>
            </div>
        </div>
        <div class="glass-card-metric" style="{card_style}border-top:3px solid rgba(6,182,212,0.35);animation-delay:0.10s;">
            <div style="{label_style}color:#06b6d4;">⚡ 极限压缩</div>
            <div style="color:#e8edf5;font-size:16px;font-weight:700;margin-bottom:4px;font-family:'Noto Sans SC',sans-serif;">剪枝 50% + LoRA + W4</div>
            <div style="{gap_style}">
                <div><span style="{sub_style}">困惑度</span><br/><span style="{val_style}color:#f43f5e;font-size:26px;">{extreme['PPL']}</span></div>
                <div><span style="{sub_style}">体积</span><br/><span style="{val_style}color:#e8edf5;font-size:26px;">{extreme['Size(MB)']}<span style="font-size:12px;color:#64748b;"> MB</span></span></div>
                <div><span style="{sub_style}">压缩比</span><br/><span style="{val_style}color:#f59e0b;font-size:26px;">{extreme['Ratio']}x</span></div>
            </div>
        </div>
        <div class="glass-card-metric" style="{card_style}border-top:3px solid rgba(244,63,94,0.25);animation-delay:0.15s;">
            <div style="{label_style}color:#f43f5e;">✕ 谨慎使用</div>
            <div style="color:#e8edf5;font-size:16px;font-weight:700;margin-bottom:4px;font-family:'Noto Sans SC',sans-serif;">Dynamic INT8</div>
            <div style="{gap_style}">
                <div><span style="{sub_style}">困惑度</span><br/><span style="{val_style}color:#f43f5e;font-size:26px;">81.82</span></div>
                <div><span style="{sub_style}">体积</span><br/><span style="{val_style}color:#e8edf5;font-size:26px;">250<span style="font-size:12px;color:#64748b;"> MB</span></span></div>
                <div><span style="{sub_style}">压缩比</span><br/><span style="{val_style}color:#f59e0b;font-size:26px;">8.8x</span></div>
            </div>
        </div>
    </div>
    """


def _table_html():
    rows = ""
    for i, r in enumerate(RESULTS):
        ppl = r["PPL"]
        if ppl < 25:
            ppl_color = "#059669"
        elif ppl < 35:
            ppl_color = "#10b981"
        elif ppl < 50:
            ppl_color = "#d97706"
        else:
            ppl_color = "#e11d48"

        ratio = r["Ratio"]
        if ratio >= 4:
            ratio_color = "#059669"
        elif ratio >= 2:
            ratio_color = "#d97706"
        else:
            ratio_color = "#64748b"

        bg = "rgba(10,18,36,0.25)" if i % 2 == 0 else "rgba(10,18,36,0.4)"
        rows += f"""
        <tr style="background:{bg};">
            <td style="padding:14px 20px;text-align:left;color:#cbd5e1;font-weight:500;border-top:1px solid rgba(59,130,246,0.04);font-family:'Noto Sans SC',sans-serif;font-size:14px;">{r['Version']}</td>
            <td style="padding:14px 20px;text-align:center;color:{ppl_color};font-weight:700;font-size:17px;font-family:'JetBrains Mono',monospace;border-top:1px solid rgba(59,130,246,0.04);">{ppl:.2f}</td>
            <td style="padding:14px 20px;text-align:center;color:#cbd5e1;font-weight:600;font-size:15px;font-family:'JetBrains Mono',monospace;border-top:1px solid rgba(59,130,246,0.04);">{r['Size(MB)']}</td>
            <td style="padding:14px 20px;text-align:center;color:{ratio_color};font-weight:700;font-size:17px;font-family:'JetBrains Mono',monospace;border-top:1px solid rgba(59,130,246,0.04);">{ratio}x</td>
        </tr>
        """

    return f"""
    <div style="border-radius:16px;overflow:hidden;border:1px solid rgba(59,130,246,0.08);box-shadow:0 1px 2px rgba(0,0,0,0.3),0 4px 12px rgba(0,0,0,0.25);margin:16px 0;">
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:rgba(12,20,44,0.7);">
                    <th style="padding:16px 20px;text-align:left;color:#94a3b8;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:1px solid rgba(59,130,246,0.1);font-family:'Noto Sans SC',sans-serif;">版本</th>
                    <th style="padding:16px 20px;text-align:center;color:#94a3b8;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:1px solid rgba(59,130,246,0.1);font-family:'Noto Sans SC',sans-serif;">困惑度 ↓</th>
                    <th style="padding:16px 20px;text-align:center;color:#94a3b8;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:1px solid rgba(59,130,246,0.1);font-family:'Noto Sans SC',sans-serif;">体积 (MB)</th>
                    <th style="padding:16px 20px;text-align:center;color:#94a3b8;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:1px solid rgba(59,130,246,0.1);font-family:'Noto Sans SC',sans-serif;">压缩比 ↑</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """


def build_dashboard():
    gr.Markdown("## 实验结果汇总\n<span style='font-size:15px;color:#64748b;'>Wikitext-2 困惑度 · 模型体积 · 压缩比</span>", elem_id="dash-title")
    gr.HTML(_cards_html())
    gr.HTML(_table_html())

    with gr.Tabs():
        with gr.TabItem("困惑度对比"):
            gr.Plot(_ppl_chart())
        with gr.TabItem("体积与压缩比"):
            gr.Plot(_size_chart())
