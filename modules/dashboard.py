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

# Design tokens
C = {
    "bg": "rgba(6, 8, 15, 0)",
    "paper": "rgba(12, 18, 32, 0.0)",
    "plot": "rgba(10, 16, 30, 0.5)",
    "grid": "rgba(100, 160, 255, 0.08)",
    "text": "#8898b0",
    "text2": "#c0cee0",
    "title": "#e0e8f4",
    "accent": "#3c8cff",
    "cyan": "#00d4ff",
    "emerald": "#34d399",
    "amber": "#fbbf24",
    "rose": "#fb7185",
    "violet": "#a78bfa",
}


def _layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=C["title"], family="Noto Sans SC"), x=0.5, y=0.97),
        paper_bgcolor=C["paper"],
        plot_bgcolor=C["plot"],
        font=dict(color=C["text"], size=11, family="JetBrains Mono, Consolas, monospace"),
        margin=dict(b=90, l=50, r=30, t=50),
        height=400,
        xaxis=dict(
            tickangle=-30,
            tickfont=dict(size=10, color=C["text"]),
            gridcolor=C["grid"],
            linecolor=C["grid"],
        ),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["grid"]),
        legend=dict(font=dict(color=C["text2"], size=11), bgcolor="rgba(0,0,0,0)"),
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
        textfont=dict(color=C["text2"], size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{x}</b><br>PPL: %{y:.2f}<extra></extra>",
        opacity=0.95,
    ))
    fig.add_hline(
        y=17.42, line_dash="dot", line_color=C["accent"], line_width=1.5,
        annotation_text="原始 PPL 17.42",
        annotation_position="top right",
        annotation_font=dict(color=C["accent"], size=10, family="Noto Sans SC"),
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
        textfont=dict(color=C["text"], size=9, family="JetBrains Mono"),
        hovertemplate="<b>%{x}</b><br>体积: %{y} MB<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=SHORT, y=ratios, name="压缩比",
        mode="lines+markers",
        line=dict(color=C["cyan"], width=3),
        marker=dict(size=7, color=C["cyan"], line=dict(width=1, color="rgba(0,212,255,0.3)")),
        hovertemplate="<b>%{x}</b><br>压缩比: %{y:.1f}x<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="体积 (MB)", secondary_y=False, gridcolor=C["grid"], title_font=dict(color=C["text"], size=11))
    fig.update_yaxes(title_text="压缩比", secondary_y=True, gridcolor="rgba(0,0,0,0)", title_font=dict(color=C["cyan"], size=11))
    return _layout(fig, "模型体积与压缩比")


def _cards_html():
    best = RESULTS[4]
    extreme = RESULTS[8]

    card_style = (
        "flex:1;min-width:200px;"
        "background:rgba(12,18,32,0.55);"
        "backdrop-filter:blur(20px);"
        "border:1px solid rgba(100,160,255,0.08);"
        "border-radius:14px;"
        "padding:22px 24px;"
        "box-shadow:0 4px 24px rgba(0,0,0,0.2),inset 0 1px 0 rgba(255,255,255,0.03);"
        "transition:border-color 0.3s ease;"
    )
    label_style = "font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;"
    val_style = "font-family:'JetBrains Mono',monospace;font-weight:700;"
    sub_style = "color:#6b7a94;font-size:10px;font-family:'Noto Sans SC',sans-serif;"
    gap_style = "display:flex;gap:24px;margin-top:14px;"

    return f"""
    <div style="display:flex;gap:16px;margin:20px 0;flex-wrap:wrap;">
        <div style="{card_style}border-top:2px solid rgba(52,211,153,0.4);">
            <div style="{label_style}color:#34d399;">推荐方案</div>
            <div style="color:#e8edf5;font-size:15px;font-weight:700;margin-bottom:2px;">剪枝 25% + LoRA + W4</div>
            <div style="{gap_style}">
                <div><span style="{sub_style}">困惑度</span><br/><span style="{val_style}color:#34d399;font-size:24px;">{best['PPL']}</span></div>
                <div><span style="{sub_style}">体积</span><br/><span style="{val_style}color:#e8edf5;font-size:24px;">{best['Size(MB)']}<span style="font-size:11px;color:#4d5e78;">MB</span></span></div>
                <div><span style="{sub_style}">压缩比</span><br/><span style="{val_style}color:#fbbf24;font-size:24px;">{best['Ratio']}x</span></div>
            </div>
        </div>
        <div style="{card_style}border-top:2px solid rgba(251,191,36,0.4);">
            <div style="{label_style}color:#fbbf24;">极限压缩</div>
            <div style="color:#e8edf5;font-size:15px;font-weight:700;margin-bottom:2px;">剪枝 50% + LoRA + W4</div>
            <div style="{gap_style}">
                <div><span style="{sub_style}">困惑度</span><br/><span style="{val_style}color:#fb7185;font-size:24px;">{extreme['PPL']}</span></div>
                <div><span style="{sub_style}">体积</span><br/><span style="{val_style}color:#e8edf5;font-size:24px;">{extreme['Size(MB)']}<span style="font-size:11px;color:#4d5e78;">MB</span></span></div>
                <div><span style="{sub_style}">压缩比</span><br/><span style="{val_style}color:#fbbf24;font-size:24px;">{extreme['Ratio']}x</span></div>
            </div>
        </div>
        <div style="{card_style}border-top:2px solid rgba(251,113,133,0.3);">
            <div style="{label_style}color:#fb7185;">不推荐</div>
            <div style="color:#e8edf5;font-size:15px;font-weight:700;margin-bottom:2px;">Dynamic INT8</div>
            <div style="{gap_style}">
                <div><span style="{sub_style}">困惑度</span><br/><span style="{val_style}color:#fb7185;font-size:24px;">81.82</span></div>
                <div><span style="{sub_style}">体积</span><br/><span style="{val_style}color:#e8edf5;font-size:24px;">250<span style="font-size:11px;color:#4d5e78;">MB</span></span></div>
                <div><span style="{sub_style}">压缩比</span><br/><span style="{val_style}color:#fbbf24;font-size:24px;">8.8x</span></div>
            </div>
        </div>
    </div>
    """


def _table_html():
    rows = ""
    for i, r in enumerate(RESULTS):
        ppl = r["PPL"]
        if ppl < 25:
            ppl_color = "#34d399"
        elif ppl < 35:
            ppl_color = "#6ee7b7"
        elif ppl < 50:
            ppl_color = "#fbbf24"
        else:
            ppl_color = "#fb7185"

        ratio = r["Ratio"]
        if ratio >= 4:
            ratio_color = "#34d399"
        elif ratio >= 2:
            ratio_color = "#fbbf24"
        else:
            ratio_color = "#a4b0c8"

        bg = "rgba(60,140,255,0.04)" if i % 2 == 0 else "rgba(10,16,30,0.4)"
        rows += f"""
        <tr style="background:{bg};">
            <td style="padding:12px 18px;text-align:left;color:#dce4f0;font-weight:500;border-top:1px solid rgba(100,160,255,0.06);font-family:'Noto Sans SC',sans-serif;font-size:13px;">{r['Version']}</td>
            <td style="padding:12px 18px;text-align:center;color:{ppl_color};font-weight:700;font-size:16px;font-family:'JetBrains Mono',monospace;border-top:1px solid rgba(100,160,255,0.06);">{ppl:.2f}</td>
            <td style="padding:12px 18px;text-align:center;color:#dce4f0;font-weight:600;font-size:14px;font-family:'JetBrains Mono',monospace;border-top:1px solid rgba(100,160,255,0.06);">{r['Size(MB)']}</td>
            <td style="padding:12px 18px;text-align:center;color:{ratio_color};font-weight:700;font-size:16px;font-family:'JetBrains Mono',monospace;border-top:1px solid rgba(100,160,255,0.06);">{ratio}x</td>
        </tr>
        """

    return f"""
    <div style="border-radius:14px;overflow:hidden;border:1px solid rgba(100,160,255,0.12);box-shadow:0 4px 24px rgba(0,0,0,0.25);margin:16px 0;">
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:rgba(16,24,44,0.85);">
                    <th style="padding:14px 18px;text-align:left;color:#c0cee0;font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:2px solid rgba(60,140,255,0.2);font-family:'Noto Sans SC',sans-serif;">版本</th>
                    <th style="padding:14px 18px;text-align:center;color:#c0cee0;font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:2px solid rgba(60,140,255,0.2);font-family:'Noto Sans SC',sans-serif;">困惑度 ↓</th>
                    <th style="padding:14px 18px;text-align:center;color:#c0cee0;font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:2px solid rgba(60,140,255,0.2);font-family:'Noto Sans SC',sans-serif;">体积 (MB)</th>
                    <th style="padding:14px 18px;text-align:center;color:#c0cee0;font-weight:700;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;border-bottom:2px solid rgba(60,140,255,0.2);font-family:'Noto Sans SC',sans-serif;">压缩比 ↑</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """


def build_dashboard():
    gr.Markdown("## 实验结果汇总\nWikitext-2 困惑度 · 模型体积 · 压缩比")
    gr.HTML(_cards_html())
    gr.HTML(_table_html())

    with gr.Tabs():
        with gr.TabItem("困惑度对比"):
            gr.Plot(_ppl_chart())
        with gr.TabItem("体积与压缩比"):
            gr.Plot(_size_chart())
