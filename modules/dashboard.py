"""Module A: Results Dashboard — 实验数据可视化"""
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Chart colors
COLORS = {
    "blue": "#4da8da",
    "green": "#3fb986",
    "orange": "#f0a050",
    "red": "#e0556a",
    "purple": "#8b7ec8",
    "teal": "#3db8b0",
    "dark_bg": "#131822",
    "paper_bg": "#131822",
    "plot_bg": "#1a2030",
    "grid": "#252d3a",
    "text": "#8896aa",
    "text_title": "#c9d1d9",
}

SHORT_NAMES = [
    "原始 FP16", "剪枝25%", "+LoRA", "+W8", "+W4",
    "剪枝50%", "+LoRA", "+W8", "+W4", "+DynamicINT8"
]


def _dark_layout(fig, title):
    """Apply dark theme consistently to all charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text_title"]), x=0.5),
        paper_bgcolor=COLORS["paper_bg"],
        plot_bgcolor=COLORS["plot_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(b=120, l=60, r=30, t=60),
        height=480,
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=10, color=COLORS["text"]),
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
        ),
        legend=dict(
            font=dict(color=COLORS["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _make_ppl_chart():
    names = SHORT_NAMES
    ppls = [d["PPL"] for d in RESULTS]
    # Color: lower PPL = better (green), higher = worse (orange/red)
    bar_colors = [
        COLORS["green"] if p < 30 else COLORS["orange"] if p < 50 else COLORS["red"]
        for p in ppls
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=ppls,
        marker_color=bar_colors,
        marker_line=dict(width=0),
        text=[f"{p:.1f}" for p in ppls],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="%{x}<br>PPL: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=17.42, line_dash="dash", line_color=COLORS["blue"],
                  annotation_text="原始 PPL (17.42)", annotation_position="top right",
                  annotation_font=dict(color=COLORS["blue"], size=11))
    return _dark_layout(fig, "Perplexity 对比 &mdash; 越低越好")


def _make_combined_size_ratio():
    """Single chart: bars for size, line for compression ratio."""
    names = SHORT_NAMES
    sizes = [d["体积(MB)"] for d in RESULTS]
    ratios = [d["压缩比"] for d in RESULTS]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=names, y=sizes, name="模型体积 (MB)",
        marker_color=COLORS["blue"],
        marker_line=dict(width=0),
        text=[f"{s}MB" for s in sizes],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=10),
        hovertemplate="%{x}<br>体积: %{y} MB<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=names, y=ratios, name="压缩比",
        mode="lines+markers",
        line=dict(color=COLORS["orange"], width=3),
        marker=dict(size=8, color=COLORS["orange"]),
        hovertemplate="%{x}<br>压缩比: %{y:.1f}x<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="体积 (MB)", secondary_y=False,
                     gridcolor=COLORS["grid"], title_font=dict(color=COLORS["text"]))
    fig.update_yaxes(title_text="压缩比", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", title_font=dict(color=COLORS["text"]))
    return _dark_layout(fig, "模型体积 & 压缩比")


def _make_summary_cards():
    """Generate HTML summary cards."""
    best = RESULTS[4]   # Pruned 25% + LoRA + W4
    extreme = RESULTS[8]  # Pruned 50% + LoRA + W4
    original = RESULTS[0]

    cards = f"""
    <div style="display:flex;gap:16px;margin:20px 0;flex-wrap:wrap;">
        <div style="flex:1;min-width:200px;background:linear-gradient(135deg,#1a2540,#1a2030);border:1px solid #252d3a;border-radius:10px;padding:20px;">
            <div style="color:#4da8da;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;">推荐方案</div>
            <div style="color:#e2e6ed;font-size:16px;font-weight:700;margin-bottom:4px;">剪枝 25% + LoRA + W4</div>
            <div style="display:flex;gap:24px;margin-top:12px;">
                <div><span style="color:#8896aa;font-size:11px;">PPL</span><br/><span style="color:#3fb986;font-size:22px;font-weight:700;">{best['PPL']}</span></div>
                <div><span style="color:#8896aa;font-size:11px;">体积</span><br/><span style="color:#e2e6ed;font-size:22px;font-weight:700;">{best['体积(MB)']}<span style="font-size:13px;color:#556070;">MB</span></span></div>
                <div><span style="color:#8896aa;font-size:11px;">压缩比</span><br/><span style="color:#f0a050;font-size:22px;font-weight:700;">{best['压缩比']}x</span></div>
            </div>
        </div>
        <div style="flex:1;min-width:200px;background:linear-gradient(135deg,#201a1a,#201a1a);border:1px solid #302525;border-radius:10px;padding:20px;">
            <div style="color:#f0a050;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;">极限压缩</div>
            <div style="color:#e2e6ed;font-size:16px;font-weight:700;margin-bottom:4px;">剪枝 50% + LoRA + W4</div>
            <div style="display:flex;gap:24px;margin-top:12px;">
                <div><span style="color:#8896aa;font-size:11px;">PPL</span><br/><span style="color:#e0556a;font-size:22px;font-weight:700;">{extreme['PPL']}</span></div>
                <div><span style="color:#8896aa;font-size:11px;">体积</span><br/><span style="color:#e2e6ed;font-size:22px;font-weight:700;">{extreme['体积(MB)']}<span style="font-size:13px;color:#556070;">MB</span></span></div>
                <div><span style="color:#8896aa;font-size:11px;">压缩比</span><br/><span style="color:#f0a050;font-size:22px;font-weight:700;">{extreme['压缩比']}x</span></div>
            </div>
        </div>
        <div style="flex:1;min-width:200px;background:linear-gradient(135deg,#1a1a20,#131320);border:1px solid #252530;border-radius:10px;padding:20px;">
            <div style="color:#e0556a;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;">不推荐</div>
            <div style="color:#e2e6ed;font-size:16px;font-weight:700;margin-bottom:4px;">Dynamic INT8</div>
            <div style="display:flex;gap:24px;margin-top:12px;">
                <div><span style="color:#8896aa;font-size:11px;">PPL</span><br/><span style="color:#e0556a;font-size:22px;font-weight:700;">81.82</span></div>
                <div><span style="color:#8896aa;font-size:11px;">体积</span><br/><span style="color:#e2e6ed;font-size:22px;font-weight:700;">250<span style="font-size:13px;color:#556070;">MB</span></span></div>
                <div><span style="color:#8896aa;font-size:11px;">压缩比</span><br/><span style="color:#f0a050;font-size:22px;font-weight:700;">8.8x</span></div>
            </div>
        </div>
    </div>
    """
    return cards


def build_dashboard():
    gr.HTML("""
    <div style="margin-bottom:8px;">
        <h2 style="color:#c9d1d9;font-size:20px;font-weight:600;margin:0;">实验结果汇总</h2>
        <p style="color:#556070;font-size:13px;margin:4px 0 0 0;">Wikitext-2 PPL &middot; 模型体积 &middot; 压缩比</p>
    </div>
    """)

    # Summary cards
    gr.HTML(_make_summary_cards())

    # Data table
    gr.Dataframe(
        RESULTS,
        label="完整实验数据",
        interactive=False,
        headers=["版本", "PPL ↓", "体积 (MB)", "压缩比 ↑"],
        datatype=["str", "number", "number", "number"],
        column_widths=["40%", "20%", "20%", "20%"],
    )

    # Charts in tabs
    with gr.Tabs():
        with gr.TabItem(" PPL 对比"):
            gr.Plot(_make_ppl_chart())

        with gr.TabItem(" 体积 & 压缩比"):
            gr.Plot(_make_combined_size_ratio())
