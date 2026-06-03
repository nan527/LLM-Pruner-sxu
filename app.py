"""LLM Compression Experiment Platform - Gradio Web UI (Frosted Glass Redesign)"""
import torch
import gradio as gr
from modules.dashboard import build_dashboard
from modules.experiment import build_experiment
from modules.inference import build_inference

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ============================================================
   🔮  CSS Custom Properties — "Crystal Frost" Palette
   ============================================================ */
:root {
    --bg-base: #080e1a;
    --bg-warm: #0c1424;
    --bg-cool: #0a1120;

    --glass-bg: rgba(12, 20, 40, 0.55);
    --glass-bg-strong: rgba(16, 28, 56, 0.75);
    --glass-bg-hover: rgba(16, 28, 56, 0.65);
    --glass-border: rgba(59, 130, 246, 0.08);
    --glass-border-strong: rgba(59, 130, 246, 0.15);
    --glass-border-hover: rgba(59, 130, 246, 0.25);
    --glass-shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 4px 12px rgba(0, 0, 0, 0.25), 0 12px 40px rgba(0, 0, 0, 0.2);
    --glass-shadow-hover: 0 2px 8px rgba(0, 0, 0, 0.35), 0 8px 28px rgba(0, 0, 0, 0.3), 0 24px 56px rgba(0, 0, 0, 0.25);
    --glass-inner: inset 0 1px 0 rgba(59, 130, 246, 0.08);

    --accent: #3b82f6;
    --accent-dark: #2563eb;
    --accent-soft: rgba(59, 130, 246, 0.15);
    --accent-light: rgba(59, 130, 246, 0.06);
    --accent-glow: 0 0 28px rgba(59, 130, 246, 0.15);
    --accent-gradient: linear-gradient(135deg, #2563eb, #3b82f6, #60a5fa);

    --amber: #f59e0b;
    --amber-soft: rgba(245, 158, 11, 0.1);
    --emerald: #10b981;
    --rose: #f43f5e;
    --rose-soft: rgba(244, 63, 94, 0.1);
    --violet: #8b5cf6;
    --violet-soft: rgba(139, 92, 246, 0.1);
    --cyan: #06b6d4;

    --text-1: #e8edf5;
    --text-2: #94a3b8;
    --text-3: #64748b;
    --text-4: #475569;
    --text-inverse: #f8fafc;

    --radius: 16px;
    --radius-sm: 10px;
    --radius-lg: 24px;

    --font-display: 'Outfit', 'Segoe UI', system-ui, sans-serif;
    --font-serif: 'Noto Serif SC', 'Georgia', 'Times New Roman', serif;
    --font-sans: 'Noto Sans SC', 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;

    font-size: 16px;
}

/* ============================================================
   🌊  Root & Background
   ============================================================ */
.gradio-container {
    background: var(--bg-base) !important;
    color: var(--text-1) !important;
    font-family: var(--font-sans) !important;
    min-height: 100vh !important;
    position: relative !important;
    overflow-x: hidden !important;
}

/* Deep space blue glow orbs */
.gradio-container::before {
    content: '' !important;
    position: fixed !important;
    top: -40% !important;
    left: -40% !important;
    width: 180% !important;
    height: 180% !important;
    background:
        radial-gradient(ellipse 900px 600px at 15% 18%, rgba(59,130,246,0.15) 0%, transparent 55%),
        radial-gradient(ellipse 700px 700px at 80% 12%, rgba(6,182,212,0.10) 0%, transparent 55%),
        radial-gradient(ellipse 600px 500px at 50% 82%, rgba(99,102,241,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 500px 600px at 88% 65%, rgba(59,130,246,0.06) 0%, transparent 55%) !important;
    pointer-events: none !important;
    z-index: 0 !important;
    animation: floatOrbs 35s ease-in-out infinite alternate !important;
}
@keyframes floatOrbs {
    0%   { transform: translate(0, 0) rotate(0deg) scale(1); }
    25%  { transform: translate(-1.8%, 1.2%) rotate(1deg) scale(1.03); }
    50%  { transform: translate(1%, -1.5%) rotate(-0.6deg) scale(0.97); }
    75%  { transform: translate(-1%, 0.8%) rotate(0.4deg) scale(1.02); }
    100% { transform: translate(1.5%, -0.8%) rotate(-0.4deg) scale(0.98); }
}

/* Dark noise overlay */
.gradio-container::after {
    content: '' !important;
    position: fixed !important;
    inset: 0 !important;
    z-index: 0 !important;
    pointer-events: none !important;
    opacity: 0.15 !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    background-repeat: repeat !important;
    background-size: 140px 140px !important;
    mix-blend-mode: overlay !important;
}

.gradio-container > * { position: relative !important; z-index: 1 !important; }

.gradio-container .contain {
    max-width: 1360px !important;
    padding: 16px 28px 48px !important;
}

/* ============================================================
   🪟  Glass Card Mixin
   ============================================================ */
.glass-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(24px) saturate(1.6) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.6) !important;
    border: 1px solid var(--glass-border) !important;
    border-bottom: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--glass-shadow), var(--glass-inner) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.glass-card:hover {
    background: var(--glass-bg-hover) !important;
    border-color: var(--glass-border-hover) !important;
    box-shadow: var(--glass-shadow-hover), var(--glass-inner) !important;
}

/* ============================================================
   📑  Tab Navigation
   ============================================================ */
.tab-nav {
    background: rgba(12, 20, 44, 0.6) !important;
    backdrop-filter: blur(28px) saturate(1.6) !important;
    -webkit-backdrop-filter: blur(28px) saturate(1.6) !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.15) !important;
    border-radius: var(--radius-lg) !important;
    padding: 5px !important;
    gap: 3px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2), 0 4px 20px rgba(0,0,0,0.15), inset 0 1px 0 rgba(59,130,246,0.06) !important;
    margin-bottom: 24px !important;
    display: flex !important;
    flex-wrap: wrap !important;
}
.tab-nav > button {
    border-radius: var(--radius) !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    color: var(--text-3) !important;
    background: transparent !important;
    border: none !important;
    font-family: var(--font-sans) !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    cursor: pointer !important;
}
.tab-nav > button:hover {
    color: var(--text-2) !important;
    background: rgba(59, 130, 246, 0.08) !important;
}
.tab-nav > button.selected {
    color: #fff !important;
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    box-shadow: 0 2px 12px rgba(59, 130, 246, 0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    font-weight: 600 !important;
}
.tab-nav > button.selected::after {
    display: none !important;
}

/* ============================================================
   📦  Block (Component Wrappers)
   ============================================================ */
.block {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.5) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.12) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--glass-shadow), var(--glass-inner) !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: cardEnter 0.6s cubic-bezier(0.4, 0, 0.2, 1) both !important;
}
.block:hover {
    border-color: var(--glass-border-hover) !important;
    box-shadow: var(--glass-shadow-hover), var(--glass-inner) !important;
}

/* Staggered card entrance */
@keyframes cardEnter {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.block:nth-child(1) { animation-delay: 0.05s !important; }
.block:nth-child(2) { animation-delay: 0.10s !important; }
.block:nth-child(3) { animation-delay: 0.15s !important; }
.block:nth-child(4) { animation-delay: 0.20s !important; }
.block:nth-child(5) { animation-delay: 0.25s !important; }
.block:nth-child(6) { animation-delay: 0.30s !important; }

/* ============================================================
   🗂️  Accordion
   ============================================================ */
.accordion {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.5) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.12) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 18px !important;
    box-shadow: var(--glass-shadow), var(--glass-inner) !important;
    overflow: hidden !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: cardEnter 0.5s cubic-bezier(0.4, 0, 0.2, 1) both !important;
}
.accordion:hover {
    border-color: var(--glass-border-hover) !important;
    box-shadow: var(--glass-shadow-hover), var(--glass-inner) !important;
}
.accordion > .label-wrap {
    background: rgba(10, 16, 32, 0.4) !important;
    padding: 18px 24px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: var(--text-1) !important;
    font-family: var(--font-sans) !important;
    border-bottom: 1px solid rgba(59, 130, 246, 0.06) !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
}
.accordion > .label-wrap:hover {
    background: rgba(15, 24, 48, 0.5) !important;
}
.accordion > .label-wrap .icon-wrap svg {
    color: var(--text-3) !important;
    transition: transform 0.3s ease !important;
}
.accordion.open > .label-wrap .icon-wrap svg {
    transform: rotate(90deg) !important;
}

/* ============================================================
   🔘  Buttons
   ============================================================ */
button.primary {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6, #60a5fa) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.02em !important;
    border-radius: var(--radius-sm) !important;
    padding: 13px 34px !important;
    color: #fff !important;
    font-family: var(--font-sans) !important;
    box-shadow: 0 2px 12px rgba(59, 130, 246, 0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    cursor: pointer !important;
}
button.primary::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent) !important;
    transition: left 0.6s ease !important;
}
button.primary:hover::before {
    left: 100% !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 24px rgba(59, 130, 246, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
button.primary:active {
    transform: translateY(0) !important;
}

button.secondary {
    background: var(--glass-bg-strong) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.12) !important;
    color: var(--text-2) !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    font-family: var(--font-sans) !important;
    border-radius: var(--radius-sm) !important;
    padding: 13px 34px !important;
    box-shadow: var(--glass-shadow), var(--glass-inner) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
button.secondary:hover {
    border-color: rgba(59, 130, 246, 0.3) !important;
    color: #60a5fa !important;
    background: rgba(14, 24, 48, 0.85) !important;
    box-shadow: var(--glass-shadow-hover), var(--glass-inner) !important;
    transform: translateY(-1px) !important;
}

/* ============================================================
   ⌨️  Inputs
   ============================================================ */
input:not([type="range"]):not([type="checkbox"]),
textarea, select {
    background: rgba(10, 18, 36, 0.5) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
    font-family: var(--font-sans) !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}
input:not([type="range"]):not([type="checkbox"]):focus,
textarea:focus, select:focus {
    border-color: rgba(59, 130, 246, 0.4) !important;
    background: rgba(14, 24, 48, 0.6) !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08), 0 0 24px rgba(59, 130, 246, 0.06), inset 0 1px 3px rgba(0,0,0,0.2) !important;
}
input::placeholder, textarea::placeholder {
    color: var(--text-4) !important;
}

label, .label-text, .block label, .form label {
    color: var(--text-3) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-family: var(--font-sans) !important;
    margin-bottom: 8px !important;
}

/* ============================================================
   📊  Table / Dataframe
   ============================================================ */
table, .gradio-dataframe table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid rgba(59, 130, 246, 0.08) !important;
    box-shadow: var(--glass-shadow) !important;
    width: 100% !important;
}
thead th {
    background: rgba(12, 20, 44, 0.7) !important;
    color: var(--text-2) !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 16px 20px !important;
    text-align: center !important;
    border-bottom: 1px solid rgba(59, 130, 246, 0.1) !important;
    font-family: var(--font-sans) !important;
}
tbody td {
    padding: 14px 20px !important;
    border-top: 1px solid rgba(59, 130, 246, 0.04) !important;
    background: rgba(10, 18, 36, 0.3) !important;
    color: var(--text-1) !important;
    text-align: center !important;
    font-size: 14.5px !important;
    font-family: var(--font-mono) !important;
    transition: background 0.2s ease !important;
}
tbody tr:hover td {
    background: rgba(59, 130, 246, 0.08) !important;
}
.gradio-dataframe {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--glass-shadow) !important;
}

/* ============================================================
   📝  Markdown / Prose
   ============================================================ */
.prose, .prose p, .prose li, .prose span, .prose td {
    color: var(--text-2) !important;
    font-family: var(--font-sans) !important;
    line-height: 1.9 !important;
    font-size: 15.5px !important;
}
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: var(--text-1) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
.prose h1 {
    font-size: 28px !important;
    font-family: var(--font-serif) !important;
    border-bottom: 1px solid rgba(59,130,246,0.1) !important;
    padding-bottom: 14px !important;
    margin-top: 36px !important;
}
.prose h2 {
    font-size: 22px !important;
    font-family: var(--font-serif) !important;
    margin-top: 32px !important;
    margin-bottom: 10px !important;
    color: #60a5fa !important;
}
.prose h3 {
    font-size: 17.5px !important;
    font-weight: 600 !important;
    margin-top: 24px !important;
}
.prose code {
    background: rgba(59, 130, 246, 0.08) !important;
    color: #2563eb !important;
    padding: 3px 8px !important;
    border-radius: 5px !important;
    font-family: var(--font-mono) !important;
    font-size: 13.5px !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
}
.prose pre {
    background: rgba(8, 14, 30, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
    border-radius: var(--radius-sm) !important;
    padding: 16px 20px !important;
    box-shadow: var(--glass-shadow) !important;
}
.prose pre code {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: var(--text-1) !important;
}
.prose strong { color: var(--text-1) !important; }
.prose em { color: var(--text-3) !important; font-style: italic !important; }
.prose a {
    color: #60a5fa !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    border-bottom: 1px solid rgba(96, 165, 250, 0.2) !important;
    transition: border-color 0.2s ease !important;
}
.prose a:hover {
    border-bottom-color: rgba(96, 165, 250, 0.5) !important;
}
.prose blockquote {
    border-left: 3px solid rgba(59, 130, 246, 0.15) !important;
    margin-left: 0 !important;
    padding-left: 20px !important;
    color: var(--text-3) !important;
    font-style: italic !important;
}
.prose ul, .prose ol {
    padding-left: 24px !important;
}
.prose li {
    margin-bottom: 6px !important;
}
.prose hr {
    border: none !important;
    border-top: 1px solid rgba(59, 130, 246, 0.08) !important;
    margin: 28px 0 !important;
}

/* ============================================================
   📈  Plotly Charts
   ============================================================ */
.js-plotly-plot {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid rgba(59, 130, 246, 0.08) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.12) !important;
    box-shadow: var(--glass-shadow), var(--glass-inner) !important;
}

/* ============================================================
   🔄  Scrollbar
   ============================================================ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(59, 130, 246, 0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(59, 130, 246, 0.25);
}

/* ============================================================
   🎚️  Sliders & Checkboxes
   ============================================================ */
input[type=range] {
    accent-color: var(--accent) !important;
    height: 4px !important;
}
input[type=range]::-webkit-slider-thumb {
    background: var(--accent) !important;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.3) !important;
}
input[type=checkbox] {
    accent-color: var(--accent) !important;
}

/* ============================================================
   📋  Dropdown
   ============================================================ */
ul.options {
    background: rgba(10, 18, 40, 0.9) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--glass-shadow-hover) !important;
    padding: 4px !important;
}
ul.options > li {
    color: var(--text-2) !important;
    padding: 10px 16px !important;
    border-radius: 6px !important;
    font-size: 14.5px !important;
}
ul.options > li:hover {
    background: rgba(59, 130, 246, 0.12) !important;
    color: var(--text-1) !important;
}

/* ============================================================
   🏷️  Footer & Misc
   ============================================================ */
footer { display: none !important; }

/* Status textareas */
textarea[aria-label*="状态"],
textarea[aria-label*="Status"],
textarea[aria-label*="运行日志"],
textarea[aria-label*="Log Output"],
textarea[aria-label*="执行日志"],
textarea[aria-label*="Execution Log"] {
    font-family: var(--font-mono) !important;
    font-size: 13.5px !important;
    line-height: 1.7 !important;
    background: rgba(8, 14, 28, 0.5) !important;
    color: #10b981 !important;
}

/* Number inputs */
input[type=number] {
    font-family: var(--font-mono) !important;
}

/* Group panels */
.gradio-group {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: 1px solid rgba(59, 130, 246, 0.12) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--glass-shadow), var(--glass-inner) !important;
}
"""

HEADER_HTML = """
<div style="text-align:center;padding:24px 0 8px 0;position:relative;">

    <!-- Top blue accent bar -->
    <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);width:80px;height:4px;border-radius:2px;background:linear-gradient(90deg,#2563eb,#60a5fa,#2563eb);box-shadow:0 0 20px rgba(59,130,246,0.3);"></div>

    <!-- Platform badge -->
    <div style="display:inline-flex;align-items:center;gap:8px;
        background:rgba(59,130,246,0.1);
        backdrop-filter:blur(20px);
        -webkit-backdrop-filter:blur(20px);
        border:1px solid rgba(59,130,246,0.15);
        border-radius:100px;
        padding:5px 18px 5px 6px;
        margin-bottom:14px;
        box-shadow:0 1px 4px rgba(0,0,0,0.2),0 4px 16px rgba(0,0,0,0.15);">
        <span style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,#2563eb,#3b82f6);">
            <span style="color:#fff;font-size:11px;font-weight:700;font-family:'Outfit',sans-serif;">L</span>
        </span>
        <span style="color:#64748b;font-size:10.5px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;font-family:'Outfit',sans-serif;">
            LLM Compression Pipeline · v2.0
        </span>
        <span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:#60a5fa;box-shadow:0 0 6px rgba(59,130,246,0.5);"></span>
    </div>

    <!-- Title -->
    <h1 style="color:#e8edf5;font-size:40px;font-weight:800;margin:0 0 8px 0;
        letter-spacing:-0.03em;
        font-family:'Outfit','Noto Sans SC',system-ui,sans-serif;
        line-height:1.2;">
        LLM <span style="background:linear-gradient(135deg,#3b82f6,#60a5fa,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-size:200% 200%;animation:gradientShift 4s ease-in-out infinite;">压缩实验</span>平台
    </h1>

    <!-- Subtitle -->
    <p style="color:#64748b;font-size:15.5px;margin:0 auto;max-width:560px;line-height:1.8;font-family:'Noto Sans SC',sans-serif;">
        结构化剪枝 · LoRA 微调恢复 · 权重量化
        <span style="display:block;color:#475569;font-size:13px;margin-top:4px;">RTX 4060 Laptop · 消费级 GPU 全流程压缩 · TinyLlama-1.1B</span>
    </p>
</div>

<style>
@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
</style>
"""


DOC_MARKDOWN = """
<div style="max-width:900px;margin:0 auto;">

# 🧊 LLM 压缩实验平台 — 使用说明

---

## 📋 平台概述

**LLM 压缩实验平台** 是一套面向消费级 GPU（如 RTX 4060 Laptop）的大语言模型全流程压缩工具。平台集成了 **结构化剪枝 → LoRA 微调恢复 → 权重量化** 三大核心模块，帮助研究者在有限资源下探索模型压缩的效率与精度权衡。

> 当前基准模型：**TinyLlama-1.1B** | 运行设备：CUDA

---

## 🚀 快速开始

### 推荐工作流

```
原始模型 → 剪枝 25% → LoRA 微调 → W4 量化
         ↘ 评估 PPL & 体积 → 推理对比 → 部署
```

### 步骤一：结果看板
浏览历史实验数据，了解不同剪枝比例、微调与量化组合下的 **困惑度 (PPL)**、**模型体积** 与 **压缩比**。

### 步骤二：实验操作
按顺序执行三个压缩阶段：

| 阶段 | 方法 | 预期耗时 | 关键参数 |
|------|------|----------|----------|
| **1. 结构化剪枝** | Taylor 一阶重要性估计 | 10–30 min | 剪枝比例 0.25–0.50 |
| **2. LoRA 微调** | Low-Rank Adaptation | 1–4 hrs | Rank=8, LR=1e-4 |
| **3. 量化评估** | GPTQ W8/W4 | 5–15 min | Group Size=128 |

### 步骤三：推理演示
加载任意压缩版本，输入 Prompt 进行文本生成，或使用 **A/B 对比** 模式并排比较两个模型的输出质量与速度。

---

## 🧩 模块详解

### 1️⃣ 结果看板

看板展示所有实验方案的汇总数据，包含：

- **核心指标卡片** — 推荐方案、极限压缩、不推荐方案的 PPL/体积/压缩比一目了然
- **详细对比表格** — 所有实验方案的完整数据
- **可视化图表** — 困惑度柱状图（含原始基线参考线）、模型体积与压缩比双轴图

> 💡 **解读提示**：PPL 越接近原始模型 (17.42) 越好；体积越小越好；压缩比越大越好。推荐在 PPL 增幅 <15 的前提下追求最小体积。

### 2️⃣ 实验操作

#### Step 1 — 结构化剪枝 (Taylor)

基于 Taylor 展开的一阶重要性估计，对 Transformer 的 Attention 与 MLP 模块进行 Block-Wise 结构化剪枝。

**参数说明：**
- **剪枝比例** — 移除参数的比例 (0.1–0.9)。建议从 0.25 开始尝试
- **MLP / Attn 起始与终止层** — 控制剪枝的层范围。例如 TinyLlama 22 层，可设置起始 4、终止 20，保留输入输出层

#### Step 2 — LoRA 微调恢复

在剪枝后的模型上应用 Low-Rank Adaptation (LoRA)，以极少的可训练参数恢复模型的语言能力。

**参数说明：**
- **LoRA Rank** — 低秩矩阵的秩。Rank 越大恢复能力越强，但显存占用也越高（建议 8–16）
- **学习率** — 建议 1e-4 至 5e-4
- **训练轮数** — 2–3 轮通常足够

#### Step 3 — 量化评估

应用 GPTQ 风格的权重量化，进一步压缩模型体积。

**量化选项：**
- **GPTQ-W8** — INT8 权重量化，几乎无精度损失，体积减少约 45%
- **GPTQ-W4** — INT4 权重量化，精度略有下降，体积减少约 75%

> ⚠️ **注意**：当前量化实现为 **Weight-Only** 模拟量化，用于快速评估量化后的精度表现。实际部署时需替换为完整的量化推理框架。

### 3️⃣ 推理演示

#### 单模型推理
1. 从下拉菜单选择模型版本（或手动输入路径）
2. 点击 **加载模型**（首次加载约 10–30 秒）
3. 输入 Prompt 并调整生成参数
4. 点击 **生成文本**

#### A/B 对比
选择两个不同版本并排对比，自动逐个加载（不占双倍显存），方便评估剪枝/微调/量化对生成质量的影响。

**生成参数：**
| 参数 | 作用 | 推荐值 |
|------|------|--------|
| Temperature | 生成随机性 | 0.7 (创意) / 0.1 (精确) |
| Top-p | 词表截断概率 | 0.9 |
| Max Tokens | 最大生成长度 | 128–256 |

---

## 🔬 技术说明

### 结构化剪枝 (Taylor)

```
重要性 Score(w) = |w · ∂L/∂w|
```

对每个参数计算 Taylor 一阶重要性，移除重要性最低的参数。Block-Wise 模式保持模型结构的规整性，利于后续硬件加速。

### Low-Rank Adaptation (LoRA)

$$W' = W + BA$$

将权重更新量 ΔW 分解为两个低秩矩阵的乘积 (B ∈ R^{d×r}, A ∈ R^{r×k})，仅训练 B 和 A。r ≪ min(d, k)，大幅减少可训练参数量。

### 权重量化 (GPTQ)

基于近似二阶信息的 **Weight-Only** 量化方法。将 FP16 权重映射到 INT4/INT8，配合 Group-Wise 缩放因子补偿精度损失。

---

## ❓ 常见问题

**Q: 显存不足怎么办？**
> 减小剪枝比例、LoRA Rank 或批量大小。RTX 4060 (6GB) 建议：剪枝 ≤50%，LoRA Rank ≤8，Batch Size ≤4。

**Q: 如何导入自定义模型？**
> 在 HuggingFace 下载模型后，在实验模块的"基础模型路径"中填入本地路径或 HF 模型名。

**Q: 量化后模型如何导出？**
> 当前版本为模拟量化评估。实际部署请使用 `bitsandbytes`、`GPTQ-for-LLaMA` 或 `AutoGPTQ` 等框架。

**Q: A/B 对比时显存不够？**
> 系统采用串行加载策略 — 加载模型 A → 生成 → 卸载 → 加载模型 B → 生成，不要求两块 GPU。

---

## 📚 参考资源

| 资源 | 链接 |
|------|------|
| TinyLlama | [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) |
| LoRA 论文 | [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| GPTQ 论文 | [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) |
| Taylor Pruner | [https://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning) |

---

<div style="text-align:center;padding:24px 0;color:#94a3b8;font-size:12px;">
    LLM 压缩实验平台 · 基于 Gradio · 使用结构化剪枝 + LoRA + 量化技术
</div>
</div>
"""

with gr.Blocks(title="LLM 压缩实验平台", css=CUSTOM_CSS,
               theme=gr.themes.Base()) as app:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.TabItem("📊  结果看板"):
            build_dashboard()
        with gr.TabItem("🧪  实验操作"):
            build_experiment()
        with gr.TabItem("🤖  推理演示"):
            build_inference()
        with gr.TabItem("📖  使用说明"):
            gr.Markdown(DOC_MARKDOWN)

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
