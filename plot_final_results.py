import os
import matplotlib.pyplot as plt

results = [
    ("Original", 17.42, 2200),
    ("Pruned25", 35.95, 1762),
    ("Pruned25+LoRA", 26.71, 1762),
    ("Pruned25+LoRA+W8", 26.71, 969),
    ("Pruned25+LoRA+W4", 30.75, 560),
    ("Pruned50", 90.72, 1426),
    ("Pruned50+LoRA", 45.18, 1426),
    ("Pruned50+LoRA+W8", 45.18, 796),
    ("Pruned50+LoRA+W4", 50.50, 471),
    ("DynamicINT8", 81.82, 250),
]

labels = [r[0] for r in results]
ppl = [r[1] for r in results]
size_mb = [r[2] for r in results]
compression = [2200 / s for s in size_mb]

os.makedirs("figures", exist_ok=True)

# Figure 1: PPL and model size bars
fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

axes[0].bar(labels, ppl, color="#3b82f6")
axes[0].set_title("PPL Comparison Across Compression Pipelines")
axes[0].set_ylabel("PPL (lower is better)")
axes[0].tick_params(axis="x", rotation=35)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

axes[1].bar(labels, size_mb, color="#10b981")
axes[1].set_title("Model Size Comparison")
axes[1].set_ylabel("Size (MB)")
axes[1].tick_params(axis="x", rotation=35)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.savefig("figures/final_results_bars.png", dpi=220)
plt.close(fig)

# Figure 2: Compression ratio vs PPL tradeoff
fig2, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax.scatter(compression, ppl, s=80, c="#ef4444")
for i, label in enumerate(labels):
    ax.annotate(label, (compression[i], ppl[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_title("Compression Ratio vs PPL Trade-off")
ax.set_xlabel("Compression Ratio (x)")
ax.set_ylabel("PPL (lower is better)")
ax.grid(True, linestyle="--", alpha=0.4)
fig2.savefig("figures/final_results_tradeoff.png", dpi=220)
plt.close(fig2)

print("Saved:")
print("- figures/final_results_bars.png")
print("- figures/final_results_tradeoff.png")
