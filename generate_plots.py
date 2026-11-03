"""
generate_plots.py
-----------------
Generates publication-quality 300 DPI vector charts (PNG and PDF)
from the full 2,469-pair evaluation logs in evaluation_results_full.json.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "evaluation_results_full.json")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set high-quality IEEE styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

with open(RESULTS_FILE) as f:
    data = json.load(f)

categories = ["Dress", "Shirt", "Top & Tee", "Average"]
cat_keys = ["dress", "shirt", "toptee"]

# Methods to plot
methods_to_plot = [
    ("image_only", "Image-Only", "#95a5a6"),
    ("text_only", "Text-Only", "#34495e"),
    ("naive_blend", "Naive Blend (CLIP)", "#7f8c8d"),
    ("cads_full", "CADS (Proposed)", "#2980b9")
]

x = np.arange(len(categories))
width = 0.2

# ─── Figure 1: Recall@10 Comparison across all 2,469 pairs ───────────────────
fig, ax = plt.subplots(figsize=(9, 4.2))

for idx, (m_key, m_label, m_color) in enumerate(methods_to_plot):
    r10_vals = [data["by_category"][k][m_key]["r10"] for k in cat_keys] + [data["overall"][m_key]["avg_r10"]]
    offset = (idx - 1.5) * width
    bars = ax.bar(x + offset, r10_vals, width, label=m_label, color=m_color, edgecolor="black", alpha=0.9)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

ax.set_ylabel("Recall@10 (%)")
ax.set_title("Recall@10 Retrieval Accuracy on Fashion-IQ (All 2,469 Valid Pairs)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 20)

plt.tight_layout()
fig1_path_png = os.path.join(FIG_DIR, "fig1_recall_comparison.png")
fig1_path_pdf = os.path.join(FIG_DIR, "fig1_recall_comparison.pdf")
plt.savefig(fig1_path_png, dpi=300)
plt.savefig(fig1_path_pdf)
plt.close()
print(f"[OK] Saved Figure 1 to: {fig1_path_png} and {fig1_path_pdf}")

# ─── Figure 2: Ablation Study Progression on Top & Tee ──────────────────────
ablation_stages = [
    "Naive Blend\n(50/50 CLIP)",
    "+ Normalizer\n(T_canon)",
    "+ Positive\nInjection (+v_pos)",
    "+ Negative\nSuppression (CADS)"
]
ablation_toptee_r10 = [
    data["by_category"]["toptee"]["naive_blend"]["r10"],
    data["by_category"]["toptee"]["ablation_normalizer"]["r10"],
    data["by_category"]["toptee"]["ablation_pos_only"]["r10"],
    data["by_category"]["toptee"]["cads_full"]["r10"]
]
colors_ablation = ["#7f8c8d", "#16a085", "#2980b9", "#8e44ad"]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.bar(ablation_stages, ablation_toptee_r10, color=colors_ablation, edgecolor="black", width=0.55)
ax.set_ylabel("Recall@10 (%)")
ax.set_title("Ablation Study: Progressive Step Gains on Top & Tee Category")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(12, 18)

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=9.5, weight="bold")

plt.tight_layout()
fig2_path_png = os.path.join(FIG_DIR, "fig2_ablation_study.png")
fig2_path_pdf = os.path.join(FIG_DIR, "fig2_ablation_study.pdf")
plt.savefig(fig2_path_png, dpi=300)
plt.savefig(fig2_path_pdf)
plt.close()
print(f"[OK] Saved Figure 2 to: {fig2_path_png} and {fig2_path_pdf}")

# ─── Figure 3: Execution Latency Breakdown ──────────────────────────────────
latency_data = data["latency_profiles"]
stages = [
    "Syntactic Parsing\n& Normalizer",
    "CLIP Text\nEncoding (Batched)",
    "Vector Delta\nFusion",
    "FAISS IndexFlatIP\nSearch (35k catalog)"
]
latencies = [
    latency_data["query_parsing_normalizer_ms"],
    latency_data["clip_text_encoding_ms"],
    latency_data["delta_fusion_ms"],
    latency_data["faiss_search_35k_catalog_ms"]
]
colors_latency = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.barh(stages, latencies, color=colors_latency, edgecolor="black", height=0.5)
ax.set_xlabel("Latency per Query (milliseconds)")
ax.set_title(f"End-to-End Latency Breakdown (Total: {latency_data['total_query_latency_ms']} ms)")
ax.grid(axis="x", linestyle="--", alpha=0.4)

for b in bars:
    w = b.get_width()
    ax.annotate(f"{w:.2f} ms", xy=(w, b.get_y() + b.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, weight="bold")

ax.set_xlim(0, 24)
plt.tight_layout()
fig3_path_png = os.path.join(FIG_DIR, "fig3_latency_breakdown.png")
fig3_path_pdf = os.path.join(FIG_DIR, "fig3_latency_breakdown.pdf")
plt.savefig(fig3_path_png, dpi=300)
plt.savefig(fig3_path_pdf)
plt.close()
print(f"[OK] Saved Figure 3 to: {fig3_path_png} and {fig3_path_pdf}")
