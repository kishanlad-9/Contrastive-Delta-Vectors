import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'

# -------------------------------------------------------------
# 1. Methodology Diagram (fig_methodology.pdf / png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Background boxes
def draw_box(ax, x, y, w, h, text, title=None, color='#f0f4f8', border='#2b6cb0', text_color='#1a202c', fontsize=9):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.5,rounding_size=2",
                                ec=border, fc=color, lw=1.5)
    ax.add_patch(box)
    if title:
        ax.text(x + w/2, y + h - 5, title, ha='center', va='top', fontsize=fontsize+1, fontweight='bold', color=border)
        ax.text(x + w/2, y + (h - 8)/2, text, ha='center', va='center', fontsize=fontsize, color=text_color)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color=text_color)

# Draw Pipeline Nodes
draw_box(ax, 3, 58, 22, 28, "Query Image ($I_q$)\n+ Conversational Prompt\n($T_{\\mathrm{user}}$)", "1. Input Stream", '#ebf8ff', '#3182ce')
draw_box(ax, 29, 58, 22, 28, "Token Filtering &\nAttribute Extraction\n(Color, Pattern, Cut)", "2. Syntactic Parser", '#edf2f7', '#4a5568')
draw_box(ax, 55, 58, 22, 28, "Canonical: $T_{\\mathrm{canon}}$\nPositive: $T_{\\mathrm{pos}}$\nNegative: $T_{\\mathrm{neg}}$", "3. Prompt Synthesis", '#fefcbf', '#d69e2e')
draw_box(ax, 81, 58, 16, 28, "512-d CLIP\nEmbeddings\n$(\\mathbf{v}_{\\mathrm{img}}, \\mathbf{v}_{\\mathrm{txt}})$", "4. Encoders", '#e6fffa', '#319795')

# Bottom Flow Nodes
draw_box(ax, 20, 12, 26, 30, "$\\mathbf{v}^* = \\alpha \\mathbf{v}_{\\mathrm{img}} + (1-\\alpha)\\mathbf{v}_{\\mathrm{canon}}$\n$+ w_{\\mathrm{pos}}\\mathbf{v}_{\\mathrm{pos}} - w_{\\mathrm{neg}}\\mathbf{v}_{\\mathrm{neg}}$\n$\\mathbf{v}_{\\mathrm{query}} = \\mathbf{v}^* / \\|\\mathbf{v}^*\\|_2$", "5. Delta Steering", '#feebc8', '#dd6b20')
draw_box(ax, 52, 12, 24, 30, "FAISS IndexFlatIP\nCosine Similarity $\\langle \\mathbf{v}_q, \\mathbf{e}_i \\rangle$\n35,726 Items (<0.5 ms)", "6. Exact Search", '#e9d8fd', '#805ad5')
draw_box(ax, 80, 12, 17, 30, "Ranked Product\nCatalog ASINs\nRecall@10 / Recall@50", "7. Output Ranking", '#c6f6d5', '#38a169')

# Connective arrows
arrow_kw = dict(arrowstyle="->", lw=1.8, color="#2d3748")
ax.annotate("", xy=(29, 72), xytext=(25, 72), arrowprops=arrow_kw)
ax.annotate("", xy=(55, 72), xytext=(51, 72), arrowprops=arrow_kw)
ax.annotate("", xy=(81, 72), xytext=(77, 72), arrowprops=arrow_kw)

# Downward & Leftward return arrow
ax.annotate("", xy=(89, 45), xytext=(89, 58), arrowprops=arrow_kw)
ax.annotate("", xy=(46, 27), xytext=(89, 45), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748", connectionstyle="angle,angleA=0,angleB=90,rad=5"))

ax.annotate("", xy=(52, 27), xytext=(46, 27), arrowprops=arrow_kw)
ax.annotate("", xy=(80, 27), xytext=(76, 27), arrowprops=arrow_kw)

plt.tight_layout()
plt.savefig('figures/fig_methodology.pdf', bbox_inches='tight')
plt.savefig('figures/fig_methodology.png', bbox_inches='tight')
plt.close()
print("Generated fig_methodology")

# -------------------------------------------------------------
# 2. Model Architecture Diagram (fig_architecture.pdf / png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.0), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Inputs on left
draw_box(ax, 2, 74, 18, 18, "Candidate Garment\nImage $I_q$", None, '#e2e8f0', '#4a5568')
draw_box(ax, 2, 50, 18, 18, "Canonical Prompt\n$T_{\\mathrm{canon}}$", None, '#e2e8f0', '#4a5568')
draw_box(ax, 2, 26, 18, 18, "Positive Attributes\n$T_{\\mathrm{pos}}$", None, '#c6f6d5', '#2f855a')
draw_box(ax, 2, 2, 18, 18, "Suppression Counter\n$T_{\\mathrm{neg}}$", None, '#fed7d7', '#c53030')

# Dual-Tower Encoders
draw_box(ax, 25, 74, 18, 18, "CLIP ViT-B/32\nVision Encoder $\\mathcal{E}_v$", None, '#bee3f8', '#2b6cb0')
draw_box(ax, 25, 50, 18, 18, "CLIP ViT-B/32\nText Encoder $\\mathcal{E}_t$", None, '#feebc8', '#c05621')
draw_box(ax, 25, 26, 18, 18, "CLIP ViT-B/32\nText Encoder $\\mathcal{E}_t$", None, '#feebc8', '#c05621')
draw_box(ax, 25, 2, 18, 18, "CLIP ViT-B/32\nText Encoder $\\mathcal{E}_t$", None, '#feebc8', '#c05621')

# Vector Normalization
draw_box(ax, 48, 74, 12, 18, "$\\mathbf{v}_{\\mathrm{img}} \\in \\mathbb{R}^{512}$\n$\\|\\mathbf{v}\\|_2 = 1$", None, '#edf2f7', '#718096')
draw_box(ax, 48, 50, 12, 18, "$\\mathbf{v}_{\\mathrm{canon}} \\in \\mathbb{R}^{512}$\n$\\|\\mathbf{v}\\|_2 = 1$", None, '#edf2f7', '#718096')
draw_box(ax, 48, 26, 12, 18, "$\\mathbf{v}_{\\mathrm{pos}} \\in \\mathbb{R}^{512}$\n$\\|\\mathbf{v}\\|_2 = 1$", None, '#edf2f7', '#718096')
draw_box(ax, 48, 2, 12, 18, "$\\mathbf{v}_{\\mathrm{neg}} \\in \\mathbb{R}^{512}$\n$\\|\\mathbf{v}\\|_2 = 1$", None, '#edf2f7', '#718096')

# Fusion Block
draw_box(ax, 66, 26, 32, 46,
         "Contrastive Steering Engine\n\n"
         "$\\mathbf{v}^* = \\alpha \\mathbf{v}_{\\mathrm{img}} + (1-\\alpha)\\mathbf{v}_{\\mathrm{canon}}$\n"
         "$+ w_{\\mathrm{pos}}\\mathbf{v}_{\\mathrm{pos}} - w_{\\mathrm{neg}}\\mathbf{v}_{\\mathrm{neg}}$\n\n"
         "$\\mathbf{v}_{\\mathrm{query}} = \\frac{\\mathbf{v}^*}{\\|\\mathbf{v}^*\\|_2} \\in \\mathbb{S}^{511}$",
         "Contrastive Fusion", '#faf5ff', '#6b46c1')

# Output to FAISS
draw_box(ax, 66, 3, 32, 16, "FAISS IndexFlatIP (35,726)\nCosine Similarity Ranking", None, '#e6fffa', '#2c7a7b')

# Arrows
for y in [83, 59, 35, 11]:
    ax.annotate("", xy=(25, y), xytext=(20, y), arrowprops=arrow_kw)
    ax.annotate("", xy=(48, y), xytext=(43, y), arrowprops=arrow_kw)

ax.annotate("", xy=(66, 60), xytext=(60, 83), arrowprops=arrow_kw)
ax.annotate("", xy=(66, 52), xytext=(60, 59), arrowprops=arrow_kw)
ax.annotate("", xy=(66, 44), xytext=(60, 35), arrowprops=arrow_kw)
ax.annotate("", xy=(66, 36), xytext=(60, 11), arrowprops=arrow_kw)

ax.annotate("", xy=(82, 19), xytext=(82, 26), arrowprops=arrow_kw)

plt.tight_layout()
plt.savefig('figures/fig_architecture.pdf', bbox_inches='tight')
plt.savefig('figures/fig_architecture.png', bbox_inches='tight')
plt.close()
print("Generated fig_architecture")

# -------------------------------------------------------------
# 3. Pie Chart / Distribution Diagram (fig4_category_distribution.pdf / png)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

colors = ['#4299e1', '#48bb78', '#ed8936']
categories = ['Dress', 'Shirt', 'Top & Tee']
catalog_sizes = [12062, 12116, 11548]
val_pairs = [1847, 463, 159]

explode = (0.04, 0.04, 0.04)
wedges1, texts1, autotexts1 = ax1.pie(catalog_sizes, explode=explode, labels=categories, colors=colors,
                                      autopct='%1.1f%%', startangle=140, textprops=dict(color="#1a202c", fontsize=9))
for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_weight('bold')
ax1.set_title("Catalog Distribution\n(Total: 35,726 items)", fontsize=11, fontweight='bold', pad=10)

explode2 = (0.05, 0.08, 0.12)
wedges2, texts2, autotexts2 = ax2.pie(val_pairs, explode=explode2, labels=categories, colors=colors,
                                      autopct='%1.1f%%', startangle=140, textprops=dict(color="#1a202c", fontsize=9))
for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_weight('bold')
ax2.set_title("Valid Evaluation Pairs\n(Total: 2,469 pairs)", fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('figures/fig4_category_distribution.pdf', bbox_inches='tight')
plt.savefig('figures/fig4_category_distribution.png', bbox_inches='tight')
plt.close()
print("Generated fig4_category_distribution")
