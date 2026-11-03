"""
app.py
------
Flask web application for CLIP-based fashion product search.

Endpoints:
    GET  /          →  Serve the main UI
    POST /search    →  Accept image + text prompt, return top-k results as JSON

Run:
    python app.py
"""

import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import CLIPProcessor, CLIPModel

# ── Config ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
IMAGES_DIR    = os.path.join(BASE_DIR, "images")
MODEL_NAME    = "openai/clip-vit-base-patch32"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K         = 12

# ── Flask App ────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model & index at startup ────────────────────────────────────────
print(f"Loading CLIP model on {DEVICE} …")
clip_model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("Loading FAISS index …")
faiss_index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "faiss_index.bin"))

with open(os.path.join(EMBEDDINGS_DIR, "metadata.json")) as f:
    metadata = json.load(f)

print(f"Ready — {faiss_index.ntotal} products indexed.\n")


import re

# ── Smart Fashion Attribute Parser ───────────────────────────────────────

def parse_fashion_query(text):
    """
    Extracts fashion attributes (patterns, sleeves, colors, garment type),
    strips conversational noise, and generates:
      - canonical: structured CLIP-friendly prompt
      - pos_modifiers: target modification tokens to boost
      - neg_modifiers: contrasting base attributes to suppress
      - detected: human-readable attributes for UI feedback
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # 1. Sleeve detection (robust to typos: 'sleves', 'sleve', 'sleevs')
    pos_sleeves = []
    neg_sleeves = []
    detected_sleeves = []
    if re.search(r'\b(full|long)\s*sle+v[a-z]*\b', text_lower):
        pos_sleeves.append("long sleeves, full sleeve, covered arms")
        neg_sleeves.append("short sleeves, half sleeve, bare arms, sleeveless")
        detected_sleeves.append("Long / Full Sleeves")
    elif re.search(r'\b(short|half)\s*sle+v[a-z]*\b', text_lower):
        pos_sleeves.append("short sleeves, half sleeve, t-shirt sleeves")
        neg_sleeves.append("long sleeves, full sleeve, covered arms")
        detected_sleeves.append("Short / Half Sleeves")
    elif re.search(r'\b(sleeveless|tank\s*top|strapless)\b', text_lower):
        pos_sleeves.append("sleeveless, tank top, bare shoulders")
        neg_sleeves.append("long sleeves, short sleeves, covered arms")
        detected_sleeves.append("Sleeveless")

    # 2. Pattern detection
    pos_patterns = []
    neg_patterns = []
    detected_patterns = []
    if re.search(r'\b(checks?|checkered|plaid|tartan|gingham|buffalo\s*check)\b', text_lower):
        pos_patterns.append("checkered plaid pattern, tartan, gingham check")
        neg_patterns.append("plain solid color without pattern, solid fabric")
        detected_patterns.append("Checks / Plaid")
    elif re.search(r'\b(stripes?|striped|pinstripe)\b', text_lower):
        pos_patterns.append("striped pattern, horizontal stripes, vertical stripes")
        neg_patterns.append("plain solid color without pattern, solid fabric")
        detected_patterns.append("Stripes")
    elif re.search(r'\b(floral|flower|flowers|botanical)\b', text_lower):
        pos_patterns.append("floral print pattern, botanical flower pattern")
        neg_patterns.append("plain solid color without pattern, solid fabric")
        detected_patterns.append("Floral")
    elif re.search(r'\b(polka\s*dots?|dots?|spotted)\b', text_lower):
        pos_patterns.append("polka dot pattern, spotted print")
        neg_patterns.append("plain solid color without pattern, solid fabric")
        detected_patterns.append("Polka Dot")
    elif re.search(r'\b(plain|solid)\b', text_lower):
        pos_patterns.append("plain solid color, simple clean fabric")
        neg_patterns.append("checkered pattern, striped pattern, floral print")
        detected_patterns.append("Solid / Plain")

    # 3. Colors
    color_names = [
        'green', 'blue', 'red', 'black', 'white', 'yellow', 'pink', 'purple',
        'orange', 'brown', 'grey', 'gray', 'navy', 'maroon', 'olive', 'teal', 'beige', 'cream', 'khaki'
    ]
    detected_colors = [c.capitalize() for c in color_names if re.search(r'\b' + c + r'\b', text_lower)]

    # 4. Garment type
    garment = "clothing"
    if re.search(r'\b(dress|dresses|gown)\b', text_lower):
        garment = "dress"
    elif re.search(r'\b(shirt|button[- ]?down|collared)\b', text_lower):
        garment = "shirt"
    elif re.search(r'\b(t-?shirt|thirt|tee|top)\b', text_lower):
        garment = "t-shirt"

    # Clean out conversational stopwords
    cleaned_prompt = text
    for stopword in [
        r'\bgive me\b', r'\bfind me\b', r'\bshow me\b', r'\bi want\b', r'\bsimilar\b',
        r'\bsomething like\b', r'\bcan you\b', r'\bwith a\b', r'\bbut with\b',
        r'\binstead of\b', r'\bif the image is\b', r'\blook like\b', r'\blooks like\b',
        r'\bplease\b', r'\bx colou?r\b', r'\bpattern or\b'
    ]:
        cleaned_prompt = re.sub(stopword, '', cleaned_prompt, flags=re.IGNORECASE)
    cleaned_prompt = ' '.join(cleaned_prompt.split())

    # Build canonical description
    desc_parts = []
    if detected_colors:
        desc_parts.append("/".join(c.lower() for c in detected_colors))
    if pos_patterns:
        desc_parts.append(pos_patterns[0].split(",")[0])
    if pos_sleeves:
        desc_parts.append(pos_sleeves[0].split(",")[0])
    desc_parts.append(garment)

    canonical = f"a photo of a {' '.join(desc_parts)}" if (detected_colors or pos_patterns or pos_sleeves) else f"a photo of {cleaned_prompt}"

    return {
        "canonical": canonical,
        "pos_modifiers": " ".join(pos_patterns + pos_sleeves),
        "neg_modifiers": " ".join(neg_patterns + neg_sleeves),
        "detected": {
            "colors": detected_colors,
            "patterns": detected_patterns,
            "sleeves": detected_sleeves,
            "garment": garment
        }
    }


# ── Composed Query Encoder ───────────────────────────────────────────────

def encode_query(image_bytes=None, text=None, alpha=0.5):
    """
    Build a multi-modal composed query embedding from an uploaded image and/or a text prompt.
    Applies attribute delta vector steering for composed image retrieval (e.g. pattern, sleeve modification).
    Returns (query_embedding, parsed_info)
    """
    img_feat = None
    if image_bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_feat = feat.cpu().numpy().astype("float32")

    parsed = parse_fashion_query(text) if text and text.strip() else None

    # Case 1: Image + Text (Composed Image Retrieval with Delta Steering)
    if img_feat is not None and parsed is not None:
        with torch.no_grad():
            c_inp = clip_processor(text=parsed["canonical"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            c_feat = clip_model.get_text_features(**c_inp)
            c_feat = (c_feat / c_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

        pos_feat = np.zeros_like(img_feat)
        if parsed["pos_modifiers"]:
            with torch.no_grad():
                p_inp = clip_processor(text=parsed["pos_modifiers"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                p_feat = clip_model.get_text_features(**p_inp)
                pos_feat = (p_feat / p_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

        neg_feat = np.zeros_like(img_feat)
        if parsed["neg_modifiers"]:
            with torch.no_grad():
                n_inp = clip_processor(text=parsed["neg_modifiers"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                n_feat = clip_model.get_text_features(**n_inp)
                neg_feat = (n_feat / n_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

        # Weighted composition:
        # Base image weighted by alpha, target canonical weighted by (1 - alpha),
        # plus targeted modification injection and contrastive suppression
        img_weight = alpha
        text_weight = 1.0 - alpha
        pos_weight = 0.50 if parsed["pos_modifiers"] else 0.0
        neg_weight = 0.30 if parsed["neg_modifiers"] else 0.0

        combined = (img_weight * img_feat) + (text_weight * c_feat) + (pos_weight * pos_feat) - (neg_weight * neg_feat)
        norm = np.linalg.norm(combined, axis=-1, keepdims=True)
        combined = (combined / norm).astype("float32")
        return combined, parsed

    # Case 2: Image Only
    elif img_feat is not None:
        return img_feat, None

    # Case 3: Text Only
    elif parsed is not None:
        with torch.no_grad():
            c_inp = clip_processor(text=parsed["canonical"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            c_feat = clip_model.get_text_features(**c_inp)
            c_feat = (c_feat / c_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
        return c_feat, parsed

    return None, None


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/images/<category>/<filename>")
def serve_image(category, filename):
    """Serve product images from the images/ directory."""
    return send_from_directory(os.path.join(IMAGES_DIR, category), filename)


@app.route("/search", methods=["POST"])
def search():
    # Get uploaded image (optional)
    image_bytes = None
    if "image" in request.files:
        file = request.files["image"]
        if file.filename:
            image_bytes = file.read()

    # Get text prompt (optional)
    text = request.form.get("prompt", "").strip()

    # Get optional alpha (image vs text weight)
    try:
        alpha = float(request.form.get("alpha", 0.5))
    except ValueError:
        alpha = 0.5

    # Get optional category filter
    category_filter = request.form.get("category", "all")

    if not image_bytes and not text:
        return jsonify({"error": "Please upload an image or enter a text prompt."}), 400

    # Encode query with attribute-aware delta steering
    query, parsed_info = encode_query(image_bytes, text, alpha)
    if query is None:
        return jsonify({"error": "Could not process the query."}), 400

    # Search FAISS — fetch extra results in case we need to filter by category
    fetch_k = TOP_K * 5 if category_filter != "all" else TOP_K
    distances, indices = faiss_index.search(query, fetch_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        if category_filter != "all" and item["category"] != category_filter:
            continue

        # Build a URL the browser can reach via /images/<cat>/<file>
        fname = os.path.basename(item["path"])
        results.append({
            "asin":     item["asin"],
            "category": item["category"],
            "image_url": f"/images/{item['category']}/{fname}",
            "score":    round(float(dist), 4),
        })
        if len(results) >= TOP_K:
            break

    return jsonify({
        "results": results,
        "parsed_query": parsed_info
    })


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
