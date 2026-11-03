"""
build_index.py
--------------
Scans every image in images/{dress,shirt,toptee}/, computes a CLIP ViT-B/32
embedding for each, and saves:
  - embeddings/faiss_index.bin   (FAISS flat L2 index)
  - embeddings/metadata.json     (ordered list of {asin, category, path})

Run once (takes ~15-30 min on CPU for ~35k images):
    python build_index.py
"""

import os
import json
import glob
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── Config ───────────────────────────────────────────────────────────────
IMAGES_ROOT   = os.path.join(os.path.dirname(__file__), "images")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "embeddings")
CATEGORIES    = ["dress", "shirt", "toptee"]
MODEL_NAME    = "openai/clip-vit-base-patch32"
BATCH_SIZE    = 64
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Helpers ──────────────────────────────────────────────────────────────

def collect_image_paths():
    """Return a list of (asin, category, full_path) for every image on disk."""
    records = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(IMAGES_ROOT, cat)
        if not os.path.isdir(cat_dir):
            print(f"[WARN] Skipping missing folder: {cat_dir}")
            continue
        for fname in os.listdir(cat_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            asin = os.path.splitext(fname)[0]
            records.append((asin, cat, os.path.join(cat_dir, fname)))
    return records


def encode_images_in_batches(model, processor, image_paths):
    """Encode a list of image file-paths into a (N, 512) numpy array."""
    all_embeddings = []
    total = len(image_paths)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_paths = image_paths[start:end]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                # Corrupt / unreadable – use a blank placeholder so indices stay aligned
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)           # (B, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True)     # L2 normalise
        all_embeddings.append(feats.cpu().numpy())

        done = min(end, total)
        print(f"  Encoded {done}/{total} images …")

    return np.vstack(all_embeddings).astype("float32")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Collect images
    print("[1/4] Collecting image paths …")
    records = collect_image_paths()
    print(f"      Found {len(records)} images across {CATEGORIES}")

    if not records:
        print("No images found.  Run download_images.py first.")
        return

    # 2. Load CLIP
    print(f"[2/4] Loading CLIP model ({MODEL_NAME}) on {DEVICE} …")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 3. Compute embeddings
    print("[3/4] Computing image embeddings …")
    paths = [r[2] for r in records]
    embeddings = encode_images_in_batches(model, processor, paths)
    print(f"      Embedding matrix shape: {embeddings.shape}")

    # 4. Build FAISS index  (Inner-Product on L2-normalised vectors == cosine similarity)
    print("[4/4] Building FAISS index …")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.bin"))

    metadata = [
        {"asin": r[0], "category": r[1], "path": r[2]}
        for r in records
    ]
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"✓ Saved FAISS index ({index.ntotal} vectors, dim={dim})")
    print(f"  → {os.path.join(OUTPUT_DIR, 'faiss_index.bin')}")
    print(f"  → {os.path.join(OUTPUT_DIR, 'metadata.json')}")


if __name__ == "__main__":
    main()
