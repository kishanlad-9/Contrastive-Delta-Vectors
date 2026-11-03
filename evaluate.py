"""
evaluate.py
-----------
Automated benchmark evaluation script for Composed Image Retrieval (CIR).
Evaluates real ground-truth validation triplets from Fashion-IQ on:
  1. Baseline Naive CLIP Linear Blend
  2. Proposed CADS (Contrastive Attribute-Delta Steering)

Outputs verifiable Recall@10, Recall@50, and latency metrics to evaluation_results.json.
"""

import os
import json
import time
import argparse
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from app import parse_fashion_query

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CAPTIONS_DIR = os.path.join(BASE_DIR, "fashion-iq", "captions")
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIR on Fashion-IQ validation triplets.")
    parser.add_argument("--limit_per_category", type=int, default=150, 
                        help="Number of validation pairs to evaluate per category (default: 150, use -1 for all).")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results JSON.")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print("   EVALUATING ZERO-SHOT COMPOSED IMAGE RETRIEVAL ON FASHION-IQ")
    print("=" * 70)

    # 1. Load FAISS index and metadata
    print("\n[1/4] Loading FAISS index and product catalog metadata...")
    faiss_index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "faiss_index.bin"))
    with open(os.path.join(EMBEDDINGS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    asin_to_idx = {item["asin"]: i for i, item in enumerate(metadata)}
    print(f"      Loaded {faiss_index.ntotal} indexed products.")

    # 2. Load CLIP text encoder
    print(f"\n[2/4] Loading CLIP model ({MODEL_NAME}) on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    categories = ["dress", "shirt", "toptee"]
    results = {
        "metadata": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "total_catalog_size": faiss_index.ntotal,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluated_per_category": args.limit_per_category
        },
        "by_category": {},
        "overall": {}
    }

    all_naive_hits_10 = []
    all_naive_hits_50 = []
    all_cads_hits_10 = []
    all_cads_hits_50 = []
    all_cads_latencies = []

    print("\n[3/4] Running automated benchmark evaluation...")

    for cat in categories:
        val_file = os.path.join(CAPTIONS_DIR, f"cap.{cat}.val.json")
        if not os.path.exists(val_file):
            print(f"[WARN] Caption file not found: {val_file}")
            continue

        with open(val_file) as f:
            pairs = json.load(f)

        # Filter pairs where candidate and target both exist in the local catalog
        valid_pairs = [
            p for p in pairs
            if p["candidate"] in asin_to_idx and p["target"] in asin_to_idx
        ]

        if args.limit_per_category > 0 and len(valid_pairs) > args.limit_per_category:
            eval_pairs = valid_pairs[:args.limit_per_category]
        else:
            eval_pairs = valid_pairs

        n_eval = len(eval_pairs)
        print(f"\n  Evaluating category: '{cat.upper()}' ({n_eval} pairs out of {len(valid_pairs)} valid in catalog)...")

        cat_naive_10 = 0
        cat_naive_50 = 0
        cat_cads_10 = 0
        cat_cads_50 = 0
        cat_latencies = []

        for p_idx, p in enumerate(eval_pairs):
            cand_idx = asin_to_idx[p["candidate"]]
            target_idx = asin_to_idx[p["target"]]
            cand_vec = faiss_index.reconstruct(cand_idx).reshape(1, -1)

            # Combined caption from annotations
            raw_text = " and ".join(p.get("captions", []))

            # ── A. Naive Linear Baseline ─────────────────────────────────────
            with torch.no_grad():
                inp_naive = processor(text=raw_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                txt_naive = model.get_text_features(**inp_naive)
                txt_naive = (txt_naive / txt_naive.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

            v_naive = 0.5 * cand_vec + 0.5 * txt_naive
            v_naive = v_naive / np.linalg.norm(v_naive, axis=-1, keepdims=True)
            _, idxs_naive = faiss_index.search(v_naive.astype("float32"), 50)
            retrieved_naive = idxs_naive[0]

            if target_idx in retrieved_naive[:10]:
                cat_naive_10 += 1
            if target_idx in retrieved_naive[:50]:
                cat_naive_50 += 1

            # ── B. Proposed CADS Steering ─────────────────────────────────────
            t_start = time.perf_counter()
            parsed = parse_fashion_query(raw_text)

            with torch.no_grad():
                c_inp = processor(text=parsed["canonical"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                c_feat = model.get_text_features(**c_inp)
                c_feat = (c_feat / c_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

                pos_feat = np.zeros_like(cand_vec)
                if parsed["pos_modifiers"]:
                    p_inp = processor(text=parsed["pos_modifiers"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    p_f = model.get_text_features(**p_inp)
                    pos_feat = (p_f / p_f.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

                neg_feat = np.zeros_like(cand_vec)
                if parsed["neg_modifiers"]:
                    n_inp = processor(text=parsed["neg_modifiers"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    n_f = model.get_text_features(**n_inp)
                    neg_feat = (n_f / n_f.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

            w_pos = 0.50 * (1 - 0.5 * 0.5)
            w_neg = 0.30 * (1 - 0.5 * 0.5)
            v_cads = (0.5 * cand_vec) + (0.5 * c_feat) + (w_pos * pos_feat) - (w_neg * neg_feat)
            v_cads = v_cads / np.linalg.norm(v_cads, axis=-1, keepdims=True)

            _, idxs_cads = faiss_index.search(v_cads.astype("float32"), 50)
            t_query = (time.perf_counter() - t_start) * 1000.0  # in ms
            cat_latencies.append(t_query)

            retrieved_cads = idxs_cads[0]
            if target_idx in retrieved_cads[:10]:
                cat_cads_10 += 1
            if target_idx in retrieved_cads[:50]:
                cat_cads_50 += 1

            if (p_idx + 1) % 50 == 0 or (p_idx + 1) == n_eval:
                print(f"    Progress: {p_idx + 1}/{n_eval} queries completed...")

        r10_naive = round(100.0 * cat_naive_10 / n_eval, 2)
        r50_naive = round(100.0 * cat_naive_50 / n_eval, 2)
        r10_cads = round(100.0 * cat_cads_10 / n_eval, 2)
        r50_cads = round(100.0 * cat_cads_50 / n_eval, 2)
        mean_latency = round(float(np.mean(cat_latencies)), 2)

        results["by_category"][cat] = {
            "n_evaluated": n_eval,
            "naive_r10": r10_naive,
            "naive_r50": r50_naive,
            "cads_r10": r10_cads,
            "cads_r50": r50_cads,
            "gain_r10": round(r10_cads - r10_naive, 2),
            "gain_r50": round(r50_cads - r50_naive, 2),
            "mean_latency_ms": mean_latency
        }

        all_naive_hits_10.append(r10_naive)
        all_naive_hits_50.append(r50_naive)
        all_cads_hits_10.append(r10_cads)
        all_cads_hits_50.append(r50_cads)
        all_cads_latencies.extend(cat_latencies)

    # Compute overall averages
    results["overall"] = {
        "avg_naive_r10": round(float(np.mean(all_naive_hits_10)), 2),
        "avg_naive_r50": round(float(np.mean(all_naive_hits_50)), 2),
        "avg_cads_r10": round(float(np.mean(all_cads_hits_10)), 2),
        "avg_cads_r50": round(float(np.mean(all_cads_hits_50)), 2),
        "avg_gain_r10": round(float(np.mean(all_cads_hits_10) - np.mean(all_naive_hits_10)), 2),
        "avg_gain_r50": round(float(np.mean(all_cads_hits_50) - np.mean(all_naive_hits_50)), 2),
        "overall_mean_latency_ms": round(float(np.mean(all_cads_latencies)), 2)
    }

    # 4. Save results to JSON
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    # 5. Display formatted summary table
    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Category':<12} | {'Naive R@10':<11} {'CADS R@10':<11} {'Gain (R@10)':<12} | {'CADS R@50':<10} {'Latency'}")
    print("-" * 70)
    for cat in categories:
        if cat in results["by_category"]:
            d = results["by_category"][cat]
            print(f"{cat.capitalize():<12} | {d['naive_r10']:<11}% {d['cads_r10']:<11}% +{d['gain_r10']:<11}% | {d['cads_r50']:<10}% {d['mean_latency_ms']} ms")
    print("-" * 70)
    ov = results["overall"]
    print(f"{'AVERAGE':<12} | {ov['avg_naive_r10']:<11}% {ov['avg_cads_r10']:<11}% +{ov['avg_gain_r10']:<11}% | {ov['avg_cads_r50']:<10}% {ov['overall_mean_latency_ms']} ms")
    print("=" * 70)
    print(f"\n[OK] Exact evaluation results successfully logged to: {args.output_file}")

if __name__ == "__main__":
    main()
