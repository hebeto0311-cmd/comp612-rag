#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_with_tfidf.py —— 用 TF-IDF+SVD 重建三档索引并评估

这是 B 的最终版索引方案：
  稠密：TF-IDF + SVD (256 维)   ← 替代失效的 hash-embed
  稀疏：BM25 (k1=1.5, b=0.75)
  融合：RRF (k=60)
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import IDX_DIR
from dense import DenseIndex
from bm25 import BM25
from dense_tfidf import build_tfidf_svd, encode_query, CACHE as TV_CACHE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

VARIANTS = {
    "small":  ("chunks_20.jsonl",  "20% 文档"),
    "medium": ("chunks_50.jsonl",  "50% 文档"),
    "large":  ("chunks_100.jsonl", "100% 文档"),
}


def load(fname):
    recs = []
    with open(os.path.join(DATA, fname), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def build_one(tag, fname, desc, n_components=256):
    recs = load(fname)
    texts = [r["text"] for r in recs]
    ids = [r["chunk_id"] for r in recs]
    print(f"\n[{tag}] {desc}: {len(recs)} chunks")

    t0 = time.time()
    out_dir = os.path.join(IDX_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # BM25
    bm = BM25(k1=1.5, b=0.75).fit(texts)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"docs": texts, "ids": ids,
                     "k1": bm.k1, "b": bm.b, "N": bm.N}, f)

    # TF-IDF + SVD（每档单独拟合，保证子集独立）
    print(f"  TF-IDF+SVD ...")
    Z, meta_tv = build_tfidf_svd(texts, n_components=n_components,
                                 cache_key=f"tfidf_{tag}")
    di = DenseIndex().build(Z, ids)
    di.save(os.path.join(out_dir, "dense"))

    meta = {
        "variant": tag,
        "desc": desc,
        "n_chunks": len(recs),
        "n_docs": len(set(c.split("_")[0] for c in ids)),
        "rrf_k": 60,
        "embed_backend": "tfidf-svd",
        "dim": int(Z.shape[1]),
        "tfidf_vocab": meta_tv["vocab"],
        "explained_variance": meta_tv["explained_variance"],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  完成 {meta['build_seconds']}s")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-components", type=int, default=256)
    args = ap.parse_args()

    metas = []
    for tag, (fname, desc) in VARIANTS.items():
        metas.append(build_one(tag, fname, desc, args.n_components))

    # 别名
    import shutil
    src = os.path.join(IDX_DIR, "large")
    dst = os.path.join(IDX_DIR, "full")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for fn in ["bm25.pkl", "dense.npy", "dense.ids.json", "meta.json"]:
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(IDX_DIR, fn))

    json.dump(metas, open(os.path.join(IDX_DIR, "variants_summary.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 66)
    print("三档索引（TF-IDF+SVD 版）")
    print("=" * 66)
    print(f"{'档位':<10}{'chunks':<10}{'维度':<8}{'解释方差':<12}{'耗时':<8}")
    for m in metas:
        print(f"{m['variant']:<10}{m['n_chunks']:<10}{m['dim']:<8}"
              f"{m['explained_variance']:<12.2%}{m['build_seconds']}s")


if __name__ == "__main__":
    main()
