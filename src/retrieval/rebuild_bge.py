#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_bge.py —— 用真正的 BGE 模型重建三档索引

BGE (BAAI/bge-small-en-v1.5):
  - 384 维，CPU 可跑
  - 对英文技术文档效果好
  - 查询时要加指令前缀 "Represent this sentence for searching relevant passages: "
    （bge 官方推荐，对短查询→长文档的检索有提升）

用法:
    python rebuild_bge.py               # 建三档
    python rebuild_bge.py --only large
"""
import os
import sys
import json
import time
import pickle
import argparse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import IDX_DIR
from dense import DenseIndex
from bm25 import BM25

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache", "embeddings")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

VARIANTS = {
    "small":  ("chunks_20.jsonl",  "20% 文档"),
    "medium": ("chunks_50.jsonl",  "50% 文档"),
    "large":  ("chunks_100.jsonl", "100% 文档"),
}

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [bge] 加载 {MODEL_NAME} ...")
        t0 = time.time()
        _model = SentenceTransformer(MODEL_NAME)
        print(f"  [bge] 加载完成 {time.time()-t0:.1f}s")
    return _model


def encode(texts, batch=64, is_query=False):
    m = get_model()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return m.encode(
        texts, batch_size=batch, normalize_embeddings=True,
        show_progress_bar=len(texts) > 200, convert_to_numpy=True,
    ).astype(np.float32)


def build_one(tag, fname, desc):
    path = os.path.join(DATA, fname)
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    texts = [r["text"] for r in recs]
    ids = [r["chunk_id"] for r in recs]
    print(f"\n[{tag}] {desc}: {len(recs)} chunks")

    t0 = time.time()
    out_dir = os.path.join(IDX_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # --- BM25（不变）---
    bm = BM25(k1=1.5, b=0.75).fit(texts)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"docs": texts, "ids": ids,
                     "k1": bm.k1, "b": bm.b, "N": bm.N}, f)
    print(f"  BM25 完成")

    # --- BGE 编码（带缓存）---
    cache_npy = os.path.join(CACHE, f"bge_{tag}.npy")
    if os.path.exists(cache_npy):
        Z = np.load(cache_npy)
        if Z.shape[0] == len(texts):
            print(f"  [cache] bge_{tag}.npy 命中")
        else:
            Z = None
    else:
        Z = None

    if Z is None:
        print(f"  [bge] 编码 {len(texts)} 条 ...")
        Z = encode(texts)
        np.save(cache_npy, Z)
        print(f"  [bge] 完成，{Z.shape}")

    di = DenseIndex().build(Z, ids)
    di.save(os.path.join(out_dir, "dense"))

    meta = {
        "variant": tag,
        "desc": desc,
        "n_chunks": len(recs),
        "n_docs": len(set(c.split("_")[0] for c in ids)),
        "rrf_k": 60,
        "embed_backend": "bge-small-en-v1.5",
        "dim": int(Z.shape[1]),
        "query_prefix": QUERY_PREFIX.strip()[:40] + "...",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  完成 {meta['build_seconds']}s")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=list(VARIANTS.keys()))
    args = ap.parse_args()

    targets = args.only or list(VARIANTS.keys())
    metas = []
    for tag in targets:
        fname, desc = VARIANTS[tag]
        metas.append(build_one(tag, fname, desc))

    # 别名
    if "large" in targets:
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
    print("三档索引（BGE 版）")
    print("=" * 66)
    print(f"{'档位':<10}{'chunks':<10}{'维度':<8}{'后端':<24}{'耗时':<8}")
    for m in metas:
        print(f"{m['variant']:<10}{m['n_chunks']:<10}{m['dim']:<8}"
              f"{m['embed_backend']:<24}{m['build_seconds']}s")


if __name__ == "__main__":
    main()
