#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_index_variants.py —— 为三档数据量分别建索引

产出:
    data/indexes/small/    ← 20%  文档 (1458 chunks)
    data/indexes/medium/   ← 50%  文档 (3016 chunks)
    data/indexes/large/    ← 100% 文档 (5836 chunks)
    data/indexes/full/     ← 兼容旧路径（= large）

用法:
    python build_index_variants.py
    python build_index_variants.py --only small medium
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import load_chunks, IDX_DIR
from dense import DenseIndex, embed_texts
from bm25 import BM25

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

VARIANTS = {
    "small":  ("chunks_20.jsonl",  "20% 文档"),
    "medium": ("chunks_50.jsonl",  "50% 文档"),
    "large":  ("chunks_100.jsonl", "100% 文档"),
}


def build_one(tag, fname, desc, force_embed=False):
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"  [skip] {fname} 不存在，先跑 build_subset.py")
        return None

    t0 = time.time()
    recs = load_chunks(path)
    texts = [r["text"] for r in recs]
    ids = [r["chunk_id"] for r in recs]
    print(f"\n[{tag}] {desc}: {len(recs)} chunks")

    out_dir = os.path.join(IDX_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # BM25
    print("  构建 BM25 ...")
    bm = BM25(k1=1.5, b=0.75).fit(texts)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"docs": texts, "ids": ids,
                     "k1": bm.k1, "b": bm.b, "N": bm.N}, f)

    # Dense
    print("  构建稠密索引 ...")
    vecs, backend = embed_texts(texts, f"{tag}_{len(texts)}", force=force_embed)
    di = DenseIndex().build(vecs, ids)
    di.save(os.path.join(out_dir, "dense"))

    meta = {
        "variant": tag,
        "desc": desc,
        "n_chunks": len(recs),
        "n_docs": len(set(c.split("_")[0] for c in ids)),
        "rrf_k": 60,
        "embed_backend": backend,
        "dim": int(vecs.shape[1]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"  完成 {meta['build_seconds']}s  →  {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=list(VARIANTS.keys()))
    ap.add_argument("--force-embed", action="store_true")
    args = ap.parse_args()

    targets = args.only or list(VARIANTS.keys())
    metas = []
    for tag in targets:
        fname, desc = VARIANTS[tag]
        m = build_one(tag, fname, desc, force_embed=args.force_embed)
        if m:
            metas.append(m)

    # 兼容旧路径：把 large 复制成 full / 根目录
    if "large" in targets and os.path.exists(os.path.join(IDX_DIR, "large")):
        import shutil
        src = os.path.join(IDX_DIR, "large")
        for alias in ["full"]:
            dst = os.path.join(IDX_DIR, alias)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        # 同时更新根目录索引（让旧的 retriever.py 默认路径也能用）
        for fn in ["bm25.pkl", "dense.npy", "dense.ids.json", "meta.json"]:
            s = os.path.join(src, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(IDX_DIR, fn))
        print("\n[alias] large → full，并同步到默认索引路径")

    # 汇总
    summary = os.path.join(IDX_DIR, "variants_summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 62)
    print("三档索引汇总")
    print("=" * 62)
    print(f"{'档位':<10}{'文档':<8}{'chunks':<10}{'embed':<16}{'耗时':<8}")
    for m in metas:
        print(f"{m['variant']:<10}{m['n_docs']:<8}{m['n_chunks']:<10}"
              f"{m['embed_backend']:<16}{m['build_seconds']}s")
    print(f"\n汇总: {summary}")


if __name__ == "__main__":
    main()
