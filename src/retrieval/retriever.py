#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retriever.py —— 统一检索接口（B 的核心交付）

契约 2:
    retrieve(query: str, mode: str = "hybrid", k: int = 5) -> list[dict]
    mode ∈ {"dense", "sparse", "hybrid"}
    返回 [{"chunk_id", "score", "text", "url"}, ...]，按 score 降序

用法:
    python retriever.py --build             # 建索引（首次必跑）
    python retriever.py --query "list append"  # 命令行测试
"""
import os
import json
import argparse
import time

import numpy as np

from bm25 import BM25, tokenize
from dense import DenseIndex, embed_texts
from rrf import rrf_fuse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS = os.path.join(ROOT, "data", "chunks.jsonl")
IDX_DIR = os.path.join(ROOT, "data", "indexes")

RRF_K = 60          # Proposal 承诺的参数
CANDIDATES = 50     # 融合前每路取多少个候选


# ============================================================
# 索引构建
# ============================================================

def load_chunks(path=CHUNKS):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_index(chunks_path=CHUNKS, force_embed=False):
    """构建 BM25 + 稠密索引并落盘"""
    t0 = time.time()
    recs = load_chunks(chunks_path)
    print(f"[build] 加载 {len(recs)} 个 chunk")

    texts = [r["text"] for r in recs]
    ids = [r["chunk_id"] for r in recs]

    # --- BM25 ---
    print("[build] 构建 BM25 ...")
    bm = BM25(k1=1.5, b=0.75).fit(texts)
    os.makedirs(IDX_DIR, exist_ok=True)
    import pickle
    with open(os.path.join(IDX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump({"docs": texts, "ids": ids,
                     "k1": bm.k1, "b": bm.b, "N": bm.N}, f)
    print(f"  BM25 完成（{bm.N} 篇，avgdl={bm.avgdl:.1f}）")

    # --- 稠密 ---
    print("[build] 构建稠密索引 ...")
    key = f"chunks_{len(texts)}"
    vecs, backend = embed_texts(texts, key, force=force_embed)
    di = DenseIndex().build(vecs, ids)
    di.save(os.path.join(IDX_DIR, "dense"))
    print(f"  稠密索引完成（backend={di.backend}, dim={vecs.shape[1]}）")

    # --- 元信息 ---
    meta = {
        "n_chunks": len(recs),
        "rrf_k": RRF_K,
        "embed_backend": backend,
        "dim": int(vecs.shape[1]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(IDX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[build] 完成，用时 {meta['build_seconds']}s")
    print(f"        {IDX_DIR}")
    return meta


# ============================================================
# 延迟加载的检索器（单例）
# ============================================================

class Retriever:
    def __init__(self, chunks_path=None, idx_dir=None, variant="large"):
        """
        variant: "small" | "medium" | "large" | "full"
                 对应 20% / 50% / 100% 文档采样
        """
        self.variant = variant
        if chunks_path is None:
            fname = {"small": "chunks_20.jsonl", "medium": "chunks_50.jsonl",
                     "large": "chunks_100.jsonl", "full": "chunks_100.jsonl"}[variant]
            chunks_path = os.path.join(ROOT, "data", fname)
        if idx_dir is None:
            idx_dir = os.path.join(IDX_DIR, variant)
        self.chunks_path = chunks_path
        self.idx_dir = idx_dir
        self._loaded = False

        self.recs = None
        self.id2idx = None
        self.texts = None
        self.ids = None

        self._bm25 = None
        self._bm25_docs = None

        self._dense = None
        self._embed_model = None
        self._embed_backend = None

    # ---------- 加载 ----------

    def _ensure_loaded(self):
        if self._loaded:
            return
        t0 = time.time()

        self.recs = load_chunks(self.chunks_path)
        self.ids = [r["chunk_id"] for r in self.recs]
        self.texts = [r["text"] for r in self.recs]
        self.id2idx = {cid: i for i, cid in enumerate(self.ids)}

        # meta
        mp = os.path.join(self.idx_dir, "meta.json")
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"索引不存在，请先运行: python retriever.py --build\n({mp})"
            )
        self.meta = json.load(open(mp, encoding="utf-8"))

        # BM25
        import pickle
        bp = os.path.join(self.idx_dir, "bm25.pkl")
        with open(bp, "rb") as f:
            bdata = pickle.load(f)
        self._bm25_docs = bdata["docs"]
        self._bm25 = BM25(k1=bdata["k1"], b=bdata["b"]).fit(self._bm25_docs)

        # 稠密
        self._dense = DenseIndex.load(os.path.join(self.idx_dir, "dense"))
        self._embed_backend = self.meta.get("embed_backend", "unknown")

        self._loaded = True
        print(f"[retriever] 加载完成 {len(self.ids)} chunks，"
              f"{time.time()-t0:.2f}s，embed={self._embed_backend}")

    def _embed_query(self, q):
        """查询向量化（与建库时同一后端）"""
        # TF-IDF + SVD 后端
        if self._embed_backend == "tfidf-svd":
            from dense_tfidf import encode_query
            return encode_query(q, cache_key=f"tfidf_{self.variant}")
        if self._embed_backend and self._embed_backend.startswith("bge"):
            return _bge_encode_query(q)
        from dense import _hash_embed
        return _hash_embed([q])[0]

    # ---------- 三种检索 ----------

    def _search_dense(self, query, k):
        qv = self._embed_query(query)
        hits = self._dense.search(qv, k=k)
        return [(self.ids[i], s) for i, s in hits]

    def _search_sparse(self, query, k):
        hits = self._bm25.top_k(query, k=k)
        return [(self.ids[i], s) for i, s in hits]

    # ---------- 统一接口 ----------

    def retrieve(self, query, mode="hybrid", k=5):
        """
        契约 2 的实现。
        返回 [{"chunk_id","score","text","url"}, ...]
        """
        self._ensure_loaded()
        if not query or not query.strip():
            return []

        if mode == "dense":
            pairs = self._search_dense(query, k)
        elif mode == "sparse":
            pairs = self._search_sparse(query, k)
        elif mode == "hybrid":
            n_cand = max(CANDIDATES, k * 3)
            d_list = [cid for cid, _ in self._search_dense(query, n_cand)]
            s_list = [cid for cid, _ in self._search_sparse(query, n_cand)]
            fused = rrf_fuse([d_list, s_list], k=RRF_K, top_n=k)
            pairs = fused
        else:
            raise ValueError(f"unknown mode: {mode}")

        out = []
        for cid, score in pairs:
            i = self.id2idx.get(cid)
            if i is None:
                continue
            r = self.recs[i]
            out.append({
                "chunk_id": cid,
                "score": float(score),
                "text": r["text"][:400],     # 契约返回截断文本，完整版查 chunks
                "url": r["url"],
            })
        return out


# ------------------------------------------------------------
# BGE 查询编码（带模型缓存 + 官方建议的指令前缀）
# ------------------------------------------------------------
_BGE_MODEL = None
_BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def _bge_encode_query(q):
    global _BGE_MODEL
    import os as _os
    _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if _BGE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _BGE_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    v = _BGE_MODEL.encode([_BGE_PREFIX + q], normalize_embeddings=True,
                          convert_to_numpy=True)
    return v[0]


# 全局单例
_default = None


_cache_by_variant = {}


def get_retriever(variant="large"):
    """按档位获取（并缓存）检索器"""
    if variant not in _cache_by_variant:
        _cache_by_variant[variant] = Retriever(variant=variant)
    return _cache_by_variant[variant]


def retrieve(query, mode="hybrid", k=5, variant="large"):
    """契约 2 的模块级函数（新增 variant 参数，默认 100% 语料）"""
    return get_retriever(variant).retrieve(query, mode=mode, k=k)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="构建索引")
    ap.add_argument("--force-embed", action="store_true", help="强制重新编码")
    ap.add_argument("--query", type=str, default=None, help="测试查询")
    ap.add_argument("--mode", type=str, default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--compare", action="store_true", help="对比三种模式")
    ap.add_argument("--variant", type=str, default="large",
                    choices=["small", "medium", "large", "full"])
    args = ap.parse_args()

    if args.build:
        build_index(force_embed=args.force_embed)
        return

    r = Retriever(variant=args.variant)

    if args.query:
        if args.compare:
            for mode in ["dense", "sparse", "hybrid"]:
                t0 = time.time()
                hits = r.retrieve(args.query, mode=mode, k=args.k)
                dt = (time.time() - t0) * 1000
                print(f"\n=== mode={mode}  ({dt:.0f} ms) ===")
                for h in hits:
                    print(f"  {h['score']:.4f}  {h['chunk_id']}  {h['url'][:70]}")
                    print(f"          {h['text'][:100]}...")
        else:
            hits = r.retrieve(args.query, mode=args.mode, k=args.k)
            for h in hits:
                print(f"{h['score']:.4f}  {h['chunk_id']}  {h['url']}")
                print(f"    {h['text'][:120]}...")
        return

    print("用法: --build 建索引 | --query 'xxx' 测试")


if __name__ == "__main__":
    main()
