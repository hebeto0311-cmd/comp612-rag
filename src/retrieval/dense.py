#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dense.py —— 稠密检索（numpy 实现，faiss 可选）

设计：
  - 优先用 faiss（如果装了），否则用 numpy 的矩阵乘法
  - 本项目数据量（几千~上万 chunk）用 numpy 完全够，且零依赖风险

向量来源（按优先级）：
  1. sentence-transformers 的 BGE 模型（Proposal 承诺用 BGE）
  2. 兜底：TF-IDF 风格的哈希投影（无网络/无模型时的可跑方案）
"""
import os
import json
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "embeddings")


# ---------------- 向量化后端 ----------------

def _try_bge(texts, model_name="BAAI/bge-small-en-v1.5", batch=32):
    """尝试用 sentence-transformers 的 BGE 编码"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        model = SentenceTransformer(model_name)
        embs = model.encode(
            texts, batch_size=batch, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        return embs.astype(np.float32)
    except Exception as e:
        print(f"  [BGE 失败] {type(e).__name__}: {str(e)[:80]}")
        return None


def _hash_embed(texts, dim=384):
    """
    兜底方案：字符 n-gram 哈希投影 + L2 归一化。
    虽然不是语义向量，但能保证 pipeline 端到端跑通。
    """
    from collections import Counter
    import hashlib
    import re

    def feats(t):
        t = t.lower()
        toks = re.findall(r"[a-z_][a-z0-9_]*|\d+", t)
        c = Counter(toks)
        # 加二元组
        for a, b in zip(toks, toks[1:]):
            c[a + "_" + b] += 1
        return c

    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok, cnt in feats(t).items():
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vecs[i, idx] += sign * (1.0 + np.log(cnt))
        # sublinear tf + L2 归一化
        n = np.linalg.norm(vecs[i])
        if n > 0:
            vecs[i] /= n
    return vecs


def embed_texts(texts, cache_key, force=False):
    """
    编码文本，带磁盘缓存。
    返回 (vectors ndarray, backend_name)
    """
    os.makedirs(CACHE, exist_ok=True)
    npy = os.path.join(CACHE, f"{cache_key}.npy")
    meta = os.path.join(CACHE, f"{cache_key}.json")

    if os.path.exists(npy) and os.path.exists(meta) and not force:
        info = json.load(open(meta, encoding="utf-8"))
        if info.get("n") == len(texts):
            print(f"  [cache hit] {cache_key}  backend={info.get('backend')}")
            return np.load(npy), info.get("backend", "unknown")

    # 尝试 BGE
    print(f"  [embed] {len(texts)} 条文本，尝试 BGE ...")
    vecs = _try_bge(texts)
    backend = "bge-small-en-v1.5"
    if vecs is None:
        print("  [embed] BGE 不可用，退回到 hash-embed（维度 384）")
        vecs = _hash_embed(texts, dim=384)
        backend = "hash-embed"

    np.save(npy, vecs)
    json.dump({"n": len(texts), "backend": backend, "dim": int(vecs.shape[1])},
              open(meta, "w", encoding="utf-8"))
    print(f"  [embed] 完成，维度 {vecs.shape}, backend={backend}")
    return vecs, backend


# ---------------- 索引 ----------------

class DenseIndex:
    """稠密索引：向量已 L2 归一化，内积即余弦相似度"""

    def __init__(self, vectors=None, ids=None):
        self.vectors = vectors          # (N, dim) float32, 已归一化
        self.ids = ids or []            # chunk_id 列表
        self._faiss = None

    def build(self, vectors, ids):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = list(ids)
        # 确保归一化
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors /= norms

        # 有 faiss 就用 faiss
        try:
            import faiss
            idx = faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(self.vectors)
            self._faiss = idx
            self.backend = "faiss"
        except Exception:
            self._faiss = None
            self.backend = "numpy"
        return self

    def search(self, query_vec, k=5):
        """返回 [(doc_idx, score)]"""
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, min(k, len(self.ids)))
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        # numpy 版
        sims = (self.vectors @ q.T).ravel()
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + ".npy", self.vectors)
        with open(path + ".ids.json", "w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        vecs = np.load(path + ".npy")
        ids = json.load(open(path + ".ids.json", encoding="utf-8"))
        inst = cls()
        inst.vectors = vecs
        inst.ids = ids
        inst._faiss = None
        inst.backend = "numpy(reloaded)"
        return inst
