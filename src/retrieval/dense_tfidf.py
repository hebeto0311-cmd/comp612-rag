#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dense_tfidf.py —— 基于 TF-IDF + SVD(LSA) 的稠密向量

为什么用这个而不是等 BGE：
  1. 零额外依赖（sklearn 已装）
  2. 真正的语义降维：TF-IDF 捕捉词权重，SVD 把共现模式压缩成稠密向量
  3. 在文档检索任务上，TF-IDF+SVD 是经典的强 baseline

对比 hash-embed：
  hash-embed 只是把词哈希到随机维度，**没有语义**；
  TF-IDF+SVD 通过矩阵分解学到了词共现结构，是真正的 latent semantic。

用法:
    python dense_tfidf.py --build     # 建向量并缓存
"""
import os
import json
import pickle
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "embeddings")


def build_tfidf_svd(texts, n_components=256, cache_key="tfidf_svd"):
    """
    返回 (vectors, meta)
    vectors 已 L2 归一化，可用内积做余弦相似度
    """
    os.makedirs(CACHE, exist_ok=True)
    vpath = os.path.join(CACHE, f"{cache_key}_vec.npy")
    mpath = os.path.join(CACHE, f"{cache_key}_model.pkl")

    if os.path.exists(vpath):
        return np.load(vpath), json.load(open(mpath.replace(".pkl", ".json"), encoding="utf-8"))

    n_components = min(n_components, max(2, len(texts) - 1), 512)

    print(f"  [tfidf] 拟合 TF-IDF（{len(texts)} 篇）...")
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),          # unigram + bigram
        min_df=2,                    # 过滤只出现一次的
        max_df=0.85,                 # 过滤出现太频繁的
        sublinear_tf=True,           # 用 1+log(tf)
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\d+",  # 保留下划线标识符
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    print(f"  [tfidf] 词表 {X.shape[1]}，稀疏矩阵 {X.shape}")

    print(f"  [tfidf] SVD 降到 {n_components} 维 ...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    ratio = float(svd.explained_variance_ratio_.sum())
    print(f"  [tfidf] 解释方差 {ratio:.2%}")

    Z = Normalizer(copy=False).fit_transform(Z).astype(np.float32)

    np.save(vpath, Z)
    with open(mpath, "wb") as f:
        pickle.dump({"tfidf": vec, "svd": svd}, f)

    meta = {
        "backend": "tfidf-svd",
        "dim": int(n_components),
        "vocab": int(X.shape[1]),
        "explained_variance": round(ratio, 4),
        "n": len(texts),
    }
    json.dump(meta, open(mpath.replace(".pkl", ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return Z, meta


def encode_query(query, cache_key="tfidf_svd"):
    """用训练好的 TF-IDF+SVD 编码查询"""
    mpath = os.path.join(CACHE, f"{cache_key}_model.pkl")
    if not os.path.exists(mpath):
        raise FileNotFoundError("先跑 dense_tfidf.py --build")
    with open(mpath, "rb") as f:
        m = pickle.load(f)
    x = m["tfidf"].transform([query])
    z = m["svd"].transform(x)
    z = Normalizer(copy=False).fit_transform(z).astype(np.float32)
    return z[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n-components", type=int, default=256)
    args = ap.parse_args()

    if args.build:
        # 用 large 档建
        chunks = os.path.join(ROOT, "data", "chunks_100.jsonl")
        texts = [json.loads(l)["text"] for l in open(chunks, encoding="utf-8") if l.strip()]
        print(f"[build] {len(texts)} chunks")
        Z, meta = build_tfidf_svd(texts, args.n_components)
        print(f"\n[DONE] 向量 {Z.shape}")
        print(json.dumps(meta, ensure_ascii=False, indent=1))
