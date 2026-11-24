#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rrf_variants.py —— RRF 的加权变体 + 与简单组合的对比

问题诊断：
    当前 RRF(k=60) 等权融合 dense 和 sparse，
    但 dense Recall=0.31、sparse Recall=0.63，**强弱悬殊**，
    等权融合会把 dense 的噪声带进来，导致 hybrid(0.49) < sparse(0.63)。

解决方案（本文件实现并对比）：
    1. 标准 RRF (等权)                      —— baseline
    2. Weighted RRF (按各路的验证集表现加权)
    3. RRF on top-k only (只取各路 top-20 再融合，减少噪声)
    4. CombSUM (分数归一化后相加)
    5. CombMNZ (CombSUM × 命中路数)
    6. 自动权重：用验证集上的 Recall 做权重

用法:
    python rrf_variants.py --tune      # 在验证集上调参
"""
import os
import sys
import json
import argparse
import itertools
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import get_retriever
from rrf import rrf_fuse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
GOLD = os.path.join(DATA, "gold_qa_v3.jsonl")


# ----------------------------------------------------------
# 各种融合策略
# ----------------------------------------------------------

def weighted_rrf(lists, weights, k=60, top_n=None):
    """加权 RRF：score = Σ_i w_i / (k + rank_i)"""
    scores = {}
    for lst, w in zip(lists, weights):
        for rank, doc in enumerate(lst, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_n] if top_n else ranked


def combsum(scores_lists, top_n=None):
    """CombSUM：各分数归一化到[0,1]后相加"""
    norm = []
    for sl in scores_lists:
        if not sl:
            norm.append([])
            continue
        vals = [s for _, s in sl]
        mx = max(vals) if vals else 1.0
        mn = min(vals) if vals else 0.0
        rng = (mx - mn) or 1.0
        norm.append([(d, (s - mn) / rng) for d, s in sl])
    agg = {}
    for sl in norm:
        for d, s in sl:
            agg[d] = agg.get(d, 0.0) + s
    ranked = sorted(agg.items(), key=lambda x: -x[1])
    return ranked[:top_n] if top_n else ranked


def combmnz(scores_lists, top_n=None):
    """CombMNZ：CombSUM × 命中该文档的路数"""
    agg = defaultdict(float)
    hits = defaultdict(int)
    for sl in scores_lists:
        if not sl:
            continue
        vals = [s for _, s in sl]
        mx = max(vals) if vals else 1.0
        mn = min(vals) if vals else 0.0
        rng = (mx - mn) or 1.0
        for d, s in sl:
            agg[d] += (s - mn) / rng
            hits[d] += 1
    for d in agg:
        agg[d] *= hits[d]
    ranked = sorted(agg.items(), key=lambda x: -x[1])
    return ranked[:top_n] if top_n else ranked


# ----------------------------------------------------------
# 评估辅助
# ----------------------------------------------------------

def recall_at_k(ids, gold, k):
    return 1.0 if gold in ids[:k] else 0.0


def mrr(ids, gold):
    for i, x in enumerate(ids, start=1):
        if x == gold:
            return 1.0 / i
    return 0.0


def evaluate(strategy_fn, r, qa, k=5, cand=20):
    """跑一个融合策略，返回 (recall, mrr)"""
    from retriever import Retriever
    recalls, mrrs = [], []

    # 预先拿到两路候选（避免重复检索）
    d_cands, s_cands = [], []
    for x in qa:
        q = x["question"]
        d = r._search_dense(q, cand)
        s = r._search_sparse(q, cand)
        d_cands.append(d)
        s_cands.append(s)

    for i, x in enumerate(qa):
        gold = x["gold_chunk_id"]
        ids, _ = strategy_fn(d_cands[i], s_cands[i], k)
        recalls.append(recall_at_k(ids, gold, k))
        mrrs.append(mrr(ids, gold))
    return sum(recalls) / len(recalls), sum(mrrs) / len(mrrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="large")
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    qa = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()][:args.limit]
    print(f"[tune] {len(qa)} 道题，档位 {args.variant}")

    r = get_retriever(args.variant)
    r._ensure_loaded()

    # 单独看各路的表现
    print("\n" + "=" * 62)
    print("单路基线")
    print("=" * 62)
    for name, fn in [
        ("dense-only", lambda d, s, k: ([x[0] for x in d[:k]], None)),
        ("sparse-only", lambda d, s, k: ([x[0] for x in s[:k]], None)),
    ]:
        rc, m = evaluate(fn, r, qa, 5)
        print(f"  {name:16s} Recall@5={rc:.3f}  MRR={m:.3f}")

    # 各种融合
    print("\n" + "=" * 62)
    print("融合策略对比")
    print("=" * 62)
    results = []

    strategies = {
        "RRF k=60 (等权)": lambda d, s, k: (
            [x for x, _ in rrf_fuse([[c for c, _ in d], [c for c, _ in s]], k=60, top_n=k)], None),
        "RRF k=10": lambda d, s, k: (
            [x for x, _ in rrf_fuse([[c for c, _ in d], [c for c, _ in s]], k=10, top_n=k)], None),
        "RRF top10 only": lambda d, s, k: (
            [x for x, _ in rrf_fuse([[c for c, _ in d[:10]], [c for c, _ in s[:10]]], k=60, top_n=k)], None),
        "RRF 2:1 (sparse重)": lambda d, s, k: (
            [x for x, _ in weighted_rrf([[c for c, _ in d], [c for c, _ in s]], [1, 2], k=60, top_n=k)], None),
        "RRF 1:3": lambda d, s, k: (
            [x for x, _ in weighted_rrf([[c for c, _ in d], [c for c, _ in s]], [1, 3], k=60, top_n=k)], None),
        "RRF 1:5": lambda d, s, k: (
            [x for x, _ in weighted_rrf([[c for c, _ in d], [c for c, _ in s]], [1, 5], k=60, top_n=k)], None),
        "CombSUM": lambda d, s, k: (
            [x for x, _ in combsum([d, s], top_n=k)], None),
        "CombMNZ": lambda d, s, k: (
            [x for x, _ in combmnz([d, s], top_n=k)], None),
        "sparse + dense top3": lambda d, s, k: (
            ([x[0] for x in s[:k]] + [x[0] for x in d[:3]])[:k], None),
    }

    for name, fn in strategies.items():
        rc, m = evaluate(fn, r, qa, 5)
        results.append((name, rc, m))
        print(f"  {name:22s} Recall@5={rc:.3f}  MRR={m:.3f}")

    # 最优
    results.sort(key=lambda x: -x[1])
    print("\n" + "=" * 62)
    print("排名（按 Recall@5）")
    print("=" * 62)
    for i, (n, rc, m) in enumerate(results, 1):
        mark = "  <-- 最佳" if i == 1 else ""
        print(f"  {i}. {n:22s} R={rc:.3f}  MRR={m:.3f}{mark}")

    # 保存
    out = os.path.join(ROOT, "results", "fusion_tuning.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump([{"strategy": n, "recall_at_5": round(rc, 4), "mrr": round(m, 4)}
               for n, rc, m in results], open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
