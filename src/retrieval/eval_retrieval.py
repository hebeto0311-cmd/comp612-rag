#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_retrieval.py —— 检索效果评估（Recall@K / MRR）

这是 D 的评估框架的「检索部分」原型，B 先做出来验证检索质量。
D 可以直接复用或扩展。

用法:
    python eval_retrieval.py                    # 默认 large 档、全部题目
    python eval_retrieval.py --variant small medium large
    python eval_retrieval.py --limit 50         # 只跑前 50 题（快）
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import get_retriever

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
GOLD = os.path.join(DATA, "gold_qa_v3.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "retrieval_eval.json")


def recall_at_k(retrieved, gold, k):
    return 1.0 if gold in retrieved[:k] else 0.0


def mrr(retrieved, gold):
    for i, cid in enumerate(retrieved, start=1):
        if cid == gold:
            return 1.0 / i
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=["large"],
                    choices=["small", "medium", "large"])
    ap.add_argument("--modes", nargs="*", default=["dense", "sparse", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(GOLD):
        print(f"gold_qa.jsonl 不存在，先跑: python gen_qa_template.py")
        return

    # 每档用各自的评测集（修复评测集与检索空间不匹配的问题）
    def load_gold(variant):
        p = os.path.join(DATA, f"gold_qa_{variant}.jsonl")
        if os.path.exists(p):
            return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]

    gold_default = load_gold(args.variant[0])
    if args.limit:
        gold_default = gold_default[:args.limit]
    print(f"[eval] {len(gold_default)} 道题（每档用各自评测集）")


    results = {}

    for variant in args.variant:
        gold = load_gold(variant)
        if args.limit:
            gold = gold[:args.limit]
        print(f"\n{'='*66}")
        print(f"数据档位: {variant}  ({len(gold)} 题)")
        print(f"{'='*66}")
        r = get_retriever(variant)
        r._ensure_loaded()

        results[variant] = {}
        for mode in args.modes:
            t0 = time.time()
            recalls, mrrs = [], []
            type_stats = defaultdict(lambda: {"n": 0, "recall": 0.0, "mrr": 0.0})
            n_empty = 0

            for x in gold:
                hits = r.retrieve(x["question"], mode=mode, k=args.k)
                ids = [h["chunk_id"] for h in hits]
                if not ids:
                    n_empty += 1

                rc = recall_at_k(ids, x["gold_chunk_id"], args.k)
                m = mrr(ids, x["gold_chunk_id"])
                recalls.append(rc)
                mrrs.append(m)

                ts = type_stats[x["type"]]
                ts["n"] += 1
                ts["recall"] += rc
                ts["mrr"] += m

            elapsed = time.time() - t0
            avg_r = sum(recalls) / len(recalls) if recalls else 0
            avg_m = sum(mrrs) / len(mrrs) if mrrs else 0

            results[variant][mode] = {
                "recall_at_k": round(avg_r, 4),
                "mrr": round(avg_m, 4),
                "n": len(gold),
                "empty_results": n_empty,
                "elapsed_s": round(elapsed, 2),
                "per_type": {
                    k: {"n": v["n"],
                        "recall": round(v["recall"] / v["n"], 4) if v["n"] else 0,
                        "mrr": round(v["mrr"] / v["n"], 4) if v["n"] else 0}
                    for k, v in type_stats.items()
                },
            }

            print(f"\n  [{mode:6s}]  Recall@{args.k}={avg_r:.3f}  MRR={avg_m:.3f}  "
                  f"空结果={n_empty}  用时={elapsed:.1f}s")
            for t, s in sorted(results[variant][mode]["per_type"].items()):
                print(f"            {t:12s} n={s['n']:3d}  R={s['recall']:.3f}  MRR={s['mrr']:.3f}")

    # 保存
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 汇总表
    print(f"\n{'='*66}")
    print("汇总")
    print(f"{'='*66}")
    print(f"{'档位':<10}{'模式':<10}{'Recall@'+str(args.k):<12}{'MRR':<10}")
    for v in args.variant:
        for m in args.modes:
            d = results[v][m]
            print(f"{v:<10}{m:<10}{d['recall_at_k']:<12.3f}{d['mrr']:<10.3f}")

    print(f"\n结果已保存: {OUT_JSON}")

    # 达标判断
    print(f"\n{'='*66}")
    print("Proposal 目标检查")
    print(f"{'='*66}")
    for v in args.variant:
        if "hybrid" in results[v]:
            r_ = results[v]["hybrid"]["recall_at_k"]
            ok = "[OK]" if r_ >= 0.75 else "[未达标]"
            print(f"  {v:10s} hybrid Recall@{args.k}={r_:.3f}  目标>=0.75  {ok}")


if __name__ == "__main__":
    main()
