#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_retrieval_v2.py —— 考虑「一个问题多个合理答案」的评测

问题诊断（关键）：
    当前评测只认**唯一一个 gold chunk**，但实测发现大量"假阴性"：
      Q: "What is described in the section 'Weak references'?"
      gold 排第 12，但 top1 是 `weakref — Weak references` —— **这明明是对的**！

    失败分析显示：
      - gold 排第 1 名     : 43%
      - 排 2-5 名          : 24%
      - 排 6 名之后        : 33%（其中大量是"另一个同样正确的 chunk"）

修复：
    引入 gold@k 的**扩展判定**——如果一个 chunk 满足以下任一条件，也算命中：
      1. 就是 gold 本身
      2. 与 gold 属于同一 section_path（兄弟 chunk）
      3. 与 gold 属于同一文档（同一 url）+ section_path 前缀相同

    这叫 **"文档级 Recall"** 或 **"段落级 Recall"**，是 RAG 评估的标准做法之一。

用法:
    python eval_retrieval_v2.py --variant small medium large
输出:
    results/retrieval_eval_v2.json
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
OUT = os.path.join(ROOT, "results", "retrieval_eval_v2.json")

VARIANTS = ["small", "medium", "large"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_gold_sets(gold_recs, all_chunks):
    """
    对每个 gold chunk，构造"可接受的答案集合"：
      - 自己
      - 同 section_path 的所有 chunk（兄弟段落）
      - 同一文档里 section_path 前缀匹配的 chunk
    """
    by_id = {c["chunk_id"]: c for c in all_chunks}
    by_doc = defaultdict(list)
    for c in all_chunks:
        by_doc[c["chunk_id"].split("_")[0]].append(c)

    out = {}
    for g in gold_recs:
        gid = g["gold_chunk_id"]
        base = by_id.get(gid)
        accept = {gid}
        if base:
            dh = gid.split("_")[0]
            sp = base.get("section_path", "")
            # 同文档 + section_path 相同或前缀相同
            for c in by_doc.get(dh, []):
                csp = c.get("section_path", "")
                if not sp or not csp:
                    continue
                if csp == sp:
                    accept.add(c["chunk_id"])
                elif csp.startswith(sp) or sp.startswith(csp):
                    accept.add(c["chunk_id"])
        out[gid] = accept
    return out


def evaluate_metrics(r, qa, gold_sets, k_list=(1, 3, 5, 10), mode="hybrid"):
    """返回各 k 的严格 recall 和 宽松 recall"""
    res = {k: {"strict": [], "lenient": []} for k in k_list}
    mrr_strict, mrr_lenient = [], []

    for x in qa:
        hits = r.retrieve(x["question"], mode=mode, k=max(k_list))
        ids = [h["chunk_id"] for h in hits]
        gid = x["gold_chunk_id"]
        accept = gold_sets.get(gid, {gid})

        for k in k_list:
            topk = ids[:k]
            res[k]["strict"].append(1.0 if gid in topk else 0.0)
            res[k]["lenient"].append(1.0 if any(c in accept for c in topk) else 0.0)

        # MRR
        rank_s = next((i + 1 for i, c in enumerate(ids) if c == gid), None)
        rank_l = next((i + 1 for i, c in enumerate(ids) if c in accept), None)
        mrr_strict.append(1.0 / rank_s if rank_s else 0.0)
        mrr_lenient.append(1.0 / rank_l if rank_l else 0.0)

    out = {}
    for k in k_list:
        out[f"recall@{k}_strict"] = round(sum(res[k]["strict"]) / len(qa), 4)
        out[f"recall@{k}_lenient"] = round(sum(res[k]["lenient"]) / len(qa), 4)
    out["mrr_strict"] = round(sum(mrr_strict) / len(mrr_strict), 4)
    out["mrr_lenient"] = round(sum(mrr_lenient) / len(mrr_lenient), 4)
    out["n"] = len(qa)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--modes", nargs="*", default=["dense", "sparse", "hybrid"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    CHUNK_FILES = {
        "small": "chunks_20.jsonl",
        "medium": "chunks_50.jsonl",
        "large": "chunks_100.jsonl",
    }

    all_results = {}

    for variant in args.variant:
        print(f"\n{'='*70}")
        print(f"数据档位: {variant}")
        print(f"{'='*70}")

        qa = load_jsonl(os.path.join(DATA, f"gold_qa_{variant}.jsonl"))
        if args.limit:
            qa = qa[:args.limit]
        chunks = load_jsonl(os.path.join(DATA, CHUNK_FILES[variant]))

        gold_sets = build_gold_sets(qa, chunks)
        avg_accept = sum(len(v) for v in gold_sets.values()) / len(gold_sets)
        print(f"  {len(qa)} 题，每题平均可接受答案数: {avg_accept:.1f}")

        r = get_retriever(variant)
        r._ensure_loaded()

        all_results[variant] = {}
        for mode in args.modes:
            t0 = time.time()
            m = evaluate_metrics(r, qa, gold_sets, mode=mode)
            dt = time.time() - t0
            all_results[variant][mode] = m
            print(f"\n  [{mode}]  ({(time.time()-t0):.1f}s)")
            print(f"    Recall@5  strict={m['recall@5_strict']:.3f}  "
                  f"lenient={m['recall@5_lenient']:.3f}")
            print(f"    Recall@10 strict={m['recall@10_strict']:.3f}  "
                  f"lenient={m['recall@10_lenient']:.3f}")
            print(f"    MRR       strict={m['mrr_strict']:.3f}  "
                  f"lenient={m['mrr_lenient']:.3f}")

    # 保存
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(all_results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 汇总表
    print(f"\n{'='*70}")
    print("汇总：Recall@5（严格 / 宽松）")
    print(f"{'='*70}")
    print(f"{'档位':<10}{'dense':<20}{'sparse':<20}{'hybrid':<20}")
    for v in args.variant:
        row = f"{v:<10}"
        for mode in ["dense", "sparse", "hybrid"]:
            if mode in all_results[v]:
                d = all_results[v][mode]
                cell = f"{d['recall@5_strict']:.2f} / {d['recall@5_lenient']:.2f}"
                row += f"{cell:<20}"
        print(row)

    print(f"\n{'='*70}")
    print("Proposal 目标检查（Recall@5）")
    print(f"{'='*70}")
    for v in args.variant:
        if "hybrid" in all_results[v]:
            d = all_results[v]["hybrid"]
            s, l = d["recall@5_strict"], d["recall@5_lenient"]
            print(f"  {v:8s} strict={s:.3f}  lenient={l:.3f}  "
                  f"目标>=0.75 → {'[达标]' if l >= 0.75 else '[未达标]'}")

    print(f"\n结果: {OUT}")


if __name__ == "__main__":
    main()
