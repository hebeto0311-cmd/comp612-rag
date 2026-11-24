#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_subset.py —— 构建三档数据量的子集索引（对齐 Proposal 的 20%/50%/100% 文档采样）

为什么要按"文档"采样而不是"chunk"：
    Proposal 原文是 "sampled at about 20%, 50% and 100% of documents from the same
    corpus: they are increasing samples of one corpus, not three datasets."
    所以必须按文档采样，保持 chunk 内部结构完整。

产出:
    data/chunks_20.jsonl    (~767 chunks)
    data/chunks_50.jsonl    (~2,918 chunks)
    data/chunks_100.jsonl   (= data/chunks.jsonl, 5836 chunks)

用法:
    python build_subset.py              # 生成全部三档
    python build_subset.py --build-all  # 生成并建索引
"""
import os
import sys
import json
import random
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CHUNKS = os.path.join(DATA, "chunks.jsonl")

RATIOS = [("20", 0.20), ("50", 0.50), ("100", 1.00)]
SEED = 42  # 固定种子，保证可复现


def group_by_doc(recs):
    """按文档分组（chunk_id 的前 8 位是文档 hash）"""
    groups = {}
    for r in recs:
        dh = r["chunk_id"].split("_")[0]
        groups.setdefault(dh, []).append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    groups = group_by_doc(recs)
    docs = sorted(groups.keys())
    print(f"[subset] 总 chunk {len(recs)}，来自 {len(docs)} 篇文档")

    rng = random.Random(args.seed)
    # 先打乱文档顺序，然后按比例取前缀（保证嵌套关系：20% ⊂ 50% ⊂ 100%）
    shuffled = docs[:]
    rng.shuffle(shuffled)

    summary = []
    for tag, ratio in RATIOS:
        n_docs = max(1, int(len(docs) * ratio))
        selected = set(shuffled[:n_docs])

        subset = []
        for dh in selected:
            subset.extend(groups[dh])
        # 保持稳定顺序
        subset.sort(key=lambda r: r["chunk_id"])

        out = os.path.join(DATA, f"chunks_{tag}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary.append((tag, n_docs, len(subset), out))
        print(f"  [{tag:>3}%] {n_docs:4d} 篇文档 → {len(subset):5d} chunks  →  {os.path.basename(out)}")

    # 验证嵌套关系
    print("\n[验证] 子集嵌套关系（20% ⊂ 50% ⊂ 100%）")
    ids = {}
    for tag, _ in RATIOS:
        ids[tag] = set(json.loads(l)["chunk_id"] for l in
                       open(os.path.join(DATA, f"chunks_{tag}.jsonl"), encoding="utf-8"))
    assert ids["20"] <= ids["50"], "20% 必须是 50% 的子集"
    assert ids["50"] <= ids["100"], "50% 必须是 100% 的子集"
    print(f"  20% ⊂ 50% ⊂ 100%  [OK]")
    print(f"  20% 占比: {len(ids['20'])/len(ids['100']):.1%}")
    print(f"  50% 占比: {len(ids['50'])/len(ids['100']):.1%}")

    print("\n[DONE] 三档子集已生成")
    print("       下一步: 为每档建索引 → python build_index_variants.py")


if __name__ == "__main__":
    main()
