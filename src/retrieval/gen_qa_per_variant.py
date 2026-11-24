#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_qa_per_variant.py —— 为每一档单独生成评测集（修复评测设计缺陷）

问题诊断：
    gold_qa_v3 是从**全量语料**（large, 5836 chunks）里选的题。
    但 small 档只有 1458 chunks，gold 在子集内的题目只有 25%。
    也就是说：**75% 的题在 small 档里根本不可能命中**，
    导致 small 档的 Recall 被严重低估（0.14 vs 实际的 ~0.6）。

    这是典型的「评测集与检索空间不匹配」的评测设计错误。

修复方案（两种，本脚本实现 B）：
    A. 固定题集，但只在"gold 在该档内"的题目子集上评估（变相提高难度）
    B. 每档单独生成评测集（本方案）—— 从该档自己的 chunk 里采样出题

    用 B 的话，三档的评测集不同，但**题目分布同质**（都从各自语料采样），
    可以公平比较"数据量增加对检索质量的影响"。

    ⚠️ 但要额外报告一件事：三档题集的重叠程度，说明可比性。

用法:
    python gen_qa_per_variant.py
输出:
    data/gold_qa_small.jsonl
    data/gold_qa_medium.jsonl
    data/gold_qa_large.jsonl
"""
import os
import re
import json
import random
import argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
SEED = 42

VARIANTS = {
    "small":  "chunks_20.jsonl",
    "medium": "chunks_50.jsonl",
    "large":  "chunks_100.jsonl",
}


def clean_title(t):
    if not t:
        return ""
    seg = t.split(">")[-1].strip()
    seg = re.sub(r"^\d+(\.\d+)*\.?\s*", "", seg)
    return seg.strip()


def make_questions(chunk):
    title = clean_title(chunk.get("section_path", ""))
    text = chunk["text"]
    qs = []

    if title and 3 <= len(title) <= 60:
        generic = {"index", "contents", "overview", "introduction", "notes",
                   "examples", "references", "see also", "summary"}
        if title.lower() not in generic:
            qs.append((f"What is described in the section \"{title}\"?", "title"))
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", title):
                qs.append((f"What does {title} provide in Python?", "title"))
            if re.match(r"^[A-Z][a-zA-Z0-9]+$", title):
                qs.append((f"What is {title} used for?", "title"))

    code_blocks = re.findall(r"```\n(.*?)```", text, re.S)
    if code_blocks:
        code = code_blocks[0]
        calls = re.findall(r"\b([a-z_][a-z0-9_]{2,}(?:\.[a-z_][a-z0-9_]+)*)\s*\(", code)
        skip = {"print", "len", "range", "str", "int", "type", "open", "isinstance"}
        cand = [c for c in calls if c.split(".")[0] not in skip]
        if cand:
            top = Counter(cand).most_common(1)[0][0]
            if len(top) >= 4:
                qs.append((f"How do I use {top}()? Show an example.", "code"))

    dotted = re.findall(r"\b([a-z_][a-z0-9_]{2,}(?:\.[a-z_][a-z0-9_]+){1,2})\b", text)
    if dotted:
        top = Counter(dotted).most_common(1)[0][0]
        if len(top) >= 7:
            qs.append((f"What does {top} do in Python?", "api"))

    return qs


def score_chunk(chunk, title):
    score = 0.0
    text = chunk["text"]
    if title and 6 <= len(title) <= 55:
        score += 3
    L = len(text)
    if 500 <= L <= 1800:
        score += 2
    elif 300 <= L < 500:
        score += 1
    if "```" in text:
        score += 1.5
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    if len(body) > 300:
        score += 1
    return score


def build_for(tag, fname, n_target, seed):
    chunks = [json.loads(l) for l in open(os.path.join(DATA, fname), encoding="utf-8") if l.strip()]
    scored = []
    for c in chunks:
        t = clean_title(c.get("section_path", ""))
        scored.append((score_chunk(c, t), c, t))
    scored.sort(key=lambda x: -x[0])

    rng = random.Random(seed)
    top_n = max(n_target * 2, int(len(scored) * 0.5))
    pool = scored[:top_n]
    rng.shuffle(pool)

    out, seen = [], set()
    for s, c, title in pool:
        if len(out) >= n_target:
            break
        for q, tag_src in make_questions(c):
            if q.lower() in seen:
                continue
            if tag_src == "title":
                words = [w for w in re.findall(r"[A-Za-z]{3,}", title)
                         if w.lower() not in ("the", "and", "for", "with", "from", "that",
                                              "this", "are", "was")]
                if len(words) < 2:
                    continue
            out.append({
                "question": q,
                "gold_chunk_id": c["chunk_id"],
                "type": "auto",
                "source": tag_src,
                "title": title,
                "url": c["url"],
                "variant": tag,
            })
            seen.add(q.lower())
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="每档多少题")
    args = ap.parse_args()

    results = {}
    for i, (tag, fname) in enumerate(VARIANTS.items()):
        qa = build_for(tag, fname, args.n, SEED + i)
        out = os.path.join(DATA, f"gold_qa_{tag}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for x in qa:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        results[tag] = qa
        print(f"[{tag}] {len(qa)} 题 → {os.path.basename(out)}")

    # 检查重叠
    print("\n[可比性] 三档题目的 gold 重合情况")
    sets = {k: set(x["gold_chunk_id"] for x in v) for k, v in results.items()}
    print(f"  small ∩ large : {len(sets['small'] & sets['large'])} 个 chunk")
    print(f"  medium ∩ large: {len(sets['medium'] & sets['large'])} 个 chunk")

    # 验证：每档的 gold 都在自己的子集里
    print("\n[验证] gold 是否都在本档子集内")
    for tag, fname in VARIANTS.items():
        ids = set(json.loads(l)["chunk_id"] for l in
                  open(os.path.join(DATA, fname), encoding="utf-8"))
        ok = sum(1 for x in results[tag] if x["gold_chunk_id"] in ids)
        pct = ok / len(results[tag]) if results[tag] else 0
        flag = "[OK]" if pct == 1.0 else "[FAIL]"
        print(f"  {tag:8s}: {ok}/{len(results[tag])} = {pct:.0%}  {flag}")

    print("\n[DONE] 三档评测集已生成")


if __name__ == "__main__":
    main()
