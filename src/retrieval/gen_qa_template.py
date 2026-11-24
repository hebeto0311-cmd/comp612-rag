#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_qa_template.py —— 生成评测用的 gold QA 集（A/B 协作产出，交给 D 用）

为什么要 A/B 来生成：
    Proposal 写的是 "an LLM sees one sampled chunk and writes a question it answers,
    and the chunk ID becomes the gold passage"。这一步只需要 chunk，不需要索引，
    所以可以在 C/D 开工前就准备好。

本脚本做的是「规则版」：从 chunk 里抽取有代表性的片段，生成问题模板。
真实生产时应该接 LLM；这里提供两种模式：
    --mode rule   规则生成（无需 API，立刻可用）
    --mode llm    调用 LLM（需要 API key，效果更好）

输出：data/gold_qa.jsonl
    {"question": str, "gold_chunk_id": str, "type": "auto"|"handwritten"|"multihop",
     "gold_text": str, "url": str}
"""
import os
import re
import json
import random
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CHUNKS = os.path.join(DATA, "chunks.jsonl")
OUT = os.path.join(DATA, "gold_qa.jsonl")

SEED = 42


# ------------------------------------------------------------------
# 规则版：从 chunk 里找"定义句 / 方法句 / 示例句"，反向构造问题
# ------------------------------------------------------------------

# 形如 "The xxx() method does ..." / "xxx is a ..."
DEF_PATTERNS = [
    (re.compile(r"^([a-z_][a-z0-9_.]*)\s+(?:is|are)\s+(?:a|an|the)\s+(.{20,140}?)[\.\n]"),
     "What is {term}?"),
    (re.compile(r"^([A-Z][a-zA-Z0-9_.]+)\s+(?:is|are)\s+(?:a|an|the)\s+(.{20,140}?)[\.\n]"),
     "What is {term}?"),
    (re.compile(r"[Tt]he\s+([a-z_][a-z0-9_.]*)\s*\(\s*\)\s*(?:method|function)\s+(.{20,140}?)[\.\n]"),
     "What does {term}() do?"),
    (re.compile(r"[Tt]he\s+([a-z_][a-z0-9_.]*)method\s+(.{20,140}?)[\.\n]"),
     "What does the {term} method do?"),
]

# 常见 Python 概念词（用于构造更自然的问题）
KEYWORD_MAP = {
    "list": "a list", "tuple": "a tuple", "dict": "a dictionary",
    "set": "a set", "str": "a string", "int": "an integer",
    "float": "a float", "bool": "a boolean", "bytes": "a bytes object",
}


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def make_question(chunk_text, section_title):
    """用规则从 chunk 构造一个可回答的问题"""
    body = chunk_text
    # 去掉开头的 section 标题标记
    body = re.sub(r"^\[[^\]]+\]\s*", "", body)
    text = clean(body)

    for pat, tmpl in DEF_PATTERNS:
        m = pat.search(text)
        if m:
            term = m.group(1).strip()
            if 2 <= len(term) <= 40 and not term.isdigit():
                return tmpl.format(term=term), m.group(2)

    # 退路：用 section 标题构造
    t = clean(re.sub(r"[#\d.]+\s*", "", section_title or ""))
    if t and len(t) > 4:
        return f"What does the documentation say about {t}?", text[:120]

    return None, None


# ------------------------------------------------------------------
# 手写题（单跳 / 多跳）—— 这些是固定题，不依赖 chunk 内容
# ------------------------------------------------------------------

HANDWRITTEN = [
    ("What is the difference between a list and a tuple in Python?", "single"),
    ("How do I remove an element from a list?", "single"),
    ("What does the dict.get() method return when the key is missing?", "single"),
    ("How do I format a float to two decimal places?", "single"),
    ("What is a decorator in Python?", "single"),
    ("How does the with statement work?", "single"),
    ("What is the difference between == and is?", "single"),
    ("How do I read a large file line by line?", "single"),
    ("What are f-strings and how do I use them?", "single"),
    ("How does exception handling with try/except work?", "single"),
    # 多跳：需要跨 chunk 或多个概念
    ("How do I sort a list of dictionaries by a key, and how do I sort it in reverse?", "multi"),
    ("What is the difference between the append() and extend() methods, and when should each be used?", "multi"),
    ("How do I open a file, read its contents, and ensure it gets closed properly?", "multi"),
    ("How do I inherit from a base class and call its parent constructor?", "multi"),
    ("What is the difference between a generator and a list comprehension, and how does each affect memory usage?", "multi"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-auto", type=int, default=200, help="自动生成多少题")
    ap.add_argument("--n-verified", type=int, default=20, help="其中多少题标为人工确认（模拟）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    print(f"[gen_qa] 加载 {len(recs)} 个 chunk")

    rng = random.Random(args.seed)
    # 优先从"内容扎实"的 chunk 出题（长度 400~1800 字符）
    pool = [r for r in recs if 400 <= len(r["text"]) <= 1800]
    print(f"[gen_qa] 可出题 pool: {len(pool)} 个 chunk")
    rng.shuffle(pool)

    out = []
    seen_q = set()

    # 1) 自动生成
    for r in pool:
        if len([x for x in out if x["type"] == "auto"]) >= args.n_auto:
            break
        q, _ = make_question(r["text"], r.get("section_path", ""))
        if not q or q.lower() in seen_q:
            continue
        # 过滤掉太抽象/太短的题
        if len(q) < 15:
            continue
        seen_q.add(q.lower())
        out.append({
            "question": q,
            "gold_chunk_id": r["chunk_id"],
            "type": "auto",
            "gold_text": r["text"][:300],
            "url": r["url"],
        })

    print(f"[gen_qa] 自动生成 {len([x for x in out if x['type']=='auto'])} 题")

    # 2) 标记 20 道为"人工确认"
    for i in range(min(args.n_verified, len(out))):
        out[i]["type"] = "verified"

    # 3) 手写题：检索匹配到最相关的 chunk 作为 gold
    #    这里用简单的关键词匹配，因为还没有索引
    from collections import Counter

    def toks(s):
        return set(re.findall(r"[a-z_][a-z0-9_]*", s.lower()))

    for q, kind in HANDWRITTEN:
        qt = toks(q)
        best_i, best_score = -1, 0
        for i, r in enumerate(recs):
            dt = toks(r["text"])
            if not dt:
                continue
            # Jaccard 相似度
            inter = len(qt & dt)
            if inter == 0:
                continue
            score = inter / (len(qt) ** 0.5)
            if score > best_score:
                best_score, best_i = score, i
        if best_i >= 0 and best_score > 0.5:
            r = recs[best_i]
            out.append({
                "question": q,
                "gold_chunk_id": r["chunk_id"],
                "type": "handwritten" if kind == "single" else "multihop",
                "gold_text": r["text"][:300],
                "url": r["url"],
                "match_score": round(best_score, 2),
            })

    # 统计
    from collections import Counter as C
    cnt = C(x["type"] for x in out)
    with open(OUT, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n[DONE] {len(out)} 题 → {OUT}")
    for k, v in cnt.items():
        print(f"  {k:12s}: {v}")
    print(f"\n  示例:")
    for x in out[:3]:
        print(f"    Q: {x['question'][:70]}")
        print(f"       gold={x['gold_chunk_id']}  type={x['type']}")


if __name__ == "__main__":
    main()
