#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_qa_v2.py —— 改进版 gold QA 生成

相比 v1 的改进：
  1. 更严格地筛"可出题"的 chunk（必须有明确的可命名实体，如函数名、类名、方法名）
  2. 问题模板更具体（"What does X return?" 而非 "What does the doc say about X"）
  3. 优先用文档里的代码示例构造问题（"How do I ... in Python?"）
  4. 过滤掉纯代码块、纯列表到底的 chunk

输出 data/gold_qa_v2.jsonl，并把 v1 留作对比
"""
import os
import re
import json
import random
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CHUNKS = os.path.join(DATA, "chunks.jsonl")
OUT = os.path.join(DATA, "gold_qa_v2.jsonl")

SEED = 42


# ------------------------------------------------------------------
# 实体抽取：从 chunk 里找可命名的东西
# ------------------------------------------------------------------

# 函数/方法定义：`list.append(x)` 或 `os.path.join(a, b)`
FUNC_CALL = re.compile(r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)\s*\(")
# 类/异常名：`ValueError` `class Foo`
CLASS_NAME = re.compile(r"\b((?:class|exception)\s+)?([A-Z][a-zA-Z0-9]{2,})\b")
# "The xxx() method does Y"
METHOD_DEF = re.compile(
    r"\b([a-z_][a-z0-9_]*)\(\)\s*(?:method|function)?\s*"
    r"(?:returns?|does|will|is used to|adds?|removes?|returns?)\s+"
    r"([^\.\n]{15,120})", re.I)
# "X is a ..." / "X is used to ..."
IS_A = re.compile(r"^([A-Za-z][A-Za-z0-9_.]{2,40})\s+(?:is|are)\s+"
                  r"(?:a|an|the|used to|called)\s+([^\.\n]{15,140})")


def strip_code_fence(t):
    return re.sub(r"```.*?```", " ", t, flags=re.S)


def is_content_rich(chunk):
    """判断 chunk 是否适合出题"""
    t = chunk["text"]
    body = strip_code_fence(t)
    # 去掉代码后还剩多少正文
    if len(body.strip()) < 200:
        return False
    # 不能是纯列表
    lines = [l for l in body.split("\n") if l.strip()]
    if lines:
        bullet = sum(1 for l in lines if l.strip().startswith(("-", "*", "|")))
        if bullet / len(lines) > 0.6:
            return False
    # 必须有一些自然语言句子（有多个空格分隔的词）
    words = len(re.findall(r"\b[a-z]{3,}\b", body))
    if words < 40:
        return False
    return True


def make_question_v2(chunk):
    """
    按优先级尝试多种模板，返回 (question, evidence) 或 None
    """
    text = chunk["text"]
    body = re.sub(r"^\[[^\]]+\]\s*", "", strip_code_fence(text))
    body = re.sub(r"\s+", " ", body).strip()

    # --- 模板 A：方法定义 ---
    for m in METHOD_DEF.finditer(body):
        name, act = m.group(1), m.group(2).strip()
        if len(name) < 2 or name in ("the", "this", "that", "it"):
            continue
        if len(act) < 12:
            continue
        q = f"What does the {name}() method do in Python?"
        return q, act[:150]

    # --- 模板 B："X is a ..." 定义 ---
    for sent in re.split(r"(?<=[\.\n])", body):
        sent = sent.strip()
        if len(sent) < 30:
            continue
        m = IS_A.match(sent)
        if m:
            name, desc = m.group(1), m.group(2).strip()
            if name.lower() in ("this", "that", "it", "there", "these"):
                continue
            if len(name) > 35 or len(desc) < 12:
                continue
            q = f"What is {name} in Python?"
            return q, sent[:150]

    # --- 模板 C：有代码示例 → "How do I ..." ---
    code_blocks = re.findall(r"```\n?(.*?)```", text, re.S)
    if code_blocks:
        code = code_blocks[0]
        # 找代码里用到的核心函数
        calls = []
        for c in FUNC_CALL.finditer(code):
            fn = c.group(1)
            if len(fn) >= 4 and not fn.startswith(("sys.", "os.environ")):
                calls.append(fn)
        if calls:
            # 取最"有名"的那个
            fn = max(set(calls), key=calls.count)
            q = f"How do I use {fn}? Show me an example."
            return q, code[:120].replace("\n", " ")

    # --- 模板 D：异常/类相关 ---
    for m in CLASS_NAME.finditer(body):
        name = m.group(2)
        if len(name) < 4 or name in ("Python", "This", "The", "Note", "When"):
            continue
        # 找包含这个类名的句子
        for sent in re.split(r"(?<=[\.\n])", body):
            if name in sent and len(sent) > 40:
                q = f"When is {name} raised or used in Python?"
                return q, sent.strip()[:150]

    return None, None


HANDWRITTEN = [
    ("How do I append an item to a list?", "single"),
    ("How do I remove an item from a list by value?", "single"),
    ("What is the difference between a list and a tuple?", "single"),
    ("How do I get a value from a dictionary without an error if the key is missing?", "single"),
    ("How do I format a number to two decimal places?", "single"),
    ("How do I write a decorator?", "single"),
    ("How do I read a file line by line?", "single"),
    ("What does the with statement do?", "single"),
    ("How do I define a class and create an instance?", "single"),
    ("How do I catch and handle an exception?", "single"),
    ("How do I sort a list of dictionaries by one of their keys?", "single"),
    ("How do I merge two dictionaries?", "single"),
    ("What is a generator expression and when should I use it?", "single"),
    ("How do I join a list of strings into one string?", "single"),
    ("How do I check whether a substring appears in a string?", "single"),
    ("How do I sort a list in reverse order?", "multi"),
    ("What is the difference between append() and extend(), and when to use each?", "multi"),
    ("How do I open a file safely and make sure it is closed after an error?", "multi"),
    ("How do I inherit from a class and call the parent constructor?", "multi"),
    ("What is the difference between a generator and a list comprehension in memory usage?", "multi"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-auto", type=int, default=200)
    ap.add_argument("--n-verified", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    print(f"[gen] 加载 {len(recs)} chunk")

    pool = [r for r in recs if is_content_rich(r)]
    print(f"[gen] 内容充足的 chunk: {len(pool)}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    out, seen = [], set()
    for r in pool:
        if len(out) >= args.n_auto:
            break
        q, ev = make_question_v2(r)
        if not q or q.lower() in seen:
            continue
        out.append({
            "question": q,
            "gold_chunk_id": r["chunk_id"],
            "type": "auto",
            "evidence": ev,
            "url": r["url"],
        })
        seen.add(q.lower())

    print(f"[gen] 自动生成 {len(out)} 题")

    # 标为人工确认（真实项目里这 20 题要人来核）
    for i in range(min(args.n_verified, len(out))):
        out[i]["type"] = "verified"

    # 手写题的 gold：用关键词匹配找最相关 chunk
    def toks(s):
        return set(re.findall(r"[a-z]{3,}", s.lower()))

    doc_toks = []
    for r in recs:
        doc_toks.append(toks(r["text"]))

    for q, kind in HANDWRITTEN:
        qt = toks(q)
        # 去掉无意义词
        qt -= {"how", "the", "what", "does", "and", "for", "with", "when",
               "should", "use", "used", "using", "into", "from", "that", "this",
               "there", "have", "are", "was", "were", "can", "will", "you"}
        best_i, best = -1, 0.0
        for i, dt in enumerate(doc_toks):
            if not dt:
                continue
            inter = len(qt & dt)
            if inter == 0:
                continue
            # 用覆盖率而非 Jaccard（问题短、文档长）
            score = inter / max(len(qt), 1)
            if score > best:
                best, best_i = score, i
        if best_i >= 0 and best >= 0.4:
            r = recs[best_i]
            out.append({
                "question": q,
                "gold_chunk_id": r["chunk_id"],
                "type": "handwritten" if kind == "single" else "multihop",
                "evidence": r["text"][:150],
                "url": r["url"],
                "match_coverage": round(best, 2),
            })

    from collections import Counter
    cnt = Counter(x["type"] for x in out)
    with open(OUT, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n[DONE] {len(out)} 题 → {OUT}")
    for k, v in cnt.items():
        print(f"  {k:12s}: {v}")

    print("\n  样例（按类型）：")
    shown = defaultdict(int)
    for x in out:
        t = x["type"]
        if shown[t] < 2:
            print(f"    [{t}] {x['question'][:80]}")
            print(f"        evidence: {x.get('evidence','')[:80]}")
            shown[t] += 1


if __name__ == "__main__":
    main()
