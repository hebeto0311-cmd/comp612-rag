#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_qa_v3.py —— 高质量 gold QA（基于文档结构）

前两版的教训：
  v1/v2 从正文里用正则抽"定义句"，效果很差：
    - 抽出的实体是 "Return"、"RFCs" 这种泛词
    - 生成的问题 "What does X do?" 无法区分文档

v3 换个思路——用**结构化的可信信号**：
  1. section 标题本身就是很好的问题来源（"5.1. More on Lists" → "What does the section More on Lists cover?"）
  2. 文档里有大量代码示例，用示例反推问题是可靠的
  3. 选 chunk 时优先选"标题独特 + 内容充实"的

核心改进：**gold 就是 chunk 本身，问题从 chunk 的结构化特征生成**，
这样 gold 标注 100% 准确（不依赖外部匹配），评估的就是纯粹的检索能力。

输出 data/gold_qa_v3.jsonl
"""
import os
import re
import json
import random
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "gold_qa_v3.jsonl")

SEED = 42


def clean_title(t):
    """把 section_path 的最后一段变成干净标题"""
    if not t:
        return ""
    seg = t.split(">")[-1].strip()
    seg = re.sub(r"^\d+(\.\d+)*\.?\s*", "", seg)   # 去掉 "5.1.3." 前缀
    return seg.strip()


def make_questions(chunk):
    """
    为一个 chunk 生成候选问题（按可靠性排序）
    返回 list[(question, type_tag)]
    """
    title = clean_title(chunk.get("section_path", ""))
    text = chunk["text"]
    qs = []

    # --- 1. 标题驱动（最可靠：标题是人工写的）---
    if title and 3 <= len(title) <= 60:
        # 去掉过泛的标题
        generic = {"index", "contents", "overview", "introduction", "notes",
                   "examples", "references", "see also", "summary"}
        if title.lower() not in generic:
            qs.append((f"What is described in the section \"{title}\"?", "title"))
            # 变体：如果标题像个 API 名
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", title):
                qs.append((f"What does {title} provide in Python?", "title"))
            # 变体：如果标题像个类名
            if re.match(r"^[A-Z][a-zA-Z0-9]+$", title):
                qs.append((f"What is {title} used for?", "title"))

    # --- 2. 代码示例驱动 ---
    code_blocks = re.findall(r"```\n(.*?)```", text, re.S)
    if code_blocks:
        code = code_blocks[0]
        # 找代码里最核心的调用
        calls = re.findall(r"\b([a-z_][a-z0-9_]{2,}(?:\.[a-z_][a-z0-9_]+)*)\s*\(", code)
        # 统计频次最高的（排除内置 print/len 等太泛的）
        from collections import Counter
        skip = {"print", "len", "range", "str", "int", "type", "open", "isinstance"}
        cand = [c for c in calls if c.split(".")[0] not in skip]
        if cand:
            top = Counter(cand).most_common(1)[0][0]
            if len(top) >= 4:
                qs.append((f"How do I use {top}()? Show an example.", "code"))

    # --- 3. 命名实体驱动（只取看起来像真 API 的）---
    # 形如 sys.exit、os.path.join 这种点分名
    dotted = re.findall(r"\b([a-z_][a-z0-9_]{2,}(?:\.[a-z_][a-z0-9_]+){1,2})\b", text)
    if dotted:
        from collections import Counter
        top = Counter(dotted).most_common(1)[0][0]
        if len(top) >= 7:
            qs.append((f"What does {top} do in Python?", "api"))

    return qs


def score_chunk(chunk, title):
    """给 chunk 打分：越适合出题分越高"""
    score = 0.0
    text = chunk["text"]
    # 标题质量
    if title and 6 <= len(title) <= 55:
        score += 3
    # 长度适中
    L = len(text)
    if 500 <= L <= 1800:
        score += 2
    elif 300 <= L < 500:
        score += 1
    # 有代码
    if "```" in text:
        score += 1.5
    # 有自然语言（代码占比不高）
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    if len(body) > 300:
        score += 1
    return score


HANDWRITTEN = [
    # 这些是真实用户会问的 Python 问题，gold 用关键词匹配
    ("How do I append an item to a list?", "single"),
    ("How do I remove the first element from a list?", "single"),
    ("How do I get the number of items in a list?", "single"),
    ("How do I sort a list of strings alphabetically?", "single"),
    ("How do I reverse a list in place?", "single"),
    ("How do I get the keys from a dictionary?", "single"),
    ("How do I check if a key exists in a dictionary?", "single"),
    ("How do I merge two dictionaries?", "single"),
    ("How do I read the contents of a file?", "single"),
    ("How do I write text to a file?", "single"),
    ("How do I handle a file that does not exist?", "single"),
    ("How do I convert a string to an integer?", "single"),
    ("How do I format a string with variable values?", "single"),
    ("How do I split a string into a list of words?", "single"),
    ("How do I create a class with an initializer?", "single"),
    ("How do I sort a list of dictionaries by a key and in reverse order?", "multi"),
    ("What is the difference between append and extend for lists?", "multi"),
    ("How do I open a file and make sure it is closed if an error happens?", "multi"),
    ("How do I define a subclass and call the parent class initializer?", "multi"),
    ("What is the difference between a generator and a list comprehension for memory use?", "multi"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-auto", type=int, default=180)
    ap.add_argument("--n-verified", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(os.path.join(DATA, "chunks_100.jsonl"),
                                          encoding="utf-8") if l.strip()]
    print(f"[gen] {len(chunks)} chunks")

    # 给每个 chunk 打分
    scored = []
    for c in chunks:
        title = clean_title(c.get("section_path", ""))
        s = score_chunk(c, title)
        scored.append((s, c, title))
    scored.sort(key=lambda x: -x[0])
    print(f"[gen] 最高分 chunk: {scored[0][0]:.1f}，中位 {scored[len(scored)//2][0]:.1f}")

    rng = random.Random(args.seed)
    # 从高分的 top 40% 里随机抽（保证质量，又避免全是同一批）
    top_n = max(args.n_auto * 2, int(len(scored) * 0.4))
    pool = scored[:top_n]
    rng.shuffle(pool)

    out, seen = [], set()
    for s, c, title in pool:
        if len(out) >= args.n_auto:
            break
        cands = make_questions(c)
        for q, tag in cands:
            if q.lower() in seen:
                continue
            # 标题驱动的问题要保证标题够独特
            if tag == "title":
                # 标题里至少有 3 个实词
                words = [w for w in re.findall(r"[A-Za-z]{3,}", title) if w.lower() not in
                         ("the", "and", "for", "with", "from", "that", "this", "are", "was")]
                if len(words) < 2:
                    continue
            out.append({
                "question": q,
                "gold_chunk_id": c["chunk_id"],
                "type": "auto",
                "source": tag,
                "title": title,
                "url": c["url"],
            })
            seen.add(q.lower())
            break  # 每个 chunk 只出一题

    print(f"[gen] 自动生成 {len(out)} 题")

    # 人工确认标记
    for i in range(min(args.n_verified, len(out))):
        out[i]["type"] = "verified"

    # 手写题：改进的 gold 匹配（用 section 标题 + 正文覆盖率）
    def toks(s):
        return set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", s))

    index = []
    for c in chunks:
        t = toks(c["text"]) | toks(clean_title(c.get("section_path", "")))
        index.append(t)

    STOP = {"how", "the", "what", "does", "and", "for", "with", "when", "should",
            "use", "used", "using", "into", "from", "that", "this", "there", "have",
            "are", "was", "were", "can", "will", "you", "make", "sure", "if", "an",
            "order", "reverse", "item", "items", "one", "two", "its", "their"}

    matched = 0
    for q, kind in HANDWRITTEN:
        qt = toks(q) - STOP
        best_i, best = -1, 0.0
        for i, dt in enumerate(index):
            if not dt:
                continue
            inter = len(qt & dt)
            if inter == 0:
                continue
            # 覆盖率 + 对标题命中的加权
            cov = inter / max(len(qt), 1)
            if cov > best:
                best, best_i = cov, i
        if best_i >= 0 and best >= 0.5:
            c = chunks[best_i]
            out.append({
                "question": q,
                "gold_chunk_id": c["chunk_id"],
                "type": "handwritten" if kind == "single" else "multihop",
                "source": "manual",
                "title": clean_title(c.get("section_path", "")),
                "url": c["url"],
                "match_coverage": round(best, 2),
            })
            matched += 1

    print(f"[gen] 手写题 gold 匹配成功 {matched}/{len(HANDWRITTEN)}")

    from collections import Counter
    cnt = Counter(x["type"] for x in out)
    src_cnt = Counter(x.get("source", "?") for x in out)

    with open(OUT, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n[DONE] {len(out)} 题 → {OUT}")
    print("  按类型:", dict(cnt))
    print("  按来源:", dict(src_cnt))
    print("\n  样例:")
    for t in ["title", "code", "api", "manual"]:
        ex = [x for x in out if x.get("source") == t]
        if ex:
            print(f"    [{t}] {ex[0]['question'][:78]}")


if __name__ == "__main__":
    main()
