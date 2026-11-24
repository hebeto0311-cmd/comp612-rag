#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chunk.py —— 按 section 边界递归分块

策略（对齐 Proposal）:
  1. 按 Markdown 标题层级切 section
  2. 每个 section 若超过 512 token，用递归切分（段落 → 句子）
  3. 相邻 chunk 之间保留 10% overlap

输出契约: data/chunks.jsonl
  {"chunk_id": "a1b2c3d4_0007", "text": "...", "url": "...",
   "section_path": "3.2 > Data Types", "version": "3.12"}

用法:
    python chunk.py                  # 处理全部
    python chunk.py --limit 200      # 只处理前 200 个 chunk（调试）
    python chunk.py --chunk-size 512
"""
import os
import re
import json
import csv
import hashlib
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT = os.path.join(ROOT, "data", "text_docs")
STRUCT = os.path.join(ROOT, "data", "doc_structure.json")
URLMAP = os.path.join(ROOT, "data", "url_map.csv")
OUT = os.path.join(ROOT, "data", "chunks.jsonl")

# 英文近似：1 token ≈ 4 字符
CHARS_PER_TOKEN = 4


def est_tokens(s):
    return len(s) / CHARS_PER_TOKEN


def doc_hash(rel_path):
    """文档路径 → 8 位 hash（chunk_id 前缀）"""
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]


def split_by_headings(text):
    """
    按 Markdown 标题切 section。
    返回 [{title, level, path, body}]
    """
    lines = text.split("\n")
    sections = []
    cur_title, cur_level, cur_body = None, 0, []
    # 维护标题栈，用于生成 path
    stack = []  # [(level, title)]

    def flush():
        if cur_title is not None or cur_body:
            body = "\n".join(cur_body).strip()
            if body or cur_title:
                path = " > ".join(t for _, t in stack) if stack else (
                    cur_title or "(root)")
                sections.append({
                    "title": cur_title or "(root)",
                    "level": cur_level,
                    "path": path,
                    "body": body,
                })

    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush()
            cur_level = len(m.group(1))
            cur_title = m.group(2).strip()
            cur_body = []
            # 更新栈
            while stack and stack[-1][0] >= cur_level:
                stack.pop()
            stack.append((cur_level, cur_title))
        else:
            cur_body.append(ln)
    flush()
    return sections


def recursive_split(text, chunk_size, overlap, sep_levels=None):
    """
    递归切分一段太长文本，返回 chunk 列表。
    sep_levels: 分隔符优先级（先段落、再句子）
    """
    if sep_levels is None:
        sep_levels = ["\n\n", "\n", "。 ", ". ", " "]
    max_chars = chunk_size * CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    # 找一个合适的分隔符
    sep = None
    for s in sep_levels:
        if s in text:
            sep = s
            break

    if sep is None:
        # 无法再切，硬切
        pieces = []
        step = max_chars
        for i in range(0, len(text), step):
            pieces.append(text[i:i + step])
        return [p.strip() for p in pieces if p.strip()]

    parts = text.split(sep)
    chunks, cur = [], ""
    for p in parts:
        cand = (cur + sep + p) if cur else p
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur.strip():
                chunks.append(cur.strip())
            # 这一段本身可能还太长 → 用下一级分隔符继续切
            if len(p) > max_chars:
                deeper = recursive_split(p, chunk_size, 0, sep_levels[sep_levels.index(sep) + 1:])
                chunks.extend(deeper)
                cur = ""
            else:
                cur = p
    if cur.strip():
        chunks.append(cur.strip())

    # 加 overlap（把上一个 chunk 的尾部拼到下一个开头）
    if overlap and len(chunks) > 1:
        ov_chars = int(overlap * CHARS_PER_TOKEN)
        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = merged[-1][-ov_chars:] if ov_chars > 0 else ""
            merged.append((tail + "\n" + chunks[i]).strip())
        return merged

    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=512, help="每个 chunk 的 token 数")
    ap.add_argument("--overlap", type=float, default=0.10, help="overlap 比例")
    ap.add_argument("--min-chars", type=int, default=100, help="丢弃过短的 chunk")
    ap.add_argument("--limit", type=int, default=0, help="调试用：最多输出多少 chunk")
    args = ap.parse_args()

    # 读 url_map
    url_of = {}
    ver_of = {}
    if os.path.exists(URLMAP):
        with open(URLMAP, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lp = row["local_path"].replace("data/raw_docs/", "")
                lp = lp[:-5] if lp.endswith(".html") else lp
                url_of[lp] = row["url"]
                ver_of[lp] = row["version"]

    # 读 section 结构（可选，用于补全 path）
    struct = {}
    if os.path.exists(STRUCT):
        struct = json.load(open(STRUCT, encoding="utf-8"))

    # 遍历纯文本
    files = []
    for dirpath, _, fns in os.walk(TEXT):
        for fn in fns:
            if fn.endswith(".txt"):
                files.append(os.path.join(dirpath, fn))
    files.sort()

    total_chunks = 0
    stats = []
    with open(OUT, "w", encoding="utf-8") as fout:
        for fp in files:
            rel = os.path.relpath(fp, TEXT).replace("\\", "/")
            key = rel[:-4]  # 去掉 .txt
            text = open(fp, encoding="utf-8").read()
            if not text.strip():
                continue

            dh = doc_hash(rel)
            url = url_of.get(key, "")
            ver = ver_of.get(key, "3.x")

            sections = split_by_headings(text)
            n_this = 0
            idx = 0
            for sec in sections:
                body = sec["body"]
                if not body.strip():
                    continue
                pieces = recursive_split(body, args.chunk_size, args.overlap)
                for piece in pieces:
                    if len(piece) < args.min_chars:
                        continue
                    # 可选：把 section 标题拼到 chunk 开头，提升检索效果
                    chunk_text = ("[" + sec["title"] + "]\n" + piece) if sec["level"] >= 2 else piece
                    rec = {
                        "chunk_id": f"{dh}_{idx:04d}",
                        "text": chunk_text,
                        "url": url,
                        "section_path": sec["path"],
                        "version": ver,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    idx += 1
                    n_this += 1
                    total_chunks += 1
                    if args.limit and total_chunks >= args.limit:
                        break
                if args.limit and total_chunks >= args.limit:
                    break
            stats.append((rel, n_this, len(text)))
            if args.limit and total_chunks >= args.limit:
                print(f"[limit] 达到 {args.limit}，提前结束")
                break

    # 报告
    tok_est = 0
    import statistics
    sizes = []
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            L = len(r["text"])
            sizes.append(L)
            tok_est += L / CHARS_PER_TOKEN

    print(f"\n[DONE] chunks 写入: {OUT}")
    print(f"  文档数      : {len(stats)}")
    print(f"  总 chunk 数 : {total_chunks:,}")
    print(f"  平均字符数  : {int(statistics.mean(sizes)) if sizes else 0}")
    print(f"  中位字符数  : {int(statistics.median(sizes)) if sizes else 0}")
    print(f"  最大/最小   : {max(sizes) if sizes else 0} / {min(sizes) if sizes else 0}")
    print(f"  估算总 token: {int(tok_est):,}")


if __name__ == "__main__":
    main()
