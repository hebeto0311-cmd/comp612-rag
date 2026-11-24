#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
to_text.py —— 把清洗后的 HTML 转成纯文本（同时保留 section 结构线索）

这是 A 交给 B 的"最终可用产物"：B 做分块时可以直接读纯文本 + section 路径，
不必再碰 HTML。

⭐ 关键点：
  1. Python 文档的代码块在 HTML 里是 <pre><span>...</span></pre>（Pygments 高亮），
     直接 text_content() 会得到正确的纯文本，但要注意 span 之间没有空格丢失问题。
  2. 每个 section 的 id 和标题会被记录下来，用于生成 section_path。

输出:
    data/text_docs/<相对路径>.txt       纯文本（Markdown 风格）
    data/doc_structure.json            每页的 section 树（B 分块时用）
"""
import os
import re
import json
import html as html_mod
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from lxml import html as LH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN = os.path.join(ROOT, "data", "raw_docs", "cleaned")
TEXT = os.path.join(ROOT, "data", "text_docs")
STRUCT = os.path.join(ROOT, "data", "doc_structure.json")

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def node_text_simple(el):
    """普通元素的文本（去多余空白）"""
    t = el.text_content()
    t = re.sub(r"[ \t]*\n[ \t]*", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def pre_text(el):
    """代码块文本：保留换行和缩进（Pygments span 拼接后 text_content 是正确的）"""
    t = el.text_content()
    # 去掉行尾空格，但保留行首缩进
    lines = [ln.rstrip() for ln in t.split("\n")]
    return "\n".join(lines).strip("\n")


def walk(el, out, path, sections, depth_guard=0):
    """递归遍历，输出 Markdown 风格文本"""
    if depth_guard > 200:
        return
    tag = el.tag if isinstance(el.tag, str) else ""

    if tag in HEADING_TAGS:
        txt = node_text_simple(el)
        if txt:
            lvl = HEADING_TAGS[tag]
            out.append("")
            out.append("#" * lvl + " " + txt)
            out.append("")
            # 记录 section
            sid = el.get("id") or (el.getparent().get("id") if el.getparent() is not None else "")
            new_path = path + ([txt] if lvl <= 3 else [])
            sections.append({"level": lvl, "title": txt,
                             "id": sid or "", "path": " > ".join(new_path)})
            for ch in el:
                walk(ch, out, new_path, sections, depth_guard + 1)
        return

    if tag == "pre":
        code = pre_text(el)
        if code:
            out.append("")
            out.append("```")
            out.append(code)
            out.append("```")
            out.append("")
        return

    if tag in ("script", "style"):
        return

    if tag == "p":
        txt = node_text_simple(el)
        if txt:
            out.append(txt)
            out.append("")
        return

    if tag == "li":
        txt = node_text_simple(el)
        if txt:
            out.append("- " + txt)
        return

    if tag == "dt":
        txt = node_text_simple(el)
        if txt:
            out.append("")
            out.append("**" + txt + "**")
        return

    if tag == "dd":
        txt = node_text_simple(el)
        if txt:
            out.append(txt)
            out.append("")
        return

    if tag == "table":
        # 表格：按行输出
        for tr in el.iter("tr"):
            cells = [node_text_simple(td) for td in tr
                     if isinstance(td.tag, str) and td.tag in ("td", "th")]
            cells = [c for c in cells if c]
            if cells:
                out.append("| " + " | ".join(cells) + " |")
        out.append("")
        return

    # 其他容器：继续遍历
    for ch in el:
        walk(ch, out, path, sections, depth_guard + 1)


def convert_one(args):
    src, dst = args
    try:
        raw = open(src, encoding="utf-8", errors="replace").read()
        tree = LH.fromstring(raw)
    except Exception as e:
        return {"ok": False, "src": src, "error": str(e)[:60]}

    title_nodes = tree.xpath("//title")
    title = title_nodes[0].text_content().strip() if title_nodes else ""

    out, sections = [], []
    body_nodes = tree.xpath("//body")
    root_el = body_nodes[0] if body_nodes else tree
    for ch in root_el:
        walk(ch, out, [], sections)

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    return {"ok": True, "src": src, "dst": dst, "title": title,
            "chars": len(text), "sections": sections}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = []
    for dirpath, _, files in os.walk(CLEAN):
        for fn in files:
            if fn.endswith(".html"):
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(src, CLEAN)
                dst = os.path.join(TEXT, rel[:-5] + ".txt")
                tasks.append((src, dst))
    tasks.sort()

    print(f"[to_text] 待转换 {len(tasks)} 个文件")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, t) for t in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    ok = [r for r in results if r.get("ok")]
    struct = {}
    for r in ok:
        rel = os.path.relpath(r["dst"], ROOT).replace("\\", "/")
        struct[rel] = {"title": r["title"], "chars": r["chars"],
                       "sections": r["sections"]}

    os.makedirs(os.path.dirname(STRUCT), exist_ok=True)
    with open(STRUCT, "w", encoding="utf-8") as f:
        json.dump(struct, f, ensure_ascii=False, indent=1)

    total_chars = sum(r["chars"] for r in ok)
    total_sec = sum(len(r["sections"]) for r in ok)
    print(f"[DONE] 成功 {len(ok)}/{len(tasks)}")
    print(f"  总字符数     : {total_chars:,}")
    print(f"  总 section 数: {total_sec:,}")
    print(f"  平均每页字符 : {total_chars // max(len(ok),1):,}")
    print(f"  纯文本目录   : {TEXT}")
    print(f"  section 结构 : {STRUCT}")


if __name__ == "__main__":
    main()
