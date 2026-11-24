#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clean_html.py —— Python 官方文档 HTML 规则清洗

策略：先精确定位正文容器 <div class="body" role="main">，只保留它内部的
      内容；然后删掉正文里残留的噪声（headerlink、版本横幅、related 链接等）；
      最后保护 <pre> 代码块不被破坏。

用法:
    python clean_html.py --limit 200
    python clean_html.py --full --workers 8

输入:  data/raw_docs/**/*.html
输出:  data/raw_docs/cleaned/**/*.html
       data/clean_stats.csv   （每页统计：原始大小/清洗后大小/代码块数/保留比例）
"""
import os
import re
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from lxml import html as LH
from lxml import etree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw_docs")
CLEAN = os.path.join(ROOT, "data", "raw_docs", "cleaned")
STATS = os.path.join(ROOT, "data", "clean_stats.csv")

# ---- 需要整块删除的噪声（在正文内部） ----
NOISE_XPATH = [
    './/a[contains(@class,"headerlink")]',          # ¶ 锚点链接
    './/div[contains(@class,"related")]',           # 上下页导航
    './/div[contains(@class,"sphinxsidebar")]',
    './/div[contains(@class,"footer")]',
    './/div[contains(@class,"clearer")]',
    './/nav',
    './/script',
    './/style',
    './/form[contains(@class,"search")]',
]

# ---- 需要删除的属性（保留结构，去掉脚本/样式/锚点） ----
STRIP_ATTRS = ["class", "style", "onclick", "data-*", "role", "aria-*",
               "title", "rel", "target"]


def find_body(tree):
    """定位正文容器。新版 Sphinx 用 <div class="body" role="main">"""
    for xp in [
        '//div[contains(@class,"body") and @role="main"]',
        '//div[contains(@class,"body")]',
        '//div[@id="content"]',
        '//main',
        '//article',
    ]:
        found = tree.xpath(xp)
        if found:
            return found[0]
    return None


def strip_attributes(el):
    """递归清理属性，但保留 href / src / id（id 用于后续分块定位）"""
    keep = {"href", "src", "alt", "id"}
    for node in el.iter():
        if not isinstance(node.tag, str):
            continue
        for attr in list(node.attrib):
            if attr in keep:
                continue
            del node.attrib[attr]


def clean_one(args):
    """清洗单个文件，返回统计 dict"""
    src_path, dst_path = args
    try:
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return {"local_path": src_path, "ok": False, "error": str(e)[:80]}

    try:
        tree = LH.fromstring(raw)
    except Exception as e:
        return {"local_path": src_path, "ok": False, "error": "parse: " + str(e)[:60]}

    body = find_body(tree)
    if body is None:
        return {"local_path": src_path, "ok": False, "error": "no body found"}

    # 1) 统计原始的代码块数量（用于校验）
    pre_count_before = len(body.xpath('.//pre'))

    # 2) 删除噪声节点
    for xp in NOISE_XPATH:
        for node in body.xpath(xp):
            try:
                node.getparent().remove(node)
            except Exception:
                pass

    # 3) 清理属性（保留 href/src/id）
    strip_attributes(body)

    # 4) 生成干净的 HTML 文档
    title_nodes = body.xpath('.//h1')
    title = (title_nodes[0].text_content().strip() if title_nodes else "") or ""

    doc = etree.Element("html")
    head = etree.SubElement(doc, "head")
    t = etree.SubElement(head, "title")
    t.text = title
    meta = etree.SubElement(head, "meta")
    meta.set("charset", "utf-8")
    body_el = etree.SubElement(doc, "body")
    # 把清洗后的内容搬进去
    for child in list(body):
        body_el.append(child)

    out_html = etree.tostring(doc, pretty_print=True,
                              encoding="utf-8", method="html").decode("utf-8")

    # 5) 保护代码块：确认 <pre> 内容完整
    pre_count_after = out_html.count("<pre")
    code_ok = (pre_count_after >= pre_count_before)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(out_html)

    return {
        "local_path": os.path.relpath(src_path, ROOT).replace("\\", "/"),
        "ok": True,
        "error": "",
        "raw_kb": len(raw) // 1024,
        "clean_kb": len(out_html) // 1024,
        "ratio": round(len(out_html) / max(len(raw), 1), 3),
        "pre_before": pre_count_before,
        "pre_after": pre_count_after,
        "code_ok": code_ok,
        "has_nav": int('<nav' in out_html),
        "has_sidebar": int('sphinxsidebar' in out_html),
        "has_footer": int('class="footer"' in out_html),
        "title": title[:60],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0=全部）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # 收集所有原始文件
    tasks = []
    for dirpath, _, files in os.walk(RAW):
        if os.path.normpath(dirpath).startswith(os.path.normpath(CLEAN)):
            continue
        for fn in files:
            if not fn.endswith(".html"):
                continue
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, RAW)
            dst = os.path.join(CLEAN, rel)
            tasks.append((src, dst))

    tasks.sort()
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"[clean] 待处理 {len(tasks)} 个文件，{args.workers} 进程并行")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(clean_one, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(tasks)}]")

    # 统计
    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]

    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    cols = ["local_path", "ok", "error", "raw_kb", "clean_kb", "ratio",
            "pre_before", "pre_after", "code_ok", "has_nav", "has_sidebar",
            "has_footer", "title"]
    with open(STATS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # 报告
    print(f"\n[DONE] 成功 {len(ok)} / {len(tasks)}，失败 {len(fail)}")
    if ok:
        avg_ratio = sum(r["ratio"] for r in ok) / len(ok)
        total_pre = sum(r["pre_before"] for r in ok)
        code_broken = [r for r in ok if not r["code_ok"]]
        nav_left = [r for r in ok if r["has_nav"] or r["has_sidebar"] or r["has_footer"]]
        print(f"  平均保留比例: {avg_ratio:.1%}")
        print(f"  代码块总数  : {total_pre}")
        print(f"  代码块受损  : {len(code_broken)} 页")
        print(f"  导航噪声残留: {len(nav_left)} 页")
        if code_broken:
            print("  ⚠️ 代码块受损示例:", [r['local_path'] for r in code_broken[:3]])
        if nav_left:
            print("  ⚠️ 噪声残留示例:", [r['local_path'] for r in nav_left[:3]])
    if fail:
        print("  ❌ 失败示例:", [(r['local_path'], r.get('error')) for r in fail[:5]])
    print(f"\n  统计表: {STATS}")


if __name__ == "__main__":
    main()
