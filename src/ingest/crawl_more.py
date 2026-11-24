#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crawl_more.py —— 续爬：从已下载页面里提取未抓的链接，继续扩展

为什么单独写：
    fetch.py 用 BFS 从种子页开始，--resume 只能跳过已下载的页面，
    但**不会恢复上次的队列**。所以续爬会立刻遇到"队列只剩种子页"的问题。

本脚本的做法：
    1. 扫描所有已下载的 HTML，重新提取所有站内链接
    2. 排除已下载的
    3. 把这些"已知但未抓"的链接作为新的种子继续爬
    4. 如果还不够，再从更深的层级（比如索引页的所有子链接）挖

用法:
    python crawl_more.py --target 620
"""
import os
import re
import csv
import time
import argparse
import urllib3
from urllib.parse import urljoin, urlparse
from collections import deque

import requests

urllib3.disable_warnings()

BASE = "https://docs.python.org/3/"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw_docs")
URL_MAP = os.path.join(ROOT, "data", "url_map.csv")

PRIORITY_PREFIXES = ["/3/tutorial/", "/3/library/", "/3/reference/", "/3/howto/"]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; COMP6132-ResearchBot/1.0; educational)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get(url, retries=2, timeout=25):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, verify=False)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r
            if r.status_code == 404:
                return None
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(1.0)
    return None


def local_path_for(url):
    p = urlparse(url)
    path = p.path.lstrip("/")
    if not path.endswith(".html"):
        path = path.rstrip("/") + ".html"
    return path


def is_doc_link(url):
    p = urlparse(url)
    if p.netloc != "docs.python.org":
        return False
    path = p.path
    if not path.endswith(".html"):
        return False
    if path in ("/3/genindex.html", "/3/py-modindex.html",
                "/3/search.html", "/3/index.html"):
        return False
    return any(path.startswith(x) for x in PRIORITY_PREFIXES)


def links_from(html, base_url):
    out = []
    for m in re.finditer(r'<a\s+[^>]*href="([^"#][^"]*)"', html):
        href = m.group(1)
        if href.startswith(("mailto:", "javascript:")):
            continue
        full = urljoin(base_url, href).split("#")[0]
        if is_doc_link(full):
            out.append(full)
    return out


def extract_version(html):
    m = re.search(r"Python (3\.\d+(?:\.\d+)?)", html)
    return m.group(1) if m else "3.x"


def extract_title(html):
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()[:100]


def scan_existing():
    """扫描已下载的 HTML，返回 (已下载url集合, 已下载的链接集合)"""
    rows = []
    urls = set()
    if os.path.exists(URL_MAP):
        with open(URL_MAP, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                urls.add(row["url"])

    # 从已下载文件里提取所有链接
    discovered = set()
    for lp, url, *_ in [(r["local_path"], r["url"]) for r in rows]:
        fp = os.path.join(ROOT, lp.replace("/", os.sep))
        if not os.path.exists(fp):
            continue
        try:
            h = open(fp, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for u in links_from(h, url):
            discovered.add(u)
    return urls, discovered, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=620, help="目标总页数")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    done_urls, discovered, rows = scan_existing()
    print(f"[scan] 已下载 {len(done_urls)} 页，从这些页面发现 {len(discovered)} 个站内链接")

    queue = deque(sorted(discovered - done_urls))
    print(f"[scan] 未抓取的已知链接: {len(queue)}")

    if not queue:
        print("[warn] 没有新链接可抓。可能需要从更深的索引页挖掘。")
        print("       尝试补充：/3/library/index.html 的完整子页列表")
        r = get("https://docs.python.org/3/library/index.html")
        if r:
            extra = set(links_from(r.text, "https://docs.python.org/3/library/index.html"))
            queue = deque(sorted(extra - done_urls))
            print(f"[scan] 从 library 索引补充到 {len(queue)} 个")

    n_new = 0
    while queue and len(rows) < args.target:
        url = queue.popleft()
        lp = local_path_for(url)
        dst = os.path.join(RAW, lp)
        if os.path.exists(dst):
            continue

        r = get(url)
        if r is None:
            continue

        html = r.text
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(html)

        rows.append({
            "local_path": os.path.relpath(dst, ROOT).replace("\\", "/"),
            "url": url,
            "version": extract_version(html),
            "page_title": extract_title(html),
        })
        n_new += 1
        if n_new % 25 == 0:
            print(f"  [{len(rows)}/{args.target}] {lp}")

        for nxt in links_from(html, url):
            if nxt not in done_urls:
                done_urls.add(nxt)
                queue.append(nxt)

    with open(URL_MAP, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["local_path", "url", "version", "page_title"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[DONE] 本次新增 {n_new} 页，累计 {len(rows)} 页")
    print(f"       队列剩余 {len(queue)}")


if __name__ == "__main__":
    main()
