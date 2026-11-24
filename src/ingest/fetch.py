#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch.py —— 下载 Python 3 官方文档 HTML

用法:
    python fetch.py --limit 200          # 先抓 200 页（Day 1 交付用）
    python fetch.py --full               # 抓全部
    python fetch.py --limit 200 --resume # 断点续传

输出:
    data/raw_docs/<相对路径>.html     原始 HTML
    data/url_map.csv                  本地路径 ↔ URL 映射
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

# 优先抓这些章节（tutorial / library / reference / howto）
PRIORITY_PREFIXES = ["/3/tutorial/", "/3/library/", "/3/reference/", "/3/howto/"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; COMP6132-ResearchBot/1.0; educational)",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get(url, retries=3, timeout=25):
    """带重试的 GET"""
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, verify=False)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r
            if r.status_code == 404:
                return None
        except Exception as e:
            if i == retries - 1:
                print(f"    [FAIL] {url} :: {type(e).__name__}")
                return None
            time.sleep(1.5 * (i + 1))
    return None


def local_path_for(url):
    """把 URL 映射为本地相对路径"""
    p = urlparse(url)
    path = p.path  # /3/library/stdtypes.html
    path = path.lstrip("/")
    if not path.endswith(".html"):
        path = path.rstrip("/") + ".html"
    return path


def extract_version(html):
    m = re.search(r"Python (3\.\d+(?:\.\d+)?)", html)
    return m.group(1) if m else "3.x"


def extract_title(html):
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if not m:
        return ""
    t = re.sub(r"\s+", " ", m.group(1)).strip()
    t = t.replace("— Python 3", "").replace("&mdash;", "").strip()
    return t


def is_doc_link(url):
    """判断是否是我们要抓的文档页（排除锚点、静态资源、其他语言版本）"""
    p = urlparse(url)
    if p.netloc != "docs.python.org":
        return False
    path = p.path
    if not path.endswith(".html"):
        return False
    if path in ("/3/genindex.html", "/3/py-modindex.html",
                "/3/search.html", "/3/index.html"):
        return False
    # 只要 tutorial / library / reference / howto
    return any(path.startswith(x) for x in PRIORITY_PREFIXES)


def links_from(html, base_url):
    """提取页面内的站内文档链接"""
    out = []
    for m in re.finditer(r'<a\s+[^>]*href="([^"#][^"]*)"', html):
        href = m.group(1)
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        full = full.split("#")[0]
        if is_doc_link(full):
            out.append(full)
    # 去重保序
    seen, res = set(), []
    for u in out:
        if u not in seen:
            seen.add(u)
            res.append(u)
    return res


def load_existing():
    """读取已下载的 URL（用于 --resume）"""
    if not os.path.exists(URL_MAP):
        return set(), []
    rows = []
    urls = set()
    with open(URL_MAP, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            urls.add(row["url"])
    return urls, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="最多抓多少页")
    ap.add_argument("--full", action="store_true", help="抓全部（忽略 limit）")
    ap.add_argument("--resume", action="store_true", help="断点续传")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    limit = 10 ** 9 if args.full else args.limit

    done_urls, rows = (load_existing() if args.resume else (set(), []))
    if args.resume:
        print(f"[resume] 已有 {len(done_urls)} 页")

    # BFS 队列
    seed = [urljoin(BASE, x) for x in
            ["tutorial/index.html", "library/index.html",
             "reference/index.html", "howto/index.html"]]
    queue = deque(u for u in seed if u not in done_urls)
    seen = set(queue) | done_urls

    n_new = 0
    while queue and n_new < limit:
        url = queue.popleft()
        lp = local_path_for(url)
        dst = os.path.join(RAW, lp)

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
        if n_new % 20 == 0:
            print(f"  [{n_new}] {lp}  ({len(html)//1024} KB)")

        # 发现新链接
        for nxt in links_from(html, url):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    # 写映射表
    with open(URL_MAP, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["local_path", "url", "version", "page_title"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[DONE] 本次新增 {n_new} 页，累计 {len(rows)} 页")
    print(f"       映射表: {URL_MAP}")
    print(f"       队列剩余: {len(queue)}")


if __name__ == "__main__":
    main()
