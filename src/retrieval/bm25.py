#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bm25.py —— 纯 Python 实现的 BM25（不依赖 rank_bm25）

为什么自己写：
  1. 项目环境装第三方包不稳定
  2. 报告里可以讲清"我们实现了 BM25"，比调库更有说服力
  3. 参数可调（k1, b）

BM25 公式:
    score(D, Q) = Σ_t IDF(t) · (f(t,D)·(k1+1)) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))
    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
"""
import re
import math
from collections import Counter

STOPWORDS = set("""
a an the and or but if then else for while of to in on at by with from as is are was were
be been being this that these those it its it's you your we our they their he she them
i me my not no nor so such than too very can will just should now
""".split())

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text):
    """分词：保留标识符和下划线（Python 文档里 `list.append` 这类有意义）"""
    toks = TOKEN_RE.findall(text)
    return [t.lower() for t in toks if t.lower() not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []   # 每篇文档的词频 Counter
        self.doc_len = []
        self.idf = {}
        self.avgdl = 0
        self.N = 0

    def fit(self, docs):
        """docs: list[str]"""
        self.N = len(docs)
        df = Counter()
        for d in docs:
            toks = tokenize(d)
            tf = Counter(toks)
            self.doc_freqs.append(tf)
            self.doc_len.append(len(toks))
            for term in tf:
                df[term] += 1
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0

        # IDF（Lucene 变体，保证非负）
        for term, n in df.items():
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
        return self

    def get_scores(self, query):
        """返回所有文档的分数（numpy array 或 list）"""
        qterms = tokenize(query)
        scores = [0.0] * self.N
        if not qterms or not self.N:
            return scores
        for term in qterms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.N):
                f = self.doc_freqs[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def top_k(self, query, k=5):
        """返回 [(doc_index, score)]，按分数降序，score>0"""
        scores = self.get_scores(query)
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: -x[1]
        )
        return ranked[:k]


if __name__ == "__main__":
    # 自测
    docs = [
        "list append method adds an element to the end of a list",
        "dict get returns the value for a key or default",
        "the append method of list objects",
        "string format method for formatting",
    ]
    bm = BM25().fit(docs)
    res = bm.top_k("list append method", k=3)
    print("query: 'list append method'")
    for i, s in res:
        print(f"  doc{i}: {s:.4f}  |  {docs[i]}")
    assert res[0][0] in (0, 2), "最相关文档应是含 list+append 的"
    print("\n[OK] BM25 自测通过")
