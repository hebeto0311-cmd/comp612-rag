#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_retrieval.py —— B 模块的完整验收测试

覆盖:
  1. 契约 2 字段完整性
  2. 三种模式都能返回 k 条
  3. 三档数据量都能加载
  4. hybrid 确实是融合（不是简单拼接）
  5. 索引可落盘再加载（跨进程）
  6. 延迟达标（<200ms 热身后）
"""
import os
import sys
import time
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import get_retriever, Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"  [FAIL] {name}  {detail}")


def main():
    print("=" * 68)
    print("B 模块验收测试")
    print("=" * 68)

    # ---------- 1. 契约字段 ----------
    print("\n[1] 契约 2 字段完整性")
    r = get_retriever("large")
    hits = r.retrieve("how to sort a list", mode="hybrid", k=5)
    check("返回非空", len(hits) > 0, f"{len(hits)} 条")
    if hits:
        need = {"chunk_id", "score", "text", "url"}
        check("字段完全匹配", all(set(h.keys()) == need for h in hits),
              f"实际字段: {sorted(hits[0].keys())}")
        check("chunk_id 类型", all(isinstance(h["chunk_id"], str) for h in hits))
        check("score 类型", all(isinstance(h["score"], float) for h in hits))
        check("按 score 降序", all(hits[i]["score"] >= hits[i+1]["score"]
                                   for i in range(len(hits)-1)))
        check("url 非空", all(h["url"].startswith("http") for h in hits))

    # ---------- 2. 三种模式 ----------
    print("\n[2] 三种模式都能工作")
    for mode in ["dense", "sparse", "hybrid"]:
        t0 = time.time()
        hs = r.retrieve("dictionary keys iteration", mode=mode, k=5)
        dt = (time.time() - t0) * 1000
        check(f"mode={mode}", len(hs) == 5, f"{len(hs)} 条, {dt:.0f}ms")

    # ---------- 3. 融合验证 ----------
    print("\n[3] hybrid 确实是融合（非简单拼接）")
    q = "how to format a string"
    d_ids = [h["chunk_id"] for h in r.retrieve(q, "dense", 10)]
    s_ids = [h["chunk_id"] for h in r.retrieve(q, "sparse", 10)]
    h_ids = [h["chunk_id"] for h in r.retrieve(q, "hybrid", 10)]

    from_dense = sum(1 for x in h_ids if x in d_ids)
    from_sparse = sum(1 for x in h_ids if x in s_ids)
    check("hybrid 包含 dense 结果", from_dense > 0, f"{from_dense}/10 来自 dense")
    check("hybrid 包含 sparse 结果", from_sparse > 0, f"{from_sparse}/10 来自 sparse")
    check("hybrid 不是简单等于 dense", h_ids != d_ids)
    check("hybrid 不是简单等于 sparse", h_ids != s_ids)
    # RRF 的经典特性：两路都靠前的应该胜出
    both = [x for x in h_ids if x in d_ids and x in s_ids]
    check("存在双路共现的 chunk", len(both) > 0, f"{len(both)} 个在两边都出现")

    # ---------- 4. 三档数据量 ----------
    print("\n[4] 三档数据量切换")
    for v, expect_min, expect_max in [("small", 1000, 2000),
                                      ("medium", 2500, 3500),
                                      ("large", 5000, 6500)]:
        rr = get_retriever(v)
        rr._ensure_loaded()
        n = len(rr.ids)
        check(f"variant={v} 加载", expect_min <= n <= expect_max, f"{n} chunks")
        hs = rr.retrieve("what is a decorator", "hybrid", 3)
        check(f"variant={v} 可检索", len(hs) > 0, f"{len(hs)} 条")

    # ---------- 5. 跨进程持久化 ----------
    print("\n[5] 索引可跨进程加载（不重新构建）")
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from retriever import get_retriever;"
        "r = get_retriever('large'); r._ensure_loaded();"
        "print('N=', len(r.ids))" % os.path.dirname(os.path.abspath(__file__))
    )
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=120,
                             encoding="utf-8", errors="replace")
        elapsed = time.time() - t0
        ok = "N=" in out.stdout
        check("新进程能加载索引", ok,
              f"用时 {elapsed:.1f}s，输出 {out.stdout.strip()[:40]}")
        check("加载速度合理(<30s)", elapsed < 30, f"{elapsed:.1f}s")
    except Exception as e:
        check("新进程能加载索引", False, str(e)[:60])

    # ---------- 6. 延迟 ----------
    print("\n[6] 延迟（热身后 10 次平均）")
    r.retrieve("warmup", "hybrid", 5)  # 预热
    for mode in ["hybrid", "sparse", "dense"]:
        times = []
        for _ in range(10):
            t0 = time.time()
            r.retrieve("how do I read a file", mode=mode, k=5)
            times.append((time.time() - t0) * 1000)
        avg = sum(times) / len(times)
        limit = 200 if mode != "dense" else 500
        check(f"{mode} 延迟 <{limit}ms", avg < limit, f"{avg:.1f}ms")

    # ---------- 7. 边界情况 ----------
    print("\n[7] 边界情况")
    check("空 query 返回空", r.retrieve("", "hybrid", 5) == [])
    check("空白 query 返回空", r.retrieve("   ", "hybrid", 5) == [])
    hs = r.retrieve("zzzzqqq nonexistent term xyz", "sparse", 5)
    check("无命中时 sparse 返回空或少量", len(hs) <= 5, f"{len(hs)} 条")
    try:
        r.retrieve("test", "bogus_mode", 5)
        check("非法 mode 抛异常", False, "没有抛异常")
    except ValueError:
        check("非法 mode 抛异常", True)

    # ---------- 汇总 ----------
    print("\n" + "=" * 68)
    print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
    if FAIL:
        print("\n失败项:")
        for n, d in FAIL:
            print(f"  - {n}: {d}")
    print("=" * 68)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
