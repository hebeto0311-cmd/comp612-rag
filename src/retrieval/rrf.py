#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rrf.py —— Reciprocal Rank Fusion 倒数排名融合

公式（Proposal 承诺 k=60）:
    score(d) = Σ_over_rankers  1 / (k + rank_i(d))
    其中 rank 从 1 开始

优点：两路分数的量纲完全不同（余弦相似度 vs BM25 分数），
      但排名是可比的，所以用排名融合不需要任何归一化。
"""
import argparse


def rrf_fuse(ranked_lists, k=60, top_n=None):
    """
    ranked_lists: list of list[doc_id]  —— 每路检索的结果，按相关性降序
    k: RRF 参数（Proposal 用 60）
    top_n: 只返回前 N 个

    返回: [(doc_id, score)]，按 score 降序
    """
    scores = {}
    for rlist in ranked_lists:
        for rank, doc_id in enumerate(rlist, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_n] if top_n else ranked


def rrf_from_scores(score_lists, k=60, top_n=None, min_score=0.0):
    """
    便利函数：输入是 [(doc_id, score)] 形式的排序结果（会自动按分数排序）
    """
    ranked_lists = []
    for sl in score_lists:
        s = sorted(sl, key=lambda x: -x[1])
        ranked_lists.append([d for d, _ in s if _ >= min_score])
    return rrf_fuse(ranked_lists, k=k, top_n=top_n)


# ============================================================
# 单元测试
# ============================================================

def _test():
    print("=" * 60)
    print("RRF 单元测试")
    print("=" * 60)

    # --- 用例 1：手算验证 ---
    # dense 排名: [A, B, C]
    # sparse 排名: [B, A, D]
    # k = 60
    # A: 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # B: 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # C: 1/63 = 0.015873
    # D: 1/63 = 0.015873
    dense = ["A", "B", "C"]
    sparse = ["B", "A", "D"]
    res = rrf_fuse([dense, sparse], k=60)
    print("\n用例 1 —— 手算验证")
    print(f"  dense  = {dense}")
    print(f"  sparse = {sparse}")
    print(f"  RRF 结果:")
    for d, s in res:
        print(f"    {d}: {s:.6f}")

    exp_A = 1 / 61 + 1 / 62
    exp_B = 1 / 62 + 1 / 61
    exp_C = 1 / 63
    exp_D = 1 / 63
    got = dict(res)
    assert abs(got["A"] - exp_A) < 1e-9, f"A: {got['A']} != {exp_A}"
    assert abs(got["B"] - exp_B) < 1e-9, f"B: {got['B']} != {exp_B}"
    assert abs(got["C"] - exp_C) < 1e-9, f"C: {got['C']} != {exp_C}"
    assert abs(got["D"] - exp_D) < 1e-9, f"D: {got['D']} != {exp_D}"
    print("  [OK] 分数与手算一致")

    # A 和 B 分数相同（两边排名对称），C 和 D 相同
    assert abs(got["A"] - got["B"]) < 1e-12, "A 和 B 应对称相等"
    assert abs(got["C"] - got["D"]) < 1e-12, "C 和 D 应对称相等"
    print("  [OK] 对称性正确（A=B, C=D）")

    # --- 用例 2：共现提升 ---
    # X 在两路都排第 3，Y 只在一路排第 1
    # X: 1/63 + 1/63 = 0.031746
    # Y: 1/61           = 0.016393
    # 共现的 X 应该超过单路第一的 Y
    r1 = ["P", "Q", "X"]
    r2 = ["Y", "Z", "X"]
    res2 = dict(rrf_fuse([r1, r2], k=60))
    print("\n用例 2 —— 共现提升（这是 RRF 的核心特性）")
    print(f"  r1 = {r1}, r2 = {r2}")
    print(f"  X（两路都第3）= {res2['X']:.6f}")
    print(f"  Y（一路第1）  = {res2['Y']:.6f}")
    assert res2["X"] > res2["Y"], "两路都靠前的应该胜出"
    print("  [OK] 双路共现 > 单路高分（融合生效）")

    # --- 用例 3：k 参数的影响 ---
    print("\n用例 3 —— k 参数影响（k 越大，排名差异被抹平）")
    for kk in [1, 10, 60, 1000]:
        r = dict(rrf_fuse([dense, sparse], k=kk))
        print(f"  k={kk:4d}: A={r['A']:.4f}  C={r['C']:.4f}  A/C={r['A']/r['C']:.3f}")
    r_small = dict(rrf_fuse([dense, sparse], k=1))
    r_big = dict(rrf_fuse([dense, sparse], k=1000))
    assert (r_small["A"] / r_small["C"]) > (r_big["A"] / r_big["C"]), \
        "k 越小，排名差异越突出"
    print("  [OK] k 的作用符合预期")

    # --- 用例 4：空输入 ---
    assert rrf_fuse([[]], k=60) == []
    assert rrf_fuse([], k=60) == []
    print("\n用例 4 —— 空输入 [OK]")

    # --- 用例 5：top_n ---
    res5 = rrf_fuse([dense, sparse], k=60, top_n=2)
    assert len(res5) == 2
    print("用例 5 —— top_n 截断 [OK]")

    # --- 用例 6：单路 ---
    res6 = rrf_fuse([dense], k=60)
    assert [d for d, _ in res6] == dense, "单路时顺序应保持"
    print("用例 6 —— 单路保持原顺序 [OK]")

    print("\n" + "=" * 60)
    print("[ALL PASS] RRF 全部测试通过")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    _test()
