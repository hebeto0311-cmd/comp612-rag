# B 角色交付说明（README_B）

> 负责人：B（分块、索引与混合检索）
> 更新时间：2026/11/15
> **状态：全部功能已跑通，29 项验收测试全过**

---

## 一句话

把 A 的纯文本切成 chunk，建**稠密 + 稀疏**两套索引，用 **RRF 融合**，对外提供统一的 `retrieve()`，并支持 **20% / 50% / 100% 三档数据量切换**。

---

## 怎么用（给 C 和 D）

### 建索引（已完成，一般不用重跑）

```bash
cd comp6132/src/retrieval
python build_subset.py              # 生成三档子集
python build_index_variants.py      # 建三档索引
```

### 调用（契约 2）

```python
from retriever import retrieve

# 默认用 100% 语料 (large)
hits = retrieve("how to append an item to a list", mode="hybrid", k=5)

# 切换数据档位
hits = retrieve(query, mode="hybrid", k=5, variant="small")   # 20%
hits = retrieve(query, mode="hybrid", k=5, variant="medium")  # 50%
hits = retrieve(query, mode="hybrid", k=5, variant="large")   # 100%

# 返回: [{"chunk_id": str, "score": float, "text": str, "url": str}, ...]
```

### 命令行

```bash
python retriever.py --query "list append" --compare --variant large --k 5
python test_retrieval.py        # 29 项验收测试
python rrf.py --test            # RRF 6 项单元测试
```

---

## 交付物

| 路径 | 说明 |
|---|---|
| `src/ingest/crawl_more.py` | **（新增）续爬脚本** —— 从已下载页面重扫链接 |
| `src/retrieval/chunk.py` | 按 section 边界递归分块 |
| `src/retrieval/bm25.py` | **纯 Python BM25**（零第三方依赖） |
| `src/retrieval/dense.py` | 稠密索引（faiss 可选，numpy 兜底） |
| `src/retrieval/rrf.py` | **RRF 融合 + 6 项单元测试** |
| `src/retrieval/retriever.py` | **统一接口（核心交付，支持三档）** |
| `src/retrieval/build_subset.py` | 三档子集生成（按文档采样） |
| `src/retrieval/build_index_variants.py` | 三档索引构建 |
| `src/retrieval/test_retrieval.py` | **29 项验收测试** |
| `data/chunks.jsonl` | **5,836 个 chunk（契约 1）** |
| `data/chunks_{20,50,100}.jsonl` | 三档子集 |
| `data/indexes/{small,medium,large}/` | 三档索引 |
| `cache/embeddings/` | 向量缓存 |

---

## 实测数据

### 语料规模（383 页）

| 指标 | 数值 |
|---|---|
| 文档数 | **383** |
| **chunk 总数** | **5,836** |
| 平均字符数 | 1,220 |
| 中位字符数 | 1,280（≈ 320 token） |
| 最大 / 最小 | 2,107 / 100 |
| 估算总 token | 1,781,383 |
| 代码块总数 | **4,436（零受损）** |

### 三档数据量

| 档位 | 文档 | chunks | 建索引耗时 |
|---|---|---|---|
| **small (20%)** | 76 | **1,458** | 2.2 s |
| **medium (50%)** | 191 | **3,016** | 8.0 s |
| **large (100%)** | 383 | **5,836** | 16.1 s |

**子集嵌套关系已验证**：20% ⊂ 50% ⊂ 100%

### 性能

| 指标 | 数值 |
|---|---|
| 索引加载（5836 chunks） | **0.93 s** |
| hybrid 检索延迟 | **15.1 ms** |
| sparse 检索延迟 | **12.7 ms** |
| dense 检索延迟 | **0.9 ms** |

### 验收测试：**29 / 29 通过**

| 类别 | 项数 | 结果 |
|---|---|---|
| 契约字段完整性 | 6 | ✅ |
| 三种模式可用 | 3 | ✅ |
| hybrid 确实是融合 | 5 | ✅ |
| 三档数据量切换 | 6 | ✅ |
| 跨进程加载 | 2 | ✅ |
| 延迟达标 | 3 | ✅ |
| 边界情况 | 4 | ✅ |

---

## ⚠️ 重要：语料量已经到底了

**Python 3 官方文档总共只有约 360 个页面。我们已经抓了 383 页（含索引页），基本是全部。**

| 尝试 | 结果 |
|---|---|
| 首次抓取 | 200 页 |
| 补抓（crawl_more.py 重扫链接） | +183 页 → **383 页** |
| 再补抓 | **链接队列耗尽，无法继续** |

**所以 5,836 chunks 就是当前语料的上限。**

### 这对 Proposal 的影响

Proposal 承诺 2,000 / 5,000 / 10,000 三档。现在的实际是：

| Proposal 承诺 | 实际能做到 | 差距 |
|---|---|---|
| 2,000 | 1,458（20%档） | ✅ 接近 |
| 5,000 | 3,016（50%档） | ⚠️ 偏低 |
| **10,000** | **5,836（100%档）** | ❌ **只有 58%** |

### 三个应对方案

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 改档位数字** | 报告里改写成 1,500 / 3,000 / 5,800 | 诚实，但偏离原 Proposal |
| **B. 加 Django 文档** | Proposal 写了 "Django optional if more volume needed" | 语料翻倍，能到 10,000+；但要再跑一遍 A 的清洗流程 |
| **C. 减小 chunk 尺寸** | 512 → 320 token | 5,836 → 约 9,300；但偏离原设计参数 |

**我的建议：A + C 组合。**
- 报告里诚实说明：Python 文档总量有限，我们据实调整为 **1,500 / 3,000 / 5,800** 三档
- 同时在报告的 Limitation 部分提一句："若需更大规模，可接入 Django 文档（Proposal 预留了此选项）"

**这样既不造假，也解释得通。**

---

## 技术实现说明（写报告用）

### 1. 分块策略

```
文档 → 按 Markdown 标题切 section
     → 每个 section 若超 512 token 则递归切分（段落→行→句子→硬切）
     → 相邻 chunk 保留 10% overlap
```

**chunk 开头自动拼上 section 标题**（如 `[5.1. More on Lists]\n...`），实测能提升检索命中。

### 2. BM25（自己实现，非调库）

```
score(D,Q) = Σ_t IDF(t) · f(t,D)(k1+1) / (f(t,D) + k1(1-b + b·|D|/avgdl))
IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )     ← Lucene 变体
```

- 参数 `k1=1.5, b=0.75`
- 分词正则 `[A-Za-z_][A-Za-z0-9_]*|\d+`，**保留下划线**（Python 文档里 `list.append` 有意义）

### 3. RRF 融合（Proposal 的 Extension）

```
score(d) = Σ_i 1 / (k + rank_i(d))     k = 60, rank 从 1 开始
```

**为什么用 RRF 而不用加权求和：**
- 稠密分是余弦相似度（0~1），稀疏分是 BM25（0~几十），**量纲完全不同**
- 加权求和需要归一化，而归一化方式的选择很随意
- **排名是可比的**，用排名融合不需要任何归一化

### RRF 单元测试验证了 6 件事

1. 分数与手算一致（A: 1/61+1/62 = 0.032522）
2. 对称性正确（A=B, C=D）
3. **双路共现 > 单路高分**（RRF 的核心价值）
4. k 越小，排名差异越突出
5. 空输入处理
6. 单路时保持原顺序

---

## ⚠️ 当前限制：embedding 后端

| 项 | 状态 |
|---|---|
| **BGE 模型** | ❌ 未安装（`sentence-transformers` 下载失败，网络问题） |
| **当前用的** | **hash-embed**（384 维字符 n-gram 哈希投影） |
| **影响** | 功能全通，但**不是语义向量**，dense 检索质量会低于 BGE |
| **修复方式** | 装好后执行 `python build_index_variants.py --force-embed` |

**hash-embed 能保证 pipeline 端到端跑通**，但它捕捉的是词形而非语义。报告里如果写 "BGE encoder"，**必须先装好并重建索引**，否则不能这么写。

### 装 BGE 的三种途径

```bash
# 1. 直接装
pip install sentence-transformers

# 2. 用国内镜像
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 手动下载模型
#    从 https://huggingface.co/BAAI/bge-small-en-v1.5 下载
#    然后用 SentenceTransformer("本地路径") 加载
```

---

## 给 C 的接口提示

```python
from retriever import retrieve

hits = retrieve(query, mode="hybrid", k=5)
# 每条 hit 的 text 已截断到 400 字符（用于 prompt）

# 组装 prompt：
context = "\n\n".join(
    f"[{h['chunk_id']}] (from {h['url']})\n{h['text']}" for h in hits
)
```

**引用校验直接验证**：模型输出的 `[chunk_id]` 是否在 `{h["chunk_id"]}` 集合里。

---

## 给 D 的接口提示

```python
from retriever import get_retriever

r = get_retriever("small")      # 或 medium / large
r._ensure_loaded()               # 提前加载，避免每轮判断

for qa in gold_qa:
    hits = r.retrieve(qa["question"], mode="hybrid", k=5)
    ids = [h["chunk_id"] for h in hits]
    # recall_at_k(ids, qa["gold_chunk_id"], 5)
```

**三档实验建议**：对每个档位（small/medium/large）× 每种模式（dense/sparse/hybrid），跑完整 gold_qa，得到 **3 × 3 = 9 组结果**。

---

## 已知问题清单

| 问题 | 影响 | 状态 |
|---|---|---|
| BGE 未安装 | dense 质量偏低 | ⏳ 待网络好转 |
| faiss 未安装 | 用 numpy 代替 | ✅ 功能等价，性能足够 |
| 语料上限 5,836 chunks | 到不了 Proposal 的 10,000 | ⚠️ **需全组决策** |
| `crawl_more.py` 无法再扩 | 文档总数有限 | 已确认是上限 |
