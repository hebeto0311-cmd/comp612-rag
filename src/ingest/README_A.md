# A 角色交付说明（README_A）

> 负责人：A（语料获取与清洗）
> 更新时间：2026/11/15

---

## 一句话

把 Python 3 官方文档变成**干净的纯文本 + section 结构**，供 B 做分块。

---

## 怎么用（给 B）

```bash
cd comp6132/src/ingest

# 1. 下载文档（已跑过 200 页，可续传/扩展）
python fetch.py --limit 200 --resume

# 2. 清洗 HTML
python clean_html.py --workers 8

# 3. 转纯文本 + section 结构
python to_text.py --workers 8
```

**B 只需要关心这两个产物：**

| 文件 | 内容 |
|---|---|
| `data/text_docs/**/*.txt` | 纯文本，Markdown 风格，403 万字符 |
| `data/doc_structure.json` | 每页的 section 树（标题层级、id、路径） |

---

## 交付物清单

| 路径 | 说明 |
|---|---|
| `src/ingest/fetch.py` | 爬虫，BFS + 断点续传 |
| `src/ingest/clean_html.py` | BeautifulSoup/lxml 规则清洗 |
| `src/ingest/to_text.py` | HTML → 纯文本 + section 结构 |
| `data/raw_docs/**/*.html` | 原始 HTML（200 页） |
| `data/raw_docs/cleaned/**/*.html` | 清洗后 HTML |
| `data/text_docs/**/*.txt` | **纯文本（B 的主要输入）** |
| `data/doc_structure.json` | **section 结构（B 分块用）** |
| `data/url_map.csv` | 本地路径 ↔ 原始 URL ↔ 版本 ↔ 标题 |
| `data/clean_stats.csv` | 每页清洗统计 |

---

## 当前结果（200 页）

| 指标 | 数值 |
|---|---|
| 下载页数 | **200**（队列还有 157 页） |
| 清洗成功 | **200 / 200** |
| 代码块总数 | **2,192** |
| **代码块受损** | **0 页** ✅ |
| **导航噪声残留** | **0 页** ✅ |
| 平均保留比例 | 39.0% |
| 纯文本总字符 | **4,127,788** |
| section 总数 | **1,529** |
| 文档版本 | Python **3.15.2** |

---

## ⚠️ 重要预警：语料量不足以支撑三档实验

**Proposal 里承诺的实验设计是 2,000 / 5,000 / 10,000 chunks。**

按当前数据估算（512 token ≈ 2048 字符）：

| 文档量 | 预估 chunk 数 |
|---|---|
| 200 页（当前） | ≈ **2,044** |
| 需要 5,000 | 约需 490 页 |
| 需要 10,000 | 约需 **980 页** |

**结论**：要继续抓，至少要到 **~1000 页**才能凑够 10,000 chunks。

当前爬虫队列里还有 **157 页**已知链接，加起来约 357 页，**仍然不够**。

**解决方案（需全组决策）**：

| 方案 | 做法 | 影响 |
|---|---|---|
| **A（推荐）** | 继续爬，抓满 1000+ 页 | A 多花 1 天；语料最真实 |
| **B** | 把 chunk 尺寸从 512 降到 256 token | 200 页就能出 ~4000 chunks；但偏离 Proposal |
| **C** | 加入 Django 文档（Proposal 里写了 optional） | 语料量翻倍；但引入异构 |

**建议选 A**，Proposal 里写的就是 "Django documentation optional if more volume is needed"，说明老师认可扩充语料。

---

## 清洗规则说明（关键决策）

### 定位正文的方式

新版 Sphinx（Python 3.15 文档）的结构是：
```html
<div class="body" role="main">
  <section id="xxx"> ... 真正的正文 ... </section>
</div>
```
我们**只保留 `<div class="body" role="main">` 内部**，外围的 nav/sidebar/footer 自然全部丢弃。

### 删除的噪声

- `a.headerlink` —— 标题旁的 ¶ 锚点
- `div.related` —— 上下页导航
- `div.sphinxsidebar`、`div.footer`、`nav`、`script`、`style`
- 所有元素的 `class/style/onclick/role/aria-*` 属性

### 保留的属性
`href`（C 的引用跳转要用）、`src`、`alt`、`id`（B 的 section 定位要用）

### 代码块保护

Python 文档的代码块是 Pygments 高亮：
```html
<pre><span>from</span><span> </span><span>turtle</span> ...
```
`text_content()` 会把所有 span 拼成完整文本，**缩进和换行保留**。已验证 2,192 个代码块无一受损。

---

## 已知问题 / 待办

| 问题 | 状态 |
|---|---|
| 语料量不足 10,000 chunks | ⚠️ 需继续爬（见上） |
| PDF 提取 | ⏳ 未做（HTML 已足够，PDF 是 Proposal 里的"variety"部分） |
| 100 页人工标注 | ⏳ 待做（需真人） |
| LLM 表格修复 | ⏳ 未做（当前表格已能正确提取，暂不需要） |
| `img` 标签保留但图片不下载 | 可接受（文本任务不需要图片） |

---

## 给 B 的接口提示

`sentences` 和 `section_path` 已经在 `doc_structure.json` 里了，分块时：

```python
struct = json.load(open("data/doc_structure.json"))
meta = struct["data/text_docs/3/library/stdtypes.txt"]
# meta["title"]     -> 页面标题
# meta["sections"]  -> [{"level":1,"title":"...","id":"...","path":"A > B"}, ...]
```

**建议分块策略**：按 `sections` 里的 level-2/3 边界切，块内超过 512 token 再用递归切分。
