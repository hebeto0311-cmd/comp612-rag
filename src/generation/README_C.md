# C 部分：生成、引用校验与网页界面

已在原项目副本中添加 C 部分；A/B 原有源码及数据保持原样。

## Mac 上启动

这台机器的项目虚拟环境位于项目目录 `.venv/`。双击项目根目录的 `启动问答.command`，或在终端运行：

```sh
cd '/Users/hebe/Documents/Codex/2026-11-21/new-chat/outputs/comp6132'
.venv/bin/python -m streamlit run src/app/streamlit_app.py --server.address 127.0.0.1
```

打开终端显示的本地网页（通常 http://localhost:8501）。无需 VS Code。
侧栏填写 API 地址、模型名、API Key，再输入问题。Key 不保存到项目或答案缓存。
默认配置为 DeepSeek 兼容接口；如果使用其他服务，请填写该服务的 API 根地址（可含 `/v1`）和可用模型名。
API Key 只在本机网页填写，不要发到聊天或提交到仓库。

其他电脑重新安装：建议 Python 3.11，代码兼容 Python 3.9+。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-c.txt
```

BGE 模型首次需要联网下载；项目自带文档向量，但不含编码查询所需模型权重。
关键词 sparse 模式无需 BGE 权重。中文提问先通过 API 翻译检索词，最终按中文问题生成中文答案。

## Python 接口

```python
import sys
sys.path.insert(0, 'src')
from generation.answer import answer
result = answer('怎么反转一个列表')
```

接口保持 `answer(query) -> {answer: str, citations: list[str], refused: bool}`。
命令行可设置 `RAG_API_KEY`、`RAG_BASE_URL`、`RAG_MODEL`，然后：

```sh
.venv/bin/python src/generation/answer.py 'How to reverse a list?' --mode sparse
```

使用标准库 HTTP 客户端调用兼容 Chat Completions 的接口，不依赖 OpenAI SDK。

## 行为与实现

- 使用 B 的 `retrieve()`，支持 sparse/dense/hybrid 与 small/medium/large。
- B 返回的片段截断为 400 字，C 按 chunk_id 从完整 chunks.jsonl 补齐正文。
- 模型仅依据检索正文作答，输出 JSON。问题及正文明确视作数据。
- 模型主动拒答、无检索结果、输出损坏、无引用、编造引用时，返回 refused=True。
- API 连接、鉴权、额度等故障抛出 LLMError，界面显示连接错误，不冒充资料不足。
- 引用按首次出现的顺序去重，只允许本次检索中的编号；界面同时展示摘要和官方原文链接。
- `cache/answers/` 保存答案；缓存键包含模型、地址、问题、模式、档位、k、提示词版本和实际正文。
- 缓存读取时再次检查引用；写入通过原子替换避免并发写坏。可取消侧栏缓存选项。
- 缓存仍需运行检索；中文问题仍需翻译调用。缓存不保存 API Key。

## 拒答策略与验收边界

没有虚构通用于 BM25/BGE/RRF 的分数阈值：三类分数尺度不同，RRF 高分也不能证明问题相关。
本版由模型判断相关性和资料是否充足，并在引用校验失败时严格拒答。
若报告要求基于分数的阈值，应使用真实相关/无关问题分别按模式及档位标定，再增加阈值门控。
引用校验验证编号合法性，**不证明每句话受到原文支持**。最终须用真实模型测试答案质量和拒答率。

当前自动测试使用假模型响应验证代码分支，不作为真实模型准确率证据。没有 API Key 时，中文回答和无关问题拒答的真实验收尚不能完成。

## 验证

```sh
.venv/bin/python -m unittest discover -s tests -v
```

`test_generation.py` 验证契约、完整上下文、合法/伪造引用、损坏输出、缓存隔离与失效、中文翻译和无 Key 错误。
`test_app.py` 使用真实 BM25 检索及显式模拟的模型响应，验证网页可操作、答案及引用展示、界面重绘。

真实验收建议在网页依次输入：
1. 怎么反转一个列表？——中文答案带引用，并可打开原文。
2. How do I append an item to a list?——重复提问第二次应显示缓存命中。
3. 今天天气怎么样？——应提示资料不足并拒答。
4. 在三种模式、三种数据档位间切换，确认配置切换有效。

## 文件

- `answer.py`：检索、生成、拒答、缓存与契约入口。
- `llm.py`：模型客户端与可读错误。
- `citation.py`：引用抽取、校验及清理。
- `../app/streamlit_app.py`：本地网页。
- `../../tests/`：自动测试。

接口资料：[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[Streamlit 输入框](https://docs.streamlit.io/develop/api-reference/widgets/st.text_input)、[链接按钮](https://docs.streamlit.io/develop/api-reference/widgets/st.link_button)。
